from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from src.common.config import load_simple_yaml
from src.common.utils import ensure_parent, read_jsonl, write_csv, write_jsonl
from src.phase2_baseline.model_client import ModelRequestError, OpenAICompatibleClient, ScriptedClient
from src.phase2_baseline.react_agent import Phase2Run, Phase2Step, ReActBaselineAgent
from src.phase2_baseline.task_loader import load_tasks
from src.phase2_baseline.terminal_environment import LocalTerminalEnvironment


def _build_environment(task: dict[str, Any], config: dict[str, Any]):
    environment = config["environment"]
    backend = str(environment.get("backend", "local_fixture"))
    if backend == "intercode_docker":
        from src.phase2_baseline.intercode_docker_environment import InterCodeDockerEnvironment

        return InterCodeDockerEnvironment(
            task=task,
            image_prefix=str(environment.get("image_prefix", "arfa/intercode-nl2bash:fs")),
            command_timeout_seconds=int(environment.get("command_timeout_seconds", 20)),
            max_observation_chars=int(environment.get("max_observation_chars", 4000)),
            container_memory=str(environment.get("container_memory", "1g")),
            network_mode=str(environment.get("network_mode", "bridge")),
        )
    if backend != "local_fixture":
        raise ValueError(f"Unknown Phase 2 environment backend: {backend}")
    return LocalTerminalEnvironment(
        workspace_root=environment["workspace_root"],
        task_id=str(task["task_id"]),
        command_timeout_seconds=int(environment.get("command_timeout_seconds", 20)),
        max_observation_chars=int(environment.get("max_observation_chars", 4000)),
        max_workspace_bytes=int(environment.get("max_workspace_bytes", 10_000_000)),
    )


def _build_summary(runs: list[Phase2Run], config: dict[str, Any], dry_run: bool) -> dict[str, Any]:
    total = len(runs)
    expected_total = int(config["data"].get("expected_task_count", 0))
    successes = sum(1 for run in runs if run.success)
    token_values = [run.total_tokens for run in runs if run.total_tokens is not None]
    reward_values = [float(run.evaluator["reward"]) for run in runs if run.evaluator.get("reward") is not None]
    return {
        "phase": "phase2_baseline",
        "dry_run": dry_run,
        "paper_protocol": not dry_run and bool(config["data"].get("paper_usable", False)),
        "paper_usable": not dry_run
        and bool(config["data"].get("paper_usable", False))
        and expected_total > 0
        and total == expected_total,
        "expected_task_count": expected_total or None,
        "dataset_role": config["data"].get("dataset_role", "unspecified"),
        "model_name": config["baseline"]["model_name"] if not dry_run else "scripted_harness_validator",
        "task_count": total,
        "success_count": successes,
        "success_rate": successes / total if total else 0.0,
        "total_model_calls": sum(run.model_calls for run in runs),
        "avg_model_calls_per_task": sum(run.model_calls for run in runs) / total if total else 0.0,
        "total_model_latency_seconds": sum(run.total_model_latency_seconds for run in runs),
        "total_command_latency_seconds": sum(run.total_command_latency_seconds for run in runs),
        "total_task_wall_seconds": sum(run.total_task_wall_seconds for run in runs),
        "avg_task_wall_seconds": sum(run.total_task_wall_seconds for run in runs) / total if total else 0.0,
        "total_tokens": sum(token_values) if token_values else None,
        "average_reward": sum(reward_values) / len(reward_values) if reward_values else None,
        "max_steps_reached_count": sum(run.max_steps_reached for run in runs),
        "parse_error_count": sum(run.parse_error_count for run in runs),
        "nonzero_exit_count": sum(run.nonzero_exit_count for run in runs),
        "repeated_action_count": sum(run.repeated_action_count for run in runs),
        "always_reason_after_observation": bool(config["baseline"].get("always_reason_after_observation", True)),
        "protocol": {
            "temperature": config["baseline"].get("temperature"),
            "max_tokens_per_call": config["baseline"].get("max_tokens"),
            "context_length": config["baseline"].get("context_length"),
            "max_steps": config["baseline"].get("max_steps"),
            "task_path": config["data"].get("task_path"),
            "environment_backend": config["environment"].get("backend"),
            "network_mode": config["environment"].get("network_mode"),
        },
        "reproducibility": config.get("reproducibility", {}),
    }


def _file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _build_model_error_run(
    task: dict[str, Any], env: Any, error: ModelRequestError, elapsed_seconds: float
) -> Phase2Run:
    message = str(error)
    evaluator = env.evaluate(task, trace_text="", final_answer="")
    step = Phase2Step(
        step_id=1,
        thought="Model request failed after retries.",
        action="",
        expected_outcome="",
        observation=message,
        exit_code=None,
        model_latency_seconds=elapsed_seconds,
        command_latency_seconds=0.0,
        prompt_tokens=None,
        completion_tokens=None,
        total_tokens=None,
        parse_error="model_request_failed",
    )
    return Phase2Run(
        task_id=str(task["task_id"]),
        instruction=str(task["instruction"]),
        benchmark=str(task.get("benchmark", "")),
        success=bool(evaluator["success"]),
        evaluator=evaluator,
        final_answer="",
        steps=[step],
        model_calls=1,
        total_model_latency_seconds=elapsed_seconds,
        total_command_latency_seconds=0.0,
        total_tokens=None,
        termination_reason="model_request_error",
        max_steps_reached=False,
        parse_error_count=1,
        nonzero_exit_count=0,
        repeated_action_count=0,
        agent_wall_seconds=elapsed_seconds,
        total_task_wall_seconds=elapsed_seconds,
    )


def _write_outputs(runs: list[Phase2Run], summary: dict[str, Any], config: dict[str, Any]) -> None:
    outputs = config["outputs"]
    write_jsonl(outputs["traces_path"], [run.to_dict() for run in runs])
    summary_path = ensure_parent(outputs["summary_path"])
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(
        outputs["metrics_path"],
        [
            {
                "task_id": run.task_id,
                "benchmark": run.benchmark,
                "success": run.success,
                "model_calls": run.model_calls,
                "total_tokens": run.total_tokens,
                "model_latency_seconds": run.total_model_latency_seconds,
                "command_latency_seconds": run.total_command_latency_seconds,
                "agent_wall_seconds": run.agent_wall_seconds,
                "total_task_wall_seconds": run.total_task_wall_seconds,
                "reward": run.evaluator.get("reward"),
                "termination_reason": run.termination_reason,
                "max_steps_reached": run.max_steps_reached,
                "parse_error_count": run.parse_error_count,
                "nonzero_exit_count": run.nonzero_exit_count,
                "repeated_action_count": run.repeated_action_count,
                "final_answer": run.final_answer,
            }
            for run in runs
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 2 standard ReAct baseline.")
    parser.add_argument("--config", default="config/phase2_baseline.yaml")
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate the harness with a scripted client; not paper-usable.")
    args = parser.parse_args()

    config = copy.deepcopy(load_simple_yaml(args.config))
    if config.get("locked"):
        raise SystemExit(f"Phase 2 is locked: {config.get('reason')}")

    if args.model_name:
        config["baseline"]["model_name"] = args.model_name
    if args.output_dir:
        output_dir = Path(args.output_dir)
        config["outputs"]["result_dir"] = str(output_dir)
        config["outputs"]["traces_path"] = str(output_dir / "traces.jsonl")
        config["outputs"]["summary_path"] = str(output_dir / "summary.json")
        config["outputs"]["metrics_path"] = str(output_dir / "metrics.csv")

    config["reproducibility"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "config_path": args.config,
        "config_sha256": _file_sha256(args.config),
        "task_manifest_sha256": _file_sha256(config["data"]["task_path"]),
        "prompt_sha256": _file_sha256("src/phase2_baseline/prompts.py"),
        "agent_sha256": _file_sha256("src/phase2_baseline/react_agent.py"),
        "environment_sha256": _file_sha256("src/phase2_baseline/intercode_docker_environment.py")
        if config["environment"].get("backend") == "intercode_docker"
        else _file_sha256("src/phase2_baseline/terminal_environment.py"),
    }

    task_limit = args.task_limit if args.task_limit is not None else int(config["data"].get("task_limit", 0))
    tasks = load_tasks(config["data"]["task_path"], limit=task_limit)
    client = ScriptedClient() if args.dry_run else OpenAICompatibleClient.from_config(config["baseline"])
    agent = ReActBaselineAgent(client=client, max_steps=int(config["baseline"].get("max_steps", 12)))

    runs: list[Phase2Run] = []
    traces_path = Path(config["outputs"]["traces_path"])
    if args.resume and traces_path.exists():
        summary_path = Path(config["outputs"]["summary_path"])
        if not summary_path.exists():
            raise SystemExit(f"Refusing to resume traces without a summary: {traces_path}")
        prior_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        expected_model = "scripted_harness_validator" if args.dry_run else config["baseline"]["model_name"]
        if prior_summary.get("model_name") != expected_model:
            raise SystemExit(
                f"Refusing to mix model runs in {traces_path}: "
                f"found {prior_summary.get('model_name')}, requested {expected_model}"
            )
        prior_repro = prior_summary.get("reproducibility", {})
        for key in ("config_sha256", "task_manifest_sha256", "prompt_sha256", "agent_sha256", "environment_sha256"):
            if prior_repro.get(key) != config["reproducibility"][key]:
                raise SystemExit(f"Refusing to resume {traces_path}: {key} changed")
        runs = [Phase2Run.from_dict(row) for row in read_jsonl(traces_path)]
        task_ids = {str(task["task_id"]) for task in tasks}
        prior_ids = [run.task_id for run in runs]
        if len(prior_ids) != len(set(prior_ids)) or not set(prior_ids) <= task_ids:
            raise SystemExit(f"Refusing to resume {traces_path}: duplicate or unexpected task IDs")
    completed_task_ids = {run.task_id for run in runs}
    for task in tasks:
        if str(task["task_id"]) in completed_task_ids:
            continue
        task_started = time.perf_counter()
        env = _build_environment(task, config)
        try:
            try:
                run = agent.run_task(task, env)
            except ModelRequestError as exc:
                run = _build_model_error_run(task, env, exc, time.perf_counter() - task_started)
        finally:
            env.close()
        run = replace(run, total_task_wall_seconds=time.perf_counter() - task_started)
        runs.append(run)
        checkpoint = _build_summary(runs, config, dry_run=args.dry_run)
        _write_outputs(runs, checkpoint, config)
        print(
            f"[phase2] {config['baseline']['model_name']}: {len(runs)}/{len(tasks)} "
            f"{run.task_id} success={run.success} wall={run.total_task_wall_seconds:.1f}s",
            flush=True,
        )

    summary = _build_summary(runs, config, dry_run=args.dry_run)
    _write_outputs(runs, summary, config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
