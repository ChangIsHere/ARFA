from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

from src.common.config import load_simple_yaml
from src.common.utils import ensure_parent, read_jsonl, write_csv, write_jsonl
from src.phase1_5_shadow.json_model_client import OllamaStructuredClient
from src.phase1_5_shadow.shadow_agent import ShadowReActAgent, ShadowRun
from src.phase2_baseline.intercode_docker_environment import InterCodeDockerEnvironment


ALLOWED_SPLITS = {"shadow_development", "shadow_validation", "shadow_test"}


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_split_tasks(task_path: str, split_path: str, split: str, limit: int) -> list[dict[str, Any]]:
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"Phase 1.5 refuses split '{split}'; allowed: {sorted(ALLOWED_SPLITS)}")
    payload = json.loads(Path(split_path).read_text(encoding="utf-8"))
    selected = set(payload["splits"][split])
    tasks = [task for task in read_jsonl(task_path) if str(task["task_id"]) in selected]
    expected = int(payload["sizes"][split])
    if len(tasks) != expected:
        raise ValueError(f"Split {split} expected {expected} tasks, found {len(tasks)}")
    return tasks[:limit] if limit > 0 else tasks


def _build_environment(task: dict[str, Any], config: dict[str, Any]) -> InterCodeDockerEnvironment:
    environment = config["environment"]
    return InterCodeDockerEnvironment(
        task=task,
        image_prefix=str(environment["image_prefix"]),
        command_timeout_seconds=int(environment.get("command_timeout_seconds", 20)),
        max_observation_chars=int(environment.get("max_observation_chars", 4000)),
        container_memory=str(environment.get("container_memory", "1g")),
        network_mode=str(environment.get("network_mode", "bridge")),
    )


def _summary(runs: list[ShadowRun], config: dict[str, Any], split: str, expected_tasks: int, provenance: dict[str, Any]) -> dict[str, Any]:
    executed = sum(run.executed_steps for run in runs)
    return {
        "phase": "phase1_5_shadow",
        "mode": "always_reason_shadow_only",
        "router_controls_execution": False,
        "paper_usable": False,
        "paper_usable_reason": "Human-blind labels and held-out residual analysis are not complete.",
        "model_name": config["shadow"]["model_name"],
        "split": split,
        "expected_task_count": expected_tasks,
        "task_count": len(runs),
        "executed_steps": executed,
        "success_count": sum(run.success for run in runs),
        "shadow_fast_candidates": sum(run.shadow_fast_candidates for run in runs),
        "shadow_fast_candidate_rate": sum(run.shadow_fast_candidates for run in runs) / executed if executed else 0.0,
        "mean_expectation_quality": (
            sum(step.expectation_quality for run in runs for step in run.steps if step.action_executed) / executed
            if executed
            else 0.0
        ),
        "parse_error_count": sum(run.parse_error_count for run in runs),
        "total_model_calls": sum(run.model_calls for run in runs),
        "total_tokens": sum(run.total_tokens or 0 for run in runs),
        "total_task_wall_seconds": sum(run.total_task_wall_seconds for run in runs),
        "protocol": {
            "temperature": config["shadow"]["temperature"],
            "max_tokens": config["shadow"]["max_tokens"],
            "context_length": config["shadow"]["context_length"],
            "max_steps": config["shadow"]["max_steps"],
            "shadow_threshold": config["shadow"]["provisional_threshold"],
            "minimum_expectation_quality": config["shadow"]["minimum_expectation_quality"],
            "maximum_identical_action_occurrences": config["shadow"]["maximum_identical_action_occurrences"],
        },
        "reproducibility": provenance,
    }


def _write_outputs(runs: list[ShadowRun], summary: dict[str, Any], output_dir: Path) -> None:
    write_jsonl(output_dir / "traces.jsonl", [run.to_dict() for run in runs])
    ensure_parent(output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_csv(
        output_dir / "metrics.csv",
        [
            {
                "task_id": run.task_id,
                "split": run.split,
                "model_name": run.model_name,
                "success": run.success,
                "model_calls": run.model_calls,
                "executed_steps": run.executed_steps,
                "shadow_fast_candidates": run.shadow_fast_candidates,
                "total_tokens": run.total_tokens,
                "total_task_wall_seconds": run.total_task_wall_seconds,
                "parse_error_count": run.parse_error_count,
            }
            for run in runs
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Phase 1.5 pre-execution expectations in shadow mode.")
    parser.add_argument("--config", default="config/phase1_5_shadow.yaml")
    parser.add_argument("--split", default="shadow_development", choices=sorted(ALLOWED_SPLITS))
    parser.add_argument("--task-limit", type=int, default=0)
    parser.add_argument("--model-name")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = copy.deepcopy(load_simple_yaml(args.config))
    if args.model_name:
        config["shadow"]["model_name"] = args.model_name
    output_dir = Path(args.output_dir or config["outputs"]["result_dir"])
    tasks = _load_split_tasks(config["data"]["task_path"], config["data"]["split_path"], args.split, args.task_limit)
    expected_tasks = len(tasks)
    provenance = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "config_sha256": _sha256(args.config),
        "task_manifest_sha256": _sha256(config["data"]["task_path"]),
        "split_manifest_sha256": _sha256(config["data"]["split_path"]),
        "prompt_sha256": _sha256("src/phase1_5_shadow/prompts.py"),
        "agent_sha256": _sha256("src/phase1_5_shadow/shadow_agent.py"),
        "json_client_sha256": _sha256("src/phase1_5_shadow/json_model_client.py"),
    }

    client = OllamaStructuredClient.from_config(config["shadow"])
    agent = ShadowReActAgent(
        client=client,
        model_name=str(config["shadow"]["model_name"]),
        split=args.split,
        max_steps=int(config["shadow"]["max_steps"]),
        shadow_threshold=float(config["shadow"]["provisional_threshold"]),
        minimum_expectation_quality=float(config["shadow"]["minimum_expectation_quality"]),
        maximum_identical_action_occurrences=int(config["shadow"]["maximum_identical_action_occurrences"]),
    )

    runs: list[ShadowRun] = []
    trace_path = output_dir / "traces.jsonl"
    if args.resume and trace_path.exists():
        runs = [ShadowRun.from_dict(row) for row in read_jsonl(trace_path)]
        mismatched = [run for run in runs if run.model_name != config["shadow"]["model_name"] or run.split != args.split]
        if mismatched:
            raise SystemExit("Refusing to mix models or splits in one shadow trace file")

    completed = {run.task_id for run in runs}
    for task in tasks:
        if str(task["task_id"]) in completed:
            continue
        task_started = time.perf_counter()
        env = _build_environment(task, config)
        try:
            run = agent.run_task(task, env)
        finally:
            env.close()
        run = ShadowRun(**{**run.to_dict(), "steps": run.steps, "total_task_wall_seconds": time.perf_counter() - task_started})
        runs.append(run)
        _write_outputs(runs, _summary(runs, config, args.split, expected_tasks, provenance), output_dir)

    summary = _summary(runs, config, args.split, expected_tasks, provenance)
    _write_outputs(runs, summary, output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
