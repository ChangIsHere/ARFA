from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

from src.common.config import load_simple_yaml
from src.common.utils import ensure_parent, read_jsonl, write_csv, write_jsonl
from src.phase1_5_shadow.json_model_client import OllamaStructuredClient
from src.phase2_baseline.intercode_docker_environment import InterCodeDockerEnvironment
from src.phase3_arfa.arfa_agent import ARFAAgent, ARFARun
from src.phase3_arfa.fast_track import FastTrack
from src.phase3_arfa.learned_observation import LearnedObservationScorer
from src.phase3_arfa.router import ResidualRouter
from src.phase3_arfa.slow_track import SlowTrack


SPLIT_NAMES = {"development", "final"}
POLICY_NAMES = ResidualRouter.POLICIES


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_value(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()


def _model_metadata(model_name: str, artifact_identity: str | None = None) -> dict[str, str]:
    if artifact_identity:
        return {
            "model_name": model_name,
            "artifact_identity": artifact_identity,
        }
    modelfile = subprocess.run(
        ["ollama", "show", model_name, "--modelfile"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {
        "model_name": model_name,
        "modelfile_sha256": hashlib.sha256(modelfile.encode("utf-8")).hexdigest(),
    }


def _docker_digests(tasks: list[dict[str, Any]], image_prefix: str) -> dict[str, str]:
    images = sorted({f"{image_prefix}{int(task['filesystem_version'])}" for task in tasks})
    return {
        image: subprocess.run(
            ["docker", "image", "inspect", "--format={{.Id}}", image],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        for image in images
    }


def phase3_status(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("locked", True):
        return {
            "ready": False,
            "stage": config.get("stage", "phase3_development"),
            "reason": config.get("reason", "Phase 3 is locked."),
        }
    return {
        "ready": True,
        "protocol_version": config.get("protocol_version"),
        "stage": config.get("stage", "phase3_development"),
        "scope": config.get("scope", "online_dual_model_evaluation"),
        "formal_claims_enabled": bool(config.get("formal_claims_enabled", False)),
        "use_phase3_final_tasks": bool(config.get("use_phase3_final_tasks", False)),
        "router": config.get("router", {}),
        "status": "controller_ready",
        "controller_implemented": True,
        "execution_started": False,
    }


def _load_split_tasks(config: dict[str, Any], split: str, limit: int) -> list[dict[str, Any]]:
    split_payload = json.loads(Path(config["data"]["split_path"]).read_text(encoding="utf-8"))
    split_map = split_payload["splits"]
    if split == "development":
        selected = set().union(
            *(set(split_map[name]) for name in ("shadow_development", "shadow_validation", "shadow_test"))
        )
        expected = int(config["data"]["development_tasks"])
    elif split == "final":
        selected = set(split_map["phase3_final"])
        expected = int(config["data"]["final_tasks"])
    else:
        raise ValueError(f"Unknown Phase 3 split: {split}")
    tasks = [task for task in read_jsonl(config["data"]["task_path"]) if str(task["task_id"]) in selected]
    if len(tasks) != expected or {str(task["task_id"]) for task in tasks} != selected:
        raise ValueError(f"Phase 3 {split} split expected exactly {expected} unique tasks")
    return tasks[:limit] if limit > 0 else tasks


def _validate_scope(config: dict[str, Any], split: str, limit: int, threshold_override: float | None) -> None:
    gate_path = Path(config["engineering_gate_result"])
    if not gate_path.is_file() or not bool(json.loads(gate_path.read_text(encoding="utf-8")).get("passed")):
        raise ValueError("Phase 3 engineering prerequisite report is missing or did not pass")
    if split == "final":
        if not bool(config.get("use_phase3_final_tasks", False)):
            raise ValueError(
                "Final tasks remain locked. Calibrate on development, freeze the controller, "
                "then set use_phase3_final_tasks: true."
            )
        if limit:
            raise ValueError("Formal final evaluation must run all reserved tasks; task-limit is not allowed")
        if threshold_override is not None:
            raise ValueError("The frozen final evaluation does not accept a threshold override")


def _client(
    config: dict[str, Any],
    model_name: str,
) -> OllamaStructuredClient:
    settings = dict(config["model"])
    settings["model_name"] = model_name
    return OllamaStructuredClient.from_config(settings)


def _provenance(
    config_path: Path,
    config: dict[str, Any],
    pair_name: str,
    policy: str,
    split: str,
    threshold: float,
    tasks: list[dict[str, Any]],
    trajectory_workers: int,
) -> dict[str, Any]:
    pair = config["pairs"][pair_name]
    source_files = [
        "src/phase3_arfa/arfa_agent.py",
        "src/phase3_arfa/router.py",
        "src/phase3_arfa/learned_observation.py",
        "src/phase3_arfa/calibrate_observation_control.py",
        "src/phase1_5_shadow/observation_baseline.py",
        "src/phase3_arfa/prompts.py",
        "src/phase1_5_shadow/residual.py",
        "src/phase1_5_shadow/json_model_client.py",
        "src/phase2_baseline/intercode_docker_environment.py",
    ]
    result: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_worktree_dirty": bool(_git_value("status", "--porcelain")),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "task_manifest_sha256": _sha256(config["data"]["task_path"]),
        "split_manifest_sha256": _sha256(config["data"]["split_path"]),
        "selected_task_ids_sha256": hashlib.sha256(
            "\n".join(str(task["task_id"]) for task in tasks).encode("utf-8")
        ).hexdigest(),
        "source_sha256": {path: _sha256(path) for path in source_files},
        "pair_name": pair_name,
        "policy": policy,
        "split": split,
        "residual_threshold": threshold,
        "trajectory_workers": trajectory_workers,
        "fast_model": _model_metadata(
            str(pair["fast_model"]),
            pair.get("fast_model_artifact_identity"),
        ),
        "slow_model": _model_metadata(
            str(pair["slow_model"]),
            pair.get("slow_model_artifact_identity"),
        ),
        "docker_image_digests": _docker_digests(tasks, str(config["environment"]["image_prefix"])),
    }
    if policy == "learned_observation_only":
        learned = config["router"]["learned_observation_only"]
        result["learned_observation_training"] = {
            "annotation_sha256": _sha256(learned["annotation_path"]),
            "source_map_sha256": _sha256(learned["source_map_path"]),
            "threshold": float(learned["threshold"]),
            "seed": int(learned.get("seed", 1507)),
            "calibration_sha256": _sha256(learned["calibration_path"]),
        }
    fingerprint_payload = {key: value for key, value in result.items() if key not in {"git_commit", "git_worktree_dirty"}}
    serialized = json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    result["fingerprint"] = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return result


def _build_environment(task: dict[str, Any], config: dict[str, Any]) -> InterCodeDockerEnvironment:
    env = config["environment"]
    return InterCodeDockerEnvironment(
        task=task,
        image_prefix=str(env["image_prefix"]),
        command_timeout_seconds=int(env["command_timeout_seconds"]),
        max_observation_chars=int(env["max_observation_chars"]),
        container_memory=str(env["container_memory"]),
        network_mode=str(env["network_mode"]),
    )


def _summary(
    runs: list[ARFARun],
    config: dict[str, Any],
    pair_name: str,
    policy: str,
    split: str,
    threshold: float,
    expected_tasks: int,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    success_count = sum(run.success for run in runs)
    total_wall = sum(run.total_task_wall_seconds for run in runs)
    total_calls = sum(run.model_calls for run in runs)
    return {
        "phase": "phase3_arfa",
        "protocol_version": config["protocol_version"],
        "stage": config["stage"],
        "pair_name": pair_name,
        "policy": policy,
        "experiment_split": split,
        "paper_primary_evaluation": split == "final" and len(runs) == expected_tasks,
        "expected_task_count": expected_tasks,
        "task_count": len(runs),
        "success_count": success_count,
        "success_rate": success_count / len(runs) if runs else 0.0,
        "fast_model": config["pairs"][pair_name]["fast_model"],
        "slow_model": config["pairs"][pair_name]["slow_model"],
        "residual_method": config["router"]["residual_method"],
        "residual_threshold": threshold,
        "trajectory_workers": provenance["trajectory_workers"],
        "total_model_calls": total_calls,
        "fast_model_calls": sum(run.fast_model_calls for run in runs),
        "slow_model_calls": sum(run.slow_model_calls for run in runs),
        "fast_call_rate": sum(run.fast_model_calls for run in runs) / total_calls if total_calls else 0.0,
        "total_tokens": sum(run.total_tokens or 0 for run in runs),
        "fast_model_tokens": sum(run.fast_model_tokens or 0 for run in runs),
        "slow_model_tokens": sum(run.slow_model_tokens or 0 for run in runs),
        "total_model_latency_seconds": sum(run.total_model_latency_seconds for run in runs),
        "total_command_latency_seconds": sum(run.total_command_latency_seconds for run in runs),
        "total_router_latency_seconds": sum(run.total_router_latency_seconds for run in runs),
        "total_task_wall_seconds": total_wall,
        "avg_task_wall_seconds": total_wall / len(runs) if runs else 0.0,
        "wall_seconds_per_success": total_wall / success_count if success_count else None,
        "parse_error_count": sum(run.parse_error_count for run in runs),
        "model_request_error_count": sum(run.termination_reason == "model_request_error" for run in runs),
        "max_steps_reached_count": sum(run.max_steps_reached for run in runs),
        "provenance": provenance,
    }


def _write_outputs(runs: list[ARFARun], summary: dict[str, Any], output_dir: Path) -> None:
    write_jsonl(output_dir / "traces.jsonl", [run.to_dict() for run in runs])
    ensure_parent(output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_csv(
        output_dir / "metrics.csv",
        [
            {
                "task_id": run.task_id,
                "experiment_split": run.experiment_split,
                "pair_name": run.pair_name,
                "policy": run.policy,
                "trajectory_workers": summary["trajectory_workers"],
                "success": run.success,
                "model_calls": run.model_calls,
                "fast_model_calls": run.fast_model_calls,
                "slow_model_calls": run.slow_model_calls,
                "fast_model_tokens": run.fast_model_tokens,
                "slow_model_tokens": run.slow_model_tokens,
                "total_tokens": run.total_tokens,
                "model_latency_seconds": run.total_model_latency_seconds,
                "command_latency_seconds": run.total_command_latency_seconds,
                "router_latency_seconds": run.total_router_latency_seconds,
                "total_task_wall_seconds": run.total_task_wall_seconds,
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


def _validate_resume(
    runs: list[ARFARun], summary_path: Path, expected_ids: set[str], provenance_fingerprint: str
) -> None:
    if not summary_path.is_file():
        raise ValueError("Resume trace exists without summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("provenance", {}).get("fingerprint") != provenance_fingerprint:
        raise ValueError("Resume artifacts do not match the current code, models, config, or split")
    ids = [run.task_id for run in runs]
    if len(ids) != len(set(ids)) or not set(ids) <= expected_ids:
        raise ValueError("Resume trace contains duplicate or unexpected task IDs")
    if any(run.provenance_fingerprint != provenance_fingerprint for run in runs):
        raise ValueError("Resume trace contains stale per-task provenance")


def _iter_completed(tasks: list[dict[str, Any]], worker: Any, workers: int):
    """Yield completed task results with a fixed upper bound on concurrency."""
    if workers < 1:
        raise ValueError("trajectory workers must be positive")
    if workers == 1:
        for task in tasks:
            yield worker(task)
        return
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="arfa-task") as executor:
        futures = [executor.submit(worker, task) for task in tasks]
        for future in as_completed(futures):
            yield future.result()


def _execute_task(
    task: dict[str, Any],
    config: dict[str, Any],
    agent: ARFAAgent,
    provenance_fingerprint: str,
) -> ARFARun:
    task_started = time.perf_counter()
    env = _build_environment(task, config)
    try:
        run = agent.run_task(task, env)
    finally:
        env.close()
    return replace(
        run,
        total_task_wall_seconds=time.perf_counter() - task_started,
        provenance_fingerprint=provenance_fingerprint,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the online ARFA Phase 3 dual-model controller.")
    parser.add_argument("--config", type=Path, default=Path("config/phase3_arfa.yaml"))
    parser.add_argument("--pair", default="qwen")
    parser.add_argument("--policy", choices=sorted(POLICY_NAMES), default="residual")
    parser.add_argument("--split", choices=sorted(SPLIT_NAMES), default="development")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--task-limit", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    config = copy.deepcopy(load_simple_yaml(args.config))
    if args.pair not in config.get("pairs", {}):
        available = ", ".join(sorted(config.get("pairs", {})))
        raise SystemExit(f"Unknown pair {args.pair!r} for {args.config}; available: {available}")
    status = phase3_status(config)
    if args.status:
        print(json.dumps(status, indent=2))
        return
    if not status["ready"]:
        raise SystemExit(status["reason"])
    try:
        _validate_scope(config, args.split, args.task_limit, args.threshold)
        tasks = _load_split_tasks(config, args.split, args.task_limit)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    threshold = float(args.threshold if args.threshold is not None else config["router"]["residual_threshold"])
    fast_call_budget = float(config["router"].get("fast_call_budget", 0.0))
    random_seed = int(config["router"].get("random_seed", 1507))
    learned_config = config["router"].get("learned_observation_only", {})
    learned_scorer = None
    if args.policy == "learned_observation_only":
        calibration = json.loads(Path(learned_config["calibration_path"]).read_text(encoding="utf-8"))
        if float(calibration["threshold"]) != float(learned_config["threshold"]):
            raise SystemExit("Learned observation threshold does not match the saved validation calibration")
        if calibration["annotation_sha256"] != _sha256(learned_config["annotation_path"]):
            raise SystemExit("Learned observation annotation source changed since validation calibration")
        if calibration["source_map_sha256"] != _sha256(learned_config["source_map_path"]):
            raise SystemExit("Learned observation source map changed since validation calibration")
        learned_scorer = LearnedObservationScorer(
            str(learned_config["annotation_path"]),
            str(learned_config["source_map_path"]),
            int(learned_config.get("seed", 1507)),
        )
    workers = int(
        args.workers
        if args.workers is not None
        else config.get("execution", {}).get("trajectory_workers", 1)
    )
    if workers < 1:
        raise SystemExit("--workers must be positive")
    pair = config["pairs"][args.pair]
    if args.output_dir:
        output_dir = args.output_dir
    elif args.task_limit:
        output_dir = (
            Path(config["outputs"]["result_root"]) / "smoke" / args.pair / args.policy / f"limit_{args.task_limit}"
        )
    else:
        output_dir = Path(config["outputs"]["result_root"]) / args.split / args.pair / args.policy
    provenance = _provenance(
        args.config,
        config,
        args.pair,
        args.policy,
        args.split,
        threshold,
        tasks,
        workers,
    )
    expected_tasks = len(tasks)
    router = ResidualRouter(
        policy=args.policy,
        residual_threshold=threshold,
        observation_threshold=float(config["router"]["observation_threshold"]),
        minimum_expectation_quality=float(config["router"]["minimum_expectation_quality"]),
        max_consecutive_fast_calls=int(config["router"]["max_consecutive_fast_calls"]),
        fast_call_budget=fast_call_budget,
        random_seed=random_seed,
        learned_observation_scorer=learned_scorer,
        learned_observation_threshold=float(learned_config.get("threshold", 0.5)),
    )
    agent = ARFAAgent(
        fast_track=FastTrack(
            str(pair["fast_model"]),
            _client(config, str(pair["fast_model"])),
        ),
        slow_track=SlowTrack(
            str(pair["slow_model"]),
            _client(config, str(pair["slow_model"])),
        ),
        router=router,
        pair_name=args.pair,
        experiment_split=args.split,
        max_steps=int(config["agent"]["max_steps"]),
        maximum_identical_action_occurrences=int(config["agent"]["maximum_identical_action_occurrences"]),
        max_history_turns=int(config["agent"].get("max_history_turns", 2)),
    )

    runs: list[ARFARun] = []
    trace_path = output_dir / "traces.jsonl"
    if args.resume and trace_path.exists():
        runs = [ARFARun.from_dict(row) for row in read_jsonl(trace_path)]
        try:
            _validate_resume(
                runs,
                output_dir / "summary.json",
                {str(task["task_id"]) for task in tasks},
                provenance["fingerprint"],
            )
        except ValueError as exc:
            raise SystemExit(f"Refusing stale Phase 3 resume: {exc}") from exc

    completed = {run.task_id for run in runs}
    pending = [task for task in tasks if str(task["task_id"]) not in completed]
    task_order = {str(task["task_id"]): index for index, task in enumerate(tasks)}
    worker = lambda task: _execute_task(task, config, agent, provenance["fingerprint"])
    for run in _iter_completed(pending, worker, workers):
        runs.append(run)
        runs.sort(key=lambda item: task_order[item.task_id])
        checkpoint = _summary(runs, config, args.pair, args.policy, args.split, threshold, expected_tasks, provenance)
        _write_outputs(runs, checkpoint, output_dir)
        print(
            f"[phase3] {args.pair}/{args.policy}/{args.split}: {len(runs)}/{expected_tasks} "
            f"{run.task_id} success={run.success} fast={run.fast_model_calls} "
            f"slow={run.slow_model_calls} wall={run.total_task_wall_seconds:.1f}s",
            flush=True,
        )

    summary = _summary(runs, config, args.pair, args.policy, args.split, threshold, expected_tasks, provenance)
    _write_outputs(runs, summary, output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
