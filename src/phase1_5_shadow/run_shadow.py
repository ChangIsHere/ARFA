from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from src.common.config import load_simple_yaml
from src.common.utils import ensure_parent, read_jsonl, write_csv, write_jsonl
from src.phase1_5_shadow.json_model_client import OllamaStructuredClient
from src.phase1_5_shadow.freeze_protocol import verify_freeze_manifest
from src.phase1_5_shadow.shadow_agent import ShadowReActAgent, ShadowRun
from src.phase2_baseline.intercode_docker_environment import InterCodeDockerEnvironment


ALLOWED_SPLITS = {"shadow_development", "shadow_validation", "shadow_test"}


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("numpy", "scikit-learn", "sentence-transformers", "PyYAML"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _ollama_model_metadata(model_name: str) -> dict[str, str | None]:
    modelfile = subprocess.run(
        ["ollama", "show", model_name, "--modelfile"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    version = subprocess.run(
        ["ollama", "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    match = re.search(r"^FROM .*sha256[-:]([0-9a-f]{64})$", modelfile, flags=re.MULTILINE)
    return {
        "model_name": model_name,
        "base_blob_sha256": match.group(1) if match else None,
        "modelfile_sha256": hashlib.sha256(modelfile.encode("utf-8")).hexdigest(),
        "ollama_version": version,
    }


def _docker_image_digests(tasks: list[dict[str, Any]], image_prefix: str) -> dict[str, str]:
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


def _freeze_provenance(
    config: dict[str, Any], tasks: list[dict[str, Any]], freeze_manifest_path: Path
) -> dict[str, Any]:
    protocol = config.get("protocol", {})
    result: dict[str, Any] = {
        "protocol_version": protocol.get("version"),
        "protocol_stage": protocol.get("stage"),
        "protocol_locked": bool(protocol.get("locked", False)),
        "freeze_manifest_sha256": None,
        "freeze_source_commit": None,
        "model_artifact": None,
        "docker_image_digests": None,
        "python_version": platform.python_version(),
        "package_versions": _package_versions(),
    }
    if not result["protocol_locked"]:
        return result
    try:
        manifest = json.loads(freeze_manifest_path.read_text(encoding="utf-8"))
        verify_freeze_manifest(config, manifest)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Formal collection provenance verification failed: {exc}") from exc
    result.update(
        {
            "freeze_manifest_sha256": _sha256(freeze_manifest_path),
            "freeze_source_commit": manifest["source_commit"],
            "model_artifact": _ollama_model_metadata(str(config["shadow"]["model_name"])),
            "docker_image_digests": _docker_image_digests(tasks, str(config["environment"]["image_prefix"])),
        }
    )
    return result


def _provenance_fingerprint(provenance: dict[str, Any], model_name: str, split: str) -> str:
    payload = {"model_name": model_name, "split": split, "provenance": provenance}
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _validate_resume(
    runs: list[ShadowRun],
    summary_path: Path,
    model_name: str,
    split: str,
    expected_task_ids: set[str],
    expected_task_count: int,
    provenance_fingerprint: str,
) -> None:
    if not summary_path.is_file():
        raise ValueError("Resume trace exists without its summary.json provenance record")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("run_provenance_fingerprint") != provenance_fingerprint:
        raise ValueError("Resume summary provenance does not match the current frozen environment")
    if int(summary.get("task_count", -1)) != len(runs):
        raise ValueError("Resume summary task count does not match traces.jsonl")
    if int(summary.get("expected_task_count", -1)) != expected_task_count:
        raise ValueError("Resume summary expected task count does not match the frozen split")
    if summary.get("model_name") != model_name or summary.get("split") != split:
        raise ValueError("Resume summary model or split does not match the requested cell")
    task_ids = [run.task_id for run in runs]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Resume trace contains duplicate task IDs")
    unexpected = sorted(set(task_ids) - expected_task_ids)
    if unexpected:
        raise ValueError(f"Resume trace contains tasks outside the frozen split: {unexpected}")
    stale = [run.task_id for run in runs if run.provenance_fingerprint != provenance_fingerprint]
    if stale:
        raise ValueError(f"Resume trace contains stale provenance for tasks: {stale}")


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


def _validate_collection_scope(config: dict[str, Any], split: str, task_limit: int) -> None:
    locked = bool(config.get("protocol", {}).get("locked", False))
    if not locked and (split != "shadow_development" or task_limit <= 0 or task_limit > 10):
        raise ValueError("Unlocked protocol permits only a shadow_development pilot of at most 10 tasks")
    if locked and task_limit != 0:
        raise ValueError("Formal collection forbids task limits; each frozen split must run in full")


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
        "run_provenance_fingerprint": provenance["run_provenance_fingerprint"],
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
            "version": config["protocol"]["version"],
            "stage": config["protocol"]["stage"],
            "locked": config["protocol"]["locked"],
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
    parser.add_argument("--freeze-manifest", default="data/phase1_5/protocol_freeze_manifest.json")
    args = parser.parse_args()

    config = copy.deepcopy(load_simple_yaml(args.config))
    if args.model_name:
        config["shadow"]["model_name"] = args.model_name
    try:
        _validate_collection_scope(config, args.split, args.task_limit)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    output_dir = Path(args.output_dir or config["outputs"]["result_dir"])
    tasks = _load_split_tasks(config["data"]["task_path"], config["data"]["split_path"], args.split, args.task_limit)
    expected_tasks = len(tasks)
    freeze_provenance = _freeze_provenance(config, tasks, Path(args.freeze_manifest))
    provenance = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "run_source_commit": _git_commit(),
        "config_sha256": _sha256(args.config),
        "task_manifest_sha256": _sha256(config["data"]["task_path"]),
        "split_manifest_sha256": _sha256(config["data"]["split_path"]),
        "prompt_sha256": _sha256("src/phase1_5_shadow/prompts.py"),
        "agent_sha256": _sha256("src/phase1_5_shadow/shadow_agent.py"),
        "json_client_sha256": _sha256("src/phase1_5_shadow/json_model_client.py"),
        **freeze_provenance,
    }
    provenance["run_provenance_fingerprint"] = _provenance_fingerprint(
        provenance,
        str(config["shadow"]["model_name"]),
        args.split,
    )

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
        try:
            _validate_resume(
                runs,
                output_dir / "summary.json",
                str(config["shadow"]["model_name"]),
                args.split,
                {str(task["task_id"]) for task in tasks},
                expected_tasks,
                provenance["run_provenance_fingerprint"],
            )
        except ValueError as exc:
            raise SystemExit(f"Refusing stale resume data: {exc}") from exc

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
        run = ShadowRun(
            **{
                **run.to_dict(),
                "steps": run.steps,
                "total_task_wall_seconds": time.perf_counter() - task_started,
                "provenance_fingerprint": provenance["run_provenance_fingerprint"],
            }
        )
        runs.append(run)
        _write_outputs(runs, _summary(runs, config, args.split, expected_tasks, provenance), output_dir)

    summary = _summary(runs, config, args.split, expected_tasks, provenance)
    _write_outputs(runs, summary, output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
