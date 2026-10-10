"""Audit the local runtime and matched Phase 2 inputs before Phase 3."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from src.common.config import load_simple_yaml
from src.common.utils import read_csv, read_jsonl
from src.phase3_arfa.run_arfa import _load_split_tasks, phase3_status


def _command_ok(command: list[str]) -> tuple[bool, str]:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    detail = (completed.stdout or completed.stderr).strip()
    return completed.returncode == 0, detail


def inspect_runtime(config_path: Path) -> dict[str, Any]:
    config = load_simple_yaml(config_path)
    tasks = read_jsonl(config["data"]["task_path"])
    task_ids = [str(task["task_id"]) for task in tasks]
    development = _load_split_tasks(config, "development", 0)
    final = _load_split_tasks(config, "final", 0)
    task_hash = hashlib.sha256(Path(config["data"]["task_path"]).read_bytes()).hexdigest()

    docker_ok, docker_detail = _command_ok(["docker", "info", "--format", "{{.ServerVersion}}"])
    required_images = sorted(
        {f"{config['environment']['image_prefix']}{int(task['filesystem_version'])}" for task in tasks}
    )
    image_checks = {}
    for image in required_images:
        ok, detail = _command_ok(["docker", "image", "inspect", "--format={{.Id}}", image])
        image_checks[image] = {"available": ok, "image_id": detail if ok else None}

    model_checks: dict[str, Any] = {}
    baseline_checks: dict[str, Any] = {}
    baseline_root = Path("results/phase2_baseline/paper")
    for pair_name, pair in config["pairs"].items():
        for role in ("fast", "slow"):
            model_name = str(pair[f"{role}_model"])
            ok, detail = _command_ok(["ollama", "show", model_name])
            model_checks[model_name] = {"available": ok, "detail": detail[:200] if not ok else "ok"}

            result_dir = baseline_root / str(pair[f"{role}_baseline_dir"])
            metrics_path = result_dir / "metrics.csv"
            summary_path = result_dir / "summary.json"
            issues: list[str] = []
            if not metrics_path.is_file() or not summary_path.is_file():
                issues.append("missing_phase2_artifacts")
                rows: list[dict[str, str]] = []
                summary: dict[str, Any] = {}
            else:
                rows = read_csv(metrics_path)
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                row_ids = [row["task_id"] for row in rows]
                if len(rows) != 200 or len(row_ids) != len(set(row_ids)) or set(row_ids) != set(task_ids):
                    issues.append("task_coverage_mismatch")
                if summary.get("model_name") != model_name:
                    issues.append("model_name_mismatch")
                if summary.get("reproducibility", {}).get("task_manifest_sha256") != task_hash:
                    issues.append("task_manifest_hash_mismatch")
            baseline_checks[f"{pair_name}.{role}"] = {
                "model_name": model_name,
                "task_count": len(rows),
                "success_count": summary.get("success_count"),
                "issues": issues,
                "ready": not issues,
            }

    gate_path = Path(config["engineering_gate_result"])
    gate = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.is_file() else {}
    checks = {
        "controller_ready": phase3_status(config)["ready"],
        "engineering_gate_passed": bool(gate.get("passed")),
        "task_manifest_has_200_unique_tasks": len(task_ids) == len(set(task_ids)) == 200,
        "development_has_100_tasks": len(development) == 100,
        "final_has_100_tasks": len(final) == 100,
        "development_final_disjoint": not ({str(t["task_id"]) for t in development} & {str(t["task_id"]) for t in final}),
        "docker_running": docker_ok,
        "all_images_available": all(row["available"] for row in image_checks.values()),
        "all_models_available": all(row["available"] for row in model_checks.values()),
        "all_phase2_baselines_matched": all(row["ready"] for row in baseline_checks.values()),
    }
    return {
        "phase": "phase3_runtime_preflight",
        "protocol_version": config.get("protocol_version"),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "ready_for_development": all(checks.values()),
        "ready_for_final": all(checks.values()) and bool(config.get("use_phase3_final_tasks", False)),
        "checks": checks,
        "docker_version": docker_detail if docker_ok else None,
        "images": image_checks,
        "models": model_checks,
        "phase2_baselines": baseline_checks,
        "task_manifest_sha256": task_hash,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/phase3_arfa.yaml"))
    parser.add_argument("--output", type=Path, default=Path("evidence/phase3_design/runtime_preflight.json"))
    args = parser.parse_args()
    report = inspect_runtime(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["ready_for_development"]:
        raise SystemExit("Phase 3 runtime preflight failed")


if __name__ == "__main__":
    main()
