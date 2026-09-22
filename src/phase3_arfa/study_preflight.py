"""Verify the Phase 3 task and Phase 2 comparison inputs before formal runs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.common.config import load_simple_yaml
from src.common.utils import read_csv, read_jsonl


def inspect_design(config_path: Path, slow_metrics: Path, fast_metrics: Path) -> dict:
    config = load_simple_yaml(config_path)
    study = config["study"]
    task_path = Path(study["task_path"])
    split_path = Path(study["split_path"])
    task_ids = [row["task_id"] for row in read_jsonl(task_path)]
    task_sha256 = hashlib.sha256(task_path.read_bytes()).hexdigest()
    splits = json.loads(split_path.read_text())["splits"]
    for name, expected in (("shadow_development", 40), ("shadow_validation", 20),
                           ("shadow_test", 40), ("phase3_final", 100)):
        if len(splits[name]) != expected or len(set(splits[name])) != expected:
            raise ValueError(f"Unexpected or duplicate task IDs in {name}")
    development = set().union(*(set(splits[name]) for name in (
        "shadow_development", "shadow_validation", "shadow_test")))
    final = set(splits["phase3_final"])
    if len(task_ids) != len(set(task_ids)) or development & final:
        raise ValueError("Duplicate task ID or overlap between development and final")
    if development | final != set(task_ids):
        raise ValueError("Splits do not cover exactly the Phase 2 task manifest")
    if len(development) != study["development_tasks"] or len(final) != study["primary_test_tasks"]:
        raise ValueError("Unexpected task count in Phase 3 split")

    outcomes = {}
    for name, path in (("slow", slow_metrics), ("fast", fast_metrics)):
        rows = read_csv(path)
        by_id = {row["task_id"]: row["success"] == "True" for row in rows}
        if len(by_id) != len(rows) or set(by_id) != set(task_ids):
            raise ValueError(f"{name} baseline task IDs do not match the task manifest")
        summary = json.loads((path.parent / "summary.json").read_text())
        if (summary["model_name"] != study[f"{name}_model"]
                or summary["reproducibility"]["task_manifest_sha256"] != task_sha256
                or summary["task_count"] != len(rows)
                or summary["success_count"] != sum(by_id.values())):
            raise ValueError(f"{name} baseline summary does not match the configured model or metrics")
        outcomes[name] = by_id

    def counts(ids: set[str]) -> dict:
        return {
            "tasks": len(ids),
            "slow_successes": sum(outcomes["slow"][task] for task in ids),
            "fast_successes": sum(outcomes["fast"][task] for task in ids),
            "slow_only_successes": sum(outcomes["slow"][task] and not outcomes["fast"][task] for task in ids),
            "fast_only_successes": sum(outcomes["fast"][task] and not outcomes["slow"][task] for task in ids),
        }

    gap = (sum(outcomes["slow"].values()) - sum(outcomes["fast"].values())) / len(task_ids)
    margin = float(config["primary"]["success_drop_margin"])
    return {
        "version": study["version"],
        "status": "design_inputs_checked_no_phase3_outcomes",
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "task_manifest_sha256": task_sha256,
        "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "slow_metrics_sha256": hashlib.sha256(slow_metrics.read_bytes()).hexdigest(),
        "fast_metrics_sha256": hashlib.sha256(fast_metrics.read_bytes()).hexdigest(),
        "all_200": counts(set(task_ids)),
        "phase3_final_100": counts(final),
        "success_gap_points": round(100 * gap, 6),
        "acceptable_drop_points": 100 * margin,
        "share_of_observed_gap_preserved": (gap - margin) / gap if gap > 0 else None,
        "decision_rule": "paired 95% bootstrap lower bound for ARFA minus matched slow-only > -margin",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/phase3_study.yaml"))
    parser.add_argument("--slow-metrics", type=Path, default=Path("results/phase2_baseline/paper/qwen2.5-coder-14b/metrics.csv"))
    parser.add_argument("--fast-metrics", type=Path, default=Path("results/phase2_baseline/paper/qwen2.5-coder-7b/metrics.csv"))
    parser.add_argument("--output", type=Path, default=Path("evidence/phase3_design/preflight.json"))
    args = parser.parse_args()
    report = inspect_design(args.config, args.slow_metrics, args.fast_metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
