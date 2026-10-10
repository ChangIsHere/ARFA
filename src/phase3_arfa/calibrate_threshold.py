"""Select a Phase 3 residual threshold using development tasks only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.common.utils import read_csv, write_csv
from src.phase3_arfa.analyze_results import _paired_comparison, _policy_summary


def _threshold_slug(value: float) -> str:
    return f"threshold_{value:.2f}".replace(".", "_")


def _trajectory_workers(metrics_path: Path) -> int | None:
    summary_path = metrics_path.parent / "summary.json"
    if not summary_path.is_file():
        return None
    value = json.loads(summary_path.read_text(encoding="utf-8")).get("trajectory_workers")
    return int(value) if value is not None else None


def calibrate(
    result_root: Path,
    pair: str,
    split: str,
    thresholds: list[float],
    expected_tasks: int,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    pair_root = result_root / split / pair
    slow_path = pair_root / "always_slow" / "metrics.csv"
    if not slow_path.is_file():
        raise FileNotFoundError(f"Missing matched always-slow metrics: {slow_path}")

    slow_rows = read_csv(slow_path)
    slow_summary = _policy_summary("always_slow", slow_rows, expected_tasks)
    workers = _trajectory_workers(slow_path)
    candidates: list[dict[str, Any]] = []

    for threshold in thresholds:
        if abs(threshold - 0.35) < 1e-9:
            metrics_path = pair_root / "residual" / "metrics.csv"
        else:
            metrics_path = (
                result_root
                / "calibration"
                / _threshold_slug(threshold)
                / split
                / pair
                / "residual"
                / "metrics.csv"
            )
        if not metrics_path.is_file():
            candidates.append(
                {
                    "threshold": threshold,
                    "metrics_path": str(metrics_path),
                    "complete": False,
                    "eligible": False,
                    "reason": "missing_metrics",
                }
            )
            continue

        rows = read_csv(metrics_path)
        candidate_workers = _trajectory_workers(metrics_path)
        if candidate_workers != workers:
            raise ValueError(
                f"Threshold {threshold} used {candidate_workers} workers; always-slow used {workers}"
            )
        summary = _policy_summary("residual", rows, expected_tasks)
        comparison = _paired_comparison(rows, slow_rows, samples, seed)
        point = comparison["point_estimates"]
        complete = bool(summary["complete"] and slow_summary["complete"])
        success_preserved = summary["success_count"] >= slow_summary["success_count"]
        wall_improved = (point["wall_time_reduction"] or 0.0) > 0.0
        slow_calls_improved = (point["slow_call_reduction"] or 0.0) > 0.0
        eligible = complete and success_preserved and wall_improved and slow_calls_improved
        reason = "eligible" if eligible else ",".join(
            name
            for name, passed in (
                ("incomplete", complete),
                ("success_below_slow", success_preserved),
                ("wall_not_reduced", wall_improved),
                ("slow_calls_not_reduced", slow_calls_improved),
            )
            if not passed
        )
        candidates.append(
            {
                "threshold": threshold,
                "metrics_path": str(metrics_path),
                "complete": complete,
                "eligible": eligible,
                "reason": reason,
                "summary": summary,
                "comparison_vs_always_slow": comparison,
            }
        )

    eligible = [row for row in candidates if row["eligible"]]
    selected = None
    if eligible:
        selected = min(
            eligible,
            key=lambda row: (
                row["comparison_vs_always_slow"]["point_estimates"]["wall_seconds_per_success_ratio"],
                -row["comparison_vs_always_slow"]["point_estimates"]["token_reduction"],
                -row["comparison_vs_always_slow"]["point_estimates"]["slow_call_reduction"],
            ),
        )["threshold"]

    return {
        "phase": "phase3_arfa",
        "purpose": "development_only_residual_threshold_calibration",
        "pair": pair,
        "split": split,
        "expected_tasks": expected_tasks,
        "trajectory_workers": workers,
        "selection_rule": {
            "constraints": [
                "candidate success count >= matched always-slow success count",
                "candidate mean wall time < matched always-slow mean wall time",
                "candidate mean slow-model calls < matched always-slow mean slow-model calls",
            ],
            "ranking": "minimum wall seconds per success, then token reduction, then slow-call reduction",
            "final_split_used": False,
        },
        "always_slow": slow_summary,
        "candidates": candidates,
        "selected_threshold": selected,
        "selection_succeeded": selected is not None,
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--pair", default="llama")
    parser.add_argument("--split", default="development")
    parser.add_argument("--thresholds", type=float, nargs="+", required=True)
    parser.add_argument("--expected-tasks", type=int, default=100)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=1507)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    report = calibrate(
        result_root=args.result_root,
        pair=args.pair,
        split=args.split,
        thresholds=args.thresholds,
        expected_tasks=args.expected_tasks,
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "threshold_calibration.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    rows = []
    for candidate in report["candidates"]:
        row = {
            "threshold": candidate["threshold"],
            "complete": candidate["complete"],
            "eligible": candidate["eligible"],
            "reason": candidate["reason"],
        }
        if "summary" in candidate:
            row.update(candidate["summary"])
            row.update(candidate["comparison_vs_always_slow"]["point_estimates"])
        rows.append(row)
    write_csv(args.output_dir / "threshold_calibration.csv", rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
