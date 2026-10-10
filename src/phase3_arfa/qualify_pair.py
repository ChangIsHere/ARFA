"""Evaluate a short mixed-pair qualification without touching the final split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.phase3_arfa.analyze_results import analyze


def _load_summary(root: Path, pair: str, policy: str) -> dict[str, Any]:
    path = root / "development" / pair / policy / "summary.json"
    if not path.is_file():
        raise ValueError(f"Missing qualification summary: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def qualify(root: Path, pair: str, expected: int = 20) -> dict[str, Any]:
    summaries = {
        policy: _load_summary(root, pair, policy)
        for policy in ("residual", "always_slow", "always_fast")
    }
    residual = summaries["residual"]
    slow = summaries["always_slow"]
    fast = summaries["always_fast"]

    def per_call(summary: dict[str, Any], key: str) -> float:
        calls = int(summary["total_model_calls"])
        return float(summary[key]) / calls if calls else float("inf")

    fast_latency = per_call(fast, "total_model_latency_seconds")
    slow_latency = per_call(slow, "total_model_latency_seconds")
    latency_reduction = 1.0 - fast_latency / slow_latency if slow_latency else None
    fast_parse_rate = int(fast["parse_error_count"]) / max(int(fast["total_model_calls"]), 1)
    residual_call_ratio = int(residual["total_model_calls"]) / max(int(slow["total_model_calls"]), 1)

    checks = {
        "all_cells_complete": all(int(row["task_count"]) == expected for row in summaries.values()),
        "no_model_request_errors": all(
            int(row["model_request_error_count"]) == 0 for row in summaries.values()
        ),
        "fast_parse_error_rate_at_most_5pct": fast_parse_rate <= 0.05,
        "fast_latency_at_least_25pct_lower": bool(
            latency_reduction is not None and latency_reduction >= 0.25
        ),
        "residual_success_within_one_task_of_slow": (
            int(residual["success_count"]) >= int(slow["success_count"]) - 1
        ),
        "residual_total_calls_at_most_110pct_of_slow": residual_call_ratio <= 1.10,
    }
    analysis = analyze(root, "development", pair, 10_000, 1507, expected)
    return {
        "purpose": "development_only_mixed_pair_qualification",
        "pair": pair,
        "expected_tasks_per_policy": expected,
        "final_split_used": False,
        "qualification_rule_frozen_before_run": True,
        "measurements": {
            "fast_latency_seconds_per_call": fast_latency,
            "slow_latency_seconds_per_call": slow_latency,
            "fast_latency_reduction": latency_reduction,
            "fast_parse_error_rate": fast_parse_rate,
            "residual_to_slow_total_call_ratio": residual_call_ratio,
        },
        "checks": checks,
        "eligible_for_full_development": all(checks.values()),
        "analysis": analysis,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--pair", required=True)
    parser.add_argument("--expected-tasks", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = qualify(args.result_root, args.pair, args.expected_tasks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
