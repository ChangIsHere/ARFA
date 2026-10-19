"""Summarize complete or in-progress Phase 3 policy runs."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from src.common.utils import read_csv, write_csv


def _truth(value: str) -> bool:
    return value.lower() == "true"


def _mean(rows: list[dict[str, str]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) not in (None, "", "None")]
    return float(np.mean(values)) if values else 0.0


def _policy_summary(policy: str, rows: list[dict[str, str]], expected: int) -> dict[str, Any]:
    successes = sum(_truth(row["success"]) for row in rows)
    total_wall = sum(float(row["total_task_wall_seconds"]) for row in rows)
    total_calls = sum(int(row["model_calls"]) for row in rows)
    fast_calls = sum(int(row["fast_model_calls"]) for row in rows)
    slow_calls = sum(int(row["slow_model_calls"]) for row in rows)
    return {
        "policy": policy,
        "task_count": len(rows),
        "expected_task_count": expected,
        "complete": len(rows) == expected,
        "success_count": successes,
        "success_rate": successes / len(rows) if rows else 0.0,
        "avg_wall_seconds": total_wall / len(rows) if rows else 0.0,
        "wall_seconds_per_success": total_wall / successes if successes else None,
        "avg_model_calls": total_calls / len(rows) if rows else 0.0,
        "avg_fast_calls": fast_calls / len(rows) if rows else 0.0,
        "avg_slow_calls": slow_calls / len(rows) if rows else 0.0,
        "fast_call_rate": fast_calls / total_calls if total_calls else 0.0,
        "avg_total_tokens": _mean(rows, "total_tokens"),
    }


def _percentile(values: list[float]) -> list[float | None]:
    if not values:
        return [None, None]
    low, high = np.percentile(values, [2.5, 97.5])
    return [float(low), float(high)]


def _exact_mcnemar_pvalue(candidate_only: int, comparator_only: int) -> float:
    discordant = candidate_only + comparator_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(candidate_only, comparator_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def _paired_comparison(
    candidate: list[dict[str, str]],
    comparator: list[dict[str, str]],
    samples: int,
    seed: int,
) -> dict[str, Any]:
    left = {row["task_id"]: row for row in candidate}
    right = {row["task_id"]: row for row in comparator}
    task_ids = sorted(set(left) & set(right))
    if not task_ids:
        return {"paired_tasks": 0}

    def observed(ids: list[str]) -> dict[str, float | None]:
        candidate_success = np.array([_truth(left[task]["success"]) for task in ids], dtype=float)
        comparator_success = np.array([_truth(right[task]["success"]) for task in ids], dtype=float)
        candidate_wall = np.array([float(left[task]["total_task_wall_seconds"]) for task in ids])
        comparator_wall = np.array([float(right[task]["total_task_wall_seconds"]) for task in ids])
        candidate_slow = np.array([float(left[task]["slow_model_calls"]) for task in ids])
        comparator_slow = np.array([float(right[task]["slow_model_calls"]) for task in ids])
        candidate_tokens = np.array([float(left[task]["total_tokens"] or 0) for task in ids])
        comparator_tokens = np.array([float(right[task]["total_tokens"] or 0) for task in ids])

        def reduction(candidate_values: np.ndarray, comparator_values: np.ndarray) -> float | None:
            baseline = float(np.mean(comparator_values))
            return 1.0 - float(np.mean(candidate_values)) / baseline if baseline else None

        candidate_successes = float(np.sum(candidate_success))
        comparator_successes = float(np.sum(comparator_success))
        candidate_cost_success = float(np.sum(candidate_wall)) / candidate_successes if candidate_successes else None
        comparator_cost_success = float(np.sum(comparator_wall)) / comparator_successes if comparator_successes else None
        return {
            "success_rate_difference": float(np.mean(candidate_success - comparator_success)),
            "wall_time_reduction": reduction(candidate_wall, comparator_wall),
            "slow_call_reduction": reduction(candidate_slow, comparator_slow),
            "token_reduction": reduction(candidate_tokens, comparator_tokens),
            "wall_seconds_per_success_ratio": (
                candidate_cost_success / comparator_cost_success
                if candidate_cost_success is not None and comparator_cost_success else None
            ),
        }

    point = observed(task_ids)
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {key: [] for key in point}
    for _ in range(samples):
        sampled = [task_ids[index] for index in rng.integers(0, len(task_ids), len(task_ids))]
        metrics = observed(sampled)
        for key, value in metrics.items():
            if value is not None and np.isfinite(value):
                draws[key].append(float(value))
    candidate_only = sum(_truth(left[t]["success"]) and not _truth(right[t]["success"]) for t in task_ids)
    comparator_only = sum(_truth(right[t]["success"]) and not _truth(left[t]["success"]) for t in task_ids)
    return {
        "paired_tasks": len(task_ids),
        "point_estimates": point,
        "bootstrap_ci95": {key: _percentile(values) for key, values in draws.items()},
        "candidate_only_successes": candidate_only,
        "comparator_only_successes": comparator_only,
        "exact_mcnemar_pvalue": _exact_mcnemar_pvalue(candidate_only, comparator_only),
    }


def _control_audit(
    policy_rows: dict[str, list[dict[str, str]]],
    provenances: dict[str, dict[str, Any]],
    expected: int,
    budget_tolerance: float = 0.05,
) -> dict[str, Any]:
    if not policy_rows:
        return {"ready": False, "reason": "no_policy_runs"}
    reference = next(iter(policy_rows))
    reference_ids = {row["task_id"] for row in policy_rows[reference]}
    shared_keys = (
        "config_sha256", "task_manifest_sha256", "split_manifest_sha256",
        "selected_task_ids_sha256", "source_sha256", "fast_model", "slow_model",
        "docker_image_digests", "trajectory_workers",
    )
    task_sets_match = all(
        {row["task_id"] for row in rows} == reference_ids for rows in policy_rows.values()
    )
    provenance_matches = all(
        all(provenances.get(policy, {}).get(key) == provenances.get(reference, {}).get(key)
            and provenances.get(policy, {}).get(key) is not None for key in shared_keys)
        for policy in policy_rows
    )
    complete = all(len(rows) == expected for rows in policy_rows.values())
    rates = {policy: _policy_summary(policy, rows, expected)["fast_call_rate"]
             for policy, rows in policy_rows.items()}
    budget_gaps = {
        policy: abs(rates[policy] - rates["residual"])
        for policy in ("random_budget", "fixed_budget")
        if policy in rates and "residual" in rates
    }
    return {
        "ready": complete and task_sets_match and provenance_matches,
        "complete": complete,
        "task_sets_match": task_sets_match,
        "provenance_matches": provenance_matches,
        "actual_fast_call_rates": rates,
        "budget_rate_gaps_vs_residual": budget_gaps,
        "budget_tolerance": budget_tolerance,
        "budget_matched": bool(budget_gaps) and complete and all(
            gap <= budget_tolerance for gap in budget_gaps.values()
        ),
    }


def analyze(
    result_root: Path,
    split: str,
    pair: str,
    samples: int,
    seed: int,
    expected: int = 100,
    strict_controls: bool = False,
) -> dict[str, Any]:
    pair_root = result_root / split / pair
    policy_rows: dict[str, list[dict[str, str]]] = {}
    policy_workers: dict[str, int | None] = {}
    provenances: dict[str, dict[str, Any]] = {}
    for policy_dir in sorted(pair_root.iterdir()) if pair_root.is_dir() else []:
        metrics = policy_dir / "metrics.csv"
        if metrics.is_file():
            rows = read_csv(metrics)
            ids = [row["task_id"] for row in rows]
            if len(ids) != len(set(ids)):
                raise ValueError(f"Duplicate task IDs in {metrics}")
            policy_rows[policy_dir.name] = rows
            summary_path = policy_dir / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
            value = summary.get("trajectory_workers")
            policy_workers[policy_dir.name] = int(value) if value is not None else None
            provenances[policy_dir.name] = summary.get("provenance", {})

    if len(set(policy_workers.values())) > 1:
        raise ValueError(f"Policy runs use different trajectory concurrency: {policy_workers}")

    control_audit = _control_audit(policy_rows, provenances, expected)
    if strict_controls and not control_audit.get("provenance_matches", False):
        raise ValueError("Policy runs do not share the same frozen config, models, code, tasks, and images")
    if strict_controls and control_audit["complete"] and not control_audit["task_sets_match"]:
        raise ValueError("Complete policy runs have different task ID sets")

    summaries = [_policy_summary(policy, rows, expected) for policy, rows in sorted(policy_rows.items())]
    comparisons: dict[str, Any] = {}
    if "residual" in policy_rows:
        for comparator in (
            "always_slow", "always_fast", "observation_only", "learned_observation_only", "fixed_switch",
            "expectation_gate_only", "random_budget", "fixed_budget",
        ):
            if comparator in policy_rows:
                comparisons[f"residual_vs_{comparator}"] = _paired_comparison(
                    policy_rows["residual"], policy_rows[comparator], samples, seed
                )
    return {
        "phase": "phase3_arfa",
        "split": split,
        "pair": pair,
        "available_policies": sorted(policy_rows),
        "trajectory_workers": next(iter(policy_workers.values()), None),
        "all_available_policies_complete": bool(summaries) and all(row["complete"] for row in summaries),
        "control_audit": control_audit,
        "policy_summaries": summaries,
        "paired_comparisons": comparisons,
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=Path("results/phase3_arfa"))
    parser.add_argument("--split", choices=("development", "final"), default="development")
    parser.add_argument("--pair", default="qwen")
    parser.add_argument("--expected-tasks", type=int, default=100)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=1507)
    parser.add_argument("--strict-controls", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    report = analyze(
        args.result_root,
        args.split,
        args.pair,
        args.bootstrap_samples,
        args.bootstrap_seed,
        args.expected_tasks,
        args.strict_controls,
    )
    output_dir = args.output_dir or args.result_root / "analysis" / args.split / args.pair
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_csv(output_dir / "policy_summary.csv", report["policy_summaries"])
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
