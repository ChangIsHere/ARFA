from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


POLICIES = ("always_slow", "always_fast", "residual")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def summarize_policy(pair: str, policy: str, traces: list[dict[str, Any]]) -> dict[str, Any]:
    steps = [step for trace in traces for step in trace.get("steps", [])]
    calls = sum(int(trace.get("model_calls", 0)) for trace in traces)
    tokens = sum(int(trace.get("total_tokens", 0)) for trace in traces)
    failed = [trace for trace in traces if not trace.get("success")]
    successful = [trace for trace in traces if trace.get("success")]
    return {
        "pair": pair,
        "policy": policy,
        "tasks": len(traces),
        "success_rate": ratio(len(successful), len(traces)),
        "calls_per_task": ratio(calls, len(traces)),
        "tokens_per_task": ratio(tokens, len(traces)),
        "tokens_per_call": ratio(tokens, calls),
        "prompt_tokens_per_call": ratio(
            sum(int(step.get("prompt_tokens", 0)) for step in steps), calls
        ),
        "completion_tokens_per_call": ratio(
            sum(int(step.get("completion_tokens", 0)) for step in steps), calls
        ),
        "failed_calls_per_task": mean(
            [int(trace.get("model_calls", 0)) for trace in failed]
        ) if failed else 0.0,
        "successful_calls_per_task": mean(
            [int(trace.get("model_calls", 0)) for trace in successful]
        ) if successful else 0.0,
        "max_step_rate": ratio(
            sum(bool(trace.get("max_steps_reached")) for trace in traces), len(traces)
        ),
        "repeated_actions_per_task": ratio(
            sum(int(trace.get("repeated_action_count", 0)) for trace in traces), len(traces)
        ),
        "parse_errors_per_task": ratio(
            sum(int(trace.get("parse_error_count", 0)) for trace in traces), len(traces)
        ),
        "nonzero_exits_per_task": ratio(
            sum(int(trace.get("nonzero_exit_count", 0)) for trace in traces), len(traces)
        ),
    }


def summarize_roles(pair: str, policy: str, traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        for step in trace.get("steps", []):
            grouped[str(step.get("model_role", "unknown"))].append(step)
    rows = []
    for role, steps in sorted(grouped.items()):
        rows.append({
            "pair": pair,
            "policy": policy,
            "role": role,
            "calls": len(steps),
            "tokens_per_call": mean(int(step.get("total_tokens", 0)) for step in steps),
            "prompt_tokens_per_call": mean(int(step.get("prompt_tokens", 0)) for step in steps),
            "completion_tokens_per_call": mean(
                int(step.get("completion_tokens", 0)) for step in steps
            ),
            "latency_seconds_per_call": mean(
                float(step.get("model_latency_seconds", 0.0)) for step in steps
            ),
        })
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percent_change(value: float, baseline: float) -> float:
    return 100.0 * ratio(value - baseline, baseline)


def paired_differences(
    pair: str,
    fast_traces: list[dict[str, Any]],
    slow_traces: list[dict[str, Any]],
    samples: int = 10000,
    seed: int = 1507,
) -> list[dict[str, Any]]:
    fast = {str(trace["task_id"]): trace for trace in fast_traces}
    slow = {str(trace["task_id"]): trace for trace in slow_traces}
    task_ids = sorted(set(fast) & set(slow))

    def task_values(trace: dict[str, Any]) -> dict[str, float]:
        calls = float(trace.get("model_calls", 0))
        return {
            "model_calls": calls,
            "tokens_per_call": ratio(float(trace.get("total_tokens", 0)), calls),
            "repeated_actions": float(trace.get("repeated_action_count", 0)),
            "parse_errors": float(trace.get("parse_error_count", 0)),
            "max_steps": float(bool(trace.get("max_steps_reached"))),
        }

    fast_values = {task: task_values(fast[task]) for task in task_ids}
    slow_values = {task: task_values(slow[task]) for task in task_ids}
    rng = np.random.default_rng(seed)
    rows = []
    for metric in next(iter(fast_values.values())):
        differences = np.array(
            [fast_values[task][metric] - slow_values[task][metric] for task in task_ids]
        )
        draws = np.array([
            float(np.mean(differences[rng.integers(0, len(task_ids), len(task_ids))]))
            for _ in range(samples)
        ])
        low, high = np.percentile(draws, [2.5, 97.5])
        rows.append({
            "pair": pair,
            "comparison": "always_fast_minus_always_slow",
            "metric": metric,
            "paired_tasks": len(task_ids),
            "mean_difference": float(np.mean(differences)),
            "ci95_low": float(low),
            "ci95_high": float(high),
        })
    return rows


def write_report(
    path: Path,
    policy_rows: list[dict[str, Any]],
    difference_rows: list[dict[str, Any]],
) -> None:
    by_pair = {
        pair: {row["policy"]: row for row in policy_rows if row["pair"] == pair}
        for pair in sorted({row["pair"] for row in policy_rows})
    }
    lines = [
        "# Fast-Model Cost Reversal",
        "",
        "This analysis uses the selected development traces. It tests whether a",
        "nominally smaller model can increase total cost by taking more reasoning",
        "steps or emitting more tokens per call.",
        "",
        "| Pair | Fast calls/task vs slow | Fast tokens/call vs slow | Fast max-step | Slow max-step | Residual calls/task vs slow |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for pair, policies in by_pair.items():
        slow = policies["always_slow"]
        fast = policies["always_fast"]
        residual = policies["residual"]
        lines.append(
            f"| {pair} | {percent_change(fast['calls_per_task'], slow['calls_per_task']):+.1f}% "
            f"| {percent_change(fast['tokens_per_call'], slow['tokens_per_call']):+.1f}% "
            f"| {fast['max_step_rate']:.1%} | {slow['max_step_rate']:.1%} "
            f"| {percent_change(residual['calls_per_task'], slow['calls_per_task']):+.1f}% |"
        )
    lines.extend([
        "",
        "Paired task differences (fast-only minus slow-only):",
        "",
        "| Pair | Metric | Mean difference | 95% bootstrap CI |",
        "| --- | --- | ---: | ---: |",
    ])
    for row in difference_rows:
        lines.append(
            f"| {row['pair']} | {row['metric']} | {row['mean_difference']:+.2f} "
            f"| [{row['ci95_low']:+.2f}, {row['ci95_high']:+.2f}] |"
        )
    lines.extend([
        "",
        "Positive values mean more work than always-slow. Failed-task calls and",
        "prompt/completion token splits are retained in `policy_metrics.csv` and",
        "`role_metrics.csv`.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pair",
        action="append",
        required=True,
        help="PAIR=directory containing always_slow, always_fast, and residual",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    policy_rows: list[dict[str, Any]] = []
    role_rows: list[dict[str, Any]] = []
    difference_rows: list[dict[str, Any]] = []
    for item in args.pair:
        pair, raw_root = item.split("=", 1)
        root = Path(raw_root)
        traces_by_policy = {}
        for policy in POLICIES:
            traces = read_jsonl(root / policy / "traces.jsonl")
            traces_by_policy[policy] = traces
            policy_rows.append(summarize_policy(pair, policy, traces))
            role_rows.extend(summarize_roles(pair, policy, traces))
        difference_rows.extend(paired_differences(
            pair, traces_by_policy["always_fast"], traces_by_policy["always_slow"]
        ))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "policy_metrics.csv", policy_rows)
    write_csv(args.output_dir / "role_metrics.csv", role_rows)
    write_csv(args.output_dir / "paired_differences.csv", difference_rows)
    write_report(args.output_dir / "README.md", policy_rows, difference_rows)


if __name__ == "__main__":
    main()
