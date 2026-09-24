from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

from src.common.utils import read_jsonl, write_csv, write_jsonl


MODEL_ORDER = {
    "arfa-qwen2.5-coder:7b-8k": 0,
    "arfa-llama3.1:8b-8k": 1,
    "arfa-qwen2.5-coder:14b-8k": 2,
    "arfa-llama3.2:3b-8k": 3,
    "arfa-gemma3:4b-8k": 4,
    "arfa-gemma3:12b-8k": 5,
    "arfa-phi4-mini:3.8b-8k": 6,
    "arfa-phi4:14b-8k": 7,
}

MODEL_LABELS = {
    "arfa-qwen2.5-coder:7b-8k": "Qwen2.5-Coder 7B",
    "arfa-llama3.1:8b-8k": "Llama 3.1 8B",
    "arfa-qwen2.5-coder:14b-8k": "Qwen2.5-Coder 14B",
    "arfa-llama3.2:3b-8k": "Llama 3.2 3B",
    "arfa-gemma3:4b-8k": "Gemma 3 4B",
    "arfa-gemma3:12b-8k": "Gemma 3 12B",
    "arfa-phi4-mini:3.8b-8k": "Phi-4 Mini 3.8B",
    "arfa-phi4:14b-8k": "Phi-4 14B",
}

PLANNED_PAIRS = (
    ("arfa-qwen2.5-coder:7b-8k", "arfa-qwen2.5-coder:14b-8k"),
    ("arfa-llama3.2:3b-8k", "arfa-llama3.1:8b-8k"),
    ("arfa-gemma3:4b-8k", "arfa-gemma3:12b-8k"),
    ("arfa-phi4-mini:3.8b-8k", "arfa-phi4:14b-8k"),
)


def _model_key(model: str) -> tuple[int, str]:
    return MODEL_ORDER.get(model, len(MODEL_ORDER)), model


def _audit_completeness(
    all_runs: dict[str, list[dict[str, Any]]],
    summaries: dict[str, dict[str, Any]],
    expected_models: set[str],
    task_ids: set[str],
    task_manifest_sha256: str,
) -> dict[str, Any]:
    protocol_hashes = ("prompt_sha256", "agent_sha256", "environment_sha256")
    reference = next((summaries[name].get("reproducibility", {}) for name in sorted(summaries)), {})
    reference_protocol = next((summaries[name].get("protocol") for name in sorted(summaries)), None)
    per_model = {}
    for model in sorted(expected_models | set(all_runs), key=_model_key):
        runs = all_runs.get(model, [])
        summary = summaries.get(model, {})
        observed = [str(run["task_id"]) for run in runs]
        observed_set = set(observed)
        provenance = summary.get("reproducibility", {})
        issues = []
        if model not in expected_models:
            issues.append("unexpected_model")
        if model not in summaries:
            issues.append("missing_summary")
        if len(observed) != len(observed_set):
            issues.append("duplicate_task_ids")
        if observed_set != task_ids:
            issues.append("task_id_set_mismatch")
        if summary.get("model_name") != model or summary.get("task_count") != len(runs):
            issues.append("summary_mismatch")
        if summary.get("dry_run") is not False or summary.get("paper_protocol") is not True:
            issues.append("not_paper_protocol")
        if summary.get("protocol") != reference_protocol:
            issues.append("protocol_mismatch")
        if provenance.get("task_manifest_sha256") != task_manifest_sha256:
            issues.append("task_manifest_hash_mismatch")
        for key in protocol_hashes:
            if not provenance.get(key) or provenance.get(key) != reference.get(key):
                issues.append(f"{key}_mismatch")
        per_model[model] = {
            "run_count": len(runs),
            "unique_task_count": len(observed_set),
            "missing_task_count": len(task_ids - observed_set),
            "unexpected_task_count": len(observed_set - task_ids),
            "issues": issues,
            "complete": not issues,
        }
    config_hashes = sorted({str(summary.get("reproducibility", {}).get("config_sha256") or "") for summary in summaries.values()})
    return {
        "expected_models": len(expected_models),
        "observed_models": len(all_runs),
        "expected_tasks_per_model": len(task_ids),
        "observed_runs": sum(len(runs) for runs in all_runs.values()),
        "missing_models": sorted(expected_models - set(all_runs)),
        "unexpected_models": sorted(set(all_runs) - expected_models),
        "config_hashes": config_hashes,
        "config_hashes_match": len(config_hashes) == 1,
        "models": per_model,
        "complete": set(all_runs) == expected_models and all(row["complete"] for row in per_model.values()),
    }


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _aggregate(model_name: str, runs: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(runs)
    successes = sum(bool(run["success"]) for run in runs)
    ci_low, ci_high = _wilson_interval(successes, total)
    token_values = [int(run["total_tokens"]) for run in runs if run.get("total_tokens") is not None]
    rewards = [
        float(run["evaluator"]["reward"])
        for run in runs
        if run.get("evaluator", {}).get("reward") is not None
    ]
    parse_labels = [
        step.get("parse_error")
        for run in runs
        for step in run.get("steps", [])
        if step.get("parse_error")
    ]
    return {
        "model": model_name,
        "tasks": total,
        "successes": successes,
        "success_rate": successes / total if total else 0.0,
        "success_ci95_low": ci_low,
        "success_ci95_high": ci_high,
        "mean_reward": sum(rewards) / len(rewards) if rewards else None,
        "total_model_calls": sum(int(run["model_calls"]) for run in runs),
        "mean_model_calls": sum(int(run["model_calls"]) for run in runs) / total if total else 0.0,
        "total_tokens": sum(token_values) if token_values else None,
        "mean_tokens": sum(token_values) / len(token_values) if token_values else None,
        "total_model_latency_seconds": sum(float(run["total_model_latency_seconds"]) for run in runs),
        "mean_model_latency_seconds": sum(float(run["total_model_latency_seconds"]) for run in runs) / total if total else 0.0,
        "total_task_wall_seconds": sum(float(run.get("total_task_wall_seconds", 0.0)) for run in runs),
        "mean_task_wall_seconds": sum(float(run.get("total_task_wall_seconds", 0.0)) for run in runs) / total if total else 0.0,
        "max_steps_rate": sum(bool(run.get("max_steps_reached")) for run in runs) / total if total else 0.0,
        "parse_error_steps": len(parse_labels),
        "recovered_format_steps": sum(label == "extracted_json_from_non_json_response" for label in parse_labels),
        "hard_parse_failure_steps": sum(label == "json_parse_failed" for label in parse_labels),
        "model_request_failure_steps": sum(label == "model_request_failed" for label in parse_labels),
        "nonzero_exit_steps": sum(int(run.get("nonzero_exit_count", 0)) for run in runs),
        "repeated_actions": sum(int(run.get("repeated_action_count", 0)) for run in runs),
        "complete_200": total == 200,
    }


def _group_rows(
    all_runs: dict[str, list[dict[str, Any]]],
    task_lookup: dict[str, dict[str, Any]],
    field: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_name, runs in all_runs.items():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for run in runs:
            grouped[str(task_lookup[run["task_id"]][field])].append(run)
        for group, group_runs in sorted(grouped.items()):
            row = _aggregate(model_name, group_runs)
            row[field] = group
            rows.append(row)
    return rows


def _exact_mcnemar_pvalue(a_only: int, b_only: int) -> float:
    discordant = a_only + b_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(a_only, b_only) + 1)) / (2 ** discordant)
    return min(1.0, 2 * tail)


def _paired_results(all_runs: dict[str, list[dict[str, Any]]], samples: int = 10_000) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    models = sorted(all_runs, key=_model_key)
    success_maps = {
        model: {str(run["task_id"]): bool(run["success"]) for run in runs}
        for model, runs in all_runs.items()
    }
    for pair_index, (model_a, model_b) in enumerate(combinations(models, 2)):
        shared = sorted(set(success_maps[model_a]) & set(success_maps[model_b]))
        outcomes = [(success_maps[model_a][task_id], success_maps[model_b][task_id]) for task_id in shared]
        total = len(outcomes)
        a_successes = sum(a for a, _ in outcomes)
        b_successes = sum(b for _, b in outcomes)
        a_only = sum(a and not b for a, b in outcomes)
        b_only = sum(b and not a for a, b in outcomes)
        rng = random.Random(20260914 + pair_index)
        bootstrap = []
        for _ in range(samples):
            selected = [outcomes[rng.randrange(total)] for _ in range(total)]
            bootstrap.append((sum(a for a, _ in selected) - sum(b for _, b in selected)) / total)
        bootstrap.sort()
        low = bootstrap[int(0.025 * (samples - 1))]
        high = bootstrap[int(0.975 * (samples - 1))]
        rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "paired_tasks": total,
                "success_rate_a": a_successes / total,
                "success_rate_b": b_successes / total,
                "success_rate_difference_a_minus_b": (a_successes - b_successes) / total,
                "bootstrap_ci95_low": low,
                "bootstrap_ci95_high": high,
                "a_only_successes": a_only,
                "b_only_successes": b_only,
                "exact_mcnemar_pvalue": _exact_mcnemar_pvalue(a_only, b_only),
                "bootstrap_samples": samples,
            }
        )
    return rows


def _failure_category(run: dict[str, Any], gold_healthy: bool) -> str:
    if not gold_healthy:
        return "benchmark_environment_anomaly"
    if run.get("max_steps_reached"):
        return "max_step_exhaustion"
    if run.get("termination_reason") == "model_request_error":
        return "model_request_error"
    if any(step.get("parse_error") == "json_parse_failed" for step in run.get("steps", [])):
        return "response_parse_failure"
    if int(run.get("repeated_action_count", 0)):
        return "repeated_action_loop"
    if int(run.get("nonzero_exit_count", 0)):
        return "command_execution_error"
    if run.get("termination_reason") == "model_done" and int(run.get("model_calls", 0)) <= 2:
        return "premature_termination"
    return "incorrect_command_or_semantics"


def _write_failure_analysis(path: Path, failures: list[dict[str, Any]]) -> None:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for failure in failures:
        counts[(str(failure["model"]), str(failure["failure_category"]))] += 1
    lines = [
        "# Phase 2 Failure-Case Analysis",
        "",
        "Categories below are deterministic diagnostic labels derived from traces, not manual causal annotations.",
        "Benchmark-environment anomalies are excluded in the gold-healthy sensitivity analysis.",
        "Recovered JSON extraction is tracked separately and is not classified as a response-parse failure.",
        "",
        "## Counts",
        "",
        "| Model | Diagnostic category | Failures |",
        "| --- | --- | ---: |",
    ]
    for (model, category), count in sorted(counts.items(), key=lambda item: (_model_key(item[0][0]), item[0][1])):
        lines.append(f"| {model} | {category.replace('_', ' ')} | {count} |")
    lines.extend(["", "## Representative Cases", ""])
    seen: set[tuple[str, str]] = set()
    for failure in sorted(failures, key=lambda row: (_model_key(str(row["model"])), str(row["failure_category"]), str(row["task_id"]))):
        key = (str(failure["model"]), str(failure["failure_category"]))
        if key in seen:
            continue
        seen.add(key)
        lines.append(
            f"- `{failure['model']}` / `{failure['failure_category']}` / `{failure['task_id']}`: "
            f"reward={float(failure['reward']):.3f}, calls={failure['model_calls']}; "
            f"instruction: {failure['instruction']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _append_result_table(lines: list[str], aggregates: list[dict[str, Any]]) -> None:
    lines.extend(
        [
            "| Model | Tasks | Success | 95% CI | Mean reward | Calls/task | Tokens/task | Model latency/task (s) | Wall/task (s) | Max-step rate |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in aggregates:
        lines.append(
            f"| {row['model']} | {row['tasks']} | {row['success_rate']:.3f} | "
            f"[{row['success_ci95_low']:.3f}, {row['success_ci95_high']:.3f}] | "
            f"{row['mean_reward']:.3f} | {row['mean_model_calls']:.2f} | "
            f"{row['mean_tokens']:.1f} | {row['mean_model_latency_seconds']:.2f} | "
            f"{row['mean_task_wall_seconds']:.2f} | {row['max_steps_rate']:.3f} |"
        )


def _write_report(
    path: Path,
    aggregates: list[dict[str, Any]],
    valid_aggregates: list[dict[str, Any]],
    completeness: dict[str, Any],
    valid_task_count: int,
    paired_results: list[dict[str, Any]],
) -> None:
    lines = [
        "# Phase 2 Paper Baseline Report",
        "",
        "Standard ReAct baseline only. No residual gate or fast/slow routing is enabled.",
        "",
        "## Main Results",
        "",
    ]
    _append_result_table(lines, aggregates)
    lines.extend(
        [
            "",
            "## Paired Model Comparisons",
            "",
            "Differences use the same 200 tasks, with a deterministic 10,000-sample paired bootstrap CI and a two-sided exact McNemar test.",
            "",
            "| Model A | Model B | A-B success | 95% paired CI | A-only | B-only | McNemar p |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in paired_results:
        lines.append(
            f"| {row['model_a']} | {row['model_b']} | {row['success_rate_difference_a_minus_b']:.3f} | "
            f"[{row['bootstrap_ci95_low']:.3f}, {row['bootstrap_ci95_high']:.3f}] | "
            f"{row['a_only_successes']} | {row['b_only_successes']} | {row['exact_mcnemar_pvalue']:.4g} |"
        )
    lines.extend(
        [
            "",
            "## Protocol Diagnostics",
            "",
            "Recovered format steps contain extractable JSON wrapped in extra text; hard parse failures contain no usable JSON object.",
            "",
            "| Model | Recovered format | Hard parse | Request failures | Nonzero exits | Repeated actions |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in aggregates:
        lines.append(
            f"| {row['model']} | {row['recovered_format_steps']} | {row['hard_parse_failure_steps']} | "
            f"{row['model_request_failure_steps']} | "
            f"{row['nonzero_exit_steps']} | {row['repeated_actions']} |"
        )
    lines.extend(
        [
            "",
            f"## Gold-Healthy Sensitivity Analysis ({valid_task_count} tasks)",
            "",
            "This subset requires the released gold command to exit zero and achieve self-reward 1.00 in the pinned environment.",
            "",
        ]
    )
    _append_result_table(lines, valid_aggregates)
    lines.extend(
        [
            "",
            "## Completeness Gate",
            "",
            "Every model must contain the exact same 200 task IDs and matching protocol hashes before these results are treated as paper-ready.",
            "",
        ]
    )
    for model, audit in completeness["models"].items():
        status = "PASS" if audit["complete"] else "INCOMPLETE"
        lines.append(f"- `{model}`: {status} ({audit['run_count']}/200; issues: {', '.join(audit['issues']) or 'none'})")
    matrix_status = "PASS" if completeness["complete"] else "INCOMPLETE"
    lines.extend(["", f"Overall matrix: **{matrix_status}** ({completeness['observed_models']}/{completeness['expected_models']} models)."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Phase 2 paper baseline runs.")
    parser.add_argument("--results-root", default="results/phase2_baseline/paper")
    parser.add_argument("--tasks", default="data/phase2/intercode_nl2bash_official_200.jsonl")
    parser.add_argument("--model-matrix", default="data/phase2/model_matrix.json")
    parser.add_argument("--expected-models", type=int, default=None)
    parser.add_argument(
        "--environment-validation",
        default="results/phase2_baseline/environment_validation/gold_validation.jsonl",
    )
    args = parser.parse_args()

    root = Path(args.results_root)
    task_lookup = {row["task_id"]: row for row in read_jsonl(args.tasks)}
    matrix = json.loads(Path(args.model_matrix).read_text(encoding="utf-8"))
    expected_model_names = {entry["name"] for entry in matrix["models"]}
    if args.expected_models is not None and args.expected_models != len(expected_model_names):
        raise SystemExit("--expected-models disagrees with the model matrix")
    all_runs: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for summary_path in sorted(root.glob("*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        traces_path = summary_path.parent / "traces.jsonl"
        if traces_path.exists():
            model_name = str(summary["model_name"])
            if model_name in all_runs:
                raise SystemExit(f"Duplicate model summary: {model_name}")
            all_runs[model_name] = read_jsonl(traces_path)
            summaries[model_name] = summary
    if not all_runs:
        raise SystemExit(f"No model runs found under {root}")

    all_runs = dict(sorted(all_runs.items(), key=lambda item: _model_key(item[0])))
    task_sha256 = hashlib.sha256(Path(args.tasks).read_bytes()).hexdigest()
    completeness = _audit_completeness(all_runs, summaries, expected_model_names, set(task_lookup), task_sha256)
    aggregates = [_aggregate(model, runs) for model, runs in all_runs.items()]
    validation_rows = read_jsonl(args.environment_validation)
    valid_task_ids = {str(row["task_id"]) for row in validation_rows if row.get("valid")}
    valid_runs = {
        model: [run for run in runs if str(run["task_id"]) in valid_task_ids]
        for model, runs in all_runs.items()
    }
    valid_aggregates = [_aggregate(model, runs) for model, runs in valid_runs.items()]
    paired_results = _paired_results(all_runs)
    analysis_dir = root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    write_csv(analysis_dir / "main_results.csv", aggregates)
    write_csv(analysis_dir / "main_results_gold_healthy.csv", valid_aggregates)
    write_csv(analysis_dir / "paired_model_comparisons.csv", paired_results)
    write_csv(analysis_dir / "results_by_filesystem.csv", _group_rows(all_runs, task_lookup, "filesystem_version"))
    write_csv(analysis_dir / "results_by_task_category.csv", _group_rows(all_runs, task_lookup, "task_category"))

    failures = []
    for model, runs in all_runs.items():
        for run in runs:
            if not run["success"]:
                gold_healthy = run["task_id"] in valid_task_ids
                failures.append(
                    {
                        "model": model,
                        "task_id": run["task_id"],
                        "instruction": run["instruction"],
                        "reward": run["evaluator"].get("reward"),
                        "model_calls": run["model_calls"],
                        "max_steps_reached": run.get("max_steps_reached"),
                        "parse_error_count": run.get("parse_error_count"),
                        "recovered_format_count": sum(
                            step.get("parse_error") == "extracted_json_from_non_json_response"
                            for step in run.get("steps", [])
                        ),
                        "hard_parse_failure_count": sum(
                            step.get("parse_error") == "json_parse_failed"
                            for step in run.get("steps", [])
                        ),
                        "nonzero_exit_count": run.get("nonzero_exit_count"),
                        "repeated_action_count": run.get("repeated_action_count"),
                        "termination_reason": run.get("termination_reason"),
                        "diff_miss": run["evaluator"].get("diff_miss"),
                        "diff_extra": run["evaluator"].get("diff_extra"),
                        "gold_healthy": gold_healthy,
                        "failure_category": _failure_category(run, gold_healthy),
                    }
                )
    write_jsonl(analysis_dir / "failure_cases.jsonl", failures)
    failure_counts = []
    grouped_failures: dict[tuple[str, str], int] = defaultdict(int)
    for failure in failures:
        grouped_failures[(str(failure["model"]), str(failure["failure_category"]))] += 1
    for (model, category), count in sorted(
        grouped_failures.items(), key=lambda item: (_model_key(item[0][0]), item[0][1])
    ):
        failure_counts.append({"model": model, "failure_category": category, "count": count})
    write_csv(analysis_dir / "failure_counts.csv", failure_counts)
    _write_failure_analysis(analysis_dir / "failure_case_analysis.md", failures)
    _write_report(
        analysis_dir / "phase2_report.md",
        aggregates,
        valid_aggregates,
        completeness,
        len(valid_task_ids),
        paired_results,
    )
    manifest = {
        **completeness,
        "environment_validation_tasks": len(validation_rows),
        "gold_healthy_tasks": len(valid_task_ids),
    }
    (analysis_dir / "completeness.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
