from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from src.phase2_baseline.analyze_paper_results import MODEL_LABELS, PLANNED_PAIRS
from src.common.utils import read_csv, read_jsonl, write_jsonl


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_readme(
    destination: Path,
    main_rows: list[dict[str, str]],
    paired_rows: list[dict[str, str]],
    completeness: dict[str, Any],
) -> None:
    pair_lookup = {frozenset((row["model_a"], row["model_b"])): row for row in paired_rows}
    task_count = int(completeness["expected_tasks_per_model"])
    lines = [
        "# Phase 2 Writing Bundle",
        "",
        "Evidence for the Phase 2 always-reason ReAct baselines. No residual routing is enabled here.",
        "",
        "## Evidence Status",
        "",
        f"- Complete matrix: {len(main_rows)} local models x {task_count} identical tasks = {completeness['observed_runs']} runs.",
        "- Data: the full released InterCode NL2Bash suite, used as ARFA's evaluation set, not an upstream official test split.",
        f"- Gold-healthy sensitivity set: {completeness['gold_healthy_tasks']} of {task_count} tasks.",
        "- Task-ID, prompt, agent, environment, and reported-protocol checks: PASS.",
        "",
        "## Main Results",
        "",
    ]
    for row in main_rows:
        label = MODEL_LABELS.get(row["model"], row["model"])
        lines.append(
            f"- {label}: {int(row['successes'])}/{task_count} = {float(row['success_rate']):.1%}; "
            f"{float(row['mean_model_calls']):.2f} calls/task; "
            f"{float(row['mean_tokens']):.1f} tokens/task; "
            f"{float(row['mean_task_wall_seconds']):.2f} s/task."
        )
    lines.extend(["", "## Planned Pair Comparisons", ""])
    for fast, slow in PLANNED_PAIRS:
        row = pair_lookup[frozenset((fast, slow))]
        sign = 1 if row["model_a"] == slow else -1
        difference = sign * float(row["success_rate_difference_a_minus_b"])
        low = sign * float(row["bootstrap_ci95_low" if sign == 1 else "bootstrap_ci95_high"])
        high = sign * float(row["bootstrap_ci95_high" if sign == 1 else "bootstrap_ci95_low"])
        lines.append(
            f"- {MODEL_LABELS[slow]} minus {MODEL_LABELS[fast]}: {difference:+.1%} success "
            f"(paired 95% CI [{low:+.1%}, {high:+.1%}]); "
            f"McNemar p={float(row['exact_mcnemar_pvalue']):.3g}."
        )
    lines.extend([
        "",
        "## Protocol And Scope",
        "",
        "- Apple M4 with 24 GB unified memory; models run serially in Ollama. Original runs used 0.34.0; expansion runs used 0.34.1.",
        "- Shared 8K context, temperature 0, 700 output tokens per call, and a 12-step cap.",
        "- Fresh Docker agent and evaluator containers per task; success means reward >= 0.99.",
        "- Phi-4 Mini 3.8B and Phi-4 14B provide the fourth within-family scale pair.",
        "- Llama 3.2 3B and Llama 3.1 8B belong to different Llama releases.",
        "- Each model has one run per task. Intervals quantify task variation, not run-to-run variation.",
        "- Phase 2 alone does not show any benefit from residual routing; Phase 3 needs matched policy runs.",
        "- Completion counts all attempted tasks, including failures. Recovered JSON wrappers are separated from hard parse failures.",
        "- McNemar p-values are unadjusted for multiple comparisons. Energy was not measured.",
        "- Token means exclude tasks with missing token usage; those tasks remain in success-rate denominators.",
        "- Two request-error tasks (Gemma 12B and Phi Mini) use a synthetic error record: token usage is unknown and call counts do not capture all retries or prior steps. Their cost totals are incomplete.",
        "- DeepSeek candidates were replaced after observed runtime and protocol failures; see ../03_phase2_baseline_agent.md for selection history.",
    ])
    if not completeness["config_hashes_match"]:
        lines.append(
            "- Original and expansion config file SHA-256 values differ. The recorded task, prompt, agent, "
            "environment, and protocol fields match; see `metrics/completeness.json` for both config hashes."
        )
    lines.extend([
        "",
        "## Files",
        "",
        "- `reports/phase2_report.md`: full results and diagnostics.",
        "- `metrics/`: per-model summaries, paired comparisons, subgroups, failures, and completeness audit.",
        "- `figures/` and `tables/`: paper-facing exhibits.",
        "- `protocol/`: config, model metadata, environment audit, and trace hashes.",
        "- `metrics/task_outcomes.jsonl`: compact outcomes for all 1,600 attempts, enabling paired reanalysis without raw conversations.",
    ])
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the tracked Phase 2 paper-writing bundle.")
    parser.add_argument("--results-root", default="results/phase2_baseline/paper")
    parser.add_argument("--output-dir", default="docs/phase2_writing_bundle")
    args = parser.parse_args()

    root = Path(args.results_root)
    analysis = root / "analysis"
    output = Path(args.output_dir)
    completeness = _read_json(analysis / "completeness.json")
    if not completeness["complete"]:
        raise SystemExit("Phase 2 matrix is incomplete; refusing to publish a writing bundle")
    copies = {
        analysis / "phase2_report.md": output / "reports/phase2_report.md",
        analysis / "failure_case_analysis.md": output / "reports/failure_case_analysis.md",
        analysis / "completeness.json": output / "metrics/completeness.json",
        analysis / "main_results.csv": output / "metrics/main_results.csv",
        analysis / "main_results_gold_healthy.csv": output / "metrics/main_results_gold_healthy.csv",
        analysis / "paired_model_comparisons.csv": output / "metrics/paired_model_comparisons.csv",
        analysis / "results_by_filesystem.csv": output / "metrics/results_by_filesystem.csv",
        analysis / "results_by_task_category.csv": output / "metrics/results_by_task_category.csv",
        analysis / "failure_cases.jsonl": output / "metrics/failure_cases.jsonl",
        analysis / "failure_counts.csv": output / "metrics/failure_counts.csv",
        Path("config/phase2_paper.yaml"): output / "protocol/phase2_paper.yaml",
        Path("data/phase2/model_matrix.json"): output / "protocol/model_matrix.json",
        Path("results/phase2_baseline/environment_validation/summary.json"): output / "protocol/environment_validation_summary.json",
    }
    for source in (analysis / "paper_figures").glob("*"):
        copies[source] = output / "figures" / source.name
    for source in (analysis / "paper_tables").glob("*"):
        copies[source] = output / "tables" / source.name
    accepted_summaries = []
    expected_models = set(completeness["models"])
    for summary_path in sorted(root.glob("*/summary.json")):
        if _read_json(summary_path).get("model_name") not in expected_models:
            continue
        if not (summary_path.parent / "traces.jsonl").is_file():
            raise ValueError(f"Missing traces for {summary_path}")
        accepted_summaries.append(summary_path)
        copies[summary_path] = output / "protocol" / f"{summary_path.parent.name}_summary.json"
    for source, destination in copies.items():
        _copy(source, destination)

    validation = read_jsonl("results/phase2_baseline/environment_validation/gold_validation.jsonl")
    write_jsonl(output / "metrics/gold_validation_anomalies.jsonl", [row for row in validation if not row.get("valid")])
    traces = []
    outcomes = []
    for summary_path in accepted_summaries:
        model = _read_json(summary_path)["model_name"]
        trace_path = summary_path.parent / "traces.jsonl"
        runs = read_jsonl(trace_path)
        for run in runs:
            outcomes.append({
                "model": model,
                **{key: run.get(key) for key in (
                    "task_id", "success", "model_calls", "total_tokens",
                    "total_model_latency_seconds", "total_task_wall_seconds",
                    "termination_reason", "max_steps_reached",
                )},
                "reward": run.get("evaluator", {}).get("reward"),
            })
        traces.append(
            {
                "path": str(trace_path),
                "bytes": trace_path.stat().st_size,
                "records": sum(1 for line in trace_path.read_text(encoding="utf-8").splitlines() if line),
                "sha256": _sha256(trace_path),
            }
        )
    write_jsonl(output / "metrics/task_outcomes.jsonl", outcomes)
    manifest = {
        "complete": _read_json(analysis / "completeness.json")["complete"],
        "source_traces": traces,
    }
    manifest_path = output / "protocol/source_artifact_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _write_readme(
        output / "README_FOR_WRITING.md",
        read_csv(analysis / "main_results.csv"),
        read_csv(analysis / "paired_model_comparisons.csv"),
        completeness,
    )
    print(f"Wrote Phase 2 writing bundle to {output}")


if __name__ == "__main__":
    main()
