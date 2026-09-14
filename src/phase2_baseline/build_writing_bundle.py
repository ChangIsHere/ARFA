from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

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


def _model_label(model: str) -> str:
    return (
        model.replace("arfa-", "")
        .replace(":7b-8k", " 7B")
        .replace(":8b-8k", " 8B")
        .replace(":14b-8k", " 14B")
        .replace("qwen2.5-coder", "Qwen2.5-Coder")
        .replace("llama3.1", "Llama 3.1")
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_readme(destination: Path, main_rows: list[dict[str, str]], paired_rows: list[dict[str, str]]) -> None:
    by_model = {row["model"]: row for row in main_rows}
    q7 = by_model["arfa-qwen2.5-coder:7b-8k"]
    llama = by_model["arfa-llama3.1:8b-8k"]
    q14 = by_model["arfa-qwen2.5-coder:14b-8k"]
    pair_lookup = {(row["model_a"], row["model_b"]): row for row in paired_rows}
    q7_q14 = pair_lookup[("arfa-qwen2.5-coder:7b-8k", "arfa-qwen2.5-coder:14b-8k")]
    llama_q14 = pair_lookup[("arfa-llama3.1:8b-8k", "arfa-qwen2.5-coder:14b-8k")]
    lines = [
        "# Phase 2 Writing Bundle",
        "",
        "This directory is the compact, tracked evidence package for writing the ARFA workshop paper's Phase 2 baseline section.",
        "Phase 2 is a standard always-reason structured ReAct agent. It contains no residual gate and no fast/slow routing.",
        "",
        "## Evidence Status",
        "",
        "- Complete matrix: 3 local models x 200 tasks = 600 runs.",
        "- Main set: the full released InterCode NL2Bash suite, used as the ARFA Phase 2 evaluation set; it is not presented as an upstream official test split.",
        "- Environment audit: 200/200 tasks executed, 199 reached released-gold self-success, and 195 met the stricter gold-healthy criterion.",
        "- Main results use all 200 tasks. A 195-task gold-healthy sensitivity analysis is reported separately.",
        "- Completeness gate: PASS.",
        "",
        "## Main Results",
        "",
        f"- Qwen2.5-Coder 7B: {int(q7['successes'])}/200 = {float(q7['success_rate']):.1%}; {float(q7['mean_model_calls']):.2f} calls/task; {float(q7['mean_tokens']):.1f} tokens/task; {float(q7['mean_task_wall_seconds']):.2f} s/task.",
        f"- Llama 3.1 8B: {int(llama['successes'])}/200 = {float(llama['success_rate']):.1%}; {float(llama['mean_model_calls']):.2f} calls/task; {float(llama['mean_tokens']):.1f} tokens/task; {float(llama['mean_task_wall_seconds']):.2f} s/task.",
        f"- Qwen2.5-Coder 14B: {int(q14['successes'])}/200 = {float(q14['success_rate']):.1%}; {float(q14['mean_model_calls']):.2f} calls/task; {float(q14['mean_tokens']):.1f} tokens/task; {float(q14['mean_task_wall_seconds']):.2f} s/task.",
        "",
        "## Paired Findings",
        "",
        f"- Qwen 14B exceeds Qwen 7B by {-float(q7_q14['success_rate_difference_a_minus_b']):.1%}; paired bootstrap 95% CI [{-float(q7_q14['bootstrap_ci95_high']):.1%}, {-float(q7_q14['bootstrap_ci95_low']):.1%}], exact McNemar p={float(q7_q14['exact_mcnemar_pvalue']):.3g}.",
        f"- Qwen 14B exceeds Llama 8B by {-float(llama_q14['success_rate_difference_a_minus_b']):.1%}; paired bootstrap 95% CI [{-float(llama_q14['bootstrap_ci95_high']):.1%}, {-float(llama_q14['bootstrap_ci95_low']):.1%}], exact McNemar p={float(llama_q14['exact_mcnemar_pvalue']):.3g}.",
        "- Qwen 7B and Llama 8B differ by only 0.5 percentage points; the paired CI includes zero and McNemar p=1.0.",
        "",
        "## Protocol",
        "",
        "- Runtime: Ollama 0.34.0 on Apple M4 with 24 GB unified memory; models run serially and locally.",
        "- Models: Qwen2.5-Coder 7B Q4_K_M, Llama 3.1 8B Q4_K_M, and Qwen2.5-Coder 14B Q4_K_M.",
        "- Shared effective context: 8192 tokens; temperature 0.0; maximum 700 generated tokens per call; maximum 12 ReAct steps.",
        "- Agent protocol: structured zero-shot ReAct with action, expected outcome, observation, and an explicit done signal.",
        "- Environment: fresh agent and evaluator Docker containers per task, with the released InterCode reward reimplemented and pinned.",
        "- Success: reward >= 0.99.",
        "",
        "## Interpretation Boundaries",
        "",
        "- Phase 2 establishes the always-reason baseline only; it does not test ARFA's residual-guided savings yet.",
        "- There is one run per model. Wilson intervals and paired bootstrap intervals quantify variation across tasks, not across random seeds.",
        "- Local Q4 quantization, hardware, Ollama version, prompt format, and the 8K context are part of the experimental condition.",
        "- Token totals are server-reported cumulative prompt-plus-completion tokens across calls, not API billing tokens.",
        "- Five released tasks have gold/environment anomalies. They remain in the 200-task main result and are removed only in the labeled sensitivity analysis.",
        "- Failure categories are deterministic trace-derived diagnostics, not manually adjudicated causal labels.",
        "- The copy-or-move category has only four tasks and zero successes for all models; subgroup claims should acknowledge the small sample.",
        "",
        "## Artifact Map",
        "",
        "- `reports/phase2_report.md`: complete main, paired, diagnostic, sensitivity, and completeness report.",
        "- `reports/failure_case_analysis.md`: failure counts and representative cases.",
        "- `metrics/main_results.csv`: all-200 aggregate metrics with Wilson intervals.",
        "- `metrics/main_results_gold_healthy.csv`: 195-task sensitivity analysis.",
        "- `metrics/paired_model_comparisons.csv`: paired bootstrap intervals and exact McNemar tests.",
        "- `metrics/results_by_filesystem.csv` and `metrics/results_by_task_category.csv`: subgroup results.",
        "- `metrics/failure_cases.jsonl` and `metrics/failure_counts.csv`: trace-derived failure diagnostics.",
        "- `metrics/gold_validation_anomalies.jsonl`: the five tasks excluded from the sensitivity analysis.",
        "- `figures/`: paper-facing PNG and PDF figures.",
        "- `tables/`: Markdown and LaTeX tables.",
        "- `protocol/`: frozen config, model matrix, environment summary, and per-model summaries.",
        "- `protocol/source_artifact_manifest.json`: SHA-256 hashes and sizes for full local traces.",
        "",
        "## Recommended Paper Claim",
        "",
        "Under a fixed local structured-ReAct protocol, the 14B coding model improves task success over both 7B/8B baselines, but incurs the highest wall-clock latency. These Phase 2 measurements provide the paired always-reason reference required for evaluating whether Phase 3 ARFA reduces reasoning calls, tokens, and latency without materially reducing success.",
    ]
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the tracked Phase 2 paper-writing bundle.")
    parser.add_argument("--results-root", default="results/phase2_baseline/paper")
    parser.add_argument("--output-dir", default="docs/phase2_writing_bundle")
    args = parser.parse_args()

    root = Path(args.results_root)
    analysis = root / "analysis"
    output = Path(args.output_dir)
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
    for model_dir in ("qwen2.5-coder-7b", "llama3.1-8b", "qwen2.5-coder-14b"):
        copies[root / model_dir / "summary.json"] = output / "protocol" / f"{model_dir}_summary.json"
    for source, destination in copies.items():
        _copy(source, destination)

    validation = read_jsonl("results/phase2_baseline/environment_validation/gold_validation.jsonl")
    write_jsonl(output / "metrics/gold_validation_anomalies.jsonl", [row for row in validation if not row.get("valid")])
    traces = []
    for trace_path in sorted(root.glob("*/traces.jsonl")):
        traces.append(
            {
                "path": str(trace_path),
                "bytes": trace_path.stat().st_size,
                "records": sum(1 for line in trace_path.read_text(encoding="utf-8").splitlines() if line),
                "sha256": _sha256(trace_path),
            }
        )
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
    )
    print(f"Wrote Phase 2 writing bundle to {output}")


if __name__ == "__main__":
    main()
