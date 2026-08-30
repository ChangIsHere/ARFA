from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.common.utils import ensure_parent, read_csv


FINAL_DIR = Path("results/phase1_residual/final")
OUT_DIR = FINAL_DIR / "paper_figures"
TABLE_DIR = FINAL_DIR / "paper_tables"

COLORS = {
    "true": "#C44536",
    "false": "#2A9D8F",
    "ambiguous": "#8D99AE",
    "hybrid": "#264653",
    "baseline": "#6C757D",
    "accent": "#E9C46A",
    "secondary": "#457B9D",
}


def _pretty(value: str) -> str:
    return value.replace("_", " ").title()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _savefig(name: str) -> None:
    ensure_parent(OUT_DIR / f"{name}.png")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{name}.png", dpi=220)
    plt.savefig(OUT_DIR / f"{name}.pdf")
    plt.close()


def _float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value in {"", "None", None}:
        return float("nan")
    return float(value)


def _fmt_rate(row: dict[str, str], key: str) -> str:
    if key == "safe_fast_precision" and _float(row, "fast_path_rate") == 0:
        return "n/a"
    return f"{_float(row, key):.3f}"


def _plot_dataset_overview(summary: dict[str, Any]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Phase 1 External Dataset Overview", fontsize=15, fontweight="bold")

    class_items = summary["class_distribution"]
    class_order = ["true", "false", "ambiguous"]
    axes[0, 0].bar(
        [_pretty(k) for k in class_order],
        [class_items.get(k, 0) for k in class_order],
        color=[COLORS[k] for k in class_order],
    )
    axes[0, 0].set_title("Annotation Labels")
    axes[0, 0].set_ylabel("Execution steps")

    split_items = summary["split_record_counts"]
    axes[0, 1].bar(
        [_pretty(k) for k in split_items],
        list(split_items.values()),
        color=[COLORS["secondary"], COLORS["accent"], COLORS["hybrid"]],
    )
    axes[0, 1].set_title("Task-Level Split Size")

    task_items = sorted(summary["task_category_distribution"].items(), key=lambda x: x[1])
    axes[1, 0].barh([_pretty(k) for k, _ in task_items], [v for _, v in task_items], color=COLORS["secondary"])
    axes[1, 0].set_title("Task Categories")
    axes[1, 0].set_xlabel("Execution steps")

    step_items = sorted(summary["step_type_distribution"].items(), key=lambda x: x[1])
    axes[1, 1].barh([_pretty(k) for k, _ in step_items], [v for _, v in step_items], color=COLORS["accent"])
    axes[1, 1].set_title("Step Types")
    axes[1, 1].set_xlabel("Execution steps")

    for ax in axes.flat:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    _savefig("fig1_dataset_overview")


def _plot_main_results(metrics: list[dict[str, str]]) -> None:
    methods = ["always_reason", "never_reason", "keyword", "tfidf", "structured", "hybrid"]
    rows = {row["method"]: row for row in metrics}
    labels = ["Always", "Never", "Keyword", "TF-IDF", "Structured", "Hybrid"]
    x = np.arange(len(methods))
    width = 0.23

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - width, [_float(rows[m], "f1") for m in methods], width, label="F1", color=COLORS["secondary"])
    ax.bar(x, [_float(rows[m], "pr_auc") for m in methods], width, label="PR-AUC", color=COLORS["accent"])
    ax.bar(x + width, [_float(rows[m], "balanced_accuracy") for m in methods], width, label="Balanced Acc.", color=COLORS["hybrid"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Held-Out Test Performance by Method")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _savefig("fig2_main_metrics")


def _plot_safety_results(metrics: list[dict[str, str]]) -> None:
    methods = ["keyword", "tfidf", "structured", "hybrid"]
    rows = {row["method"]: row for row in metrics}
    labels = ["Keyword", "TF-IDF", "Structured", "Hybrid"]
    x = np.arange(len(methods))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, [_float(rows[m], "false_fast_rate") for m in methods], width, label="False-fast rate", color=COLORS["true"])
    ax.bar(x, [_float(rows[m], "fast_path_rate") for m in methods], width, label="Fast-path rate", color=COLORS["false"])
    ax.bar(x + width, [_float(rows[m], "safe_fast_precision") for m in methods], width, label="Safe-fast precision", color=COLORS["hybrid"])
    ax.axhline(0.05, color="black", linestyle="--", linewidth=1, label="5% false-fast gate")
    ax.axhline(0.15, color="gray", linestyle=":", linewidth=1, label="15% fast-path gate")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title("Safety-Efficiency Tradeoff on Held-Out Test")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _savefig("fig3_safety_tradeoff")


def _plot_ablation(ablation: list[dict[str, str]]) -> None:
    labels = {
        "semantic_component_only": "Semantic only",
        "structured_component_only": "Structured only",
        "hybrid_without_exit_code": "No exit code",
        "hybrid_without_test_status": "No test status",
        "hybrid_without_error_keywords": "No error keywords",
        "hybrid_without_expectation_text": "No expectation",
        "full_hybrid": "Full hybrid",
    }
    rows = [row for row in ablation if row["ablation"] in labels]
    x = np.arange(len(rows))

    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    ax1.bar(x, [_float(row, "pr_auc") for row in rows], color=COLORS["secondary"], label="PR-AUC")
    ax1.set_ylabel("PR-AUC")
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    ax2.plot(x, [_float(row, "false_fast_rate") for row in rows], color=COLORS["true"], marker="o", label="False-fast")
    ax2.plot(x, [_float(row, "fast_path_rate") for row in rows], color=COLORS["false"], marker="s", label="Fast-path")
    ax2.set_ylabel("Rate")
    ax2.set_ylim(0, 1.05)

    ax1.set_title("Hybrid Component Ablation")
    ax1.set_xticks(x)
    ax1.set_xticklabels([labels[row["ablation"]] for row in rows], rotation=25, ha="right")
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    lines, line_labels = ax1.get_legend_handles_labels()
    lines2, line_labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, line_labels + line_labels2, frameon=False, loc="upper left", ncol=3)
    _savefig("fig4_ablation")


def _plot_subgroup(category_rows: list[dict[str, str]], step_rows: list[dict[str, str]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, rows, title in [
        (axes[0], category_rows, "By Task Category"),
        (axes[1], step_rows, "By Step Type"),
    ]:
        usable = [row for row in rows if row.get("flag", "") == ""]
        usable = sorted(usable, key=lambda r: _float(r, "f1"))
        labels = [_pretty(row["group"]) for row in usable]
        y = np.arange(len(usable))
        ax.barh(y - 0.18, [_float(row, "f1") for row in usable], height=0.36, label="F1", color=COLORS["secondary"])
        ax.barh(y + 0.18, [_float(row, "false_fast_rate") for row in usable], height=0.36, label="False-fast", color=COLORS["true"])
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlim(0, 1.05)
        ax.set_title(title)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, loc="lower right")
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("Held-Out Hybrid Performance Across Subgroups", fontsize=14, fontweight="bold")
    _savefig("fig5_subgroup_results")


def _plot_threshold_operating_points(thresholds: dict[str, Any]) -> None:
    ops = thresholds["methods"]["hybrid"]["operating_points"]
    labels = ["1%", "5%", "10%"]
    keys = ["0.01", "0.05", "0.1"]
    fast = [ops[k]["validation_metrics"]["fast_path_rate"] for k in keys]
    ff = [ops[k]["validation_metrics"]["false_fast_rate"] for k in keys]
    safe = [ops[k]["validation_metrics"]["safe_fast_precision"] for k in keys]

    x = np.arange(len(keys))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, ff, width, label="False-fast", color=COLORS["true"])
    ax.bar(x, fast, width, label="Fast-path", color=COLORS["false"])
    ax.bar(x + width, safe, width, label="Safe-fast precision", color=COLORS["hybrid"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Validation false-fast constraint")
    ax.set_ylabel("Validation rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Selected Hybrid Operating Points")
    ax.legend(frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _savefig("fig6_operating_points")


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out) + "\n"


def _latex_table(headers: list[str], rows: list[list[str]]) -> str:
    cols = "l" + "r" * (len(headers) - 1)
    lines = [f"\\begin{{tabular}}{{{cols}}}", "\\toprule", " & ".join(headers) + " \\\\", "\\midrule"]
    lines.extend(" & ".join(row) + " \\\\" for row in rows)
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def _write_tables(summary: dict[str, Any], metrics: list[dict[str, str]], ablation: list[dict[str, str]]) -> None:
    ensure_parent(TABLE_DIR / "placeholder")
    dataset_rows = [
        ["Total execution steps", str(summary["record_count"])],
        ["Binary-labeled steps", str(summary["binary_record_count"])],
        ["Ambiguous steps retained", str(summary["ambiguous_count"])],
        ["Distinct tasks", str(summary["distinct_task_count"])],
        ["Distinct trajectories", str(summary["distinct_trajectory_count"])],
        ["Reasoning needed", str(summary["class_distribution"].get("true", 0))],
        ["Reasoning not needed", str(summary["class_distribution"].get("false", 0))],
    ]
    (TABLE_DIR / "table1_dataset_summary.md").write_text(_markdown_table(["Statistic", "Value"], dataset_rows), encoding="utf-8")
    (TABLE_DIR / "table1_dataset_summary.tex").write_text(_latex_table(["Statistic", "Value"], dataset_rows), encoding="utf-8")

    method_order = ["always_reason", "never_reason", "keyword", "tfidf", "structured", "hybrid"]
    by_method = {row["method"]: row for row in metrics}
    result_rows = []
    for method in method_order:
        row = by_method[method]
        result_rows.append(
            [
                method.replace("_", "-"),
                f"{_float(row, 'f1'):.3f}",
                f"{_float(row, 'pr_auc'):.3f}",
                f"{_float(row, 'false_fast_rate'):.3f}",
                f"{_float(row, 'fast_path_rate'):.3f}",
                _fmt_rate(row, "safe_fast_precision"),
            ]
        )
    headers = ["Method", "F1", "PR-AUC", "False-fast", "Fast-path", "Safe-fast Prec."]
    (TABLE_DIR / "table2_main_results.md").write_text(_markdown_table(headers, result_rows), encoding="utf-8")
    (TABLE_DIR / "table2_main_results.tex").write_text(_latex_table(headers, result_rows), encoding="utf-8")

    ablation_rows = []
    for row in ablation:
        ablation_rows.append(
            [
                row["ablation"].replace("_", " "),
                f"{_float(row, 'pr_auc'):.3f}",
                f"{_float(row, 'false_fast_rate'):.3f}",
                f"{_float(row, 'fast_path_rate'):.3f}",
            ]
        )
    headers = ["Ablation", "PR-AUC", "False-fast", "Fast-path"]
    (TABLE_DIR / "table3_ablation.md").write_text(_markdown_table(headers, ablation_rows), encoding="utf-8")
    (TABLE_DIR / "table3_ablation.tex").write_text(_latex_table(headers, ablation_rows), encoding="utf-8")


def _write_index(summary: dict[str, Any], backend: str) -> None:
    caveat = (
        "The final data source is external InterCode Bash trajectories, and this run uses the intended MiniLM embedding backend. Remaining caveats are annotation quality and benchmark scope."
        if "sentence-transformers/all-MiniLM-L6-v2" in backend
        else f"The final data source is external InterCode Bash trajectories, but the current semantic backend is `{backend}`. Do not describe this run as a final MiniLM embedding result until `sentence-transformers/all-MiniLM-L6-v2` is installed and the experiment is rerun."
    )
    text = f"""# Phase 1 Paper Artifacts

Generated from `results/phase1_residual/final/`.

## Important Caveat

{caveat}

## Dataset Snapshot

- Records: {summary['record_count']}
- Binary records: {summary['binary_record_count']}
- Ambiguous retained: {summary['ambiguous_count']}
- Distinct tasks: {summary['distinct_task_count']}
- Distinct trajectories: {summary['distinct_trajectory_count']}

## Suggested Figure Use

- `fig1_dataset_overview`: dataset and split description.
- `fig2_main_metrics`: main held-out method comparison.
- `fig3_safety_tradeoff`: ARFA-relevant safety/efficiency tradeoff.
- `fig4_ablation`: component contribution analysis.
- `fig5_subgroup_results`: subgroup robustness and collapse check.
- `fig6_operating_points`: validation threshold policy.

Tables are in `results/phase1_residual/final/paper_tables/`.
"""
    (FINAL_DIR / "paper_artifacts.md").write_text(text, encoding="utf-8")


def main() -> None:
    global FINAL_DIR, OUT_DIR, TABLE_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument("--final-dir", default=str(FINAL_DIR))
    args = parser.parse_args()

    FINAL_DIR = Path(args.final_dir)
    OUT_DIR = FINAL_DIR / "paper_figures"
    TABLE_DIR = FINAL_DIR / "paper_tables"

    summary = _load_json(FINAL_DIR / "dataset_summary.json")
    thresholds = _load_json(FINAL_DIR / "thresholds.json")
    metrics = read_csv(FINAL_DIR / "test_metrics.csv")
    ablation = read_csv(FINAL_DIR / "ablation_results.csv")
    category_rows = read_csv(FINAL_DIR / "results_by_task_category.csv")
    step_rows = read_csv(FINAL_DIR / "results_by_step_type.csv")
    scores = read_csv(FINAL_DIR / "residual_scores.csv")
    backend_counts = Counter(row.get("embedding_backend", "unknown") for row in scores)
    backend = ", ".join(f"{k}: {v}" for k, v in backend_counts.items())

    _plot_dataset_overview(summary)
    _plot_main_results(metrics)
    _plot_safety_results(metrics)
    _plot_ablation(ablation)
    _plot_subgroup(category_rows, step_rows)
    _plot_threshold_operating_points(thresholds)
    _write_tables(summary, metrics, ablation)
    _write_index(summary, backend)
    print(f"Wrote paper figures to {OUT_DIR}")
    print(f"Wrote paper tables to {TABLE_DIR}")


if __name__ == "__main__":
    main()
