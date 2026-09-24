from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.common.utils import read_csv
from src.phase2_baseline.analyze_paper_results import MODEL_LABELS, PLANNED_PAIRS


def _label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def _colors(count: int):
    palette = plt.get_cmap("tab10")
    return [palette(index % 10) for index in range(count)]


def _save(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _main_results(rows: list[dict[str, str]], output_dir: Path) -> None:
    labels = [_label(row["model"]) for row in rows]
    rates = np.array([float(row["success_rate"]) for row in rows])
    lows = np.array([float(row["success_ci95_low"]) for row in rows])
    highs = np.array([float(row["success_ci95_high"]) for row in rows])
    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 1.65), 5.3))
    x = np.arange(len(rows))
    ax.bar(x, rates, color=_colors(len(rows)), width=0.62)
    ax.errorbar(x, rates, yerr=np.vstack([rates - lows, highs - rates]), fmt="none", color="#202020", capsize=5)
    ax.set_xticks(x, labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0, min(1.0, float(max(highs)) + 0.10))
    ax.set_ylabel("Task success rate")
    ax.set_title("Standard ReAct Baseline on InterCode NL2Bash")
    ax.spines[["top", "right"]].set_visible(False)
    for idx, rate in enumerate(rates):
        ax.text(idx, highs[idx] + 0.015, f"{rate:.1%}", ha="center", fontsize=10)
    _save(fig, output_dir, "fig1_phase2_success")


def _efficiency(rows: list[dict[str, str]], output_dir: Path) -> None:
    labels = [_label(row["model"]) for row in rows]
    metrics = [
        ("mean_model_calls", "Calls per task"),
        ("mean_tokens", "Tokens per task"),
        ("mean_task_wall_seconds", "Wall time per task (s)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(max(13, len(rows) * 2.3), 5.2))
    x = np.arange(len(rows))
    for ax, (field, title) in zip(axes, metrics):
        values = [float(row[field]) for row in rows]
        ax.bar(x, values, color=_colors(len(rows)), width=0.62)
        ax.set_xticks(x, labels, rotation=35, ha="right", fontsize=8)
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Baseline Computation Cost", fontsize=14, fontweight="bold")
    _save(fig, output_dir, "fig2_phase2_efficiency")


def _category_heatmap(rows: list[dict[str, str]], output_dir: Path) -> None:
    models = list(dict.fromkeys(row["model"] for row in rows))
    categories = sorted(dict.fromkeys(row["task_category"] for row in rows))
    lookup = {(row["model"], row["task_category"]): float(row["success_rate"]) for row in rows}
    matrix = np.array([[lookup.get((model, category), np.nan) for category in categories] for model in models])
    fig, ax = plt.subplots(figsize=(11, max(4.2, len(models) * 0.55 + 1.8)))
    image = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(categories)), [c.replace("_", " ").title() for c in categories], rotation=28, ha="right")
    ax.set_yticks(np.arange(len(models)), [_label(model) for model in models])
    ax.set_title("Success Rate by Task Category")
    for i in range(len(models)):
        for j in range(len(categories)):
            value = matrix[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white" if value > 0.55 else "#202020", fontsize=8)
    fig.colorbar(image, ax=ax, label="Success rate", fraction=0.025, pad=0.03)
    _save(fig, output_dir, "fig3_phase2_categories")


def _failure_modes(rows: list[dict[str, str]], output_dir: Path) -> None:
    models = list(dict.fromkeys(row["model"] for row in rows))
    categories = sorted(dict.fromkeys(row["failure_category"] for row in rows))
    lookup = {(row["model"], row["failure_category"]): int(row["count"]) for row in rows}
    fig, ax = plt.subplots(figsize=(12, max(4.8, len(models) * 0.58 + 2)))
    left = np.zeros(len(models))
    palette = plt.get_cmap("tab10")
    for index, category in enumerate(categories):
        values = np.array([lookup.get((model, category), 0) for model in models])
        ax.barh(np.arange(len(models)), values, left=left, label=category.replace("_", " "), color=palette(index))
        left += values
    ax.set_yticks(np.arange(len(models)), [_label(model) for model in models])
    ax.invert_yaxis()
    ax.set_xlabel("Failed tasks")
    ax.set_title("Trace-Derived Failure Diagnostics")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    _save(fig, output_dir, "fig4_phase2_failures")


def _write_tables(
    rows: list[dict[str, str]],
    paired_rows: list[dict[str, str]],
    table_dir: Path,
) -> None:
    table_dir.mkdir(parents=True, exist_ok=True)
    planned = {frozenset(pair) for pair in PLANNED_PAIRS}
    paired_rows = [row for row in paired_rows if frozenset((row["model_a"], row["model_b"])) in planned]
    headers = ["Model", "Success", "95% CI", "Mean reward", "Calls/task", "Tokens/task", "Wall/task (s)"]
    body = [
        [
            _label(row["model"]),
            f"{float(row['success_rate']):.3f}",
            f"[{float(row['success_ci95_low']):.3f}, {float(row['success_ci95_high']):.3f}]",
            f"{float(row['mean_reward']):.3f}",
            f"{float(row['mean_model_calls']):.2f}",
            f"{float(row['mean_tokens']):.1f}",
            f"{float(row['mean_task_wall_seconds']):.2f}",
        ]
        for row in rows
    ]
    markdown = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |"]
    markdown.extend("| " + " | ".join(row) + " |" for row in body)
    (table_dir / "table1_phase2_main_results.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    latex_headers = ["Model", "Success", "95\\% CI", "Mean reward", "Calls/task", "Tokens/task", "Wall/task (s)"]
    latex = ["\\begin{tabular}{lrrrrrr}", "\\toprule", " & ".join(latex_headers) + " \\\\", "\\midrule"]
    latex.extend(" & ".join(row) + " \\\\" for row in body)
    latex.extend(["\\bottomrule", "\\end{tabular}", ""])
    (table_dir / "table1_phase2_main_results.tex").write_text("\n".join(latex), encoding="utf-8")

    paired_headers = ["Model A", "Model B", "A-B success", "95% paired CI", "McNemar p"]
    paired_body = [
        [
            _label(row["model_a"]),
            _label(row["model_b"]),
            f"{float(row['success_rate_difference_a_minus_b']):.3f}",
            f"[{float(row['bootstrap_ci95_low']):.3f}, {float(row['bootstrap_ci95_high']):.3f}]",
            f"{float(row['exact_mcnemar_pvalue']):.4g}",
        ]
        for row in paired_rows
    ]
    paired_markdown = [
        "| " + " | ".join(paired_headers) + " |",
        "| " + " | ".join(["---", "---", "---:", "---:", "---:"]) + " |",
    ]
    paired_markdown.extend("| " + " | ".join(row) + " |" for row in paired_body)
    (table_dir / "table2_phase2_paired_comparisons.md").write_text(
        "\n".join(paired_markdown) + "\n", encoding="utf-8"
    )
    paired_latex = [
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "Model A & Model B & A-B success & 95\\% paired CI & McNemar $p$ \\\\",
        "\\midrule",
    ]
    paired_latex.extend(" & ".join(row) + " \\\\" for row in paired_body)
    paired_latex.extend(["\\bottomrule", "\\end{tabular}", ""])
    (table_dir / "table2_phase2_paired_comparisons.tex").write_text(
        "\n".join(paired_latex), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Phase 2 paper-facing figures and tables.")
    parser.add_argument("--analysis-dir", default="results/phase2_baseline/paper/analysis")
    args = parser.parse_args()
    analysis_dir = Path(args.analysis_dir)
    main_rows = read_csv(analysis_dir / "main_results.csv")
    paired_rows = read_csv(analysis_dir / "paired_model_comparisons.csv")
    category_rows = read_csv(analysis_dir / "results_by_task_category.csv")
    failure_rows = read_csv(analysis_dir / "failure_counts.csv")
    figures = analysis_dir / "paper_figures"
    _main_results(main_rows, figures)
    _efficiency(main_rows, figures)
    _category_heatmap(category_rows, figures)
    _failure_modes(failure_rows, figures)
    _write_tables(main_rows, paired_rows, analysis_dir / "paper_tables")
    print(f"Wrote Phase 2 paper artifacts to {analysis_dir}")


if __name__ == "__main__":
    main()
