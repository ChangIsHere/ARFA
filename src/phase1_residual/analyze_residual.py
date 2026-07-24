from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from src.common.config import load_simple_yaml
from src.common.utils import ensure_parent, read_csv, write_jsonl


METHODS = {
    "embedding": "embedding_residual",
    "rule": "rule_residual",
    "hybrid": "hybrid_residual",
}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def _metrics(y_true: list[bool], y_score: list[float], threshold: float) -> dict[str, Any]:
    y_pred = [score >= threshold for score in y_score]
    tp = sum(pred and true for pred, true in zip(y_pred, y_true))
    tn = sum((not pred) and (not true) for pred, true in zip(y_pred, y_true))
    fp = sum(pred and (not true) for pred, true in zip(y_pred, y_true))
    fn = sum((not pred) and true for pred, true in zip(y_pred, y_true))
    total = len(y_true)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    false_fast_rate = fn / (tp + fn) if tp + fn else 0.0
    false_slow_rate = fp / (tn + fp) if tn + fp else 0.0
    try:
        roc_auc = float(roc_auc_score([int(v) for v in y_true], y_score))
    except Exception:
        roc_auc = None

    return {
        "threshold": threshold,
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_fast_rate": false_fast_rate,
        "false_slow_rate": false_slow_rate,
        "roc_auc": roc_auc,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def _candidate_thresholds(scores: list[float]) -> list[float]:
    unique = sorted(set(scores))
    if not unique:
        return [0.5]
    candidates = [max(0.0, unique[0] - 1e-6), min(1.0, unique[-1] + 1e-6)]
    candidates.extend(unique)
    for left, right in zip(unique, unique[1:]):
        candidates.append((left + right) / 2)
    return sorted(set(round(candidate, 6) for candidate in candidates))


def _select_threshold(y_true: list[bool], y_score: list[float]) -> tuple[float, dict[str, Any]]:
    evaluated = [_metrics(y_true, y_score, threshold) for threshold in _candidate_thresholds(y_score)]
    # For this project, unsafe false-fast errors matter first; then optimize F1 and accuracy.
    best = max(
        evaluated,
        key=lambda item: (
            -item["false_fast_rate"],
            item["f1"],
            item["accuracy"],
            -item["false_slow_rate"],
        ),
    )
    return float(best["threshold"]), best


def _threshold_for(config_value: Any, y_true: list[bool], y_score: list[float]) -> tuple[float, dict[str, Any], bool]:
    if str(config_value).lower() == "auto":
        threshold, metrics = _select_threshold(y_true, y_score)
        return threshold, metrics, True
    threshold = float(config_value)
    return threshold, _metrics(y_true, y_score, threshold), False


def _plot_distribution(rows: list[dict[str, str]], output: Path) -> None:
    needed = [float(row["hybrid_residual"]) for row in rows if _as_bool(row["reasoning_needed"])]
    not_needed = [float(row["hybrid_residual"]) for row in rows if not _as_bool(row["reasoning_needed"])]

    ensure_parent(output)
    plt.figure(figsize=(8, 5))
    plt.hist(not_needed, bins=10, alpha=0.70, label="reasoning not needed")
    plt.hist(needed, bins=10, alpha=0.70, label="reasoning needed")
    plt.xlabel("Hybrid residual")
    plt.ylabel("Count")
    plt.title("Phase 1 Residual Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def _plot_confusion(confusion: dict[str, int], output: Path) -> None:
    matrix = np.array([[confusion["tn"], confusion["fp"]], [confusion["fn"], confusion["tp"]]])
    ensure_parent(output)
    plt.figure(figsize=(5, 4))
    plt.imshow(matrix, cmap="Blues")
    plt.xticks([0, 1], ["pred false", "pred true"])
    plt.yticks([0, 1], ["actual false", "actual true"])
    for row in range(2):
        for col in range(2):
            plt.text(col, row, str(matrix[row, col]), ha="center", va="center", color="black")
    plt.title("Hybrid Residual Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def _write_report(path: Path, rows: list[dict[str, str]], metrics: dict[str, Any], false_fast: list[dict[str, Any]]) -> None:
    ensure_parent(path)
    backend = rows[0].get("embedding_backend", "unknown") if rows else "unknown"
    hybrid = metrics["hybrid"]
    conclusion = (
        "Residual shows enough signal to continue to Phase 2 review."
        if hybrid["f1"] >= 0.70 and hybrid["false_fast_rate"] <= 0.25
        else "Residual needs more dataset work or threshold tuning before moving forward."
    )
    lines = [
        "# Phase 1 Report",
        "",
        "## Dataset",
        "",
        f"- Records: {len(rows)}",
        f"- Reasoning needed: {sum(_as_bool(row['reasoning_needed']) for row in rows)}",
        f"- Reasoning not needed: {sum(not _as_bool(row['reasoning_needed']) for row in rows)}",
        f"- Embedding backend: `{backend}`",
        "",
        "## Metrics",
        "",
        "| Method | Threshold | Source | Accuracy | Precision | Recall | F1 | False-fast | False-slow | ROC-AUC |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method_name, method_metrics in metrics.items():
        roc_auc = method_metrics["roc_auc"]
        roc_text = f"{roc_auc:.3f}" if roc_auc is not None else "n/a"
        threshold_source = "auto" if method_metrics.get("threshold_auto") else "config"
        lines.append(
            f"| {method_name} | {method_metrics['threshold']:.3f} | {threshold_source} | {method_metrics['accuracy']:.3f} | "
            f"{method_metrics['precision']:.3f} | {method_metrics['recall']:.3f} | {method_metrics['f1']:.3f} | "
            f"{method_metrics['false_fast_rate']:.3f} | {method_metrics['false_slow_rate']:.3f} | {roc_text} |"
        )
    lines.extend(
        [
            "",
            "## False-fast Cases",
            "",
            "False-fast means the residual method predicted that reasoning was unnecessary when the label says reasoning was needed.",
            "",
        ]
    )
    if false_fast:
        for case in false_fast[:10]:
            lines.append(f"- `{case['task_id']}` `{case['scenario']}` score={case['hybrid_residual']}: {case['label_reason']}")
    else:
        lines.append("- No false-fast cases under the current hybrid threshold.")
    lines.extend(
        [
            "",
            "## Failure-Case Analysis",
            "",
            "Manual review should focus first on false-fast cases because they represent unsafe skipping of reasoning. False-slow cases are less dangerous but reduce efficiency gains.",
            "",
            "## Conclusion",
            "",
            conclusion,
            "",
            "Phase 2 remains locked until this report and the false-fast cases are reviewed.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(config_path: str) -> dict[str, Any]:
    config = load_simple_yaml(config_path)
    rows = read_csv(config["analysis"]["scores_path"])
    y_true = [_as_bool(row["reasoning_needed"]) for row in rows]

    threshold_config = {
        "embedding": config["residual"]["embedding_threshold"],
        "rule": config["residual"]["rule_threshold"],
        "hybrid": config["residual"]["hybrid_threshold"],
    }
    all_metrics = {}
    thresholds = {}
    for method, column in METHODS.items():
        scores = [float(row[column]) for row in rows]
        threshold, method_metrics, threshold_auto = _threshold_for(threshold_config[method], y_true, scores)
        method_metrics["threshold_auto"] = threshold_auto
        all_metrics[method] = method_metrics
        thresholds[method] = threshold

    hybrid_threshold = thresholds["hybrid"]
    false_fast = [
        row
        for row in rows
        if _as_bool(row["reasoning_needed"]) and float(row["hybrid_residual"]) < hybrid_threshold
    ]

    metrics_path = ensure_parent(config["analysis"]["metrics_path"])
    metrics_payload = {
        "record_count": len(rows),
        "method_metrics": all_metrics,
        "false_fast_count": len(false_fast),
        "false_fast_cases_path": config["analysis"]["false_fast_path"],
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    write_jsonl(config["analysis"]["false_fast_path"], false_fast)
    _plot_distribution(rows, Path(config["analysis"]["distribution_plot_path"]))
    _plot_confusion(all_metrics["hybrid"]["confusion"], Path(config["analysis"]["confusion_matrix_path"]))
    _write_report(Path(config["analysis"]["report_path"]), rows, all_metrics, false_fast)
    return metrics_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/phase1_residual.yaml")
    args = parser.parse_args()
    payload = analyze(args.config)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
