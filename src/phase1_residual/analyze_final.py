from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve

from src.common.config import load_simple_yaml
from src.common.utils import ensure_parent, read_csv, write_csv, write_jsonl


METHOD_COLUMNS = {
    "always_reason": "always_reason_score",
    "never_reason": "never_reason_score",
    "keyword": "keyword_residual",
    "tfidf": "tfidf_residual",
    "embedding": "embedding_residual",
    "structured": "structured_residual",
    "hybrid": "hybrid_residual",
}

ABLATION_COLUMNS = {
    "semantic_component_only": "embedding_residual",
    "structured_component_only": "structured_residual",
    "hybrid_without_exit_code": "hybrid_no_exit_code",
    "hybrid_without_test_status": "hybrid_no_test_status",
    "hybrid_without_error_keywords": "hybrid_no_error_keywords",
    "hybrid_without_expectation_text": "hybrid_no_expectation_text",
    "full_hybrid": "hybrid_residual",
}


def _as_label(row: dict[str, str]) -> bool | None:
    value = row["reasoning_needed"].lower()
    if value == "true":
        return True
    if value == "false":
        return False
    return None


def _split_rows(rows: list[dict[str, str]], split_task_ids: list[str]) -> list[dict[str, str]]:
    tasks = set(split_task_ids)
    return [row for row in rows if row["task_id"] in tasks]


def _binary_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if _as_label(row) is not None]


def _scores(rows: list[dict[str, str]], column: str) -> list[float]:
    return [float(row[column]) for row in rows]


def _labels(rows: list[dict[str, str]]) -> list[bool]:
    return [bool(_as_label(row)) for row in rows]


def _confusion(y_true: list[bool], y_pred: list[bool]) -> dict[str, int]:
    return {
        "tp": sum(p and t for p, t in zip(y_pred, y_true)),
        "tn": sum((not p) and (not t) for p, t in zip(y_pred, y_true)),
        "fp": sum(p and (not t) for p, t in zip(y_pred, y_true)),
        "fn": sum((not p) and t for p, t in zip(y_pred, y_true)),
    }


def _metrics(rows: list[dict[str, str]], column: str, threshold: float) -> dict[str, Any]:
    y_true = _labels(rows)
    y_score = _scores(rows, column)
    y_pred = [score >= threshold for score in y_score]
    c = _confusion(y_true, y_pred)
    tp, tn, fp, fn = c["tp"], c["tn"], c["fp"], c["fn"]
    total = len(rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    false_fast_rate = fn / (tp + fn) if tp + fn else 0.0
    false_slow_rate = fp / (tn + fp) if tn + fp else 0.0
    fast_path = sum(not pred for pred in y_pred)
    safe_fast = tn / (tn + fn) if tn + fn else 0.0
    try:
        roc_auc = float(roc_auc_score([int(v) for v in y_true], y_score))
    except Exception:
        roc_auc = None
    try:
        precision_curve, recall_curve, _ = precision_recall_curve([int(v) for v in y_true], y_score)
        pr_auc = float(auc(recall_curve, precision_curve))
    except Exception:
        pr_auc = None
    return {
        "threshold": threshold,
        "accuracy": (tp + tn) / total if total else 0.0,
        "balanced_accuracy": (recall + specificity) / 2,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "false_fast_count": fn,
        "false_fast_rate": false_fast_rate,
        "false_slow_count": fp,
        "false_slow_rate": false_slow_rate,
        "fast_path_rate": fast_path / total if total else 0.0,
        "safe_fast_precision": safe_fast,
        "confusion_matrix": c,
    }


def _candidate_thresholds(rows: list[dict[str, str]], column: str) -> list[float]:
    unique = sorted(set(_scores(rows, column)))
    if not unique:
        return [0.5]
    candidates = [max(0.0, unique[0] - 1e-6), min(1.0, unique[-1] + 1e-6)]
    candidates.extend(unique)
    candidates.extend((left + right) / 2 for left, right in zip(unique, unique[1:]))
    return sorted(set(round(value, 6) for value in candidates if math.isfinite(value)))


def _select_threshold(rows: list[dict[str, str]], column: str, false_fast_limit: float) -> tuple[float, dict[str, Any]]:
    candidates = []
    for threshold in _candidate_thresholds(rows, column):
        item = _metrics(rows, column, threshold)
        if item["false_fast_rate"] <= false_fast_limit:
            candidates.append(item)
    if not candidates:
        threshold = min(_scores(rows, column)) - 1e-6
        return threshold, _metrics(rows, column, threshold)
    best = max(candidates, key=lambda item: (item["fast_path_rate"], item["safe_fast_precision"], item["f1"]))
    return float(best["threshold"]), best


def _write_thresholds(validation_rows: list[dict[str, str]], output: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {"selection_split": "validation", "policy": "maximize fast_path_rate subject to validation false_fast_rate <= constraint", "methods": {}}
    for method, column in METHOD_COLUMNS.items():
        constraints = {}
        for limit in [0.01, 0.05, 0.10]:
            if method == "always_reason":
                threshold = 0.5
                metrics = _metrics(validation_rows, column, threshold)
            elif method == "never_reason":
                threshold = 0.5
                metrics = _metrics(validation_rows, column, threshold)
            else:
                threshold, metrics = _select_threshold(validation_rows, column, limit)
            constraints[str(limit)] = {"threshold": threshold, "validation_metrics": metrics}
        payload["methods"][method] = {"primary_constraint": "0.05", "operating_points": constraints}
    ensure_parent(output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _metrics_rows(test_rows: list[dict[str, str]], thresholds: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for method, column in METHOD_COLUMNS.items():
        threshold = thresholds["methods"][method]["operating_points"]["0.05"]["threshold"]
        metrics = _metrics(test_rows, column, threshold)
        rows.append({"method": method, **_flat_metrics(metrics)})
    return rows


def _flat_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    flat = {key: value for key, value in metrics.items() if key != "confusion_matrix"}
    for key, value in metrics["confusion_matrix"].items():
        flat[f"cm_{key}"] = value
    return flat


def _bootstrap_ci(test_rows: list[dict[str, str]], thresholds: dict[str, Any], seed: int, n_boot: int = 200) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_task: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in test_rows:
        by_task[row["task_id"]].append(row)
    tasks = sorted(by_task)
    output = []
    for method, column in METHOD_COLUMNS.items():
        threshold = thresholds["methods"][method]["operating_points"]["0.05"]["threshold"]
        samples: dict[str, list[float]] = defaultdict(list)
        for _ in range(n_boot):
            sampled_rows = []
            for task_id in (rng.choice(tasks) for _ in tasks):
                sampled_rows.extend(by_task[task_id])
            metrics = _metrics(sampled_rows, column, threshold)
            for metric in ["accuracy", "balanced_accuracy", "f1", "pr_auc", "false_fast_rate", "fast_path_rate", "safe_fast_precision"]:
                if metrics[metric] is not None:
                    samples[metric].append(float(metrics[metric]))
        for metric, values in samples.items():
            output.append(
                {
                    "method": method,
                    "metric": metric,
                    "ci_low": float(np.percentile(values, 2.5)),
                    "ci_high": float(np.percentile(values, 97.5)),
                    "bootstrap_unit": "task_id",
                    "n_bootstrap": n_boot,
                }
            )
    return output


def _subgroup(rows: list[dict[str, str]], group_column: str, threshold: float, score_column: str) -> list[dict[str, Any]]:
    output = []
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row[group_column]].append(row)
    for group, group_rows in sorted(groups.items()):
        binary = _binary_rows(group_rows)
        if len(binary) < 10 or len(set(_labels(binary))) < 2:
            output.append({"group": group, "n": len(binary), "flag": "too_few_or_single_class"})
            continue
        output.append({"group": group, "n": len(binary), "flag": "", **_flat_metrics(_metrics(binary, score_column, threshold))})
    return output


def _ablation_rows(validation_rows: list[dict[str, str]], test_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output = []
    for name, column in ABLATION_COLUMNS.items():
        threshold, validation_metrics = _select_threshold(validation_rows, column, 0.05)
        test_metrics = _metrics(test_rows, column, threshold)
        output.append({"ablation": name, "threshold": threshold, "validation_fast_path_rate": validation_metrics["fast_path_rate"], **_flat_metrics(test_metrics)})
    return output


def _prediction(row: dict[str, str], threshold: float, column: str = "hybrid_residual") -> bool:
    return float(row[column]) >= threshold


def _write_case_analysis(test_rows: list[dict[str, str]], threshold: float, paths: dict[str, str]) -> None:
    false_fast = []
    false_slow = []
    for row in test_rows:
        label = _as_label(row)
        pred = _prediction(row, threshold)
        if label is True and pred is False:
            false_fast.append(_case_payload(row, threshold, "missing_structured_rule"))
        if label is False and pred is True:
            false_slow.append(_case_payload(row, threshold, "conservative_threshold_or_semantic_mismatch"))
    write_jsonl(paths["false_fast"], false_fast)
    false_slow = sorted(false_slow, key=lambda item: abs(float(item["final_residual_score"]) - threshold))[:25]
    write_jsonl(paths["false_slow"], false_slow)
    lines = ["# Phase 1 Failure-Case Analysis", "", f"Hybrid selected threshold: `{threshold:.6f}`", "", f"False-fast cases on held-out test: {len(false_fast)}", f"False-slow sample size: {len(false_slow)}", ""]
    if false_fast:
        lines.append("## False-fast Cases")
        for case in false_fast:
            lines.append(f"- `{case['task_id']}` step {case['step_id']}: {case['suspected_failure_cause']}; mitigation: {case['proposed_mitigation']}")
    else:
        lines.append("No false-fast cases were observed under the selected operating point. This should still be rechecked after human annotation.")
    ensure_parent(paths["failure_analysis"]).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _case_payload(row: dict[str, str], threshold: float, suspected: str) -> dict[str, Any]:
    return {
        "task_id": row["task_id"],
        "step_id": row["step_id"],
        "task_category": row["task_category"],
        "step_type": row["step_type"],
        "action": row["action"],
        "expectation": row["expected_outcome"],
        "observation": row["actual_observation"],
        "exit_code": row["exit_code"],
        "component_residual_scores": {
            "keyword": row["keyword_residual"],
            "tfidf": row["tfidf_residual"],
            "embedding": row["embedding_residual"],
            "structured": row["structured_residual"],
        },
        "final_residual_score": row["hybrid_residual"],
        "selected_threshold": threshold,
        "why_reasoning_was_actually_necessary": row["label_reason"],
        "suspected_failure_cause": suspected,
        "proposed_mitigation": "Review expectation specificity, add source-specific structured signal, or require human adjudication for similar ambiguous states.",
    }


def _write_leakage_audit(path: Path, rows: list[dict[str, str]]) -> None:
    forbidden = ["reasoning_needed", "label_reason", "annotation_confidence"]
    leaked_columns = [column for column in forbidden if any(column in method_column for method_column in METHOD_COLUMNS.values())]
    source_counts = Counter(row["data_source"] for row in rows)
    lines = [
        "# Phase 1 Leakage Audit",
        "",
        "Residual feature extraction uses only expected outcome, action, raw observation, inferred exit code, and observable command text.",
        "",
        f"Forbidden label columns referenced by method score columns: `{leaked_columns}`",
        "",
        "Known risks:",
        "",
        "- Expectations are heuristic annotations derived from action text because InterCode trajectories do not store pre-execution expectations.",
        "- Exit codes are inferred from InterCode valid-action metadata and terminal text, not directly recorded shell exit statuses.",
        "- Thresholds are selected on validation only, then frozen for held-out test metrics.",
        "- The dataset uses external InterCode trajectories, but labels remain an ARFA annotation layer and need human review.",
        "",
        f"Data sources: `{dict(source_counts)}`",
    ]
    ensure_parent(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_curves(validation_rows: list[dict[str, str]], test_rows: list[dict[str, str]], threshold: float, paths: dict[str, str]) -> None:
    score_column = "hybrid_residual"
    y_true = [int(v) for v in _labels(test_rows)]
    y_score = _scores(test_rows, score_column)

    plt.figure(figsize=(7, 5))
    for label_value, name in [(False, "reasoning not needed"), (True, "reasoning needed")]:
        values = [float(row[score_column]) for row in test_rows if _as_label(row) is label_value]
        plt.hist(values, bins=12, alpha=0.65, label=name)
    plt.axvline(threshold, color="black", linestyle="--", label="selected threshold")
    plt.xlabel("Hybrid residual")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ensure_parent(paths["distribution"]), dpi=160)
    plt.close()

    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.tight_layout()
    plt.savefig(ensure_parent(paths["roc"]), dpi=160)
    plt.close()

    p, r, _ = precision_recall_curve(y_true, y_score)
    plt.figure(figsize=(6, 5))
    plt.plot(r, p)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.savefig(ensure_parent(paths["pr"]), dpi=160)
    plt.close()

    thresholds = _candidate_thresholds(validation_rows, score_column)
    tradeoff = [_metrics(validation_rows, score_column, value) for value in thresholds]
    plt.figure(figsize=(7, 5))
    plt.plot(thresholds, [item["false_fast_rate"] for item in tradeoff], label="false-fast")
    plt.plot(thresholds, [item["false_slow_rate"] for item in tradeoff], label="false-slow")
    plt.axvline(threshold, color="black", linestyle="--")
    plt.xlabel("Threshold")
    plt.ylabel("Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ensure_parent(paths["tradeoff"]), dpi=160)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot([item["false_fast_rate"] for item in tradeoff], [item["fast_path_rate"] for item in tradeoff])
    plt.scatter([_metrics(validation_rows, score_column, threshold)["false_fast_rate"]], [_metrics(validation_rows, score_column, threshold)["fast_path_rate"]], color="black")
    plt.xlabel("Validation false-fast rate")
    plt.ylabel("Validation fast-path rate")
    plt.tight_layout()
    plt.savefig(ensure_parent(paths["fast_safety"]), dpi=160)
    plt.close()

    c = _metrics(test_rows, score_column, threshold)["confusion_matrix"]
    matrix = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
    plt.figure(figsize=(5, 4))
    plt.imshow(matrix, cmap="Blues")
    plt.xticks([0, 1], ["pred false", "pred true"])
    plt.yticks([0, 1], ["actual false", "actual true"])
    for row in range(2):
        for col in range(2):
            plt.text(col, row, str(matrix[row, col]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(ensure_parent(paths["confusion"]), dpi=160)
    plt.close()


def _write_report(path: Path, dataset_summary: dict[str, Any], test_metric_rows: list[dict[str, Any]], thresholds: dict[str, Any], gate: dict[str, Any]) -> None:
    hybrid = next(row for row in test_metric_rows if row["method"] == "hybrid")
    lines = [
        "# Phase 1 Final Report",
        "",
        "## Status",
        "",
        "This is the external-data Phase 1 experiment. It tests whether residual is useful as a reasoning-trigger signal; it does not prove residual is necessary.",
        "",
        "## Data Source",
        "",
        "The final dataset is derived from external InterCode Bash trajectories downloaded from the official Princeton NLP InterCode repository.",
        "",
        f"- Records: {dataset_summary['record_count']}",
        f"- Binary records: {dataset_summary['binary_record_count']}",
        f"- Ambiguous records retained: {dataset_summary['ambiguous_count']}",
        f"- Distinct tasks: {dataset_summary['distinct_task_count']}",
        f"- Distinct trajectories: {dataset_summary['distinct_trajectory_count']}",
        f"- Class distribution: `{dataset_summary['class_distribution']}`",
        "",
        "## Threshold Policy",
        "",
        "Thresholds are selected on the validation split by maximizing predicted fast-path rate subject to a false-fast constraint. The primary constraint is 5%.",
        "",
        f"Hybrid primary validation threshold: `{thresholds['methods']['hybrid']['operating_points']['0.05']['threshold']:.6f}`",
        "",
        "## Held-Out Test Metrics",
        "",
        "| Method | Accuracy | Balanced Acc. | F1 | PR-AUC | False-fast | Fast-path | Safe-fast precision |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in test_metric_rows:
        pr_auc = row["pr_auc"]
        lines.append(
            f"| {row['method']} | {float(row['accuracy']):.3f} | {float(row['balanced_accuracy']):.3f} | "
            f"{float(row['f1']):.3f} | {float(pr_auc):.3f} | {float(row['false_fast_rate']):.3f} | "
            f"{float(row['fast_path_rate']):.3f} | {float(row['safe_fast_precision']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Evidence Gate",
            "",
            f"Gate passed: `{gate['passed']}`",
            "",
        ]
    )
    for item in gate["checks"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Expectations are generated from action text before reading each observation, because InterCode logs do not contain agent-written expectations.",
            "- Labels are an ARFA annotation layer over external trajectories and need human review before manuscript claims.",
            "- `sentence-transformers` is not installed in the current environment, so the current semantic backend may be incomplete unless the embedding model is installed and rerun.",
            "",
            "Phase 2 remains locked until this report, the leakage audit, and the acceptance gate are reviewed.",
        ]
    )
    ensure_parent(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _gate(test_metric_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_method = {row["method"]: row for row in test_metric_rows}
    hybrid = by_method["hybrid"]
    baseline_names = ["always_reason", "never_reason", "keyword", "tfidf"]
    better = all(float(hybrid["f1"]) >= float(by_method[name]["f1"]) or float(hybrid["pr_auc"]) >= float(by_method[name]["pr_auc"]) for name in baseline_names)
    checks = [
        f"Hybrid beats always/never/keyword/TF-IDF on F1 or PR-AUC: {better}",
        f"Held-out false-fast rate <= 5%: {float(hybrid['false_fast_rate']) <= 0.05}",
        f"Fast-path rate >= 15%: {float(hybrid['fast_path_rate']) >= 0.15}",
        f"Safe-fast precision >= 95%: {float(hybrid['safe_fast_precision']) >= 0.95}",
        "No leakage source identified in the automated audit: review_required",
    ]
    return {
        "passed": better and float(hybrid["false_fast_rate"]) <= 0.05 and float(hybrid["fast_path_rate"]) >= 0.15 and float(hybrid["safe_fast_precision"]) >= 0.95,
        "checks": checks,
    }


def analyze_final(config_path: str) -> dict[str, Any]:
    config = load_simple_yaml(config_path)
    paths = config["final_analysis"]
    rows = read_csv(paths["scores_path"])
    splits = json.loads(Path(paths["splits_path"]).read_text(encoding="utf-8"))
    dataset_summary = json.loads(Path(paths["dataset_summary_path"]).read_text(encoding="utf-8"))

    dev_rows = _binary_rows(_split_rows(rows, splits["development"]))
    validation_rows = _binary_rows(_split_rows(rows, splits["validation"]))
    test_rows = _binary_rows(_split_rows(rows, splits["test"]))
    thresholds = _write_thresholds(validation_rows, Path(paths["thresholds_path"]))
    test_metric_rows = _metrics_rows(test_rows, thresholds)
    write_csv(paths["test_metrics_path"], test_metric_rows)
    write_csv(paths["bootstrap_ci_path"], _bootstrap_ci(test_rows, thresholds, int(splits["seed"])))

    hybrid_threshold = thresholds["methods"]["hybrid"]["operating_points"]["0.05"]["threshold"]
    write_csv(paths["task_category_path"], _subgroup(test_rows, "task_category", hybrid_threshold, "hybrid_residual"))
    write_csv(paths["step_type_path"], _subgroup(test_rows, "step_type", hybrid_threshold, "hybrid_residual"))
    write_csv(paths["ablation_path"], _ablation_rows(validation_rows, test_rows))
    _write_case_analysis(
        test_rows,
        hybrid_threshold,
        {
            "false_fast": paths["false_fast_path"],
            "false_slow": paths["false_slow_path"],
            "failure_analysis": paths["failure_analysis_path"],
        },
    )
    _write_leakage_audit(Path(paths["leakage_audit_path"]), rows)
    _plot_curves(
        validation_rows,
        test_rows,
        hybrid_threshold,
        {
            "distribution": paths["residual_distribution_plot_path"],
            "roc": paths["roc_curve_path"],
            "pr": paths["precision_recall_curve_path"],
            "tradeoff": paths["threshold_tradeoff_path"],
            "fast_safety": paths["fast_path_safety_curve_path"],
            "confusion": paths["confusion_matrix_path"],
        },
    )
    gate = _gate(test_metric_rows)
    _write_report(Path(paths["report_path"]), dataset_summary, test_metric_rows, thresholds, gate)
    return {"dev_binary_records": len(dev_rows), "validation_binary_records": len(validation_rows), "test_binary_records": len(test_rows), "gate": gate}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/phase1_residual.yaml")
    args = parser.parse_args()
    print(json.dumps(analyze_final(args.config), indent=2))


if __name__ == "__main__":
    main()
