from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, cohen_kappa_score, f1_score, roc_auc_score

from src.common.config import load_simple_yaml
from src.common.utils import read_jsonl, write_csv, write_jsonl
from src.phase1_5_shadow.residual import continuation_is_self_contained
from src.phase1_residual.residual_calculator import embedding_residuals


LABELS = {
    "slow_reasoning_needed": {"yes": 1, "no": 0},
    "expectation_match": {"mismatch": 1, "match": 0},
}


def _binary_label(row: dict[str, Any], target: str) -> int | None:
    return LABELS[target].get(str(row.get(target, "")).strip().lower())


def _metrics(rows: list[dict[str, Any]], score_key: str, threshold: float, target: str) -> dict[str, Any]:
    usable = [(row, _binary_label(row, target)) for row in rows]
    usable = [(row, label) for row, label in usable if label is not None]
    y_true = np.array([label for _, label in usable], dtype=int)
    scores = np.array([float(row[score_key]) for row, _ in usable], dtype=float)
    y_pred = (scores >= threshold).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    positives = tp + fn
    negatives = tn + fp
    fast = tn + fn
    both_classes = len(set(y_true.tolist())) == 2
    return {
        "n": len(usable),
        "threshold": threshold,
        "accuracy": (tp + tn) / len(usable) if usable else 0.0,
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred) if both_classes else None,
        "f1": f1_score(y_true, y_pred, zero_division=0) if usable else 0.0,
        "roc_auc": roc_auc_score(y_true, scores) if both_classes else None,
        "pr_auc": average_precision_score(y_true, scores) if positives and usable else None,
        "false_fast_count": fn,
        "false_fast_rate": fn / positives if positives else 0.0,
        "fast_path_count": fast,
        "fast_path_rate": fast / len(usable) if usable else 0.0,
        "safe_fast_precision": tn / fast if fast else None,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "positive_count": positives,
        "negative_count": negatives,
    }


def _select_threshold(rows: list[dict[str, Any]], score_key: str, target: str, false_fast_limit: float) -> tuple[float, dict[str, Any]]:
    scores = sorted({float(row[score_key]) for row in rows})
    candidates = [min(scores) - 1e-9, *scores, max(scores) + 1e-9] if scores else [0.5]
    evaluated = [_metrics(rows, score_key, threshold, target) for threshold in candidates]
    feasible = [row for row in evaluated if row["false_fast_rate"] <= false_fast_limit]
    pool = feasible or evaluated
    selected = max(
        pool,
        key=lambda row: (
            row["fast_path_rate"],
            row["safe_fast_precision"] if row["safe_fast_precision"] is not None else -1.0,
            row["f1"],
        ),
    )
    return float(selected["threshold"]), selected


def _join_annotations(annotations: list[dict[str, Any]], sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in sources:
        grouped[str(source["annotation_id"])].append(source)

    rows: list[dict[str, Any]] = []
    for annotation in annotations:
        linked = grouped.get(str(annotation["annotation_id"]), [])
        if not linked:
            raise ValueError(f"No private source mapping for {annotation['annotation_id']}")
        splits = {str(source["split"]) for source in linked}
        tasks = {str(source["task_id"]) for source in linked}
        if len(splits) != 1 or len(tasks) != 1:
            raise ValueError(f"Deduplicated item crosses task or split boundary: {annotation['annotation_id']}")
        structured = [float(source["structured_residual"]) for source in linked if source.get("structured_residual") is not None]
        observation = [float(source["observation_only_score"]) for source in linked if source.get("observation_only_score") is not None]
        qualities = [float(source["expectation_quality"]) for source in linked if source.get("expectation_quality") is not None]
        categories = {str(source.get("task_category", "unknown")) for source in linked}
        if len(categories) != 1:
            raise ValueError(f"Deduplicated item crosses task-category boundary: {annotation['annotation_id']}")
        next_action = str(annotation.get("next_action_if_expected") or "").strip()
        action = str(annotation.get("action") or "").strip()
        action_safety_gate = not action or not continuation_is_self_contained(action)
        expectation_safety_gate = (
            action_safety_gate
            or not next_action
            or next_action.upper() == "<REASON>"
            or not continuation_is_self_contained(next_action)
        )
        rows.append(
            {
                **annotation,
                "split": next(iter(splits)),
                "task_id": next(iter(tasks)),
                "source_models": ",".join(sorted({str(source["model_name"]) for source in linked})),
                "source_count": len(linked),
                "task_category": next(iter(categories)),
                "structured_expectation_score": float(np.mean(structured)) if structured else 0.0,
                "observation_only_raw_score": float(np.mean(observation)) if observation else 0.0,
                "expectation_quality": float(np.mean(qualities)) if qualities else 0.0,
                "action_safety_gate": float(action_safety_gate),
                "expectation_safety_gate": float(expectation_safety_gate),
            }
        )
    return rows


def _apply_scores(rows: list[dict[str, Any]], semantic_scores: list[float], weight: float, min_quality: float) -> None:
    for row, semantic in zip(rows, semantic_scores):
        action_gate = float(row["action_safety_gate"])
        expectation_gate = float(row["expectation_safety_gate"])
        quality_gate = float(float(row["expectation_quality"]) < min_quality)
        expectation_gate = max(expectation_gate, quality_gate)
        structured = float(row["structured_expectation_score"])
        observation = float(row["observation_only_raw_score"])
        row["semantic_residual_score"] = semantic
        row["semantic_with_safety_score"] = max(expectation_gate, semantic)
        row["structured_with_safety_score"] = max(expectation_gate, structured)
        row["no_expectation_score"] = max(action_gate, observation)
        row["full_residual_score"] = max(expectation_gate, weight * semantic + (1.0 - weight) * structured)


def _choose_weight(rows: list[dict[str, Any]], semantic_scores: list[float], weights: list[float], min_quality: float) -> float:
    best: tuple[float, float] | None = None
    selected = weights[0]
    for weight in weights:
        _apply_scores(rows, semantic_scores, weight, min_quality)
        usable = [row for row in rows if row["split"] == "shadow_development" and _binary_label(row, "slow_reasoning_needed") is not None]
        y_true = [_binary_label(row, "slow_reasoning_needed") for row in usable]
        if len(set(y_true)) < 2:
            score = -1.0
        else:
            score = roc_auc_score(y_true, [row["full_residual_score"] for row in usable])
        candidate = (score, -abs(weight - 0.5))
        if best is None or candidate > best:
            best = candidate
            selected = weight
    _apply_scores(rows, semantic_scores, selected, min_quality)
    return selected


def _bootstrap_auc_difference(rows: list[dict[str, Any]], target: str, samples: int, seed: int) -> dict[str, Any]:
    usable = [row for row in rows if _binary_label(row, target) is not None]
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in usable:
        by_task[str(row["task_id"])].append(row)
    task_ids = sorted(by_task)
    rng = np.random.default_rng(seed)
    differences: list[float] = []
    for _ in range(samples):
        sampled = rng.choice(task_ids, size=len(task_ids), replace=True)
        draw = [row for task_id in sampled for row in by_task[str(task_id)]]
        labels = [_binary_label(row, target) for row in draw]
        if len(set(labels)) < 2:
            continue
        full = roc_auc_score(labels, [row["full_residual_score"] for row in draw])
        baseline = roc_auc_score(labels, [row["no_expectation_score"] for row in draw])
        differences.append(full - baseline)
    observed_labels = [_binary_label(row, target) for row in usable]
    observed = None
    if len(set(observed_labels)) == 2:
        observed = roc_auc_score(observed_labels, [row["full_residual_score"] for row in usable]) - roc_auc_score(
            observed_labels, [row["no_expectation_score"] for row in usable]
        )
    return {
        "observed_auc_gain": observed,
        "bootstrap_ci95_low": float(np.percentile(differences, 2.5)) if differences else None,
        "bootstrap_ci95_high": float(np.percentile(differences, 97.5)) if differences else None,
        "bootstrap_samples_requested": samples,
        "bootstrap_samples_valid": len(differences),
        "cluster_unit": "task_id",
    }


def _annotation_agreement(
    primary: list[dict[str, Any]], secondary: list[dict[str, Any]], minimum_fraction: float
) -> dict[str, Any]:
    primary_by_id = {str(row["annotation_id"]): row for row in primary}
    overlap = [
        (primary_by_id[str(row["annotation_id"])], row)
        for row in secondary
        if str(row["annotation_id"]) in primary_by_id
    ]
    result: dict[str, Any] = {
        "primary_items": len(primary),
        "secondary_items": len(secondary),
        "overlap_items": len(overlap),
        "overlap_fraction": len(overlap) / len(primary) if primary else 0.0,
        "minimum_fraction": minimum_fraction,
    }
    complete = True
    for target in LABELS:
        pairs = [(_binary_label(a, target), _binary_label(b, target)) for a, b in overlap]
        pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
        if len(pairs) < len(overlap) or not pairs:
            complete = False
            kappa = None
        else:
            left, right = zip(*pairs)
            raw_kappa = float(cohen_kappa_score(left, right))
            kappa = raw_kappa if math.isfinite(raw_kappa) else None
        result[target] = {"binary_pairs": len(pairs), "cohen_kappa": kappa}
    result["complete"] = complete and result["overlap_fraction"] >= minimum_fraction
    return result


def _subgroup_metrics(
    rows: list[dict[str, Any]], threshold: float, target: str, minimum_items: int
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    dimensions: dict[str, set[str]] = {
        "task_category": {str(row["task_category"]) for row in rows},
        "source_model": {model for row in rows for model in str(row["source_models"]).split(",")},
    }
    for dimension, values in dimensions.items():
        for value in sorted(values):
            if dimension == "task_category":
                selected = [row for row in rows if str(row["task_category"]) == value]
            else:
                selected = [row for row in rows if value in str(row["source_models"]).split(",")]
            metrics = _metrics(selected, "full_residual_score", threshold, target)
            groups.append(
                {
                    "dimension": dimension,
                    "value": value,
                    "eligible_for_gate": metrics["n"] >= minimum_items,
                    **metrics,
                }
            )
    return groups


def analyze(
    annotations: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    config: dict[str, Any],
    secondary_annotations: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = _join_annotations(annotations, sources)
    incomplete_by_target = {
        target: [row["annotation_id"] for row in rows if _binary_label(row, target) is None]
        for target in LABELS
    }
    agreement = _annotation_agreement(
        annotations,
        secondary_annotations or [],
        float(config["annotation"]["secondary_fraction"]),
    )
    readiness = {
        "annotation_items": len(rows),
        "complete_labels": {target: len(rows) - len(items) for target, items in incomplete_by_target.items()},
        "incomplete_labels": {target: len(items) for target, items in incomplete_by_target.items()},
        "agreement": agreement,
        "ready": not any(incomplete_by_target.values()) and agreement["complete"],
    }
    if not readiness["ready"]:
        return rows, {"readiness": readiness}

    residual_config = config["residual"]
    semantic_scores, backend = embedding_residuals(
        rows,
        str(residual_config["embedding_model"]),
        require_model=bool(residual_config.get("require_embedding_model", True)),
    )
    weights = [float(value) for value in str(residual_config["candidate_semantic_weights"]).split(",")]
    selected_weight = _choose_weight(
        rows,
        semantic_scores,
        weights,
        float(config["shadow"]["minimum_expectation_quality"]),
    )
    methods = [
        "no_expectation_score",
        "semantic_with_safety_score",
        "structured_with_safety_score",
        "full_residual_score",
    ]
    validation = [row for row in rows if row["split"] == "shadow_validation"]
    test = [row for row in rows if row["split"] == "shadow_test"]
    thresholds: dict[str, Any] = {}
    test_metrics: list[dict[str, Any]] = []
    false_fast_limit = float(config["acceptance"]["held_out_false_fast_rate"])
    for method in methods:
        threshold, validation_metrics = _select_threshold(validation, method, "slow_reasoning_needed", false_fast_limit)
        thresholds[method] = {"threshold": threshold, "selection_split": "shadow_validation", "metrics": validation_metrics}
        test_metrics.append({"method": method, **_metrics(test, method, threshold, "slow_reasoning_needed")})

    full = next(row for row in test_metrics if row["method"] == "full_residual_score")
    no_expectation = next(row for row in test_metrics if row["method"] == "no_expectation_score")
    bootstrap = _bootstrap_auc_difference(
        test,
        "slow_reasoning_needed",
        int(config["analysis"]["bootstrap_samples"]),
        int(config["analysis"]["bootstrap_seed"]),
    )
    expectation_metrics = {
        method: _metrics(test, method, 0.5, "expectation_match")
        for method in (
            "no_expectation_score",
            "semantic_with_safety_score",
            "structured_with_safety_score",
            "full_residual_score",
        )
    }
    minimum_subgroup_items = int(config["acceptance"]["minimum_subgroup_items"])
    subgroup_metrics = _subgroup_metrics(
        test,
        float(thresholds["full_residual_score"]["threshold"]),
        "slow_reasoning_needed",
        minimum_subgroup_items,
    )
    eligible_subgroups = [row for row in subgroup_metrics if row["eligible_for_gate"]]
    subgroup_passed = bool(eligible_subgroups) and all(
        row["safe_fast_precision"] is not None
        and row["safe_fast_precision"] >= float(config["acceptance"]["minimum_subgroup_safe_fast_precision"])
        and row["false_fast_rate"] <= float(config["acceptance"]["maximum_subgroup_false_fast_rate"])
        for row in eligible_subgroups
    )
    required_labels = int(config["acceptance"]["minimum_test_binary_labels"])
    minimum_gain = float(config["acceptance"]["minimum_auc_gain_over_no_expectation"])
    gate = {
        "minimum_test_labels": full["n"] >= required_labels,
        "safe_fast_precision": full["safe_fast_precision"] is not None
        and full["safe_fast_precision"] >= float(config["acceptance"]["held_out_safe_fast_precision"]),
        "fast_path_rate": full["fast_path_rate"] >= float(config["acceptance"]["held_out_fast_path_rate"]),
        "false_fast_rate": full["false_fast_rate"] <= false_fast_limit,
        "auc_gain_over_no_expectation": bootstrap["observed_auc_gain"] is not None
        and bootstrap["observed_auc_gain"] >= minimum_gain,
        "auc_gain_ci_excludes_zero": bootstrap["bootstrap_ci95_low"] is not None and bootstrap["bootstrap_ci95_low"] > 0,
        "expectation_mismatch_auc": expectation_metrics["full_residual_score"]["roc_auc"] is not None
        and expectation_metrics["full_residual_score"]["roc_auc"]
        >= float(config["acceptance"]["minimum_expectation_mismatch_auc"]),
        "annotation_agreement": all(
            agreement[target]["cohen_kappa"] is not None
            and agreement[target]["cohen_kappa"] >= float(config["annotation"]["minimum_cohen_kappa"])
            for target in LABELS
        ),
        "no_major_subgroup_collapse": subgroup_passed,
    }
    gate["passed"] = all(gate.values())
    return rows, {
        "readiness": readiness,
        "embedding_backend": backend,
        "selected_semantic_weight": selected_weight,
        "thresholds": thresholds,
        "test_metrics": test_metrics,
        "expectation_mismatch_test_metrics": expectation_metrics,
        "subgroup_metrics": subgroup_metrics,
        "incremental_value": bootstrap,
        "full_minus_no_expectation_fast_path_rate": full["fast_path_rate"] - no_expectation["fast_path_rate"],
        "acceptance_gate": gate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze completed Phase 1.5 blind annotations.")
    parser.add_argument("--config", default="config/phase1_5_shadow.yaml")
    parser.add_argument("--annotations", default="data/phase1_5/annotation/blind_annotation_packet.jsonl")
    parser.add_argument("--source-map", default="data/phase1_5/annotation/private_source_map.jsonl")
    parser.add_argument(
        "--secondary-annotations",
        default="data/phase1_5/annotation/blind_annotation_packet_secondary_25pct.jsonl",
    )
    parser.add_argument("--output-dir", default="results/phase1_5_shadow/analysis")
    args = parser.parse_args()

    secondary_path = Path(args.secondary_annotations)
    secondary = read_jsonl(secondary_path) if secondary_path.exists() else []
    rows, result = analyze(
        read_jsonl(args.annotations),
        read_jsonl(args.source_map),
        load_simple_yaml(args.config),
        secondary,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "analysis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not result["readiness"]["ready"]:
        print(json.dumps(result["readiness"], indent=2))
        raise SystemExit("Blind annotations or independent agreement labels are incomplete; no residual metrics were computed.")

    write_csv(output / "test_metrics.csv", result["test_metrics"])
    write_jsonl(
        output / "false_fast_cases.jsonl",
        [
            row
            for row in rows
            if row["split"] == "shadow_test"
            and _binary_label(row, "slow_reasoning_needed") == 1
            and float(row["full_residual_score"])
            < float(result["thresholds"]["full_residual_score"]["threshold"])
        ],
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
