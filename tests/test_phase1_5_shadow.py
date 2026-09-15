import json

import pytest

from src.phase1_5_shadow.build_annotation_packet import (
    ALLOWED_PACKET_FIELDS,
    build_packets,
    secondary_annotation_sample,
)
from src.phase1_5_shadow.build_splits import DEFAULT_SIZES, build_split
from src.phase1_5_shadow.audit_collection import audit
from src.phase1_5_shadow.freeze_protocol import build_freeze_manifest
from src.phase1_5_shadow.analyze_annotations import (
    _annotation_agreement,
    _apply_scores,
    _cluster_bootstrap_intervals,
    _metrics,
    _select_threshold,
    _validate_label_values,
    analyze,
)
from src.phase1_5_shadow.json_model_client import OllamaStructuredClient
from src.phase1_5_shadow.residual import (
    continuation_is_self_contained,
    expectation_quality,
    normalize_expected_signals,
    shadow_decision,
    structured_residual,
)
from src.phase1_5_shadow.shadow_agent import ShadowReActAgent


def _tasks():
    categories = ["rare"] * 4 + ["common"] * 196
    return [
        {
            "task_id": f"task-{index:03d}",
            "task_category": category,
            "filesystem_version": index % 4 + 1,
        }
        for index, category in enumerate(categories)
    ]


def test_frozen_split_is_exact_deterministic_and_disjoint():
    first = build_split(_tasks(), seed=1507)
    second = build_split(_tasks(), seed=1507)

    assert first == second
    assert {name: len(ids) for name, ids in first["splits"].items()} == DEFAULT_SIZES
    flattened = [task_id for task_ids in first["splits"].values() for task_id in task_ids]
    assert len(flattened) == len(set(flattened)) == 200


def test_structured_residual_uses_precommitted_signals():
    signals = normalize_expected_signals({"exit_code": 0, "stdout": "nonempty", "stderr": "empty"})
    matched, matched_checks = structured_residual(signals, "Print one result successfully", 0, "one\n", "", "exit_code=0\nstdout:\none")
    mismatched, mismatched_checks = structured_residual(signals, "Print one result successfully", 1, "", "boom", "exit_code=1\nstderr:\nboom")

    assert matched == 0.0
    assert not any(matched_checks.values())
    assert mismatched > 0.8
    assert any(mismatched_checks.values())

    missing_output, checks = structured_residual(
        normalize_expected_signals({"exit_code": 0, "stdout": "any", "stderr": "any"}),
        "The command will output the calculated result.",
        0,
        "",
        "",
        "exit_code=0",
    )
    assert missing_output > 0
    assert checks["textual_output_missing"] is True


def test_quality_gate_prevents_uncommitted_fast_path():
    signals = normalize_expected_signals({"exit_code": "any", "stdout": "any", "stderr": "any"})
    quality = expectation_quality(signals, "vague", "")
    decision, reason = shadow_decision(0.0, quality, "", threshold=0.35, min_quality=0.70)

    assert decision == "reason"
    assert reason == "missing_precommitted_continuation"

    decision, reason = shadow_decision(0.0, 1.0, "<REASON>", threshold=0.35, min_quality=0.70)
    assert decision == "reason"
    assert reason == "model_declared_observation_dependent_continuation"

    decision, reason = shadow_decision(
        0.0,
        1.0,
        "<DONE>",
        threshold=0.35,
        min_quality=0.70,
        action="awk '{print $1}'",
    )
    assert decision == "reason"
    assert reason == "executed_action_depended_on_missing_stdin"


def test_continuation_must_not_depend_on_previous_stdout():
    assert continuation_is_self_contained("awk '{sum += $1} END {print sum}'") is False
    assert continuation_is_self_contained("xargs -0 cp -t $TESTBED/tmp") is False
    assert continuation_is_self_contained("awk '$1 < 20 {print $1}'") is False
    assert continuation_is_self_contained("find $TESTBED -type f | xargs wc -l") is True
    assert continuation_is_self_contained("<DONE>") is True


def test_annotation_packet_is_blind_and_deduplicated(tmp_path):
    run = {
        "task_id": "task-1",
        "instruction": "List files",
        "model_name": "hidden-model",
        "split": "shadow_development",
        "success": True,
        "evaluator": {"reward": 1.0},
        "steps": [
            {
                "step_id": 1,
                "current_plan": "List then finish",
                "action": "ls",
                "expected_outcome": "Files are listed",
                "expected_signals": {"exit_code": "0", "stdout": "nonempty", "stderr": "empty"},
                "next_action_if_expected": "<DONE>",
                "observation": "exit_code=0\nstdout:\na.txt",
                "exit_code": 0,
                "expectation_quality": 1.0,
                "structured_residual": 0.0,
                "observation_only_score": 0.0,
                "shadow_decision": "fast_candidate",
                "shadow_reason": "shadow_only_expectation_matched",
                "parse_error": None,
            }
        ],
    }
    path = tmp_path / "traces.jsonl"
    path.write_text(json.dumps(run) + "\n" + json.dumps(run) + "\n", encoding="utf-8")

    annotations, sources = build_packets([path])

    assert len(annotations) == 1
    assert len(sources) == 2
    assert set(annotations[0]) == ALLOWED_PACKET_FIELDS
    assert "model_name" not in annotations[0]
    assert "reward" not in annotations[0]
    assert sources[0]["model_name"] == "hidden-model"


def test_secondary_annotation_sample_is_frozen_and_large_enough():
    rows = [{"annotation_id": f"a-{index}"} for index in range(23)]
    first = secondary_annotation_sample(rows)
    second = secondary_annotation_sample(rows)

    assert first == second
    assert len(first) == 6


def test_no_expectation_baseline_cannot_read_continuation_commitment():
    rows = [
        {
            "action_safety_gate": 0.0,
            "expectation_safety_gate": 1.0,
            "expectation_quality": 1.0,
            "structured_expectation_score": 0.0,
            "observation_only_raw_score": 0.0,
        }
    ]
    _apply_scores(rows, [0.0], weight=0.5, min_quality=0.7)

    assert rows[0]["no_expectation_score"] == 0.0
    assert rows[0]["full_residual_score"] == 1.0


def test_annotation_agreement_requires_complete_independent_labels():
    primary = [
        {"annotation_id": "a", "slow_reasoning_needed": "yes", "expectation_match": "mismatch"},
        {"annotation_id": "b", "slow_reasoning_needed": "no", "expectation_match": "match"},
    ]
    secondary = [dict(row) for row in primary]
    agreement = _annotation_agreement(primary, secondary, minimum_fraction=0.25)

    assert agreement["complete"] is True
    assert agreement["slow_reasoning_needed"]["multiclass_cohen_kappa"] == 1.0
    assert agreement["expectation_match"]["multiclass_cohen_kappa"] == 1.0


def test_ambiguous_is_complete_for_agreement_but_excluded_from_binary_metrics():
    primary = [
        {"annotation_id": "a", "slow_reasoning_needed": "ambiguous", "expectation_match": "ambiguous"},
        {"annotation_id": "b", "slow_reasoning_needed": "yes", "expectation_match": "mismatch"},
        {"annotation_id": "c", "slow_reasoning_needed": "no", "expectation_match": "match"},
    ]
    secondary = [dict(row) for row in primary]
    for index, row in enumerate(primary):
        row["score"] = float(index) / 2
    agreement = _annotation_agreement(primary, secondary, minimum_fraction=0.25)
    metrics = _metrics(primary, "score", 0.5, "slow_reasoning_needed")

    assert agreement["complete"] is True
    assert agreement["slow_reasoning_needed"]["ambiguous_pair_count"] == 1
    assert agreement["slow_reasoning_needed"]["binary_pair_coverage"] == pytest.approx(2 / 3)
    assert metrics["n"] == 2


def test_invalid_annotation_label_is_rejected():
    with pytest.raises(ValueError, match="Invalid expectation_match"):
        _validate_label_values([{"annotation_id": "a", "expectation_match": "maybe"}])


def test_split_rejects_wrong_task_count():
    with pytest.raises(ValueError, match="Expected 200"):
        build_split(_tasks()[:10])


def test_protocol_freeze_rejects_unlocked_pilot_config():
    config = {
        "protocol": {"version": "phase1.5-v1.1", "stage": "pilot_review", "locked": False},
        "annotation": {"minimum_cohen_kappa": 0.7},
    }
    with pytest.raises(ValueError, match="protocol.locked=true"):
        build_freeze_manifest(config, {"readiness": {"ready": False}})


def test_shadow_parser_extracts_balanced_nested_json():
    wrapped = 'prefix {"action":"ls","done":false,"expected_signals":{"exit_code":0,"stdout":"nonempty"}} suffix {ignored}'
    parsed, error = ShadowReActAgent._parse_model_json(wrapped)

    assert parsed["action"] == "ls"
    assert parsed["expected_signals"]["exit_code"] == 0
    assert error == "extracted_json_from_non_json_response"


def test_shadow_parser_rejects_nested_fragment():
    parsed, error = ShadowReActAgent._parse_model_json('bad {"expected_signals":{"exit_code":0}}')
    assert parsed["action"] == ""
    assert error == "json_parse_failed"


def test_json_mode_client_requests_json_schema():
    client = OllamaStructuredClient("http://localhost:11434/v1", "test-model")
    payload = client._payload([{"role": "user", "content": "test"}])
    assert payload["format"]["properties"]["expected_signals"]["properties"]["stdout"]["enum"] == [
        "empty",
        "nonempty",
        "any",
    ]
    assert payload["options"]["num_predict"] == 900


def test_collection_audit_does_not_claim_accuracy(tmp_path):
    trace = {
        "task_id": "t1",
        "instruction": "List files",
        "model_name": "m1",
        "split": "shadow_development",
        "steps": [
            {
                "step_id": 1,
                "action": "find $TESTBED -type f",
                "action_executed": True,
                "current_plan": "List then finish",
                "expected_outcome": "Files will be listed",
                "expected_signals": {"exit_code": "0", "stdout": "nonempty", "stderr": "empty"},
                "next_action_if_expected": "<DONE>",
                "observation": "exit_code=0\nstdout:\na.txt",
                "exit_code": 0,
                "expectation_quality": 1.0,
                "structured_residual": 0.0,
                "observation_only_score": 0.0,
                "shadow_decision": "fast_candidate",
                "shadow_reason": "shadow_only_expectation_matched",
                "parse_error": None,
            }
        ],
    }
    path = tmp_path / "traces.jsonl"
    path.write_text(json.dumps(trace) + "\n", encoding="utf-8")
    result = audit([path], expected_tasks=1)

    assert result["complete"] is True
    assert result["parse_error_count"] == 0
    assert result["shadow_fast_candidate_rate"] == 1.0
    assert "accuracy" not in result


def test_threshold_selection_uses_false_fast_constraint():
    rows = [
        {"task_id": "a", "slow_reasoning_needed": "yes", "score": 0.9},
        {"task_id": "b", "slow_reasoning_needed": "yes", "score": 0.8},
        {"task_id": "c", "slow_reasoning_needed": "no", "score": 0.2},
        {"task_id": "d", "slow_reasoning_needed": "no", "score": 0.1},
    ]
    threshold, validation = _select_threshold(
        rows,
        "score",
        "slow_reasoning_needed",
        false_fast_limit=0.0,
        safe_fast_minimum=0.95,
    )
    test = _metrics(rows, "score", threshold, "slow_reasoning_needed")

    assert validation["selection_feasible"] is True
    assert validation["false_fast_rate"] == 0.0
    assert test["safe_fast_precision"] == 1.0
    assert test["fast_path_rate"] == 0.5


def test_threshold_selection_marks_infeasible_and_falls_back_to_all_reason():
    rows = [
        {"task_id": "a", "slow_reasoning_needed": "yes", "score": 0.1},
        {"task_id": "b", "slow_reasoning_needed": "no", "score": 0.9},
    ]
    threshold, validation = _select_threshold(
        rows,
        "score",
        "slow_reasoning_needed",
        false_fast_limit=0.0,
        safe_fast_minimum=0.95,
    )
    fallback = _metrics(rows, "score", threshold, "slow_reasoning_needed")

    assert validation["selection_feasible"] is False
    assert validation["selection_status"] == "infeasible_conservative_all_reason_fallback"
    assert fallback["fast_path_rate"] == 0.0


def test_task_clustered_bootstrap_reports_main_safety_intervals():
    rows = [
        {"task_id": "a", "slow_reasoning_needed": "yes", "score": 0.9},
        {"task_id": "b", "slow_reasoning_needed": "yes", "score": 0.8},
        {"task_id": "c", "slow_reasoning_needed": "no", "score": 0.2},
        {"task_id": "d", "slow_reasoning_needed": "no", "score": 0.1},
    ]
    intervals = _cluster_bootstrap_intervals(
        rows,
        "score",
        threshold=0.5,
        target="slow_reasoning_needed",
        samples=100,
        seed=1507,
    )

    assert intervals["cluster_unit"] == "task_id"
    assert intervals["cluster_count"] == 4
    assert intervals["fast_path_rate"]["valid_bootstrap_samples"] == 100
    assert intervals["roc_auc"]["valid_bootstrap_samples"] > 0


def test_completed_annotations_run_full_held_out_analysis():
    annotations = []
    sources = []
    for split_index, split in enumerate(("shadow_development", "shadow_validation", "shadow_test")):
        for item_index in range(4):
            mismatch = item_index % 2 == 0
            annotation_id = f"{split}-{item_index}"
            annotations.append(
                {
                    "annotation_id": annotation_id,
                    "task_instruction": "Inspect a file",
                    "current_plan": "Inspect and finish",
                    "action": "cat $TESTBED/file.txt",
                    "expected_outcome": "The command prints alpha successfully",
                    "expected_signals": {"exit_code": "0", "stdout": "nonempty", "stderr": "empty"},
                    "next_action_if_expected": "<DONE>",
                    "actual_observation": "exit_code=1 error" if mismatch else "exit_code=0 alpha",
                    "exit_code": 1 if mismatch else 0,
                    "expectation_match": "mismatch" if mismatch else "match",
                    "slow_reasoning_needed": "yes" if mismatch else "no",
                    "annotation_confidence": "high",
                    "annotation_notes": "",
                    "annotator": "a",
                }
            )
            sources.append(
                {
                    "annotation_id": annotation_id,
                    "split": split,
                    "task_id": f"task-{split_index}-{item_index}",
                    "model_name": "model-a",
                    "task_category": "inspect",
                    "structured_residual": 1.0 if mismatch else 0.0,
                    "observation_only_score": 0.8 if mismatch else 0.0,
                    "expectation_quality": 1.0,
                }
            )
    config = {
        "residual": {
            "embedding_model": "tfidf",
            "require_embedding_model": False,
            "candidate_semantic_weights": "0.5",
        },
        "shadow": {"minimum_expectation_quality": 0.7},
        "acceptance": {
            "held_out_false_fast_rate": 0.05,
            "held_out_safe_fast_precision": 0.95,
            "held_out_fast_path_rate": 0.15,
            "minimum_subgroup_items": 2,
            "minimum_subgroup_safe_fast_precision": 0.85,
            "maximum_subgroup_false_fast_rate": 0.10,
            "minimum_test_binary_labels": 4,
            "minimum_auc_gain_over_no_expectation": 0.02,
            "minimum_expectation_mismatch_auc": 0.8,
        },
        "analysis": {"bootstrap_samples": 20, "bootstrap_seed": 1507},
        "annotation": {"secondary_fraction": 0.25, "minimum_cohen_kappa": 0.7},
    }

    _, result = analyze(annotations, sources, config, secondary_annotations=[dict(row) for row in annotations])

    assert result["readiness"]["ready"] is True
    assert result["thresholds"]["full_residual_score"]["metrics"]["selection_feasible"] is True
    assert result["test_metrics"][3]["confidence_intervals"]["cluster_count"] == 4
    assert result["subgroup_metrics"]
