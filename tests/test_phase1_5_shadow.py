import json

import pytest

from src.phase1_5_shadow.build_annotation_packet import (
    ALLOWED_PACKET_FIELDS,
    build_packets,
    secondary_annotation_sample,
)
from src.phase1_5_shadow.build_splits import DEFAULT_SIZES, build_split
from src.phase1_5_shadow.audit_collection import audit
from src.phase1_5_shadow.analyze_annotations import (
    _annotation_agreement,
    _apply_scores,
    _metrics,
    _select_threshold,
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
    assert agreement["slow_reasoning_needed"]["cohen_kappa"] == 1.0
    assert agreement["expectation_match"]["cohen_kappa"] == 1.0


def test_split_rejects_wrong_task_count():
    with pytest.raises(ValueError, match="Expected 200"):
        build_split(_tasks()[:10])


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
    threshold, validation = _select_threshold(rows, "score", "slow_reasoning_needed", false_fast_limit=0.0)
    test = _metrics(rows, "score", threshold, "slow_reasoning_needed")

    assert validation["false_fast_rate"] == 0.0
    assert test["safe_fast_precision"] == 1.0
    assert test["fast_path_rate"] == 0.5
