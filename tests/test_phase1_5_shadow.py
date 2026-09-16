import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.phase1_5_shadow.build_annotation_packet import (
    ALLOWED_PACKET_FIELDS,
    build_packets,
    secondary_annotation_sample,
)
from src.phase1_5_shadow.build_splits import DEFAULT_SIZES, build_split
from src.phase1_5_shadow.audit_collection import audit
from src.phase1_5_shadow.freeze_protocol import FROZEN_FILES, build_freeze_manifest, verify_freeze_manifest
from src.phase1_5_shadow.run_shadow import _freeze_provenance, _validate_collection_scope
from src.phase1_5_shadow.run_shadow import _provenance_fingerprint, _validate_resume
from src.phase1_5_shadow.audit_formal_collection import audit_formal_collection
from src.phase1_5_shadow.observation_baseline import fit_observation_only_scores
from src.phase1_5_shadow.review_pilot_annotations import review
from src.phase1_5_shadow.analyze_annotations import (
    _annotation_agreement,
    _apply_scores,
    _cluster_bootstrap_intervals,
    _metrics,
    _select_threshold,
    _validate_label_values,
    analyze,
)
from src.phase1_5_shadow.ai_annotate_packet import _blind_item, _make_batches
from src.phase1_5_shadow.evaluate_exploratory_gate import evaluate_engineering_gate
from src.phase1_5_shadow.json_model_client import OllamaStructuredClient
from src.phase1_5_shadow.residual import (
    continuation_is_self_contained,
    expectation_quality,
    normalize_expected_signals,
    shadow_decision,
    structured_residual,
)
from src.phase1_5_shadow.shadow_agent import ShadowReActAgent


def test_ai_annotation_input_is_strictly_blinded():
    row = {
        "annotation_id": "item-1",
        "task_instruction": "Inspect a file",
        "current_plan": "Run cat",
        "action": "cat file.txt",
        "expected_outcome": "Print alpha",
        "expected_signals": {"exit_code": "0"},
        "next_action_if_expected": "<DONE>",
        "actual_observation": "exit_code=0\nstdout:\nalpha",
        "exit_code": 0,
        "residual_score": 0.99,
        "split": "shadow_test",
        "model_name": "hidden-model",
        "reward": 1,
        "gold_command": "cat file.txt",
        "success": True,
    }

    blinded = _blind_item(row)

    assert set(blinded) == {
        "annotation_id",
        "task_instruction",
        "current_plan",
        "action",
        "expected_outcome",
        "expected_signals",
        "next_action_if_expected",
        "actual_observation",
        "exit_code",
    }
    assert not ({"residual_score", "split", "model_name", "reward", "gold_command", "success"} & set(blinded))


def test_ai_annotation_batches_respect_count_and_character_limits():
    rows = [
        {"annotation_id": f"item-{index}", "actual_observation": "x" * 20}
        for index in range(5)
    ]

    batches = _make_batches(rows, batch_size=2, max_chars=10_000)

    assert [len(batch) for batch in batches] == [2, 2, 1]
    assert [row["annotation_id"] for batch in batches for row in batch] == [
        "item-0",
        "item-1",
        "item-2",
        "item-3",
        "item-4",
    ]


def test_exploratory_engineering_gate_unlocks_pilot_without_rewriting_formal_gate():
    analysis = {
        "readiness": {"ready": True, "annotation_items": 751},
        "embedding_backend": "sentence-transformers/all-MiniLM-L6-v2",
        "test_metrics": [
            {
                "method": "full_residual_score",
                "roc_auc": 0.743,
                "confidence_intervals": {"roc_auc": {"ci95_low": 0.68, "ci95_high": 0.81}},
            }
        ],
        "incremental_value": {
            "full_vs_expectation_gate_only": {
                "observed_auc_gain": 0.124,
                "bootstrap_ci95_low": 0.092,
                "bootstrap_ci95_high": 0.159,
            }
        },
        "acceptance_gate": {"passed": False},
    }
    collection = {"passed": True, "observed_total_runs": 300, "expected_total_runs": 300}
    phase2 = {"complete": True, "observed_runs": 600}
    config = {
        "gate": {
            "version": "test-engineering-v1",
            "scope": "exploratory_phase3_readiness",
            "minimum_annotation_items": 700,
            "minimum_full_residual_auc": 0.70,
            "minimum_auc_gain_over_expectation_gate_only": 0.05,
            "require_gain_ci_excludes_zero": True,
        },
        "review": {
            "mode": "human_reviewed_ai_assisted",
            "reviewer_count_reported_by_project_owner": 6,
            "independently_partitioned_review_reported": True,
            "per_reviewer_raw_files_available": False,
        },
        "unlock": {"phase3_exploratory_pilot": True, "phase3_formal_evaluation": False},
    }

    result = evaluate_engineering_gate(analysis, collection, phase2, config)

    assert result["passed"] is True
    assert result["phase3_p_exploratory_unlocked"] is True
    assert result["phase3_formal_unlocked"] is False
    assert result["formal_evidence_gate_passed"] is False


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
    assert rows[0]["expectation_gate_only_score"] == 1.0
    assert rows[0]["raw_full_residual_score"] == 0.0
    assert rows[0]["full_residual_score"] == 1.0


def test_annotation_agreement_requires_complete_independent_labels():
    primary = [
        {"annotation_id": "a", "slow_reasoning_needed": "yes", "expectation_match": "mismatch", "annotator": "a"},
        {"annotation_id": "b", "slow_reasoning_needed": "no", "expectation_match": "match", "annotator": "a"},
    ]
    secondary = [{**row, "annotator": "b"} for row in primary]
    agreement = _annotation_agreement(primary, secondary, minimum_fraction=0.25)

    assert agreement["complete"] is True
    assert agreement["slow_reasoning_needed"]["multiclass_cohen_kappa"] == 1.0
    assert agreement["expectation_match"]["multiclass_cohen_kappa"] == 1.0


def test_ambiguous_is_complete_for_agreement_but_excluded_from_binary_metrics():
    primary = [
        {"annotation_id": "a", "slow_reasoning_needed": "ambiguous", "expectation_match": "ambiguous", "annotator": "a"},
        {"annotation_id": "b", "slow_reasoning_needed": "yes", "expectation_match": "mismatch", "annotator": "a"},
        {"annotation_id": "c", "slow_reasoning_needed": "no", "expectation_match": "match", "annotator": "a"},
    ]
    secondary = [{**row, "annotator": "b"} for row in primary]
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


def _valid_freeze_fixture():
    config = {
        "protocol": {
            "version": "phase1.5-v1.1",
            "stage": "formal_shadow_collection",
            "locked": True,
        }
    }
    manifest = {
        "protocol_version": "phase1.5-v1.1",
        "stage": "formal_shadow_collection",
        "locked": True,
        "files": {
            path: {"sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
            for path in FROZEN_FILES
        },
    }
    return config, manifest


def test_freeze_verification_rejects_missing_and_unexpected_entries():
    config, manifest = _valid_freeze_fixture()
    missing = copy.deepcopy(manifest)
    missing["files"].pop(FROZEN_FILES[0])
    with pytest.raises(ValueError, match="file set mismatch"):
        verify_freeze_manifest(config, missing)

    unexpected = copy.deepcopy(manifest)
    unexpected["files"]["README.md"] = {"sha256": hashlib.sha256(Path("README.md").read_bytes()).hexdigest()}
    with pytest.raises(ValueError, match="file set mismatch"):
        verify_freeze_manifest(config, unexpected)


def test_freeze_verification_rejects_modified_hash_and_version():
    config, manifest = _valid_freeze_fixture()
    modified = copy.deepcopy(manifest)
    modified["files"][FROZEN_FILES[0]]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="artifact changed"):
        verify_freeze_manifest(config, modified)

    wrong_version = copy.deepcopy(manifest)
    wrong_version["protocol_version"] = "phase1.5-other"
    with pytest.raises(ValueError, match="version"):
        verify_freeze_manifest(config, wrong_version)


def test_unlocked_pilot_provenance_does_not_require_external_runtimes(tmp_path):
    config = {"protocol": {"version": "phase1.5-v1.1", "stage": "pilot_review", "locked": False}}
    provenance = _freeze_provenance(config, [], tmp_path / "missing.json")

    assert provenance["protocol_locked"] is False
    assert provenance["freeze_manifest_sha256"] is None
    assert provenance["model_artifact"] is None


def test_collection_scope_blocks_unfrozen_or_partial_formal_runs():
    pilot = {"protocol": {"locked": False}}
    _validate_collection_scope(pilot, "shadow_development", 10)
    with pytest.raises(ValueError, match="Unlocked protocol"):
        _validate_collection_scope(pilot, "shadow_test", 10)
    with pytest.raises(ValueError, match="Unlocked protocol"):
        _validate_collection_scope(pilot, "shadow_development", 0)

    formal = {"protocol": {"locked": True}}
    _validate_collection_scope(formal, "shadow_test", 0)
    with pytest.raises(ValueError, match="forbids task limits"):
        _validate_collection_scope(formal, "shadow_test", 10)


def test_resume_rejects_stale_provenance_and_duplicate_tasks(tmp_path):
    provenance = {"freeze_manifest_sha256": "abc", "run_source_commit": "commit"}
    fingerprint = _provenance_fingerprint(provenance, "model-a", "shadow_test")
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "model_name": "model-a",
                "split": "shadow_test",
                "task_count": 1,
                "expected_task_count": 2,
                "run_provenance_fingerprint": fingerprint,
            }
        ),
        encoding="utf-8",
    )
    valid = [SimpleNamespace(task_id="task-a", provenance_fingerprint=fingerprint)]
    _validate_resume(valid, summary_path, "model-a", "shadow_test", {"task-a", "task-b"}, 2, fingerprint)

    stale = [SimpleNamespace(task_id="task-a", provenance_fingerprint="old")]
    with pytest.raises(ValueError, match="stale provenance"):
        _validate_resume(stale, summary_path, "model-a", "shadow_test", {"task-a", "task-b"}, 2, fingerprint)

    duplicate = [
        SimpleNamespace(task_id="task-a", provenance_fingerprint=fingerprint),
        SimpleNamespace(task_id="task-a", provenance_fingerprint=fingerprint),
    ]
    summary_path.write_text(
        json.dumps(
            {
                "model_name": "model-a",
                "split": "shadow_test",
                "task_count": 2,
                "expected_task_count": 2,
                "run_provenance_fingerprint": fingerprint,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate task IDs"):
        _validate_resume(duplicate, summary_path, "model-a", "shadow_test", {"task-a", "task-b"}, 2, fingerprint)


def test_formal_matrix_audit_requires_exact_tasks_and_provenance(tmp_path, monkeypatch):
    import src.phase1_5_shadow.audit_formal_collection as formal_audit_module

    monkeypatch.setattr(formal_audit_module, "verify_freeze_manifest", lambda config, manifest: None)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("frozen config\n", encoding="utf-8")
    task_path = tmp_path / "tasks.jsonl"
    split_path = tmp_path / "splits.json"
    freeze_path = tmp_path / "freeze.json"
    result_root = tmp_path / "full"
    split_tasks = {
        "shadow_development": "dev-task",
        "shadow_validation": "val-task",
        "shadow_test": "test-task",
    }
    task_path.write_text(
        "".join(
            json.dumps({"task_id": task_id, "filesystem_version": 1}) + "\n"
            for task_id in split_tasks.values()
        ),
        encoding="utf-8",
    )
    split_path.write_text(
        json.dumps(
            {
                "sizes": {split: 1 for split in split_tasks},
                "splits": {split: [task_id] for split, task_id in split_tasks.items()},
            }
        ),
        encoding="utf-8",
    )
    freeze_path.write_text(json.dumps({"source_commit": "freeze-commit"}), encoding="utf-8")
    config = {
        "protocol": {"version": "test-v1", "stage": "formal_shadow_collection", "locked": True},
        "formal": {"model_1": "model-a|model-a", "expected_total_runs": 3},
        "data": {"task_path": str(task_path), "split_path": str(split_path)},
        "environment": {"image_prefix": "image-fs"},
    }

    for split, task_id in split_tasks.items():
        cell = result_root / "model-a" / split
        cell.mkdir(parents=True)
        provenance = {
            "run_source_commit": "run-commit",
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "task_manifest_sha256": hashlib.sha256(task_path.read_bytes()).hexdigest(),
            "split_manifest_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
            "freeze_manifest_sha256": hashlib.sha256(freeze_path.read_bytes()).hexdigest(),
            "freeze_source_commit": "freeze-commit",
            "protocol_version": "test-v1",
            "protocol_stage": "formal_shadow_collection",
            "protocol_locked": True,
            "model_artifact": {
                "model_name": "model-a",
                "base_blob_sha256": "a" * 64,
                "modelfile_sha256": "b" * 64,
            },
            "docker_image_digests": {"image-fs1": "sha256:" + "c" * 64},
        }
        fingerprint = _provenance_fingerprint(provenance, "model-a", split)
        trace = {
            "task_id": task_id,
            "model_name": "model-a",
            "split": split,
            "provenance_fingerprint": fingerprint,
            "steps": [],
        }
        (cell / "traces.jsonl").write_text(json.dumps(trace) + "\n", encoding="utf-8")
        provenance["run_provenance_fingerprint"] = fingerprint
        (cell / "summary.json").write_text(
            json.dumps(
                {
                    "model_name": "model-a",
                    "split": split,
                    "task_count": 1,
                    "expected_task_count": 1,
                    "run_provenance_fingerprint": fingerprint,
                    "reproducibility": provenance,
                }
            ),
            encoding="utf-8",
        )

    result = audit_formal_collection(config, config_path, result_root, freeze_path)
    assert result["passed"] is True
    assert result["observed_total_runs"] == 3

    test_trace = result_root / "model-a" / "shadow_test" / "traces.jsonl"
    tampered = json.loads(test_trace.read_text(encoding="utf-8"))
    tampered["task_id"] = "wrong-task"
    test_trace.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
    rejected = audit_formal_collection(config, config_path, result_root, freeze_path)
    assert rejected["passed"] is False
    assert any("missing task IDs" in error for error in rejected["errors"])


def test_pilot_review_reports_agreement_without_residual_metrics():
    primary = []
    for index in range(8):
        positive = index % 2 == 0
        primary.append(
            {
                "annotation_id": f"a-{index}",
                "expectation_match": "mismatch" if positive else "match",
                "slow_reasoning_needed": "yes" if positive else "no",
                "annotation_confidence": "high",
                "annotation_notes": "",
                "annotator": "primary",
            }
        )
    secondary = [{**primary[index], "annotator": "secondary"} for index in (0, 1)]
    result, disagreements = review(
        primary,
        secondary,
        {"annotation": {"secondary_fraction": 0.25, "minimum_cohen_kappa": 0.70}},
    )

    assert result["readiness"]["ready"] is True
    assert result["readiness"]["paper_agreement_result"] is False
    assert disagreements == []
    assert "residual" not in result

    same_annotator = [{**row, "annotator": "primary"} for row in secondary]
    rejected, _ = review(
        primary,
        same_annotator,
        {"annotation": {"secondary_fraction": 0.25, "minimum_cohen_kappa": 0.70}},
    )
    assert rejected["readiness"]["ready"] is False
    assert rejected["readiness"]["agreement"]["independent_annotator_ids"] is False


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


def test_learned_observation_baseline_cannot_read_expectations_or_actions():
    rows = [
        {
            "split": "shadow_development" if index < 4 else "shadow_test",
            "slow_reasoning_needed": "yes" if index % 2 else "no",
            "actual_observation": "exit_code=1\nstderr:\nerror" if index % 2 else "exit_code=0\nstdout:\nok",
            "exit_code": 1 if index % 2 else 0,
            "expected_outcome": "original",
            "current_plan": "original",
            "action": "original",
            "next_action_if_expected": "original",
        }
        for index in range(6)
    ]
    changed = copy.deepcopy(rows)
    for row in changed:
        row["expected_outcome"] = "completely changed expectation"
        row["current_plan"] = "completely changed plan"
        row["action"] = "rm -rf irrelevant"
        row["next_action_if_expected"] = "completely changed continuation"

    original_scores, info = fit_observation_only_scores(rows, "tfidf", False, 1507)
    changed_scores, _ = fit_observation_only_scores(changed, "tfidf", False, 1507)

    assert original_scores == pytest.approx(changed_scores)
    assert info["allowed_inputs"] == [
        "actual_observation",
        "exit_code",
        "stderr_derived_from_actual_observation",
    ]
    assert "expectation" in info["forbidden_inputs"]


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
            "minimum_raw_residual_auc_gain_over_no_expectation": 0.02,
            "minimum_full_auc_gain_over_expectation_gate_only": 0.02,
            "minimum_raw_residual_auc_gain_over_learned_observation_only": 0.02,
            "minimum_full_auc_gain_over_learned_observation_only": 0.02,
            "minimum_expectation_mismatch_auc": 0.8,
            "minimum_fast_path_decisions": 3,
            "minimum_safe_fast_precision_ci95_low": 1.01,
        },
        "analysis": {"bootstrap_samples": 20, "bootstrap_seed": 1507},
        "annotation": {"secondary_fraction": 0.25, "minimum_cohen_kappa": 0.7},
    }

    secondary = [{**row, "annotator": "b"} for row in annotations]
    _, result = analyze(annotations, sources, config, secondary_annotations=secondary)

    assert result["readiness"]["ready"] is True
    assert result["thresholds"]["full_residual_score"]["metrics"]["selection_feasible"] is True
    full = next(row for row in result["test_metrics"] if row["method"] == "full_residual_score")
    methods = {row["method"] for row in result["test_metrics"]}
    assert {
        "learned_observation_only_score",
        "expectation_gate_only_score",
        "raw_full_residual_score",
    } <= methods
    assert set(result["incremental_value"]) == {
        "full_vs_no_expectation",
        "raw_residual_vs_no_expectation",
        "full_vs_expectation_gate_only",
        "raw_residual_vs_learned_observation_only",
        "full_vs_learned_observation_only",
    }
    assert full["confidence_intervals"]["cluster_count"] == 4
    assert result["learned_observation_only_baseline"]["training_split"] == "shadow_development"
    assert "minimum_fast_path_decisions" in result["acceptance_gate"]
    assert "safe_fast_precision_ci_lower_bound" in result["acceptance_gate"]
    assert result["acceptance_gate"]["minimum_fast_path_decisions"] is False
    assert result["acceptance_gate"]["safe_fast_precision_ci_lower_bound"] is False
    assert result["subgroup_metrics"]
