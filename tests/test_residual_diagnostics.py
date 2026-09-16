import json

import pytest

from src.phase1_5_shadow.ai_annotate_packet import (
    SYSTEM_PROMPT, EXPECTATION_CODES, REASONING_CODES, CONFIDENCE_CODES, EVIDENCE_CODES_SHORT,
)
from src.phase1_5_shadow.export_diagnostics import (
    collect_steps, failure_auc, paired_comparisons, summarize_runs,
)


def test_short_label_codes_are_explained_to_judge():
    for mapping in (EXPECTATION_CODES, REASONING_CODES, CONFIDENCE_CODES, EVIDENCE_CODES_SHORT):
        for code, meaning in mapping.items():
            assert f"{code}={meaning}" in SYSTEM_PROMPT
    assert "e = expectation_match" in SYSTEM_PROMPT
    assert "r = slow_reasoning_needed" in SYSTEM_PROMPT


def test_zero_step_run_is_not_a_zero_residual():
    runs = [{"model": "a", "split": "test", "task_id": "x", "success": False}]
    summarize_runs([], runs)
    assert runs[0]["executed_steps"] == 0
    assert runs[0]["first_structured_residual"] is None
    assert runs[0]["mean_structured_residual"] is None
    assert failure_auc(runs, "first_structured_residual") is None


def test_task_means_and_first_step_do_not_weight_long_runs_more():
    run = {"model": "a", "split": "test", "task_id": "x"}
    steps = [{**run, "step_id": step, "structured_expectation_score": score,
              "semantic_residual_score": score, "observation_only_raw_score": score,
              "full_residual_score": score} for step, score in ((2, 1.0), (1, 0.0))]
    runs = [{**run, "success": False}]
    summarize_runs(steps, runs)
    assert runs[0]["first_structured_residual"] == 0.0
    assert runs[0]["mean_structured_residual"] == 0.5


def test_paired_model_comparison_requires_identical_tasks():
    runs = [{"model": "a", "split": "test", "task_id": "x"},
            {"model": "b", "split": "test", "task_id": "y"}]
    with pytest.raises(ValueError, match="Unpaired"):
        paired_comparisons(runs, 0.35)


def test_paired_model_comparison_keeps_denominators():
    runs = [{"model": model, "split": "test", "task_id": task,
             "first_structured_residual": score, "success": success}
            for model, task, score, success in (("a", "x", 0.5, False), ("a", "y", None, False),
                                               ("b", "x", 0.0, True), ("b", "y", 0.0, False))]
    result = paired_comparisons(runs, 0.35)
    selected = next(r for r in result if r["base_model"] == "a" and r["group"] == "high_first_residual")
    assert selected["paired_tasks"] == 1
    assert selected["peer_minus_base_success"] == 1.0
    assert next(r for r in result if r["base_model"] == "a" and r["group"] == "all")["paired_tasks"] == 2


def test_duplicate_runs_are_rejected(tmp_path):
    trace = {"model_name": "a", "split": "test", "task_id": "x", "success": False,
             "model_calls": 1, "total_tokens": 10, "steps": []}
    path = tmp_path / "traces.jsonl"
    path.write_text((json.dumps(trace) + "\n") * 2)
    with pytest.raises(ValueError, match="Duplicate run"):
        collect_steps([path])


def test_export_verifies_recorded_residual_and_skips_done(tmp_path):
    step = {"step_id": 1, "action_executed": True, "expected_signals": {
        "exit_code": "0", "stdout": "nonempty", "stderr": "empty"},
        "expected_outcome": "print the result", "exit_code": 0, "stdout": "ok", "stderr": "",
        "observation": "ok", "structured_residual": 0.0, "action": "echo ok",
        "next_action_if_expected": "<DONE>", "expectation_quality": 0.8,
        "observation_only_score": 0.0, "shadow_decision": "fast_candidate"}
    trace = {"model_name": "a", "split": "test", "task_id": "x", "success": True,
             "model_calls": 2, "total_tokens": 10,
             "steps": [step, {"action_executed": False, "structured_residual": None}]}
    path = tmp_path / "traces.jsonl"
    path.write_text(json.dumps(trace) + "\n")
    steps, runs = collect_steps([path])
    assert len(steps) == len(runs) == 1
    assert steps[0]["structured_expectation_score"] == 0.0
    step["structured_residual"] = 1.0
    path.write_text(json.dumps(trace) + "\n")
    with pytest.raises(ValueError, match="Recorded residual"):
        collect_steps([path])
