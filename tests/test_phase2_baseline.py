import json

from src.phase2_baseline.model_client import ModelRequestError, ScriptedClient
from src.phase2_baseline.run_baseline import _build_model_error_run
from src.phase2_baseline.analyze_paper_results import (
    PLANNED_PAIRS,
    _audit_completeness,
    _exact_mcnemar_pvalue,
    _paired_results,
)
from src.phase2_baseline.react_agent import ReActBaselineAgent
from src.phase2_baseline.task_loader import load_tasks
from src.phase2_baseline.terminal_environment import LocalTerminalEnvironment


def test_phase2_scripted_react_baseline(tmp_path):
    tasks = load_tasks("data/phase2/intercode_bash_local_tasks.jsonl", limit=2)
    agent = ReActBaselineAgent(ScriptedClient(), max_steps=3)
    runs = []
    for task in tasks:
        env = LocalTerminalEnvironment(tmp_path, task["task_id"])
        try:
            runs.append(agent.run_task(task, env))
        finally:
            env.close()

    assert [run.success for run in runs] == [True, True]
    assert all(run.model_calls == 2 for run in runs)
    assert all(run.termination_reason == "model_done" for run in runs)
    assert all(run.max_steps_reached is False for run in runs)
    assert all(run.parse_error_count == 0 for run in runs)


def test_phase2_run_round_trip(tmp_path):
    task = load_tasks("data/phase2/intercode_bash_local_tasks.jsonl", limit=1)[0]
    env = LocalTerminalEnvironment(tmp_path, task["task_id"])
    try:
        run = ReActBaselineAgent(ScriptedClient(), max_steps=3).run_task(task, env)
    finally:
        env.close()

    restored = type(run).from_dict(run.to_dict())
    assert restored == run


def test_exact_mcnemar_and_paired_difference():
    runs = {
        "arfa-qwen2.5-coder:7b-8k": [
            {"task_id": "a", "success": True},
            {"task_id": "b", "success": False},
        ],
        "arfa-llama3.1:8b-8k": [
            {"task_id": "a", "success": False},
            {"task_id": "b", "success": False},
        ],
    }
    row = _paired_results(runs, samples=100)[0]

    assert row["success_rate_difference_a_minus_b"] == 0.5
    assert row["a_only_successes"] == 1
    assert row["b_only_successes"] == 0
    assert _exact_mcnemar_pvalue(1, 0) == 1.0


def test_paper_matrix_audit_requires_same_unique_tasks_and_hashes():
    hashes = {
        "task_manifest_sha256": "tasks",
        "config_sha256": "config",
        "prompt_sha256": "prompt",
        "agent_sha256": "agent",
        "environment_sha256": "environment",
    }
    names = {"fast", "slow"}
    runs = {name: [{"task_id": "a"}, {"task_id": "b"}] for name in names}
    summaries = {
        name: {
            "model_name": name,
            "task_count": 2,
            "dry_run": False,
            "paper_protocol": True,
            "reproducibility": dict(hashes),
        }
        for name in names
    }
    assert _audit_completeness(runs, summaries, names, {"a", "b"}, "tasks")["complete"]

    runs["fast"] = [{"task_id": "a"}, {"task_id": "a"}]
    summaries["slow"]["reproducibility"]["prompt_sha256"] = "changed"
    audit = _audit_completeness(runs, summaries, names, {"a", "b"}, "tasks")
    assert not audit["complete"]
    assert "duplicate_task_ids" in audit["models"]["fast"]["issues"]
    assert "task_id_set_mismatch" in audit["models"]["fast"]["issues"]
    assert "prompt_sha256_mismatch" in audit["models"]["slow"]["issues"]


def test_phase2_expansion_matrix_has_eight_models_and_four_pairs():
    with open("data/phase2/model_matrix.json", encoding="utf-8") as handle:
        matrix = json.load(handle)
    names = [row["name"] for row in matrix["models"]]

    assert len(names) == len(set(names)) == 8
    assert len(PLANNED_PAIRS) == 4
    assert all(fast in names and slow in names for fast, slow in PLANNED_PAIRS)
    assert all(row["experiment_context_length"] == 8192 for row in matrix["models"])


def test_model_request_failure_becomes_a_scored_task_record(tmp_path):
    task = load_tasks("data/phase2/intercode_bash_local_tasks.jsonl", limit=1)[0]
    env = LocalTerminalEnvironment(tmp_path, task["task_id"])
    try:
        run = _build_model_error_run(task, env, ModelRequestError("repeat limit"), 12.5)
    finally:
        env.close()

    assert run.task_id == task["task_id"]
    assert run.termination_reason == "model_request_error"
    assert run.model_calls == 1
    assert run.parse_error_count == 1
    assert run.steps[0].parse_error == "model_request_failed"
    assert "repeat limit" in run.steps[0].observation
