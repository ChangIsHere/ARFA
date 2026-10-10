import json
import threading
import time

from src.phase1_5_shadow.json_model_client import JSONModelRequestError
from src.phase2_baseline.model_client import ChatResult
from src.phase2_baseline.terminal_environment import LocalTerminalEnvironment
from src.phase3_arfa.arfa_agent import ARFAAgent
from src.phase3_arfa.analyze_results import _paired_comparison
from src.phase3_arfa.fast_track import FastTrack
from src.phase3_arfa.router import ResidualRouter
from src.phase3_arfa.run_arfa import _iter_completed, _model_metadata, phase3_status
from src.phase3_arfa.slow_track import SlowTrack


def test_formal_phase3_remains_locked():
    status = phase3_status({"locked": True, "reason": "formal gate pending"})

    assert status["ready"] is False
    assert status["reason"] == "formal gate pending"


def test_phase3_controller_can_be_unlocked():
    status = phase3_status(
        {
            "locked": False,
            "stage": "phase3_p_exploratory",
            "scope": "engineering_pilot",
            "formal_claims_enabled": False,
            "use_phase3_final_tasks": False,
            "router": {"fail_open_to_reasoning": True},
        }
    )

    assert status["ready"] is True
    assert status["formal_claims_enabled"] is False
    assert status["use_phase3_final_tasks"] is False
    assert status["router"]["fail_open_to_reasoning"] is True
    assert status["controller_implemented"] is True
    assert status["execution_started"] is False


def test_remote_model_metadata_uses_frozen_artifact_identity():
    metadata = _model_metadata("remote-model", "base_id=abc;modelfile_sha256=def")

    assert metadata == {
        "model_name": "remote-model",
        "artifact_identity": "base_id=abc;modelfile_sha256=def",
    }


class _SequenceClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def chat(self, messages):
        del messages
        response = self.responses[self.calls]
        self.calls += 1
        return ChatResult(content=json.dumps(response), latency_seconds=0.01, total_tokens=10)


class _FailingClient:
    def chat(self, messages):
        del messages
        raise JSONModelRequestError("HTTP 500: failed to generate structured response")


def _response(action, expected_signals, done=False):
    return {
        "thought": "test",
        "current_plan": "test the route",
        "action": action,
        "expected_outcome": "Print a result successfully",
        "expected_signals": expected_signals,
        "next_action_if_expected": "<DONE>",
        "done": done,
        "final_answer": "done" if done else "",
    }


def _task():
    return {
        "task_id": "phase3-route-test",
        "instruction": "Print ok",
        "benchmark": "test",
        "evaluator": {"kind": "output_contains", "oracle_command": "printf ok"},
    }


def _router():
    return ResidualRouter(
        policy="residual",
        residual_threshold=0.35,
        observation_threshold=0.55,
        minimum_expectation_quality=0.70,
        max_consecutive_fast_calls=1,
    )


def test_low_residual_routes_next_turn_to_fast_model(tmp_path):
    slow = _SequenceClient([_response("printf ok", {"exit_code": "0", "stdout": "nonempty", "stderr": "empty"})])
    fast = _SequenceClient([_response("", {"exit_code": "any", "stdout": "any", "stderr": "any"}, done=True)])
    agent = ARFAAgent(
        FastTrack("fast", fast), SlowTrack("slow", slow), _router(), "test", "development", max_steps=3
    )
    env = LocalTerminalEnvironment(tmp_path, "low-residual")
    try:
        run = agent.run_task(_task(), env)
    finally:
        env.close()

    assert [step.model_role for step in run.steps] == ["slow", "fast"]
    assert run.steps[0].structured_residual == 0.0
    assert run.success is True


def test_high_residual_keeps_next_turn_on_slow_model(tmp_path):
    slow = _SequenceClient(
        [
            _response("false", {"exit_code": "0", "stdout": "nonempty", "stderr": "empty"}),
            _response("", {"exit_code": "any", "stdout": "any", "stderr": "any"}, done=True),
        ]
    )
    fast = _SequenceClient([])
    agent = ARFAAgent(
        FastTrack("fast", fast), SlowTrack("slow", slow), _router(), "test", "development", max_steps=3
    )
    env = LocalTerminalEnvironment(tmp_path, "high-residual")
    try:
        run = agent.run_task(_task(), env)
    finally:
        env.close()

    assert [step.model_role for step in run.steps] == ["slow", "slow"]
    assert run.steps[0].structured_residual >= 0.35
    assert fast.calls == 0


def test_model_request_error_is_recorded_without_aborting_matrix(tmp_path):
    agent = ARFAAgent(
        FastTrack("fast", _FailingClient()),
        SlowTrack("slow", _FailingClient()),
        _router(),
        "test",
        "development",
        max_steps=3,
    )
    env = LocalTerminalEnvironment(tmp_path, "model-error")
    try:
        run = agent.run_task(_task(), env)
    finally:
        env.close()

    assert run.termination_reason == "model_request_error"
    assert run.model_calls == 1
    assert run.parse_error_count == 1
    assert run.steps[0].parse_error == "model_request_failed"
    assert "HTTP 500" in run.steps[0].observation


def test_phase3_paired_analysis_uses_matching_task_ids():
    residual = [
        {"task_id": "a", "success": "True", "total_task_wall_seconds": "5", "slow_model_calls": "1", "total_tokens": "50"},
        {"task_id": "b", "success": "False", "total_task_wall_seconds": "5", "slow_model_calls": "1", "total_tokens": "50"},
    ]
    always_slow = [
        {"task_id": "a", "success": "True", "total_task_wall_seconds": "10", "slow_model_calls": "2", "total_tokens": "100"},
        {"task_id": "b", "success": "True", "total_task_wall_seconds": "10", "slow_model_calls": "2", "total_tokens": "100"},
    ]
    result = _paired_comparison(residual, always_slow, samples=100, seed=1)

    assert result["paired_tasks"] == 2
    assert result["point_estimates"]["success_rate_difference"] == -0.5
    assert result["point_estimates"]["wall_time_reduction"] == 0.5
    assert result["point_estimates"]["slow_call_reduction"] == 0.5
    assert result["candidate_only_successes"] == 0
    assert result["comparator_only_successes"] == 1
    assert result["exact_mcnemar_pvalue"] == 1.0


def test_parallel_task_iterator_is_bounded():
    lock = threading.Lock()
    active = 0
    maximum_active = 0

    def worker(task):
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return task["task_id"]

    tasks = [{"task_id": str(index)} for index in range(12)]
    completed = list(_iter_completed(tasks, worker, workers=3))

    assert sorted(completed) == sorted(task["task_id"] for task in tasks)
    assert maximum_active == 3


def test_agent_context_keeps_system_and_recent_turns_only():
    messages = [{"role": "system", "content": "system"}]
    messages.extend(
        {"role": "user" if index % 2 == 0 else "assistant", "content": str(index)}
        for index in range(10)
    )

    bounded = ARFAAgent._bounded_messages(messages, max_history_turns=2)

    assert bounded[0] == messages[0]
    assert bounded[1:] == messages[-4:]
