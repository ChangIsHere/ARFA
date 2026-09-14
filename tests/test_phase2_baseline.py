from src.phase2_baseline.model_client import ScriptedClient
from src.phase2_baseline.analyze_paper_results import _exact_mcnemar_pvalue, _paired_results
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
