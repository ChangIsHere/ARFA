import json
from subprocess import CompletedProcess

import pytest

from src.phase2_baseline.build_official_dataset import build_records
from src.phase2_baseline.intercode_docker_environment import InterCodeDockerEnvironment, _decode_stream


def test_build_official_records(tmp_path):
    for fs_version, count in ((1, 2), (2, 1), (3, 1), (4, 1)):
        rows = [{"query": f"Count sample {index}", "gold": "true"} for index in range(count)]
        (tmp_path / f"nl2bash_fs_{fs_version}.json").write_text(json.dumps(rows), encoding="utf-8")

    records = build_records(tmp_path)

    assert len(records) == 5
    assert records[0]["task_id"] == "intercode-nl2bash-fs1-000"
    assert records[-1]["filesystem_version"] == 4
    assert all(record["source_commit"] for record in records)
    assert all(record["source_partition"] == "full_intercode_nl2bash_suite" for record in records)


def test_intercode_text_similarity():
    assert InterCodeDockerEnvironment._text_similarity("same output", "same output") == pytest.approx(1.0)
    assert InterCodeDockerEnvironment._text_similarity("", "") == 1.0
    assert InterCodeDockerEnvironment._text_similarity("alpha", "beta") == 0.0


def test_intercode_binary_output_uses_replacement_decoding():
    assert _decode_stream(b"gzip:\x8bpayload") == "gzip:\ufffdpayload"


def test_intercode_rejects_nul_action_without_crashing(monkeypatch):
    env = object.__new__(InterCodeDockerEnvironment)

    def unexpected_subprocess(*args, **kwargs):
        del args, kwargs
        pytest.fail("A command containing NUL must not reach subprocess.run")

    monkeypatch.setattr("src.phase2_baseline.intercode_docker_environment.subprocess.run", unexpected_subprocess)

    exit_code, stdout, stderr = env._exec_action("agent-container", "printf 'bad\x00command'", agent=True)

    assert exit_code == 126
    assert stdout == ""
    assert stderr == "Invalid command: NUL byte is not allowed."


def test_git_status_reports_missing_evaluator_tool(monkeypatch):
    def missing_git(*args, **kwargs):
        del args, kwargs
        return CompletedProcess([], 127, stdout="", stderr="executable file not found")

    monkeypatch.setattr("src.phase2_baseline.intercode_docker_environment.subprocess.run", missing_git)

    with pytest.raises(RuntimeError, match="Git status failed"):
        InterCodeDockerEnvironment._git_status("agent-container")


def test_evaluate_restores_git_before_scoring(monkeypatch):
    env = object.__new__(InterCodeDockerEnvironment)
    env.agent_container = "agent"
    env.eval_container = "reference"
    env.last_raw_output = ""
    env.max_observation_chars = 4000
    status_attempts = {"agent": 0}

    monkeypatch.setattr(env, "_exec_action", lambda *args, **kwargs: (0, "", ""))
    monkeypatch.setattr(env, "_fingerprint", lambda *args, **kwargs: "same")
    monkeypatch.setattr(env, "_restore_git_from_reference", lambda: None)

    def git_status(container):
        if container == "agent" and status_attempts["agent"] == 0:
            status_attempts["agent"] += 1
            raise RuntimeError("git missing")
        return []

    monkeypatch.setattr(env, "_git_status", git_status)
    result = env.evaluate({"gold_command": "true"}, trace_text="", final_answer="")

    assert result["success"] is True
    assert result["evaluator_git_restored"] is True
