from src.phase2_baseline.terminal_environment import LocalTerminalEnvironment


def test_phase2_environment_runs_commands(tmp_path):
    env = LocalTerminalEnvironment(tmp_path, "task")
    try:
        result = env.run("tail -n 1 \"$TESTBED/dir1/textfile1.txt\"")
        assert result.exit_code == 0
        assert "banana" in result.observation
    finally:
        env.close()


def test_phase2_environment_blocks_network_commands(tmp_path):
    env = LocalTerminalEnvironment(tmp_path, "task")
    try:
        result = env.run("curl https://example.com")
        assert result.blocked is True
        assert result.exit_code == 126
    finally:
        env.close()


def test_phase2_environment_limits_created_file_size(tmp_path):
    env = LocalTerminalEnvironment(tmp_path, "task", max_workspace_bytes=10_000)
    try:
        result = env.run(
            "while true; do cat \"$TESTBED/dir1/textfile1.txt\" >> \"$TESTBED/growing.txt\"; done"
        )
        target = env.testbed / "growing.txt"

        assert result.exit_code != 0
        assert result.exit_code == 125
        assert target.stat().st_size < 100_000
    finally:
        env.close()
