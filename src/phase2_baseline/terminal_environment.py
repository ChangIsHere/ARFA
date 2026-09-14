from __future__ import annotations

import os
import resource
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CommandResult:
    action: str
    stdout: str
    stderr: str
    observation: str
    exit_code: int
    latency_seconds: float
    blocked: bool = False


class LocalTerminalEnvironment:
    def __init__(
        self,
        workspace_root: str | Path,
        task_id: str,
        command_timeout_seconds: int = 20,
        max_observation_chars: int = 4000,
        max_workspace_bytes: int = 10_000_000,
    ) -> None:
        self.workspace_root = Path(workspace_root)
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.workspace_root = self.workspace_root.resolve()
        self.root = Path(tempfile.mkdtemp(prefix=f"{task_id}-", dir=self.workspace_root))
        self.testbed = self.root / "testbed"
        self.command_timeout_seconds = command_timeout_seconds
        self.max_observation_chars = max_observation_chars
        self.max_workspace_bytes = max_workspace_bytes
        self._create_default_testbed()

    def close(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def run(self, action: str) -> CommandResult:
        if self._is_blocked(action):
            observation = "Command blocked by ARFA Phase 2 sandbox policy."
            return CommandResult(action, "", observation, observation, 126, 0.0, blocked=True)

        env = {
            "PATH": "/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin",
            "TESTBED": str(self.testbed),
            "HOME": str(self.root),
        }
        started = time.perf_counter()
        process = subprocess.Popen(
            action,
            shell=True,
            cwd=self.root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            preexec_fn=self._limit_created_file_size,
        )
        deadline = time.monotonic() + self.command_timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._kill_process_group(process)
                stdout, stderr = process.communicate()
                stderr += "\nCommand timed out."
                exit_code = 124
                break
            try:
                stdout, stderr = process.communicate(timeout=min(0.1, remaining))
                exit_code = process.returncode
                break
            except subprocess.TimeoutExpired:
                if self._workspace_size_bytes() > self.max_workspace_bytes:
                    self._kill_process_group(process)
                    stdout, stderr = process.communicate()
                    stderr += "\nCommand stopped: workspace size limit exceeded."
                    exit_code = 125
                    break
        latency = time.perf_counter() - started
        observation = self._format_observation(stdout, stderr, exit_code)
        return CommandResult(action, stdout, stderr, observation, exit_code, latency)

    def evaluate(self, task: dict[str, Any], trace_text: str, final_answer: str) -> dict[str, Any]:
        evaluator = task["evaluator"]
        kind = evaluator["kind"]
        if kind == "output_contains":
            oracle = self._run_oracle(str(evaluator["oracle_command"]))
            passed = self._normalize(oracle) in self._normalize(trace_text + "\n" + final_answer)
            return {"success": passed, "oracle": oracle, "kind": kind}
        if kind == "output_contains_all":
            haystack = self._normalize(trace_text + "\n" + final_answer)
            missing = [s for s in evaluator["snippets"] if self._normalize(str(s)) not in haystack]
            return {"success": not missing, "missing": missing, "kind": kind}
        if kind == "file_equals":
            target = self.testbed / str(evaluator["target_path"])
            actual = self._read_evaluation_file(target)
            expected = str(evaluator["expected_content"])
            return {"success": actual == expected, "actual": actual, "expected": expected, "kind": kind}
        if kind == "file_contains_all":
            target = self.testbed / str(evaluator["target_path"])
            actual = self._read_evaluation_file(target)
            missing = [s for s in evaluator["snippets"] if str(s) not in actual]
            return {"success": not missing, "missing": missing, "kind": kind}
        if kind == "no_empty_dirs":
            empty_dirs = [str(path.relative_to(self.testbed)) for path in self.testbed.rglob("*") if path.is_dir() and not any(path.iterdir())]
            return {"success": not empty_dirs, "empty_dirs": empty_dirs, "kind": kind}
        raise ValueError(f"Unknown evaluator kind: {kind}")

    def _create_default_testbed(self) -> None:
        files = {
            "dir1/textfile1.txt": "zebra\napple\nlemon\nbanana\n",
            "dir1/subdir1/jsonfile1.json": '{\n  "key1": "value1",\n  "key2": "value2",\n  "key3": "value3"\n}\n',
            "dir1/script.py": "print('hello from dir1')\n",
            "dir2/textfile2.txt": "alpha\nbeta\ngamma\nerror: missing value\n",
            "dir2/notes.txt": "first note\nsecond note\n",
            "dir2/tool.py": "print('tool')\n",
            "dir3/subdir1/subsubdir1/textfile3.txt": "".join(f"line{i:02d}\n" for i in range(1, 13)),
            "logs/error.log": "error: failed sample job\n",
        }
        for dirname in ["empty_a", "dir3/empty_b"]:
            (self.testbed / dirname).mkdir(parents=True, exist_ok=True)
        for relative, content in files.items():
            path = self.testbed / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

    def _run_oracle(self, command: str) -> str:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=self.root,
            env={"TESTBED": str(self.testbed), "PATH": "/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"},
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=self.command_timeout_seconds,
            preexec_fn=self._limit_created_file_size,
        )
        return completed.stdout if completed.returncode == 0 else completed.stdout + completed.stderr

    def _limit_created_file_size(self) -> None:
        resource.setrlimit(
            resource.RLIMIT_FSIZE,
            (self.max_workspace_bytes, self.max_workspace_bytes),
        )

    @staticmethod
    def _kill_process_group(process: subprocess.Popen[str]) -> None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def _workspace_size_bytes(self) -> int:
        total = 0
        for path in self.root.rglob("*"):
            try:
                if path.is_file():
                    total += path.stat().st_size
            except OSError:
                continue
        return total

    def _read_evaluation_file(self, path: Path) -> str:
        if not path.exists():
            return ""
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return handle.read(self.max_workspace_bytes + 1)

    def _format_observation(self, stdout: str, stderr: str, exit_code: int) -> str:
        parts = [f"exit_code={exit_code}"]
        if stdout:
            parts.append(f"stdout:\n{stdout}")
        if stderr:
            parts.append(f"stderr:\n{stderr}")
        observation = "\n".join(parts)
        if len(observation) > self.max_observation_chars:
            return observation[: self.max_observation_chars] + "\n...[truncated]"
        return observation

    @staticmethod
    def _normalize(text: str) -> str:
        return "\n".join(line.strip() for line in text.strip().splitlines() if line.strip())

    @staticmethod
    def _is_blocked(action: str) -> bool:
        lowered = action.lower()
        blocked_fragments = [
            "sudo ",
            " ssh ",
            "curl ",
            "wget ",
            "git push",
            "rm -rf /",
            "> /users/",
            "> /etc/",
            "> /var/",
        ]
        return any(fragment in f" {lowered}" for fragment in blocked_fragments)
