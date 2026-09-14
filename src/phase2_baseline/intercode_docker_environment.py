from __future__ import annotations

import math
import re
import shlex
import subprocess
import uuid
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer

from src.phase2_baseline.terminal_environment import CommandResult


FILESYSTEM_ROOTS = {1: "/testbed", 2: "/system", 3: "/workspace", 4: "/"}


def _decode_stream(value: bytes | str) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


class InterCodeDockerEnvironment:
    """Isolated InterCode-compatible NL2Bash task environment."""

    def __init__(
        self,
        task: dict[str, Any],
        image_prefix: str = "arfa/intercode-nl2bash:fs",
        command_timeout_seconds: int = 20,
        max_observation_chars: int = 4000,
        container_memory: str = "1g",
        network_mode: str = "bridge",
    ) -> None:
        self.task = task
        self.filesystem_version = int(task["filesystem_version"])
        self.testbed = FILESYSTEM_ROOTS[self.filesystem_version]
        self.image = f"{image_prefix}{self.filesystem_version}"
        self.command_timeout_seconds = command_timeout_seconds
        self.max_observation_chars = max_observation_chars
        suffix = uuid.uuid4().hex[:10]
        self.agent_container = f"arfa-p2-agent-{suffix}"
        self.eval_container = f"arfa-p2-eval-{suffix}"
        self.workdir = "/"
        self.eval_workdir = "/"
        self.last_raw_output = ""
        self._start_container(self.agent_container, container_memory, network_mode)
        try:
            self._start_container(self.eval_container, container_memory, network_mode)
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        for name in (self.agent_container, self.eval_container):
            subprocess.run(
                ["docker", "rm", "-f", name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )

    def run(self, action: str) -> CommandResult:
        started = __import__("time").perf_counter()
        exit_code, stdout, stderr = self._exec_action(self.agent_container, action, agent=True)
        latency = __import__("time").perf_counter() - started
        self.last_raw_output = stdout + stderr
        observation = self._format_observation(stdout, stderr, exit_code)
        return CommandResult(action, stdout, stderr, observation, exit_code, latency)

    def evaluate(self, task: dict[str, Any], trace_text: str, final_answer: str) -> dict[str, Any]:
        del trace_text, final_answer
        gold = task["gold_command"]
        gold_commands = [gold] if isinstance(gold, str) else list(gold)
        gold_exit_codes: list[int] = []
        gold_output_parts: list[str] = []
        for command in gold_commands:
            exit_code, stdout, stderr = self._exec_action(self.eval_container, str(command), agent=False)
            gold_exit_codes.append(exit_code)
            gold_output_parts.append(stdout + stderr)
        gold_output = "".join(gold_output_parts)

        agent_diff = self._git_status(self.agent_container)
        eval_diff = self._git_status(self.eval_container)
        diff_miss = sorted(set(eval_diff) - set(agent_diff))
        diff_extra = sorted(set(agent_diff) - set(eval_diff))
        p1_score = round(0.33 * (1 - math.erf(len(diff_miss) + len(diff_extra))), 2)

        common = sorted(set(agent_diff) & set(eval_diff))
        # Preserve the released InterCode evaluator's status filter exactly.
        comparable = [item for item in common if item[0] in {"A", "C", "??"}]
        matching_paths: list[str] = []
        for _, path in comparable:
            if self._fingerprint(self.agent_container, path) == self._fingerprint(self.eval_container, path):
                matching_paths.append(path)
        p2_score = 0.33 if not comparable else round(0.33 * len(matching_paths) / len(comparable), 2)

        similarity = self._text_similarity(self.last_raw_output, gold_output)
        p3_score = round(0.33 * similarity, 2)
        reward = round(0.01 + p1_score + p2_score + p3_score, 2)
        return {
            "success": reward >= 0.99,
            "reward": reward,
            "kind": "intercode_official_reward_reimplementation",
            "reward_components": {
                "file_diff": p1_score,
                "file_changes": p2_score,
                "answer_similarity": p3_score,
            },
            "answer_similarity": similarity,
            "diff_miss": diff_miss,
            "diff_extra": diff_extra,
            "matching_changed_paths": matching_paths,
            "changed_path_count": len(comparable),
            "agent_observation": self.last_raw_output[: self.max_observation_chars],
            "gold_observation": gold_output[: self.max_observation_chars],
            "gold_exit_codes": gold_exit_codes,
        }

    def _start_container(self, name: str, memory: str, network_mode: str) -> None:
        completed = subprocess.run(
            [
                "docker",
                "run",
                "--detach",
                "--rm",
                "--network",
                network_mode,
                "--memory",
                memory,
                "--pids-limit",
                "256",
                "--name",
                name,
                self.image,
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"Failed to start {self.image}: {completed.stderr.strip()}")

    def _exec_action(self, container: str, action: str, agent: bool) -> tuple[int, str, str]:
        workdir = self.workdir if agent else self.eval_workdir
        command = (
            "ulimit -f 20480; "
            f"timeout --signal=KILL {self.command_timeout_seconds}s "
            f"/bin/bash -lc {shlex.quote(action)}"
        )
        completed = subprocess.run(
            ["docker", "exec", "--workdir", workdir, "--env", f"TESTBED={self.testbed}", container, "/bin/bash", "-lc", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=self.command_timeout_seconds + 10,
        )
        if completed.returncode == 0 and re.fullmatch(r"\s*cd(?:\s+.+)?\s*", action):
            target = action.strip()[2:].strip() or "/root"
            resolved = self._resolve_directory(container, workdir, target)
            if resolved:
                if agent:
                    self.workdir = resolved
                else:
                    self.eval_workdir = resolved
        return completed.returncode, _decode_stream(completed.stdout), _decode_stream(completed.stderr)

    @staticmethod
    def _resolve_directory(container: str, workdir: str, target: str) -> str | None:
        command = f"cd {shlex.quote(target)} && pwd -P"
        completed = subprocess.run(
            ["docker", "exec", "--workdir", workdir, container, "/bin/bash", "-lc", command],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    @staticmethod
    def _git_status(container: str) -> list[tuple[str, str]]:
        completed = subprocess.run(
            ["docker", "exec", "--workdir", "/", container, "git", "status", "--short"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=True,
        )
        rows: list[tuple[str, str]] = []
        for line in completed.stdout.splitlines():
            if len(line) < 4:
                continue
            status = line[:2].strip()
            path = line[3:].strip()
            if " -> " in path:
                path = path.rsplit(" -> ", 1)[1]
            if status and path:
                rows.append((status, "/" + path.lstrip("/")))
        return rows

    @staticmethod
    def _fingerprint(container: str, path: str) -> str:
        quoted = shlex.quote(path)
        command = (
            f"if [ -e {quoted} ] || [ -L {quoted} ]; then "
            f"tar --sort=name --mtime='UTC 1970-01-01' --owner=0 --group=0 --numeric-owner -cf - {quoted} 2>/dev/null | sha256sum; "
            "else echo MISSING; fi"
        )
        completed = subprocess.run(
            ["docker", "exec", container, "/bin/bash", "-lc", command],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
        return completed.stdout.strip()

    def _format_observation(self, stdout: str, stderr: str, exit_code: int) -> str:
        parts = [f"exit_code={exit_code}"]
        if stdout:
            parts.append(f"stdout:\n{stdout}")
        if stderr:
            parts.append(f"stderr:\n{stderr}")
        observation = "\n".join(parts)
        if len(observation) > self.max_observation_chars:
            observation = observation[: self.max_observation_chars] + "\n...[truncated]"
        return observation

    @staticmethod
    def _text_similarity(first: str, second: str) -> float:
        if not first and not second:
            return 1.0
        try:
            matrix = TfidfVectorizer().fit_transform([first, second])
            return float((matrix * matrix.T).toarray()[0][1])
        except ValueError:
            return 1.0 if first == second else 0.0
