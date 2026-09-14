from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ChatResult:
    content: str
    latency_seconds: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    raw: dict[str, Any] | None = None


class ChatClient(Protocol):
    def chat(self, messages: list[dict[str, str]]) -> ChatResult:
        ...


class OpenAICompatibleClient:
    """Small OpenAI-compatible chat client for local model servers."""

    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 700,
        timeout_seconds: int = 120,
        request_attempts: int = 3,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key or "ollama"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.request_attempts = request_attempts

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "OpenAICompatibleClient":
        api_key_env = str(config.get("api_key_env") or "")
        return cls(
            base_url=str(config["base_url"]),
            model_name=str(config["model_name"]),
            api_key=os.environ.get(api_key_env) if api_key_env else None,
            temperature=float(config.get("temperature", 0.0)),
            max_tokens=int(config.get("max_tokens", 700)),
            timeout_seconds=int(config.get("request_timeout_seconds", 120)),
            request_attempts=int(config.get("request_attempts", 3)),
        )

    def chat(self, messages: list[dict[str, str]]) -> ChatResult:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        started = time.perf_counter()
        last_error: Exception | None = None
        for attempt in range(1, self.request_attempts + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    raw = json.loads(response.read().decode("utf-8"))
                break
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = exc
                if attempt < self.request_attempts:
                    time.sleep(2 ** (attempt - 1))
        else:
            raise RuntimeError(
                "Could not reach the local OpenAI-compatible model server after retries. "
                "Start Ollama/LM Studio/vLLM and check the Phase 2 config."
            ) from last_error
        latency = time.perf_counter() - started
        content = raw["choices"][0]["message"]["content"]
        usage = raw.get("usage") or {}
        return ChatResult(
            content=content,
            latency_seconds=latency,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            raw=raw,
        )


class ScriptedClient:
    """Deterministic client used only to validate the Phase 2 harness."""

    def chat(self, messages: list[dict[str, str]]) -> ChatResult:
        prompt = messages[-1]["content"].lower()
        action = "find \"$TESTBED\" | wc -l"
        if "reverse alphabetical" in prompt:
            action = "sort -r \"$TESTBED/dir1/textfile1.txt\" > \"$TESTBED/dir1/textfile1_reverse_sorted.txt\""
        elif "number of lines" in prompt:
            action = "find \"$TESTBED/dir2\" -type f -name '*.txt' -print0 | xargs -0 wc -l"
        elif "last 10 lines" in prompt:
            action = "tail -n 10 \"$TESTBED/dir3/subdir1/subsubdir1/textfile3.txt\""
        elif "value3" in prompt:
            action = "grep -n 'value3' \"$TESTBED/dir1/subdir1/jsonfile1.json\""
        elif "empty directories" in prompt:
            action = "find \"$TESTBED\" -type d -empty -delete"
        elif "concatenate" in prompt:
            action = "find \"$TESTBED\" -type f -name '*.txt' ! -name 'concatenated_text_files.txt' -exec cat {} \\; > \"$TESTBED/concatenated_text_files.txt\""
        elif "word error" in prompt:
            action = "grep -rl 'error' \"$TESTBED\""
        content = json.dumps(
            {
                "thought": "Use one shell command that directly addresses the task.",
                "action": action,
                "expected_outcome": "The command should complete successfully and show or create the requested result.",
                "done": False,
                "final_answer": "",
            }
        )
        if "previous observation" in prompt:
            content = json.dumps(
                {
                    "thought": "The requested command has run; submit the observed result.",
                    "action": "",
                    "expected_outcome": "No more terminal output is needed.",
                    "done": True,
                    "final_answer": "Done.",
                }
            )
        return ChatResult(content=content, latency_seconds=0.0, total_tokens=0)
