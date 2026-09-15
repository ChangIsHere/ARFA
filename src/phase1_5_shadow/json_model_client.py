from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from src.phase2_baseline.model_client import ChatResult


SHADOW_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "thought": {"type": "string"},
        "current_plan": {"type": "string"},
        "action": {"type": "string"},
        "expected_outcome": {"type": "string"},
        "expected_signals": {
            "type": "object",
            "properties": {
                "exit_code": {"enum": ["0", "nonzero", "any"]},
                "stdout": {"enum": ["empty", "nonempty", "any"]},
                "stderr": {"enum": ["empty", "nonempty", "any"]},
            },
            "required": ["exit_code", "stdout", "stderr"],
        },
        "next_action_if_expected": {"type": "string"},
        "done": {"type": "boolean"},
        "final_answer": {"type": "string"},
    },
    "required": [
        "thought",
        "current_plan",
        "action",
        "expected_outcome",
        "expected_signals",
        "next_action_if_expected",
        "done",
        "final_answer",
    ],
}


class OllamaStructuredClient:
    """Ollama-native client with schema-constrained decoding."""

    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 900,
        timeout_seconds: int = 180,
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
    def from_config(cls, config: dict[str, Any]) -> "OllamaStructuredClient":
        api_key_env = str(config.get("api_key_env") or "")
        return cls(
            base_url=str(config["base_url"]),
            model_name=str(config["model_name"]),
            api_key=os.environ.get(api_key_env) if api_key_env else None,
            temperature=float(config.get("temperature", 0.0)),
            max_tokens=int(config.get("max_tokens", 900)),
            timeout_seconds=int(config.get("request_timeout_seconds", 180)),
            request_attempts=int(config.get("request_attempts", 3)),
        )

    def _payload(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "format": SHADOW_RESPONSE_SCHEMA,
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
        }

    def chat(self, messages: list[dict[str, str]]) -> ChatResult:
        request = urllib.request.Request(
            f"{self.base_url.removesuffix('/v1')}/api/chat",
            data=json.dumps(self._payload(messages)).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
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
            raise RuntimeError("Could not reach the JSON-mode local model endpoint after retries") from last_error

        content = str(raw["message"]["content"])
        prompt_tokens = raw.get("prompt_eval_count")
        completion_tokens = raw.get("eval_count")
        return ChatResult(
            content=content,
            latency_seconds=time.perf_counter() - started,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=(prompt_tokens + completion_tokens) if prompt_tokens is not None and completion_tokens is not None else None,
            raw=raw,
        )
