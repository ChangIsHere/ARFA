"""Slow-track model wrapper used by the Phase 3 agent."""
from __future__ import annotations

from src.phase2_baseline.model_client import ChatClient, ChatResult


class SlowTrack:
    def __init__(self, model_name: str, client: ChatClient) -> None:
        self.model_name = model_name
        self.client = client

    def reason(self, messages: list[dict[str, str]]) -> ChatResult:
        return self.client.chat(messages)
