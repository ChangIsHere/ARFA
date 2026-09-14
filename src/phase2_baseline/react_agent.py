from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, asdict
from typing import Any

from src.phase2_baseline.model_client import ChatClient
from src.phase2_baseline.prompts import SYSTEM_PROMPT, build_user_prompt
from src.phase2_baseline.terminal_environment import LocalTerminalEnvironment


@dataclass(frozen=True)
class Phase2Step:
    step_id: int
    thought: str
    action: str
    expected_outcome: str
    observation: str
    exit_code: int | None
    model_latency_seconds: float
    command_latency_seconds: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    parse_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Phase2Run:
    task_id: str
    instruction: str
    benchmark: str
    success: bool
    evaluator: dict[str, Any]
    final_answer: str
    steps: list[Phase2Step]
    model_calls: int
    total_model_latency_seconds: float
    total_command_latency_seconds: float
    total_tokens: int | None
    termination_reason: str
    max_steps_reached: bool
    parse_error_count: int
    nonzero_exit_count: int
    repeated_action_count: int
    agent_wall_seconds: float
    total_task_wall_seconds: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["steps"] = [step.to_dict() for step in self.steps]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Phase2Run":
        values = dict(payload)
        values["steps"] = [Phase2Step(**step) for step in payload.get("steps", [])]
        values.setdefault("termination_reason", "legacy_unknown")
        values.setdefault("max_steps_reached", False)
        values.setdefault("parse_error_count", 0)
        values.setdefault("nonzero_exit_count", 0)
        values.setdefault("repeated_action_count", 0)
        values.setdefault(
            "agent_wall_seconds",
            float(values.get("total_model_latency_seconds", 0.0))
            + float(values.get("total_command_latency_seconds", 0.0)),
        )
        values.setdefault("total_task_wall_seconds", values["agent_wall_seconds"])
        return cls(**values)


class ReActBaselineAgent:
    def __init__(self, client: ChatClient, max_steps: int = 12) -> None:
        self.client = client
        self.max_steps = max_steps

    def run_task(self, task: dict[str, Any], env: LocalTerminalEnvironment) -> Phase2Run:
        run_started = time.perf_counter()
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        previous_observation: str | None = None
        steps: list[Phase2Step] = []
        final_answer = ""
        termination_reason = "max_steps"

        for step_id in range(1, self.max_steps + 1):
            messages.append(
                {
                    "role": "user",
                    "content": build_user_prompt(
                        instruction=str(task["instruction"]),
                        testbed_path=str(env.testbed),
                        previous_observation=previous_observation,
                    ),
                }
            )
            result = self.client.chat(messages)
            parsed, parse_error = self._parse_model_json(result.content)
            messages.append({"role": "assistant", "content": result.content})

            if bool(parsed.get("done", False)):
                termination_reason = "model_done"
                final_answer = str(parsed.get("final_answer") or "")
                steps.append(
                    Phase2Step(
                        step_id=step_id,
                        thought=str(parsed.get("thought") or ""),
                        action="",
                        expected_outcome=str(parsed.get("expected_outcome") or ""),
                        observation="",
                        exit_code=None,
                        model_latency_seconds=result.latency_seconds,
                        command_latency_seconds=0.0,
                        prompt_tokens=result.prompt_tokens,
                        completion_tokens=result.completion_tokens,
                        total_tokens=result.total_tokens,
                        parse_error=parse_error,
                    )
                )
                break

            action = str(parsed.get("action") or "").strip()
            if not action:
                previous_observation = "No action was provided by the model."
                command_result = None
            else:
                command_result = env.run(action)
                previous_observation = command_result.observation

            steps.append(
                Phase2Step(
                    step_id=step_id,
                    thought=str(parsed.get("thought") or ""),
                    action=action,
                    expected_outcome=str(parsed.get("expected_outcome") or ""),
                    observation=previous_observation,
                    exit_code=command_result.exit_code if command_result else None,
                    model_latency_seconds=result.latency_seconds,
                    command_latency_seconds=command_result.latency_seconds if command_result else 0.0,
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    total_tokens=result.total_tokens,
                    parse_error=parse_error,
                )
            )

        trace_text = "\n".join(step.observation for step in steps if step.observation)
        evaluator = env.evaluate(task, trace_text=trace_text, final_answer=final_answer)
        token_values = [step.total_tokens for step in steps if step.total_tokens is not None]
        actions = [step.action for step in steps if step.action]
        return Phase2Run(
            task_id=str(task["task_id"]),
            instruction=str(task["instruction"]),
            benchmark=str(task.get("benchmark", "")),
            success=bool(evaluator["success"]),
            evaluator=evaluator,
            final_answer=final_answer,
            steps=steps,
            model_calls=len(steps),
            total_model_latency_seconds=sum(step.model_latency_seconds for step in steps),
            total_command_latency_seconds=sum(step.command_latency_seconds for step in steps),
            total_tokens=sum(token_values) if token_values else None,
            termination_reason=termination_reason,
            max_steps_reached=termination_reason == "max_steps",
            parse_error_count=sum(step.parse_error is not None for step in steps),
            nonzero_exit_count=sum(step.exit_code not in (None, 0) for step in steps),
            repeated_action_count=len(actions) - len(set(actions)),
            agent_wall_seconds=time.perf_counter() - run_started,
            total_task_wall_seconds=0.0,
        )

    @staticmethod
    def _parse_model_json(content: str) -> tuple[dict[str, Any], str | None]:
        try:
            return json.loads(content), None
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, flags=re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0)), "extracted_json_from_non_json_response"
                except json.JSONDecodeError:
                    pass
        return {
            "thought": "Could not parse model response as JSON.",
            "action": "",
            "expected_outcome": "",
            "done": False,
            "final_answer": "",
        }, "json_parse_failed"
