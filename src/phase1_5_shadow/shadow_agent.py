from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Any

from src.phase1_5_shadow.prompts import SYSTEM_PROMPT, build_user_prompt
from src.phase1_5_shadow.residual import (
    expectation_quality,
    normalize_expected_signals,
    observation_only_score,
    shadow_decision,
    structured_residual,
)
from src.phase2_baseline.model_client import ChatClient


@dataclass(frozen=True)
class ShadowStep:
    step_id: int
    thought: str
    current_plan: str
    action: str
    expected_outcome: str
    expected_signals: dict[str, str]
    next_action_if_expected: str
    observation: str
    stdout: str
    stderr: str
    exit_code: int | None
    expectation_quality: float
    structured_residual: float | None
    observation_only_score: float | None
    residual_checks: dict[str, bool]
    shadow_decision: str
    shadow_reason: str
    model_latency_seconds: float
    command_latency_seconds: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    action_executed: bool
    raw_model_response: str = ""
    parse_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ShadowRun:
    task_id: str
    instruction: str
    benchmark: str
    split: str
    model_name: str
    success: bool
    evaluator: dict[str, Any]
    final_answer: str
    steps: list[ShadowStep]
    model_calls: int
    total_tokens: int | None
    total_model_latency_seconds: float
    total_command_latency_seconds: float
    termination_reason: str
    max_steps_reached: bool
    shadow_fast_candidates: int
    executed_steps: int
    parse_error_count: int
    total_task_wall_seconds: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["steps"] = [step.to_dict() for step in self.steps]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ShadowRun":
        values = dict(payload)
        values["steps"] = [ShadowStep(**step) for step in values.get("steps", [])]
        return cls(**values)


class ShadowReActAgent:
    """Always-reason collector that records, but never applies, residual decisions."""

    def __init__(
        self,
        client: ChatClient,
        model_name: str,
        split: str,
        max_steps: int = 12,
        shadow_threshold: float = 0.35,
        minimum_expectation_quality: float = 0.70,
        maximum_identical_action_occurrences: int = 2,
    ) -> None:
        self.client = client
        self.model_name = model_name
        self.split = split
        self.max_steps = max_steps
        self.shadow_threshold = shadow_threshold
        self.minimum_expectation_quality = minimum_expectation_quality
        self.maximum_identical_action_occurrences = maximum_identical_action_occurrences

    def run_task(self, task: dict[str, Any], env: Any) -> ShadowRun:
        started = time.perf_counter()
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        previous_observation: str | None = None
        previous_commitment: str | None = None
        steps: list[ShadowStep] = []
        final_answer = ""
        termination_reason = "max_steps"
        action_counts: dict[str, int] = {}

        for step_id in range(1, self.max_steps + 1):
            messages.append(
                {
                    "role": "user",
                    "content": build_user_prompt(
                        instruction=str(task["instruction"]),
                        testbed_path=str(env.testbed),
                        previous_observation=previous_observation,
                        previous_commitment=previous_commitment,
                    ),
                }
            )
            result = self.client.chat(messages)
            parsed, parse_error = self._parse_model_json(result.content)
            messages.append({"role": "assistant", "content": result.content})

            signals = normalize_expected_signals(parsed.get("expected_signals"))
            current_plan = str(parsed.get("current_plan") or "")
            expected_outcome = str(parsed.get("expected_outcome") or "")
            next_action = str(parsed.get("next_action_if_expected") or "").strip()
            quality = expectation_quality(signals, expected_outcome, next_action)

            if bool(parsed.get("done", False)):
                termination_reason = "model_done"
                final_answer = str(parsed.get("final_answer") or "")
                steps.append(
                    ShadowStep(
                        step_id=step_id,
                        thought=str(parsed.get("thought") or ""),
                        current_plan=current_plan,
                        action="",
                        expected_outcome=expected_outcome,
                        expected_signals=signals,
                        next_action_if_expected=next_action,
                        observation="",
                        stdout="",
                        stderr="",
                        exit_code=None,
                        expectation_quality=quality,
                        structured_residual=None,
                        observation_only_score=None,
                        residual_checks={},
                        shadow_decision="not_applicable",
                        shadow_reason="model_terminated",
                        model_latency_seconds=result.latency_seconds,
                        command_latency_seconds=0.0,
                        prompt_tokens=result.prompt_tokens,
                        completion_tokens=result.completion_tokens,
                        total_tokens=result.total_tokens,
                        action_executed=False,
                        raw_model_response=result.content,
                        parse_error=parse_error,
                    )
                )
                break

            action = str(parsed.get("action") or "").strip()
            if action and action_counts.get(action, 0) >= self.maximum_identical_action_occurrences:
                termination_reason = "collector_repeated_action_stop"
                steps.append(
                    ShadowStep(
                        step_id=step_id,
                        thought=str(parsed.get("thought") or ""),
                        current_plan=current_plan,
                        action=action,
                        expected_outcome=expected_outcome,
                        expected_signals=signals,
                        next_action_if_expected=next_action,
                        observation="",
                        stdout="",
                        stderr="",
                        exit_code=None,
                        expectation_quality=quality,
                        structured_residual=None,
                        observation_only_score=None,
                        residual_checks={},
                        shadow_decision="reason",
                        shadow_reason="collector_stopped_repeated_action",
                        model_latency_seconds=result.latency_seconds,
                        command_latency_seconds=0.0,
                        prompt_tokens=result.prompt_tokens,
                        completion_tokens=result.completion_tokens,
                        total_tokens=result.total_tokens,
                        action_executed=False,
                        raw_model_response=result.content,
                        parse_error=parse_error,
                    )
                )
                break
            if action:
                action_counts[action] = action_counts.get(action, 0) + 1
                command_result = env.run(action)
                observation = command_result.observation
                exit_code = command_result.exit_code
                stdout = command_result.stdout
                stderr = command_result.stderr
                command_latency = command_result.latency_seconds
                residual, checks = structured_residual(
                    signals,
                    expected_outcome,
                    exit_code,
                    stdout,
                    stderr,
                    observation,
                )
                observation_score = observation_only_score(exit_code, stderr, observation)
                decision, reason = shadow_decision(
                    residual,
                    quality,
                    next_action,
                    self.shadow_threshold,
                    self.minimum_expectation_quality,
                    action=action,
                )
            else:
                observation = "No action was provided by the model."
                exit_code = None
                stdout = ""
                stderr = ""
                command_latency = 0.0
                residual = None
                observation_score = None
                checks = {}
                decision = "reason"
                reason = "missing_action"

            steps.append(
                ShadowStep(
                    step_id=step_id,
                    thought=str(parsed.get("thought") or ""),
                    current_plan=current_plan,
                    action=action,
                    expected_outcome=expected_outcome,
                    expected_signals=signals,
                    next_action_if_expected=next_action,
                    observation=observation,
                    stdout=stdout,
                    stderr=stderr,
                    exit_code=exit_code,
                    expectation_quality=quality,
                    structured_residual=residual,
                    observation_only_score=observation_score,
                    residual_checks=checks,
                    shadow_decision=decision,
                    shadow_reason=reason,
                    model_latency_seconds=result.latency_seconds,
                    command_latency_seconds=command_latency,
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    total_tokens=result.total_tokens,
                    action_executed=bool(action),
                    raw_model_response=result.content,
                    parse_error=parse_error,
                )
            )
            previous_observation = observation
            previous_commitment = json.dumps(
                {
                    "current_plan": current_plan,
                    "action": action,
                    "expected_outcome": expected_outcome,
                    "expected_signals": signals,
                    "next_action_if_expected": next_action,
                },
                ensure_ascii=False,
            )

        trace_text = "\n".join(step.observation for step in steps if step.observation)
        evaluator = env.evaluate(task, trace_text=trace_text, final_answer=final_answer)
        tokens = [step.total_tokens for step in steps if step.total_tokens is not None]
        return ShadowRun(
            task_id=str(task["task_id"]),
            instruction=str(task["instruction"]),
            benchmark=str(task.get("benchmark", "")),
            split=self.split,
            model_name=self.model_name,
            success=bool(evaluator["success"]),
            evaluator=evaluator,
            final_answer=final_answer,
            steps=steps,
            model_calls=len(steps),
            total_tokens=sum(tokens) if tokens else None,
            total_model_latency_seconds=sum(step.model_latency_seconds for step in steps),
            total_command_latency_seconds=sum(step.command_latency_seconds for step in steps),
            termination_reason=termination_reason,
            max_steps_reached=termination_reason == "max_steps",
            shadow_fast_candidates=sum(step.shadow_decision == "fast_candidate" for step in steps),
            executed_steps=sum(step.action_executed for step in steps),
            parse_error_count=sum(step.parse_error is not None for step in steps),
            total_task_wall_seconds=time.perf_counter() - started,
        )

    @staticmethod
    def _parse_model_json(content: str) -> tuple[dict[str, Any], str | None]:
        try:
            value = json.loads(content)
            if isinstance(value, dict) and {"action", "done"}.issubset(value):
                return value, None
        except json.JSONDecodeError:
            pass

        decoder = json.JSONDecoder()
        for index, character in enumerate(content):
            if character != "{":
                continue
            try:
                value, _ = decoder.raw_decode(content[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and {"action", "done"}.issubset(value):
                return value, "extracted_json_from_non_json_response"

        return {
            "thought": "Could not parse model response as JSON.",
            "current_plan": "",
            "action": "",
            "expected_outcome": "",
            "expected_signals": {},
            "next_action_if_expected": "",
            "done": False,
            "final_answer": "",
        }, "json_parse_failed"
