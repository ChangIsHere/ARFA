from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Any

from src.phase1_5_shadow.json_model_client import JSONModelRequestError
from src.phase1_5_shadow.residual import (
    expectation_quality,
    normalize_expected_signals,
    observation_only_score,
    structured_residual,
)
from src.phase3_arfa.fast_track import FastTrack
from src.phase3_arfa.prompts import SYSTEM_PROMPT, build_user_prompt
from src.phase3_arfa.router import ResidualRouter, RoutingDecision
from src.phase3_arfa.slow_track import SlowTrack


@dataclass(frozen=True)
class ARFAStep:
    step_id: int
    model_role: str
    model_name: str
    route_into_step: dict[str, Any]
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
    route_after_step: dict[str, Any]
    model_latency_seconds: float
    command_latency_seconds: float
    router_latency_seconds: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    action_executed: bool
    raw_model_response: str
    parse_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ARFARun:
    task_id: str
    instruction: str
    benchmark: str
    experiment_split: str
    pair_name: str
    policy: str
    fast_model: str
    slow_model: str
    success: bool
    evaluator: dict[str, Any]
    final_answer: str
    steps: list[ARFAStep]
    model_calls: int
    fast_model_calls: int
    slow_model_calls: int
    fast_model_tokens: int | None
    slow_model_tokens: int | None
    total_tokens: int | None
    total_model_latency_seconds: float
    total_command_latency_seconds: float
    total_router_latency_seconds: float
    total_task_wall_seconds: float
    termination_reason: str
    max_steps_reached: bool
    parse_error_count: int
    nonzero_exit_count: int
    repeated_action_count: int
    fast_route_count: int
    slow_route_count: int
    provenance_fingerprint: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["steps"] = [step.to_dict() for step in self.steps]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ARFARun":
        values = dict(payload)
        values["steps"] = [ARFAStep(**step) for step in payload.get("steps", [])]
        return cls(**values)


class ARFAAgent:
    """Online dual-model agent whose next model is selected by execution residual."""

    def __init__(
        self,
        fast_track: FastTrack,
        slow_track: SlowTrack,
        router: ResidualRouter,
        pair_name: str,
        experiment_split: str,
        max_steps: int = 12,
        maximum_identical_action_occurrences: int = 2,
        max_history_turns: int = 2,
    ) -> None:
        if max_history_turns < 1:
            raise ValueError("max_history_turns must be positive")
        self.fast_track = fast_track
        self.slow_track = slow_track
        self.router = router
        self.pair_name = pair_name
        self.experiment_split = experiment_split
        self.max_steps = max_steps
        self.maximum_identical_action_occurrences = maximum_identical_action_occurrences
        self.max_history_turns = max_history_turns

    def run_task(self, task: dict[str, Any], env: Any) -> ARFARun:
        run_started = time.perf_counter()
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        previous_observation: str | None = None
        previous_commitment: str | None = None
        route = self.router.initial()
        steps: list[ARFAStep] = []
        final_answer = ""
        termination_reason = "max_steps"
        action_counts: dict[str, int] = {}
        consecutive_fast_calls = 0

        for step_id in range(1, self.max_steps + 1):
            role = route.selected_role
            track = self.fast_track if role == "fast" else self.slow_track
            messages.append(
                {
                    "role": "user",
                    "content": build_user_prompt(
                        instruction=str(task["instruction"]),
                        testbed_path=str(env.testbed),
                        selected_role=role,
                        previous_observation=previous_observation,
                        previous_commitment=previous_commitment,
                    ),
                }
            )
            model_started = time.perf_counter()
            try:
                result = track.reason(self._bounded_messages(messages, self.max_history_turns))
            except JSONModelRequestError as exc:
                termination_reason = "model_request_error"
                steps.append(
                    ARFAStep(
                        step_id=step_id,
                        model_role=role,
                        model_name=track.model_name,
                        route_into_step=route.to_dict(),
                        thought="Model request failed after retries.",
                        current_plan="",
                        action="",
                        expected_outcome="",
                        expected_signals={},
                        next_action_if_expected="",
                        observation=str(exc),
                        stdout="",
                        stderr="",
                        exit_code=None,
                        expectation_quality=0.0,
                        structured_residual=None,
                        observation_only_score=None,
                        residual_checks={},
                        route_after_step=RoutingDecision(
                            self.router.policy, role, None, None, "model_request_error"
                        ).to_dict(),
                        model_latency_seconds=time.perf_counter() - model_started,
                        command_latency_seconds=0.0,
                        router_latency_seconds=0.0,
                        prompt_tokens=None,
                        completion_tokens=None,
                        total_tokens=None,
                        action_executed=False,
                        raw_model_response="",
                        parse_error="model_request_failed",
                    )
                )
                break
            parsed, parse_error = self._parse_model_json(result.content)
            messages.append({"role": "assistant", "content": result.content})

            signals = normalize_expected_signals(parsed.get("expected_signals"))
            current_plan = str(parsed.get("current_plan") or "")
            expected_outcome = str(parsed.get("expected_outcome") or "")
            next_action = str(parsed.get("next_action_if_expected") or "").strip()
            quality = expectation_quality(signals, expected_outcome, next_action)
            if role == "fast":
                consecutive_fast_calls += 1
            else:
                consecutive_fast_calls = 0

            if bool(parsed.get("done", False)):
                termination_reason = "model_done"
                final_answer = str(parsed.get("final_answer") or "")
                steps.append(
                    self._step_without_execution(
                        step_id, role, track.model_name, route, parsed, signals,
                        quality, result, parse_error, "task_complete",
                    )
                )
                break

            action = str(parsed.get("action") or "").strip()
            if action and action_counts.get(action, 0) >= self.maximum_identical_action_occurrences:
                termination_reason = "repeated_action_stop"
                steps.append(
                    self._step_without_execution(
                        step_id, role, track.model_name, route, parsed, signals,
                        quality, result, parse_error, "repeated_action_stop",
                    )
                )
                break

            if action:
                action_counts[action] = action_counts.get(action, 0) + 1
                command_result = env.run(action)
                observation = command_result.observation
                stdout = command_result.stdout
                stderr = command_result.stderr
                exit_code = command_result.exit_code
                command_latency = command_result.latency_seconds
                residual, checks = structured_residual(
                    signals, expected_outcome, exit_code, stdout, stderr, observation
                )
                obs_score = observation_only_score(exit_code, stderr, observation)
            else:
                observation = "No action was provided by the model."
                stdout = ""
                stderr = ""
                exit_code = None
                command_latency = 0.0
                residual = None
                checks = {}
                obs_score = None
                parse_error = parse_error or "missing_action"

            router_started = time.perf_counter()
            next_route = self.router.after_observation(
                residual=residual,
                expectation_quality=quality,
                exit_code=exit_code,
                stderr=stderr,
                observation=observation,
                parse_error=parse_error,
                consecutive_fast_calls=consecutive_fast_calls,
                step_id=step_id,
            )
            router_latency = time.perf_counter() - router_started
            steps.append(
                ARFAStep(
                    step_id=step_id,
                    model_role=role,
                    model_name=track.model_name,
                    route_into_step=route.to_dict(),
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
                    observation_only_score=obs_score,
                    residual_checks=checks,
                    route_after_step=next_route.to_dict(),
                    model_latency_seconds=result.latency_seconds,
                    command_latency_seconds=command_latency,
                    router_latency_seconds=router_latency,
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
                    "action": action,
                    "expected_outcome": expected_outcome,
                    "expected_signals": signals,
                    "next_action_if_expected": next_action,
                },
                ensure_ascii=False,
            )
            route = next_route

        trace_text = "\n".join(step.observation for step in steps if step.observation)
        evaluator = env.evaluate(task, trace_text=trace_text, final_answer=final_answer)
        return self._build_run(task, evaluator, final_answer, steps, termination_reason, run_started)

    @staticmethod
    def _bounded_messages(
        messages: list[dict[str, str]], max_history_turns: int
    ) -> list[dict[str, str]]:
        if not messages:
            return []
        return [messages[0], *messages[1:][-(2 * max_history_turns):]]

    def _step_without_execution(
        self,
        step_id: int,
        role: str,
        model_name: str,
        route: RoutingDecision,
        parsed: dict[str, Any],
        signals: dict[str, str],
        quality: float,
        result: Any,
        parse_error: str | None,
        reason: str,
    ) -> ARFAStep:
        return ARFAStep(
            step_id=step_id,
            model_role=role,
            model_name=model_name,
            route_into_step=route.to_dict(),
            thought=str(parsed.get("thought") or ""),
            current_plan=str(parsed.get("current_plan") or ""),
            action=str(parsed.get("action") or "").strip(),
            expected_outcome=str(parsed.get("expected_outcome") or ""),
            expected_signals=signals,
            next_action_if_expected=str(parsed.get("next_action_if_expected") or "").strip(),
            observation="",
            stdout="",
            stderr="",
            exit_code=None,
            expectation_quality=quality,
            structured_residual=None,
            observation_only_score=None,
            residual_checks={},
            route_after_step=RoutingDecision(self.router.policy, role, None, None, reason).to_dict(),
            model_latency_seconds=result.latency_seconds,
            command_latency_seconds=0.0,
            router_latency_seconds=0.0,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
            action_executed=False,
            raw_model_response=result.content,
            parse_error=parse_error,
        )

    def _build_run(
        self,
        task: dict[str, Any],
        evaluator: dict[str, Any],
        final_answer: str,
        steps: list[ARFAStep],
        termination_reason: str,
        run_started: float,
    ) -> ARFARun:
        fast_tokens = [s.total_tokens for s in steps if s.model_role == "fast" and s.total_tokens is not None]
        slow_tokens = [s.total_tokens for s in steps if s.model_role == "slow" and s.total_tokens is not None]
        actions = [s.action for s in steps if s.action_executed]
        return ARFARun(
            task_id=str(task["task_id"]),
            instruction=str(task["instruction"]),
            benchmark=str(task.get("benchmark", "")),
            experiment_split=self.experiment_split,
            pair_name=self.pair_name,
            policy=self.router.policy,
            fast_model=self.fast_track.model_name,
            slow_model=self.slow_track.model_name,
            success=bool(evaluator["success"]),
            evaluator=evaluator,
            final_answer=final_answer,
            steps=steps,
            model_calls=len(steps),
            fast_model_calls=sum(s.model_role == "fast" for s in steps),
            slow_model_calls=sum(s.model_role == "slow" for s in steps),
            fast_model_tokens=sum(fast_tokens) if fast_tokens else None,
            slow_model_tokens=sum(slow_tokens) if slow_tokens else None,
            total_tokens=sum(fast_tokens + slow_tokens) if fast_tokens or slow_tokens else None,
            total_model_latency_seconds=sum(s.model_latency_seconds for s in steps),
            total_command_latency_seconds=sum(s.command_latency_seconds for s in steps),
            total_router_latency_seconds=sum(s.router_latency_seconds for s in steps),
            total_task_wall_seconds=time.perf_counter() - run_started,
            termination_reason=termination_reason,
            max_steps_reached=termination_reason == "max_steps",
            parse_error_count=sum(s.parse_error is not None for s in steps),
            nonzero_exit_count=sum(s.exit_code not in (None, 0) for s in steps),
            repeated_action_count=len(actions) - len(set(actions)),
            fast_route_count=sum(s.model_role == "fast" for s in steps),
            slow_route_count=sum(s.model_role == "slow" for s in steps),
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
