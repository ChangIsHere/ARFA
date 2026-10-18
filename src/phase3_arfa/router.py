"""Residual-guided model selection for the online Phase 3 controller."""
from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass

from src.phase1_5_shadow.residual import observation_only_score
from src.phase3_arfa.learned_observation import LearnedObservationScorer


@dataclass(frozen=True)
class RoutingDecision:
    policy: str
    selected_role: str
    score: float | None
    threshold: float | None
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


class ResidualRouter:
    """Choose the model for the next reasoning turn.

    The residual policy routes low expectation-observation residuals to the fast
    model and high residuals to the slow model. Missing or low-quality
    expectations fail open to the slow model.
    """

    POLICIES = {
        "residual", "observation_only", "always_slow", "always_fast", "fixed_switch",
        "expectation_gate_only", "random_budget", "fixed_budget", "learned_observation_only",
    }

    def __init__(
        self,
        policy: str,
        residual_threshold: float,
        observation_threshold: float,
        minimum_expectation_quality: float,
        max_consecutive_fast_calls: int,
        fast_call_budget: float = 0.0,
        random_seed: int = 1507,
        learned_observation_scorer: LearnedObservationScorer | None = None,
        learned_observation_threshold: float = 0.5,
    ) -> None:
        if policy not in self.POLICIES:
            raise ValueError(f"Unknown Phase 3 routing policy: {policy}")
        if not 0.0 <= residual_threshold <= 1.0:
            raise ValueError("Residual threshold must be in [0, 1]")
        if max_consecutive_fast_calls < 1:
            raise ValueError("max_consecutive_fast_calls must be positive")
        if not 0.0 <= fast_call_budget <= 0.5:
            raise ValueError("fast_call_budget must be in [0, 0.5] with the one-fast-call limit")
        self.policy = policy
        self.residual_threshold = residual_threshold
        self.observation_threshold = observation_threshold
        self.minimum_expectation_quality = minimum_expectation_quality
        self.max_consecutive_fast_calls = max_consecutive_fast_calls
        self.fast_call_budget = fast_call_budget
        self.random_seed = random_seed
        self.learned_observation_scorer = learned_observation_scorer
        self.learned_observation_threshold = learned_observation_threshold
        if policy == "learned_observation_only" and learned_observation_scorer is None:
            raise ValueError("learned_observation_only requires a fitted scorer")

    def initial(self) -> RoutingDecision:
        role = "fast" if self.policy == "always_fast" else "slow"
        return RoutingDecision(self.policy, role, None, None, "initial_model")

    def after_observation(
        self,
        *,
        residual: float | None,
        expectation_quality: float,
        exit_code: int | None,
        stderr: str,
        observation: str,
        parse_error: str | None,
        consecutive_fast_calls: int,
        step_id: int,
        task_id: str = "",
    ) -> RoutingDecision:
        if self.policy == "always_slow":
            return RoutingDecision(self.policy, "slow", None, None, "always_slow_policy")
        if self.policy == "always_fast":
            return RoutingDecision(self.policy, "fast", None, None, "always_fast_policy")
        if self.policy == "fixed_switch":
            role = "fast" if step_id % 2 == 1 else "slow"
            return RoutingDecision(self.policy, role, None, None, "alternating_fixed_switch")

        if parse_error:
            return RoutingDecision(self.policy, "slow", residual, self.residual_threshold, "model_response_parse_error")
        if exit_code is None:
            return RoutingDecision(self.policy, "slow", residual, self.residual_threshold, "missing_execution_signal")
        if self.policy == "observation_only":
            score = observation_only_score(exit_code, stderr, observation)
            if score >= self.observation_threshold:
                return RoutingDecision(self.policy, "slow", score, self.observation_threshold, "observation_score_high")
            if consecutive_fast_calls >= self.max_consecutive_fast_calls:
                return RoutingDecision(self.policy, "slow", score, self.observation_threshold, "fast_call_limit")
            return RoutingDecision(self.policy, "fast", score, self.observation_threshold, "observation_score_low")

        if self.policy == "learned_observation_only":
            score = self.learned_observation_scorer.score(observation, exit_code)
            if score >= self.learned_observation_threshold:
                return RoutingDecision(self.policy, "slow", score, self.learned_observation_threshold, "learned_observation_score_high")
            if consecutive_fast_calls >= self.max_consecutive_fast_calls:
                return RoutingDecision(self.policy, "slow", score, self.learned_observation_threshold, "fast_call_limit")
            return RoutingDecision(self.policy, "fast", score, self.learned_observation_threshold, "learned_observation_score_low")

        if self.policy == "expectation_gate_only":
            if expectation_quality < self.minimum_expectation_quality:
                return RoutingDecision(self.policy, "slow", expectation_quality, self.minimum_expectation_quality, "expectation_quality_low")
            if consecutive_fast_calls >= self.max_consecutive_fast_calls:
                return RoutingDecision(self.policy, "slow", expectation_quality, self.minimum_expectation_quality, "fast_call_limit")
            return RoutingDecision(self.policy, "fast", expectation_quality, self.minimum_expectation_quality, "expectation_quality_high")

        if self.policy in {"random_budget", "fixed_budget"}:
            if consecutive_fast_calls >= self.max_consecutive_fast_calls:
                return RoutingDecision(self.policy, "slow", None, self.fast_call_budget, "fast_call_limit")
            if self.policy == "random_budget":
                key = f"{self.random_seed}:{task_id}:{step_id}".encode("utf-8")
                draw = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2**64
                use_fast = draw < min(1.0, 2.0 * self.fast_call_budget)
            else:
                offset = 1.0 - self.fast_call_budget
                use_fast = math.floor(step_id * self.fast_call_budget + offset) > math.floor(
                    (step_id - 1) * self.fast_call_budget + offset
                )
            return RoutingDecision(
                self.policy, "fast" if use_fast else "slow", None, self.fast_call_budget,
                "budget_fast" if use_fast else "budget_slow",
            )

        if residual is None:
            return RoutingDecision(self.policy, "slow", None, self.residual_threshold, "missing_residual")

        if expectation_quality < self.minimum_expectation_quality:
            return RoutingDecision(
                self.policy,
                "slow",
                residual,
                self.residual_threshold,
                "insufficient_expectation_quality",
            )
        if residual >= self.residual_threshold:
            return RoutingDecision(self.policy, "slow", residual, self.residual_threshold, "residual_high")
        if consecutive_fast_calls >= self.max_consecutive_fast_calls:
            return RoutingDecision(self.policy, "slow", residual, self.residual_threshold, "fast_call_limit")
        return RoutingDecision(self.policy, "fast", residual, self.residual_threshold, "residual_low")
