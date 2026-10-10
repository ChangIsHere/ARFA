"""Residual-guided model selection for the online Phase 3 controller."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from src.phase1_5_shadow.residual import observation_only_score


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

    POLICIES = {"residual", "observation_only", "always_slow", "always_fast", "fixed_switch"}

    def __init__(
        self,
        policy: str,
        residual_threshold: float,
        observation_threshold: float,
        minimum_expectation_quality: float,
        max_consecutive_fast_calls: int,
    ) -> None:
        if policy not in self.POLICIES:
            raise ValueError(f"Unknown Phase 3 routing policy: {policy}")
        if not 0.0 <= residual_threshold <= 1.0:
            raise ValueError("Residual threshold must be in [0, 1]")
        if max_consecutive_fast_calls < 1:
            raise ValueError("max_consecutive_fast_calls must be positive")
        self.policy = policy
        self.residual_threshold = residual_threshold
        self.observation_threshold = observation_threshold
        self.minimum_expectation_quality = minimum_expectation_quality
        self.max_consecutive_fast_calls = max_consecutive_fast_calls

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
        if residual is None or exit_code is None:
            return RoutingDecision(self.policy, "slow", residual, self.residual_threshold, "missing_execution_signal")
        if self.policy == "observation_only":
            score = observation_only_score(exit_code, stderr, observation)
            if score >= self.observation_threshold:
                return RoutingDecision(self.policy, "slow", score, self.observation_threshold, "observation_score_high")
            if consecutive_fast_calls >= self.max_consecutive_fast_calls:
                return RoutingDecision(self.policy, "slow", score, self.observation_threshold, "fast_call_limit")
            return RoutingDecision(self.policy, "fast", score, self.observation_threshold, "observation_score_low")

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
