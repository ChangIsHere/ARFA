from __future__ import annotations

import re
from typing import Any


ERROR_PATTERN = re.compile(
    r"\b(error|failed|failure|exception|traceback|syntaxerror|permission denied|command not found|no such file)\b",
    flags=re.IGNORECASE,
)


def normalize_expected_signals(value: Any) -> dict[str, str]:
    raw = value if isinstance(value, dict) else {}
    exit_code = str(raw.get("exit_code", "any")).strip().lower()
    stdout = str(raw.get("stdout", "any")).strip().lower()
    stderr = str(raw.get("stderr", "any")).strip().lower()
    if exit_code not in {"0", "nonzero", "any"}:
        exit_code = "any"
    if stdout not in {"empty", "nonempty", "any"}:
        stdout = "any"
    if stderr not in {"empty", "nonempty", "any"}:
        stderr = "any"
    return {"exit_code": exit_code, "stdout": stdout, "stderr": stderr}


def expectation_quality(signals: dict[str, str], expected_outcome: str, next_action: str) -> float:
    specified = sum(signals[key] != "any" for key in ("exit_code", "stdout", "stderr"))
    components = [specified / 3, min(len(expected_outcome.strip()) / 60, 1.0), float(bool(next_action.strip()))]
    return sum(components) / len(components)


def observation_only_score(exit_code: int, stderr: str, observation: str) -> float:
    score = 0.0
    if exit_code != 0:
        score += 0.55
    if stderr.strip():
        score += 0.20
    if ERROR_PATTERN.search(observation):
        score += 0.25
    return min(score, 1.0)


def continuation_is_self_contained(next_action: str) -> bool:
    continuation = next_action.strip()
    if continuation.upper() in {"<DONE>", "<REASON>"}:
        return True
    first = continuation.split(maxsplit=1)[0] if continuation else ""
    stdin_consumers = {"awk", "xargs", "sort", "uniq", "wc", "head", "tail", "cut", "tr"}
    has_explicit_input = "|" in continuation or bool(
        re.search(r"\b(find|cat|printf|echo)\b", continuation)
    )
    return not (first in stdin_consumers and not has_explicit_input)


def structured_residual(
    signals: dict[str, str],
    expected_outcome: str,
    exit_code: int,
    stdout: str,
    stderr: str,
    observation: str,
) -> tuple[float, dict[str, bool]]:
    checks: dict[str, bool] = {}
    weights: dict[str, float] = {}

    if signals["exit_code"] != "any":
        checks["exit_code_mismatch"] = (signals["exit_code"] == "0" and exit_code != 0) or (
            signals["exit_code"] == "nonzero" and exit_code == 0
        )
        weights["exit_code_mismatch"] = 0.45
    if signals["stdout"] != "any":
        checks["stdout_presence_mismatch"] = (signals["stdout"] == "empty") != (not stdout.strip())
        weights["stdout_presence_mismatch"] = 0.20
    if signals["stderr"] != "any":
        checks["stderr_presence_mismatch"] = (signals["stderr"] == "empty") != (not stderr.strip())
        weights["stderr_presence_mismatch"] = 0.20

    expects_success = any(token in expected_outcome.lower() for token in ("success", "without error", "exit code 0", "complete"))
    checks["textual_error_contradiction"] = expects_success and bool(ERROR_PATTERN.search(observation))
    weights["textual_error_contradiction"] = 0.15

    expects_visible_output = any(
        token in expected_outcome.lower()
        for token in ("output", "print", "display", "list", "show", "return", "calculate the mean", "result")
    )
    checks["textual_output_missing"] = expects_visible_output and not stdout.strip()
    weights["textual_output_missing"] = 0.20

    total_weight = sum(weights.values())
    score = sum(weights[key] for key, mismatch in checks.items() if mismatch) / total_weight if total_weight else 0.0
    return score, checks


def shadow_decision(
    residual: float,
    quality: float,
    next_action: str,
    threshold: float,
    min_quality: float,
    action: str = "",
) -> tuple[str, str]:
    if action and not continuation_is_self_contained(action):
        return "reason", "executed_action_depended_on_missing_stdin"
    continuation = next_action.strip()
    if not continuation:
        return "reason", "missing_precommitted_continuation"
    if continuation.upper() == "<REASON>":
        return "reason", "model_declared_observation_dependent_continuation"
    if not continuation_is_self_contained(continuation):
        return "reason", "continuation_depends_on_missing_stdin"
    if quality < min_quality:
        return "reason", "insufficient_expectation_quality"
    if residual >= threshold:
        return "reason", "structured_residual_above_threshold"
    return "fast_candidate", "shadow_only_expectation_matched"
