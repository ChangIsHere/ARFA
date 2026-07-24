from __future__ import annotations


FAILURE_PATTERNS = [
    "failed",
    "failure",
    "error",
    "exception",
    "traceback",
    "syntaxerror",
    "modulenotfounderror",
    "no such file",
    "not found",
    "permission denied",
    "command not found",
]


def heuristic_reasoning_label(observation: str, exit_code: int) -> tuple[bool, str]:
    text = observation.lower()
    if exit_code != 0:
        return True, "Non-zero exit code"
    for pattern in FAILURE_PATTERNS:
        if pattern in text:
            return True, f"Failure pattern detected: {pattern}"
    return False, "Output appears to confirm expected progress"
