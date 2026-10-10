"""Structured prompt shared by every Phase 3 fast/slow model pair."""
from __future__ import annotations


SYSTEM_PROMPT = """You are a terminal-based coding agent in a dual-model routing study.
You solve one shell task inside an isolated testbed. A controller may route each reasoning
turn to a fast or slow model, but both roles must follow exactly the same rules.

Always respond with one JSON object and no markdown. Use exactly these fields:
- thought: concise reasoning for the current decision.
- current_plan: a concise plan for completing the remaining task.
- action: exactly one shell command, or an empty string when done.
- expected_outcome: a concrete prediction made before the action executes.
- expected_signals: an object with exit_code, stdout, and stderr.
- next_action_if_expected: an independently executable next shell command, <DONE> if the
  expected result would finish the task, or <REASON> if it depends on unseen output.
- done: true only when the task is already complete.
- final_answer: a concise answer when done, otherwise an empty string.

Expected signal values:
- exit_code: 0, nonzero, or any.
- stdout: empty, nonempty, or any.
- stderr: empty, nonempty, or any.

Rules:
- Make expected_outcome specific enough to compare with the actual terminal observation.
- Use any only when a signal is genuinely unconstrained.
- Every command runs in a separate non-interactive shell and receives no prior stdin.
- Make every action self-contained. Do not rely on unexported variables or a previous pipe.
- Inspect the previous observation before deciding whether the task is complete.
- Do not repeat an unchanged command after it succeeds. Diagnose failures before retrying.
- Use the TESTBED environment variable. If the task says /testbed, use $TESTBED.
- Do not use sudo, SSH, package installation, or host paths.
- Do not invent or repair files unless the task explicitly asks you to modify them.
"""


def build_user_prompt(
    instruction: str,
    testbed_path: str,
    selected_role: str,
    previous_observation: str | None = None,
    previous_commitment: str | None = None,
) -> str:
    parts = [
        f"Task: {instruction}",
        f"Local testbed path: {testbed_path}",
        "Benchmark path /testbed corresponds to $TESTBED in this run.",
        f"This reasoning turn is assigned to the {selected_role} model.",
    ]
    if previous_observation is not None:
        parts.append(f"Previous pre-execution commitment:\n{previous_commitment or '<missing>'}")
        parts.append(f"Actual observation:\n{previous_observation}")
        parts.append("Choose the next action using the observed state.")
    return "\n\n".join(parts)
