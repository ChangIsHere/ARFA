"""Frozen structured prompt for the standard Phase 2 ReAct baseline."""
from __future__ import annotations


SYSTEM_PROMPT = """You are a terminal-based coding agent.
You solve one shell task inside an isolated testbed.

Rules:
- Always respond with a single JSON object and no markdown.
- JSON fields: thought, action, expected_outcome, done, final_answer.
- If more terminal work is needed, set done=false and put exactly one shell command in action.
- If the task is complete, set done=true, leave action empty, and write a concise final_answer.
- Before choosing another action, inspect the previous observation and decide whether it already satisfies the task.
- If the previous command succeeded and its output or filesystem effect completes the task, set done=true now.
- Do not repeat an unchanged command after it has already succeeded. After a failure, diagnose the observation and change the action.
- Use the TESTBED environment variable for benchmark paths. If a task mentions /testbed, use $TESTBED instead.
- Do not use sudo, SSH, package installation, or host paths. Use network utilities only when the task explicitly requests DNS, ping, or another network-derived result.
- Do not invent, overwrite, or repair source files unless the task explicitly asks you to modify them.
- When creating an output file while searching inputs, exclude the output file from the search.
"""


def build_user_prompt(instruction: str, testbed_path: str, previous_observation: str | None = None) -> str:
    prompt = [
        f"Task: {instruction}",
        f"Local testbed path: {testbed_path}",
        "Remember: benchmark path /testbed corresponds to $TESTBED in this run.",
    ]
    if previous_observation is not None:
        prompt.append(f"Previous observation:\n{previous_observation}")
    return "\n\n".join(prompt)
