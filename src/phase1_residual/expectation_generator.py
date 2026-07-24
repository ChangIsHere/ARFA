from __future__ import annotations


def heuristic_expectation_for_action(action: str) -> str:
    """Small helper for later trace imports that do not already contain expectations."""
    lowered = action.lower()
    if "pytest" in lowered:
        return "The tests should pass and the command should exit successfully."
    if "py_compile" in lowered or "compile" in lowered:
        return "The code should compile without syntax or compilation errors."
    if lowered.startswith("cat "):
        return "The requested file should be printed successfully."
    if "grep" in lowered or "rg " in lowered:
        return "The search should find the expected content."
    if lowered.startswith("ls "):
        return "The requested files or directories should be listed."
    return "The command should complete successfully and produce the expected terminal output."
