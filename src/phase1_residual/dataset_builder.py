from __future__ import annotations

import argparse
from pathlib import Path

from src.common.config import load_simple_yaml
from src.common.schemas import TerminalStepRecord
from src.common.utils import write_jsonl


SCENARIOS = [
    ("pytest_pass", "Run the unit tests and verify the change.", "pytest -q", "All unit tests should pass and the command should exit successfully.", "12 passed in 0.41s", 0, False, "Tests passed as expected"),
    ("pytest_fail", "Run the unit tests and verify the change.", "pytest -q", "All unit tests should pass and the command should exit successfully.", "2 failed, 10 passed in 0.66s", 1, True, "Unexpected test failure"),
    ("syntax_pass", "Check whether the Python module parses.", "python -m py_compile src/app.py", "The file should compile without syntax errors.", "", 0, False, "Compilation succeeded with no output"),
    ("syntax_fail", "Check whether the Python module parses.", "python -m py_compile src/app.py", "The file should compile without syntax errors.", "SyntaxError: invalid syntax at line 18", 1, True, "Syntax error blocks continuation"),
    ("missing_file", "Inspect the required configuration file.", "cat config/settings.yaml", "The configuration file should be printed.", "cat: config/settings.yaml: No such file or directory", 1, True, "Expected file is missing"),
    ("file_found", "Inspect the required configuration file.", "cat config/settings.yaml", "The configuration file should be printed.", "debug: false\nport: 8000", 0, False, "Expected file is available"),
    ("grep_found", "Find the target function definition.", "grep -R \"def parse_args\" src", "The search should find the target function.", "src/cli.py:def parse_args():", 0, False, "Search found expected symbol"),
    ("grep_missing", "Find the target function definition.", "grep -R \"def parse_args\" src", "The search should find the target function.", "", 1, True, "Expected symbol was not found"),
    ("wrong_dir", "List project files from the repository root.", "ls src tests", "The source and test directories should be listed.", "ls: src: No such file or directory\nls: tests: No such file or directory", 1, True, "Command appears to run from the wrong directory"),
    ("ls_ok", "List project files from the repository root.", "ls src tests", "The source and test directories should be listed.", "src:\napp.py\ntests:\ntest_app.py", 0, False, "Directory listing confirms expected layout"),
    ("runtime_ok", "Run the small smoke script.", "python scripts/smoke.py", "The script should finish successfully.", "smoke ok", 0, False, "Smoke script passed"),
    ("runtime_exception", "Run the small smoke script.", "python scripts/smoke.py", "The script should finish successfully.", "Traceback (most recent call last):\nValueError: invalid literal", 1, True, "Runtime exception requires reasoning"),
    ("dependency_ok", "Check whether the dependency imports.", "python -c \"import numpy\"", "The import should succeed without output.", "", 0, False, "Dependency import succeeded"),
    ("dependency_missing", "Check whether the dependency imports.", "python -c \"import numpy\"", "The import should succeed without output.", "ModuleNotFoundError: No module named 'numpy'", 1, True, "Missing dependency blocks execution"),
    ("empty_expected", "Confirm there are no lint errors.", "ruff check src", "The linter should report no issues.", "All checks passed!", 0, False, "Lint output confirms success"),
    ("unexpected_empty", "Confirm generated report contains a summary.", "grep -n \"Summary\" report.md", "The output should include a Summary heading.", "", 1, True, "Expected output is missing"),
    ("permission_error", "Run the executable helper script.", "./scripts/build.sh", "The helper should run successfully.", "Permission denied", 126, True, "Permission error prevents continuation"),
    ("command_not_found", "Run the package manager check.", "poetry check", "The project metadata should validate successfully.", "zsh: command not found: poetry", 127, True, "Required command is unavailable"),
    ("test_xfail_expected", "Run tests for a known xfail case.", "pytest tests/test_parser.py -q", "The suite should pass with the expected xfail.", "7 passed, 1 xfailed in 0.32s", 0, False, "Expected xfail is acceptable"),
    ("warning_only", "Run the migration dry run.", "python manage.py migrate --check", "The dry run should finish without applying migrations.", "System check identified 1 warning (0 silenced)", 0, False, "Warning does not block the plan"),
]


def build_records(min_records: int) -> list[TerminalStepRecord]:
    records: list[TerminalStepRecord] = []
    i = 0
    while len(records) < min_records:
        scenario = SCENARIOS[i % len(SCENARIOS)]
        batch = i // len(SCENARIOS)
        task_id = f"phase1_{len(records) + 1:03d}"
        step_id = 1 + batch
        observation = scenario[4]
        if batch:
            observation = observation.replace("0.41s", f"0.{41 + batch}s").replace("0.66s", f"0.{66 + batch}s")
        records.append(
            TerminalStepRecord(
                task_id=task_id,
                step_id=step_id,
                instruction=scenario[1],
                action=scenario[2],
                expectation=scenario[3],
                observation=observation,
                exit_code=scenario[5],
                reasoning_needed=scenario[6],
                label_reason=scenario[7],
                scenario=scenario[0],
            )
        )
        i += 1
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/phase1_residual.yaml")
    parser.add_argument("--output")
    parser.add_argument("--min-records", type=int)
    args = parser.parse_args()

    config = load_simple_yaml(args.config)
    min_records = args.min_records or int(config["dataset"]["min_records"])
    output = Path(args.output or config["dataset"]["output_path"])

    records = [record.to_dict() for record in build_records(min_records)]
    write_jsonl(output, records)
    print(f"Wrote {len(records)} records to {output}")


if __name__ == "__main__":
    main()
