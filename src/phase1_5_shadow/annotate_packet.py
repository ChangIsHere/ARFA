from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path
from typing import Any

from src.common.utils import read_jsonl, write_jsonl


EXPECTATION_CHOICES = {"m": "match", "x": "mismatch", "a": "ambiguous"}
REASONING_CHOICES = {"y": "yes", "n": "no", "a": "ambiguous"}
CONFIDENCE_CHOICES = {"h": "high", "m": "medium", "l": "low"}


def _choice(prompt: str, choices: dict[str, str]) -> str:
    options = "/".join(f"{key}={value}" for key, value in choices.items())
    while True:
        value = input(f"{prompt} [{options}]: ").strip().lower()
        if value in choices:
            return choices[value]
        print("Invalid choice; use one of the displayed keys.")


def _display(index: int, total: int, row: dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print(f"Item {index}/{total} | {row['annotation_id']}")
    for label, key in (
        ("Task", "task_instruction"),
        ("Current plan", "current_plan"),
        ("Action", "action"),
        ("Expected outcome", "expected_outcome"),
        ("Expected signals", "expected_signals"),
        ("Committed continuation", "next_action_if_expected"),
        ("Actual observation", "actual_observation"),
        ("Exit code", "exit_code"),
    ):
        print(f"\n{label}:\n{row.get(key, '')}")


def annotate(packet: Path, output: Path, annotator: str, shuffle_seed: int) -> None:
    if output.exists():
        rows = read_jsonl(output)
        existing_annotators = {
            str(row.get("annotator", "")).strip()
            for row in rows
            if str(row.get("annotator", "")).strip()
        }
        if existing_annotators - {annotator}:
            raise ValueError(f"Output already contains a different annotator ID: {sorted(existing_annotators)}")
    else:
        rows = read_jsonl(packet)
        output.parent.mkdir(parents=True, exist_ok=True)
        (output.with_suffix(output.suffix + ".source.sha256")).write_text(
            hashlib.sha256(packet.read_bytes()).hexdigest() + "\n",
            encoding="utf-8",
        )
        write_jsonl(output, rows)

    pending = [row for row in rows if not str(row.get("expectation_match", "")).strip() or not str(row.get("slow_reasoning_needed", "")).strip()]
    rng = random.Random(shuffle_seed)
    rng.shuffle(pending)
    by_id = {str(row["annotation_id"]): row for row in rows}
    for index, pending_row in enumerate(pending, start=1):
        row = by_id[str(pending_row["annotation_id"])]
        _display(index, len(pending), row)
        row["expectation_match"] = _choice("Expectation", EXPECTATION_CHOICES)
        row["slow_reasoning_needed"] = _choice("Slow reasoning needed", REASONING_CHOICES)
        row["annotation_confidence"] = _choice("Confidence", CONFIDENCE_CHOICES)
        notes = input("Notes (required for ambiguous, otherwise optional): ").strip()
        if "ambiguous" in {row["expectation_match"], row["slow_reasoning_needed"]}:
            while not notes:
                notes = input("Please explain the ambiguous label: ").strip()
        row["annotation_notes"] = notes
        row["annotator"] = annotator
        write_jsonl(output, rows)
    print(f"Annotation complete: {output} ({len(rows)} items)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Independently label a blinded Phase 1.5 packet.")
    parser.add_argument("--packet", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--annotator", required=True)
    parser.add_argument("--shuffle-seed", type=int, required=True)
    args = parser.parse_args()

    packet = Path(args.packet)
    output = Path(args.output)
    if packet.resolve() == output.resolve():
        raise SystemExit("Output must differ from the pristine blank packet")
    try:
        annotate(packet, output, args.annotator, args.shuffle_seed)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
