from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from src.common.utils import read_jsonl, write_jsonl


ALLOWED_INPUT_FIELDS = (
    "annotation_id",
    "task_instruction",
    "current_plan",
    "action",
    "expected_outcome",
    "expected_signals",
    "next_action_if_expected",
    "actual_observation",
    "exit_code",
)
EXPECTATION_LABELS = {"match", "mismatch", "ambiguous"}
REASONING_LABELS = {"yes", "no", "ambiguous"}
CONFIDENCE_LABELS = {"high", "medium", "low"}
EXPECTATION_CODES = {"m": "match", "x": "mismatch", "a": "ambiguous"}
REASONING_CODES = {"y": "yes", "n": "no", "a": "ambiguous"}
CONFIDENCE_CODES = {"h": "high", "m": "medium", "l": "low"}
EVIDENCE_CODES = {
    "observation_supports_expectation",
    "expected_signal_conflict",
    "semantic_output_conflict",
    "execution_error",
    "committed_continuation_invalid",
    "explicit_reason_required",
    "task_complete",
    "insufficient_evidence",
    "other",
}
EVIDENCE_NOTES = {
    "observation_supports_expectation": "Observation supports the stated expectation.",
    "expected_signal_conflict": "Observed signals conflict with the expected signals.",
    "semantic_output_conflict": "Observation content conflicts with the expected outcome.",
    "execution_error": "The observation exposes an execution error.",
    "committed_continuation_invalid": "The observation invalidates the committed continuation.",
    "explicit_reason_required": "The commitment explicitly requests renewed reasoning.",
    "task_complete": "The blind evidence supports task completion.",
    "insufficient_evidence": "The blind evidence is insufficient for a definitive label.",
    "other": "The blind evidence supports this label through another listed condition.",
}
EVIDENCE_CODES_SHORT = {
    "s": "observation_supports_expectation",
    "g": "expected_signal_conflict",
    "c": "semantic_output_conflict",
    "e": "execution_error",
    "i": "committed_continuation_invalid",
    "r": "explicit_reason_required",
    "t": "task_complete",
    "u": "insufficient_evidence",
    "o": "other",
}


SYSTEM_PROMPT = """You are a blind research annotator for terminal-agent traces.
Use only the fields supplied for each item. Do not infer or request a residual score,
model identity, split, reward, gold command, future action, or final task outcome.

Expectation match:
- match: the observation is compatible with the concrete expected outcome and signals.
- mismatch: the observation contradicts a material part of the expectation.
- ambiguous: the expectation or observation is insufficient to decide.

Slow reasoning needed:
- no: the committed self-contained next_action_if_expected (or <DONE>) remains safe
  and appropriate after the observation.
- yes: the observation invalidates the commitment, exposes an error, changes a
  required assumption, or makes the next action unsafe.
- ambiguous: the supplied fields do not contain enough evidence to judge safely.

<REASON> means slow reasoning is needed unless the task is already clearly complete
and <DONE> should have been used. A nonzero exit can be expected, and a zero exit can
still hide an error. Judge semantic content, not exit code alone. Select the single
evidence_code that best identifies the decisive visible evidence. Do not provide
chain-of-thought."""


def _schema(expected_ids: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "annotations": {
                "type": "array",
                "minItems": len(expected_ids),
                "maxItems": len(expected_ids),
                "items": {
                    "type": "object",
                    "properties": {
                        "i": {"type": "string", "enum": expected_ids},
                        "e": {"type": "string", "enum": sorted(EXPECTATION_CODES)},
                        "r": {"type": "string", "enum": sorted(REASONING_CODES)},
                        "c": {"type": "string", "enum": sorted(CONFIDENCE_CODES)},
                        "v": {"type": "string", "enum": sorted(EVIDENCE_CODES_SHORT)},
                    },
                    "required": [
                        "i",
                        "e",
                        "r",
                        "c",
                        "v",
                    ],
                },
            }
        },
        "required": ["annotations"],
    }


def _blind_item(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row.get(key) for key in ALLOWED_INPUT_FIELDS}


def _input_hash(items: list[dict[str, Any]]) -> str:
    payload = json.dumps(items, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _make_batches(rows: list[dict[str, Any]], batch_size: int, max_chars: int) -> list[list[dict[str, Any]]]:
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_chars = 0
    for row in rows:
        item_chars = len(json.dumps(_blind_item(row), ensure_ascii=False))
        if current and (len(current) >= batch_size or current_chars + item_chars > max_chars):
            batches.append(current)
            current = []
            current_chars = 0
        current.append(row)
        current_chars += item_chars
    if current:
        batches.append(current)
    return batches


def _request_batch(
    *,
    base_url: str,
    model: str,
    rows: list[dict[str, Any]],
    timeout: int,
    attempts: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    items = [_blind_item(row) for row in rows]
    expected_ids = [str(row["annotation_id"]) for row in rows]
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "Label every item in this JSON array independently:\n"
                + json.dumps(items, ensure_ascii=False),
            },
        ],
        "stream": False,
        "format": _schema(expected_ids),
        "options": {"temperature": 0, "num_predict": max(700, 180 * len(rows))},
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = json.loads(response.read().decode("utf-8"))
            parsed = json.loads(str(raw["message"]["content"]))
            compact_annotations = list(parsed["annotations"])
            annotations = [
                {
                    "annotation_id": item["i"],
                    "expectation_match": EXPECTATION_CODES[item["e"]],
                    "slow_reasoning_needed": REASONING_CODES[item["r"]],
                    "annotation_confidence": CONFIDENCE_CODES[item["c"]],
                    "evidence_code": EVIDENCE_CODES_SHORT[item["v"]],
                }
                for item in compact_annotations
            ]
            by_id = {str(item["annotation_id"]): item for item in annotations}
            if set(by_id) != set(expected_ids) or len(annotations) != len(expected_ids):
                raise ValueError("Judge returned missing, duplicate, or unexpected annotation IDs")
            for annotation_id in expected_ids:
                item = by_id[annotation_id]
                if item["expectation_match"] not in EXPECTATION_LABELS:
                    raise ValueError(f"Invalid expectation label for {annotation_id}")
                if item["slow_reasoning_needed"] not in REASONING_LABELS:
                    raise ValueError(f"Invalid reasoning label for {annotation_id}")
                if item["annotation_confidence"] not in CONFIDENCE_LABELS:
                    raise ValueError(f"Invalid confidence label for {annotation_id}")
                if item["evidence_code"] not in EVIDENCE_CODES:
                    raise ValueError(f"Invalid evidence code for {annotation_id}")
            metadata = {
                "prompt_eval_count": raw.get("prompt_eval_count"),
                "eval_count": raw.get("eval_count"),
                "total_duration": raw.get("total_duration"),
            }
            return [by_id[annotation_id] for annotation_id in expected_ids], metadata
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(2 ** (attempt - 1))
    raise RuntimeError(f"Judge failed after {attempts} attempts") from last_error


def annotate(
    *,
    packet: Path,
    output: Path,
    audit: Path,
    annotator: str,
    model: str,
    base_url: str,
    batch_size: int,
    max_chars: int,
    timeout: int,
    attempts: int,
) -> None:
    packet_rows = read_jsonl(packet)
    if output.exists():
        rows = read_jsonl(output)
        if [row["annotation_id"] for row in rows] != [row["annotation_id"] for row in packet_rows]:
            raise SystemExit("Existing output does not match the frozen packet order")
    else:
        rows = packet_rows
        output.parent.mkdir(parents=True, exist_ok=True)
        output.with_suffix(output.suffix + ".source.sha256").write_text(
            hashlib.sha256(packet.read_bytes()).hexdigest() + "\n", encoding="utf-8"
        )
        write_jsonl(output, rows)

    existing_annotators = {
        str(row.get("annotator", "")).strip()
        for row in rows
        if str(row.get("annotator", "")).strip()
    }
    if existing_annotators - {annotator}:
        raise SystemExit(f"Output contains another annotator: {sorted(existing_annotators)}")

    pending = [
        row
        for row in rows
        if not str(row.get("expectation_match", "")).strip()
        or not str(row.get("slow_reasoning_needed", "")).strip()
    ]
    batches = _make_batches(pending, batch_size=batch_size, max_chars=max_chars)
    by_id = {str(row["annotation_id"]): row for row in rows}
    audit.parent.mkdir(parents=True, exist_ok=True)
    completed_before = len(rows) - len(pending)
    print(f"AI blind annotation: {completed_before}/{len(rows)} complete; {len(batches)} batches pending", flush=True)

    for batch_index, batch in enumerate(batches, start=1):
        blind_items = [_blind_item(row) for row in batch]
        started = time.perf_counter()
        annotations, metadata = _request_batch(
            base_url=base_url,
            model=model,
            rows=batch,
            timeout=timeout,
            attempts=attempts,
        )
        for annotation in annotations:
            row = by_id[str(annotation["annotation_id"])]
            evidence_code = str(annotation.pop("evidence_code"))
            row.update(annotation)
            row["annotation_notes"] = EVIDENCE_NOTES[evidence_code]
            row["annotation_evidence_code"] = evidence_code
            row["annotator"] = annotator
            row["annotation_method"] = "ai_blind_judge"
            row["annotation_model"] = model
        write_jsonl(output, rows)
        with audit.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "batch": batch_index,
                        "annotation_ids": [row["annotation_id"] for row in batch],
                        "blind_input_sha256": _input_hash(blind_items),
                        "model": model,
                        "annotator": annotator,
                        "temperature": 0,
                        "elapsed_seconds": time.perf_counter() - started,
                        **metadata,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        completed = completed_before + sum(len(item) for item in batches[:batch_index])
        print(f"[{batch_index}/{len(batches)}] {completed}/{len(rows)} complete", flush=True)

    print(f"Annotation complete: {output} ({len(rows)} items)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Blind AI annotation with auditable local judges.")
    parser.add_argument("--packet", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--annotator", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--max-chars", type=int, default=18000)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--attempts", type=int, default=3)
    args = parser.parse_args()
    annotate(
        packet=Path(args.packet),
        output=Path(args.output),
        audit=Path(args.audit),
        annotator=args.annotator,
        model=args.model,
        base_url=args.base_url,
        batch_size=args.batch_size,
        max_chars=args.max_chars,
        timeout=args.timeout,
        attempts=args.attempts,
    )


if __name__ == "__main__":
    main()
