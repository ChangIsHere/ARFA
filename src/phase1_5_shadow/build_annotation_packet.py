from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

from src.common.utils import read_jsonl, write_jsonl


ALLOWED_PACKET_FIELDS = {
    "annotation_id",
    "task_instruction",
    "current_plan",
    "action",
    "expected_outcome",
    "expected_signals",
    "next_action_if_expected",
    "actual_observation",
    "exit_code",
    "expectation_match",
    "slow_reasoning_needed",
    "annotation_confidence",
    "annotation_notes",
    "annotator",
}
FORBIDDEN_PACKET_FIELDS = {
    "model_name",
    "split",
    "success",
    "reward",
    "evaluator",
    "structured_residual",
    "observation_only_score",
    "shadow_decision",
    "future_action",
    "final_answer",
    "raw_model_response",
}


def _content_key(run: dict[str, Any], step: dict[str, Any]) -> str:
    content = {
        "task_instruction": run["instruction"],
        "current_plan": step.get("current_plan", ""),
        "action": step.get("action", ""),
        "expected_outcome": step.get("expected_outcome", ""),
        "expected_signals": step.get("expected_signals", {}),
        "next_action_if_expected": step.get("next_action_if_expected", ""),
        "actual_observation": step.get("observation", ""),
        "exit_code": step.get("exit_code"),
    }
    return json.dumps(content, sort_keys=True, ensure_ascii=False)


def build_packets(
    trace_paths: Iterable[Path],
    task_metadata: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    annotations_by_content: dict[str, dict[str, Any]] = {}
    sources: list[dict[str, Any]] = []

    for trace_path in sorted(trace_paths):
        for run in read_jsonl(trace_path):
            for step in run.get("steps", []):
                if not step.get("action_executed", True) or not str(step.get("action") or "").strip():
                    continue
                key = _content_key(run, step)
                annotation_id = "p15-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
                if key not in annotations_by_content:
                    annotations_by_content[key] = {
                        "annotation_id": annotation_id,
                        "task_instruction": run["instruction"],
                        "current_plan": step.get("current_plan", ""),
                        "action": step.get("action", ""),
                        "expected_outcome": step.get("expected_outcome", ""),
                        "expected_signals": step.get("expected_signals", {}),
                        "next_action_if_expected": step.get("next_action_if_expected", ""),
                        "actual_observation": step.get("observation", ""),
                        "exit_code": step.get("exit_code"),
                        "expectation_match": "",
                        "slow_reasoning_needed": "",
                        "annotation_confidence": "",
                        "annotation_notes": "",
                        "annotator": "",
                    }
                source_id = hashlib.sha256(
                    f"{run['model_name']}|{run['task_id']}|{step['step_id']}".encode("utf-8")
                ).hexdigest()[:20]
                metadata = (task_metadata or {}).get(str(run["task_id"]), {})
                sources.append(
                    {
                        "source_id": source_id,
                        "annotation_id": annotation_id,
                        "trace_path": str(trace_path),
                        "model_name": run["model_name"],
                        "task_id": run["task_id"],
                        "task_category": metadata.get("task_category", "unknown"),
                        "filesystem_version": metadata.get("filesystem_version", "unknown"),
                        "step_id": step["step_id"],
                        "split": run["split"],
                        "expectation_quality": step.get("expectation_quality"),
                        "structured_residual": step.get("structured_residual"),
                        "observation_only_score": step.get("observation_only_score"),
                        "shadow_decision": step.get("shadow_decision"),
                        "shadow_reason": step.get("shadow_reason"),
                        "parse_error": step.get("parse_error"),
                    }
                )

    annotations = sorted(annotations_by_content.values(), key=lambda row: row["annotation_id"])
    for row in annotations:
        if set(row) != ALLOWED_PACKET_FIELDS or set(row) & FORBIDDEN_PACKET_FIELDS:
            raise RuntimeError("Annotation packet privacy contract was violated")
    return annotations, sorted(sources, key=lambda row: row["source_id"])


def secondary_annotation_sample(
    annotations: list[dict[str, Any]], fraction: float = 0.25, seed: int = 1507
) -> list[dict[str, Any]]:
    if not annotations:
        return []
    count = max(1, round(len(annotations) * fraction))
    rng = random.Random(seed)
    selected = rng.sample(annotations, min(count, len(annotations)))
    return sorted(({**row, "annotator": ""} for row in selected), key=lambda row: row["annotation_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a blinded, content-deduplicated Phase 1.5 annotation packet.")
    parser.add_argument("--results-root", default="results/phase1_5_shadow/full")
    parser.add_argument("--output-dir", default="data/phase1_5/annotation")
    parser.add_argument("--tasks", default="data/phase2/intercode_nl2bash_official_200.jsonl")
    args = parser.parse_args()

    trace_paths = list(Path(args.results_root).glob("**/traces.jsonl"))
    if not trace_paths:
        raise SystemExit(f"No traces found below {args.results_root}")
    task_metadata = {str(row["task_id"]): row for row in read_jsonl(args.tasks)}
    annotations, sources = build_packets(trace_paths, task_metadata)
    secondary = secondary_annotation_sample(annotations)
    output = Path(args.output_dir)
    write_jsonl(output / "blind_annotation_packet.jsonl", annotations)
    write_jsonl(output / "blind_annotation_packet_secondary_25pct.jsonl", secondary)
    write_jsonl(output / "private_source_map.jsonl", sources)
    manifest = {
        "phase": "phase1_5_shadow",
        "trace_files": [str(path) for path in sorted(trace_paths)],
        "source_steps": len(sources),
        "unique_annotation_items": len(annotations),
        "secondary_annotation_items": len(secondary),
        "secondary_annotation_fraction": len(secondary) / len(annotations) if annotations else 0.0,
        "deduplicated_steps": len(sources) - len(annotations),
        "packet_fields": sorted(ALLOWED_PACKET_FIELDS),
        "forbidden_fields": sorted(FORBIDDEN_PACKET_FIELDS),
        "labels_complete": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
