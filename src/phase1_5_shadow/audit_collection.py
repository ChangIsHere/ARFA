from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.common.utils import read_jsonl
from src.phase1_5_shadow.build_annotation_packet import build_packets
from src.phase1_5_shadow.residual import continuation_is_self_contained


def audit(trace_paths: list[Path], expected_tasks: int | None = None) -> dict[str, Any]:
    runs = [run for path in trace_paths for run in read_jsonl(path)]
    executed = [step for run in runs for step in run.get("steps", []) if step.get("action_executed", False)]
    annotations, sources = build_packets(trace_paths)
    decisions = Counter(str(step.get("shadow_decision")) for step in executed)
    reasons = Counter(str(step.get("shadow_reason")) for step in executed)
    signal_any = Counter(
        key
        for step in executed
        for key, value in (step.get("expected_signals") or {}).items()
        if str(value) == "any"
    )
    complete_commitments = [
        step
        for step in executed
        if str(step.get("expected_outcome") or "").strip()
        and str(step.get("next_action_if_expected") or "").strip()
        and isinstance(step.get("expected_signals"), dict)
    ]
    parse_errors = sum(bool(step.get("parse_error")) for run in runs for step in run.get("steps", []))
    self_contained_actions = sum(continuation_is_self_contained(str(step.get("action") or "")) for step in executed)
    next_actions = [str(step.get("next_action_if_expected") or "") for step in executed]
    executable_next = sum(continuation_is_self_contained(value) for value in next_actions if value.upper() != "<REASON>")
    non_reason_next = sum(value.upper() != "<REASON>" for value in next_actions)
    task_count = len(runs)
    completeness = expected_tasks is None or task_count == expected_tasks
    parse_rate = parse_errors / sum(len(run.get("steps", [])) for run in runs) if runs else 0.0
    commitment_rate = len(complete_commitments) / len(executed) if executed else 0.0
    payload = {
        "task_count": task_count,
        "expected_task_count": expected_tasks,
        "complete": completeness,
        "executed_steps": len(executed),
        "unique_annotation_items": len(annotations),
        "deduplicated_source_steps": len(sources) - len(annotations),
        "parse_error_count": parse_errors,
        "parse_error_rate": parse_rate,
        "complete_preexecution_commitment_rate": commitment_rate,
        "mean_expectation_quality": sum(float(step.get("expectation_quality", 0.0)) for step in executed) / len(executed)
        if executed
        else 0.0,
        "self_contained_action_rate": self_contained_actions / len(executed) if executed else 0.0,
        "self_contained_non_reason_continuation_rate": executable_next / non_reason_next if non_reason_next else 1.0,
        "shadow_fast_candidate_rate": decisions["fast_candidate"] / len(executed) if executed else 0.0,
        "decision_counts": dict(sorted(decisions.items())),
        "reason_counts": dict(sorted(reasons.items())),
        "expected_signal_any_counts": dict(sorted(signal_any.items())),
        "collection_gate": {
            "complete": completeness,
            "zero_parse_errors": parse_errors == 0,
            "commitment_rate_at_least_95_percent": commitment_rate >= 0.95,
            "at_least_20_unique_annotation_items": len(annotations) >= 20,
        },
    }
    payload["collection_gate"]["passed"] = all(payload["collection_gate"].values())
    return payload


def _write_report(path: Path, result: dict[str, Any]) -> None:
    gate = result["collection_gate"]
    lines = [
        "# Phase 1.5 Collection Audit",
        "",
        "This audit covers collection quality only. It does not estimate residual accuracy.",
        "",
        f"- Tasks: {result['task_count']}/{result['expected_task_count']}",
        f"- Executed steps: {result['executed_steps']}",
        f"- Unique annotation items: {result['unique_annotation_items']}",
        f"- Parse errors: {result['parse_error_count']}",
        f"- Complete commitment rate: {result['complete_preexecution_commitment_rate']:.3f}",
        f"- Mean expectation quality: {result['mean_expectation_quality']:.3f}",
        f"- Self-contained action rate: {result['self_contained_action_rate']:.3f}",
        f"- Shadow fast-candidate rate: {result['shadow_fast_candidate_rate']:.3f}",
        "",
        f"Collection gate: **{'PASS' if gate['passed'] else 'FAIL'}**",
        "",
        "A passing collection gate only permits annotation and larger shadow collection. Human labels are required before any residual claim.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase 1.5 shadow trace collection quality.")
    parser.add_argument("--results-root", default="results/phase1_5_shadow/pilot")
    parser.add_argument("--output-dir", default="results/phase1_5_shadow/pilot_audit")
    parser.add_argument("--expected-tasks", type=int, default=10)
    args = parser.parse_args()

    paths = list(Path(args.results_root).glob("**/traces.jsonl"))
    if not paths:
        raise SystemExit(f"No trace files found below {args.results_root}")
    result = audit(paths, expected_tasks=args.expected_tasks)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _write_report(output / "audit.md", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
