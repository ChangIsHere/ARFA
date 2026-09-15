from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from src.common.utils import read_jsonl, write_jsonl


PUBLIC_PACKET_FIELDS = (
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


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _file_metadata(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    return {
        "path": str(target),
        "sha256": _sha256(target),
        "bytes": target.stat().st_size,
        "records": sum(1 for line in target.read_text(encoding="utf-8").splitlines() if line.strip()),
    }


def build_evidence_bundle(
    trace_paths: list[Path],
    audit_path: Path,
    annotation_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    annotation_manifest_path = annotation_dir / "manifest.json"
    annotation_manifest = json.loads(annotation_manifest_path.read_text(encoding="utf-8"))
    public_annotation_manifest = {
        key: value for key, value in annotation_manifest.items() if key != "trace_files"
    }
    public_annotation_manifest["private_trace_file_count"] = len(annotation_manifest.get("trace_files", []))
    blind_path = annotation_dir / "blind_annotation_packet.jsonl"
    source_path = annotation_dir / "private_source_map.jsonl"
    public_packet = [
        {field: row.get(field) for field in PUBLIC_PACKET_FIELDS}
        for row in read_jsonl(blind_path)
    ]
    source_index = [
        {
            "source_id": row["source_id"],
            "annotation_id": row["annotation_id"],
            "task_id_sha256": hashlib.sha256(str(row["task_id"]).encode("utf-8")).hexdigest(),
            "step_id": row["step_id"],
        }
        for row in read_jsonl(source_path)
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_output = output_dir / "pilot_audit.json"
    annotation_output = output_dir / "annotation_manifest.json"
    packet_output = output_dir / "pilot_blind_packet_unlabeled.jsonl"
    source_output = output_dir / "pilot_source_index_redacted.jsonl"
    audit_output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    annotation_output.write_text(json.dumps(public_annotation_manifest, indent=2) + "\n", encoding="utf-8")
    write_jsonl(packet_output, public_packet)
    write_jsonl(source_output, source_index)

    protocol_files = (
        "config/phase1_5_shadow.yaml",
        "data/phase1_5/task_splits.json",
        "docs/phase1_5_annotation_guideline.md",
        "docs/phase1_5_protocol.md",
        "src/phase1_5_shadow/analyze_annotations.py",
        "src/phase1_5_shadow/prompts.py",
    )
    manifest = {
        "phase": "phase1_5_shadow",
        "scope": "10-task engineering pilot; no residual-effect labels or metrics",
        "audit": audit,
        "source_trace_files": [
            {**_file_metadata(path), "path": f"private://pilot_trace_{index:03d}"}
            for index, path in enumerate(sorted(trace_paths), start=1)
        ],
        "private_annotation_artifacts": [
            {**_file_metadata(blind_path), "path": "private://blind_annotation_packet"},
            {**_file_metadata(source_path), "path": "private://source_map"},
        ],
        "protocol_artifacts": [_file_metadata(path) for path in protocol_files],
        "published_artifacts": [
            _file_metadata(path)
            for path in (audit_output, annotation_output, packet_output, source_output)
        ],
        "redaction": {
            "published_packet_fields": list(PUBLIC_PACKET_FIELDS),
            "excluded": [
                "model identity",
                "split",
                "residual scores",
                "shadow decision",
                "raw model response",
                "reward",
                "success",
                "labels and annotator identity",
            ],
        },
    }
    (output_dir / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish a redacted, hash-verifiable Phase 1.5 pilot bundle.")
    parser.add_argument("--results-root", default="results/phase1_5_shadow/pilot")
    parser.add_argument("--audit", default="results/phase1_5_shadow/pilot_audit/audit.json")
    parser.add_argument("--annotation-dir", default="results/phase1_5_shadow/pilot_annotation")
    parser.add_argument("--output-dir", default="evidence/phase1_5_pilot")
    args = parser.parse_args()

    trace_paths = list(Path(args.results_root).glob("**/traces.jsonl"))
    if not trace_paths:
        raise SystemExit(f"No pilot traces found below {args.results_root}")
    manifest = build_evidence_bundle(
        trace_paths,
        Path(args.audit),
        Path(args.annotation_dir),
        Path(args.output_dir),
    )
    print(json.dumps({"output_dir": args.output_dir, "audit": manifest["audit"]}, indent=2))


if __name__ == "__main__":
    main()
