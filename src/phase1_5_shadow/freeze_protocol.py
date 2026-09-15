from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from src.common.config import load_simple_yaml


FROZEN_FILES = (
    "config/phase1_5_shadow.yaml",
    "data/phase1_5/task_splits.json",
    "docs/phase1_5_annotation_guideline.md",
    "docs/phase1_5_protocol.md",
    "src/phase1_5_shadow/analyze_annotations.py",
    "src/phase1_5_shadow/build_annotation_packet.py",
    "src/phase1_5_shadow/json_model_client.py",
    "src/phase1_5_shadow/prompts.py",
    "src/phase1_5_shadow/residual.py",
    "src/phase1_5_shadow/run_shadow.py",
    "src/phase1_5_shadow/shadow_agent.py",
)


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_freeze_manifest(config: dict[str, Any], pilot_analysis: dict[str, Any]) -> dict[str, Any]:
    protocol = config.get("protocol", {})
    if not protocol.get("locked", False):
        raise ValueError("Set protocol.locked=true only after pilot review, then create the freeze manifest")
    if protocol.get("stage") != "formal_shadow_collection":
        raise ValueError("Set protocol.stage=formal_shadow_collection only after pilot review")
    readiness = pilot_analysis.get("readiness", {})
    if not readiness.get("ready", False):
        raise ValueError("Pilot annotations and independent agreement labels are not complete")
    minimum_kappa = float(config["annotation"]["minimum_cohen_kappa"])
    agreement = readiness.get("agreement", {})
    for target in ("slow_reasoning_needed", "expectation_match"):
        kappa = agreement.get(target, {}).get("multiclass_cohen_kappa")
        if kappa is None or float(kappa) < minimum_kappa:
            raise ValueError(f"Pilot agreement for {target} is below the frozen minimum")
    missing = [path for path in FROZEN_FILES if not Path(path).is_file()]
    if missing:
        raise ValueError(f"Cannot freeze missing protocol files: {missing}")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "phase": "phase1_5_shadow",
        "protocol_version": protocol["version"],
        "stage": "formal_shadow_collection",
        "locked": True,
        "source_commit": commit,
        "files": {path: {"sha256": _sha256(path), "bytes": Path(path).stat().st_size} for path in FROZEN_FILES},
        "pilot_annotation_readiness": readiness,
    }


def verify_freeze_manifest(config: dict[str, Any], manifest: dict[str, Any]) -> None:
    protocol = config.get("protocol", {})
    if not protocol.get("locked", False) or not manifest.get("locked", False):
        raise ValueError("Formal collection requires a locked protocol and freeze manifest")
    if manifest.get("protocol_version") != protocol.get("version"):
        raise ValueError("Protocol version does not match the freeze manifest")
    if protocol.get("stage") != "formal_shadow_collection":
        raise ValueError("Formal collection requires protocol.stage=formal_shadow_collection")
    for path, metadata in manifest.get("files", {}).items():
        if not Path(path).is_file() or _sha256(path) != metadata.get("sha256"):
            raise ValueError(f"Frozen protocol artifact changed or is missing: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the immutable Phase 1.5 formal-collection manifest.")
    parser.add_argument("--config", default="config/phase1_5_shadow.yaml")
    parser.add_argument("--pilot-analysis", default="results/phase1_5_shadow/pilot_analysis/analysis.json")
    parser.add_argument("--output", default="data/phase1_5/protocol_freeze_manifest.json")
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()

    config = load_simple_yaml(args.config)
    output = Path(args.output)
    if args.verify_existing:
        try:
            manifest = json.loads(output.read_text(encoding="utf-8"))
            verify_freeze_manifest(config, manifest)
        except FileNotFoundError as exc:
            raise SystemExit(f"Freeze manifest not found: {output}") from exc
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        print(f"Verified frozen protocol: {manifest['protocol_version']}")
        return

    analysis = json.loads(Path(args.pilot_analysis).read_text(encoding="utf-8"))
    try:
        manifest = build_freeze_manifest(config, analysis)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
