from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from src.common.config import load_simple_yaml
from src.phase1_5_shadow.audit_formal_collection import audit_formal_collection
from src.phase1_5_shadow.formal_matrix import FORMAL_SPLITS, formal_models
from src.phase1_5_shadow.freeze_protocol import verify_freeze_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and audit the frozen 3x3 Phase 1.5 shadow matrix.")
    parser.add_argument("--config", default="config/phase1_5_shadow.yaml")
    parser.add_argument("--results-root", default="results/phase1_5_shadow/full")
    parser.add_argument("--freeze-manifest", default="data/phase1_5/protocol_freeze_manifest.json")
    parser.add_argument("--annotation-dir", default="data/phase1_5/annotation")
    args = parser.parse_args()

    config = load_simple_yaml(args.config)
    try:
        freeze_manifest = json.loads(Path(args.freeze_manifest).read_text(encoding="utf-8"))
        verify_freeze_manifest(config, freeze_manifest)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(f"Formal matrix remains locked: {exc}") from exc
    result_root = Path(args.results_root)

    for model in formal_models(config):
        for split in FORMAL_SPLITS:
            print(f"Running formal cell model={model.name} split={split}", flush=True)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.phase1_5_shadow.run_shadow",
                    "--config",
                    args.config,
                    "--freeze-manifest",
                    args.freeze_manifest,
                    "--split",
                    split,
                    "--model-name",
                    model.name,
                    "--output-dir",
                    str(result_root / model.slug / split),
                    "--resume",
                ],
                check=True,
            )

    audit = audit_formal_collection(config, args.config, result_root, args.freeze_manifest)
    audit_path = result_root / "formal_collection_audit.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    if not audit["passed"]:
        raise SystemExit("Formal matrix audit failed; annotation packet generation remains locked")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.phase1_5_shadow.build_annotation_packet",
            "--config",
            args.config,
            "--freeze-manifest",
            args.freeze_manifest,
            "--results-root",
            str(result_root),
            "--output-dir",
            args.annotation_dir,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
