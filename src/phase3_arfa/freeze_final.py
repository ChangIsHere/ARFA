from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from src.common.config import load_simple_yaml


COMMON_FILES = (
    "data/phase2/intercode_nl2bash_official_200.jsonl",
    "data/phase1_5/task_splits.json",
    "evidence/phase3_design/supplementary/learned_observation_validation.json",
    "requirements.txt",
    "src/common/config.py",
    "src/common/utils.py",
    "src/phase1_5_shadow/observation_baseline.py",
    "src/phase1_5_shadow/residual.py",
    "src/phase2_baseline/intercode_docker_environment.py",
    "src/phase3_arfa/analyze_results.py",
    "src/phase3_arfa/arfa_agent.py",
    "src/phase3_arfa/audit_policy_run.py",
    "src/phase3_arfa/freeze_final.py",
    "src/phase3_arfa/prompts.py",
    "src/phase3_arfa/router.py",
    "src/phase3_arfa/run_arfa.py",
    "deployment/nrp/gpu_job_lib.sh",
    "deployment/nrp/run_phase3_final.sh",
)

PAIR_FILES = {
    "qwen": (
        "config/phase3_final_qwen.yaml",
        "deployment/nrp/25-qwen-controls-configmap.yaml",
        "deployment/nrp/63-qwen-final-service.yaml",
    ),
    "gemma": (
        "config/phase3_final_gemma.yaml",
        "deployment/nrp/24-gemma-controls-configmap.yaml",
        "deployment/nrp/64-gemma-final-service.yaml",
    ),
}


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_files(pair: str) -> tuple[str, ...]:
    return tuple(sorted((*COMMON_FILES, *PAIR_FILES[pair])))


def build_manifest(pair: str) -> dict[str, Any]:
    config_path = Path(PAIR_FILES[pair][0])
    config = load_simple_yaml(config_path)
    if not config.get("locked") or config.get("stage") != "phase3_final":
        raise ValueError("Final config must be locked and use stage=phase3_final")
    if not config.get("formal_claims_enabled") or not config.get("use_phase3_final_tasks"):
        raise ValueError("Final claims and the reserved final split must be enabled")
    files = frozen_files(pair)
    missing = [path for path in files if not Path(path).is_file()]
    if missing:
        raise ValueError(f"Missing frozen files: {missing}")
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", *files],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise ValueError("Commit all Phase 3 final files before creating the freeze manifest")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    return {
        "phase": "phase3_final",
        "pair": pair,
        "protocol_version": config["protocol_version"],
        "locked": True,
        "source_commit": commit,
        "files": {
            path: {"sha256": sha256(path), "bytes": Path(path).stat().st_size}
            for path in files
        },
    }


def verify_manifest(pair: str, manifest: dict[str, Any]) -> None:
    config = load_simple_yaml(PAIR_FILES[pair][0])
    if manifest.get("pair") != pair or not manifest.get("locked"):
        raise ValueError("Freeze manifest pair or lock state does not match")
    if manifest.get("protocol_version") != config.get("protocol_version"):
        raise ValueError("Freeze manifest protocol version does not match")
    files = manifest.get("files")
    expected = set(frozen_files(pair))
    if not isinstance(files, dict) or set(files) != expected:
        raise ValueError("Freeze manifest file set does not match the final protocol")
    for path, metadata in files.items():
        if not Path(path).is_file() or sha256(path) != metadata.get("sha256"):
            raise ValueError(f"Frozen file changed or is missing: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", choices=sorted(PAIR_FILES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        manifest = json.loads(args.output.read_text(encoding="utf-8"))
        verify_manifest(args.pair, manifest)
        print(f"Verified frozen Phase 3 final protocol: {args.pair}")
        return
    manifest = build_manifest(args.pair)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
