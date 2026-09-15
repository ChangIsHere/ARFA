from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.common.config import load_simple_yaml
from src.common.utils import read_jsonl
from src.phase1_5_shadow.formal_matrix import FORMAL_SPLITS, expected_trace_paths, formal_models
from src.phase1_5_shadow.freeze_protocol import verify_freeze_manifest
from src.phase1_5_shadow.run_shadow import _provenance_fingerprint


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_formal_collection(
    config: dict[str, Any],
    config_path: str | Path,
    result_root: str | Path,
    freeze_manifest_path: str | Path,
) -> dict[str, Any]:
    errors: list[str] = []
    root = Path(result_root)
    freeze_path = Path(freeze_manifest_path)
    try:
        freeze_manifest = json.loads(freeze_path.read_text(encoding="utf-8"))
        verify_freeze_manifest(config, freeze_manifest)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        return {"passed": False, "errors": [f"freeze verification failed: {exc}"], "cells": []}

    try:
        models = formal_models(config)
        expected_paths = expected_trace_paths(config, root)
    except ValueError as exc:
        return {"passed": False, "errors": [str(exc)], "cells": []}

    split_manifest = json.loads(Path(config["data"]["split_path"]).read_text(encoding="utf-8"))
    expected_ids = {
        split: {str(task_id) for task_id in split_manifest["splits"][split]}
        for split in FORMAL_SPLITS
    }
    expected_sizes = {split: int(split_manifest["sizes"][split]) for split in FORMAL_SPLITS}
    for split in FORMAL_SPLITS:
        raw_ids = [str(task_id) for task_id in split_manifest["splits"][split]]
        if len(raw_ids) != expected_sizes[split] or len(set(raw_ids)) != expected_sizes[split]:
            errors.append(f"frozen split {split} does not contain {expected_sizes[split]} unique task IDs")
    split_id_counts = Counter(task_id for split in FORMAL_SPLITS for task_id in expected_ids[split])
    overlapping_ids = sorted(task_id for task_id, count in split_id_counts.items() if count > 1)
    if overlapping_ids:
        errors.append(f"task IDs occur in multiple formal splits: {overlapping_ids}")
    expected_total = int(config["formal"]["expected_total_runs"])
    calculated_total = len(models) * sum(expected_sizes.values())
    if calculated_total != expected_total:
        errors.append(
            f"configured expected_total_runs={expected_total}, but the frozen matrix requires {calculated_total}"
        )

    expected_trace_set = {path.resolve() for path in expected_paths.values()}
    actual_trace_set = {path.resolve() for path in root.glob("**/traces.jsonl")}
    for path in sorted(expected_trace_set - actual_trace_set):
        errors.append(f"missing formal trace: {path}")
    for path in sorted(actual_trace_set - expected_trace_set):
        errors.append(f"unexpected trace below formal result root: {path}")

    expected_summary_set = {path.with_name("summary.json").resolve() for path in expected_paths.values()}
    actual_summary_set = {path.resolve() for path in root.glob("**/summary.json")}
    for path in sorted(expected_summary_set - actual_summary_set):
        errors.append(f"missing formal summary: {path}")
    for path in sorted(actual_summary_set - expected_summary_set):
        errors.append(f"unexpected summary below formal result root: {path}")

    freeze_sha = _sha256(freeze_path)
    task_sha = _sha256(config["data"]["task_path"])
    split_sha = _sha256(config["data"]["split_path"])
    config_sha = _sha256(config_path)
    tasks = read_jsonl(config["data"]["task_path"])
    task_ids_in_dataset = {str(task["task_id"]) for task in tasks}
    missing_dataset_ids = sorted(set(split_id_counts) - task_ids_in_dataset)
    if missing_dataset_ids:
        errors.append(f"frozen formal task IDs are absent from the task dataset: {missing_dataset_ids}")
    cell_results: list[dict[str, Any]] = []
    run_commits: set[str] = set()
    model_artifacts: dict[str, set[str]] = defaultdict(set)
    docker_digests: dict[str, set[str]] = defaultdict(set)
    total_runs = 0

    for model in models:
        for split in FORMAL_SPLITS:
            trace_path = expected_paths[(model.name, split)]
            summary_path = trace_path.with_name("summary.json")
            cell_errors: list[str] = []
            runs: list[dict[str, Any]] = []
            summary: dict[str, Any] = {}
            if trace_path.is_file():
                try:
                    runs = read_jsonl(trace_path)
                except (json.JSONDecodeError, OSError) as exc:
                    cell_errors.append(f"invalid traces.jsonl: {exc}")
            if summary_path.is_file():
                try:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError) as exc:
                    cell_errors.append(f"invalid summary.json: {exc}")

            expected = expected_ids[split]
            task_ids = [str(run.get("task_id", "")) for run in runs]
            counts = Counter(task_ids)
            duplicates = sorted(task_id for task_id, count in counts.items() if count > 1)
            if len(runs) != expected_sizes[split]:
                cell_errors.append(f"expected {expected_sizes[split]} runs, found {len(runs)}")
            if duplicates:
                cell_errors.append(f"duplicate task IDs: {duplicates}")
            missing_ids = sorted(expected - set(task_ids))
            unexpected_ids = sorted(set(task_ids) - expected)
            if missing_ids:
                cell_errors.append(f"missing task IDs: {missing_ids}")
            if unexpected_ids:
                cell_errors.append(f"unexpected task IDs: {unexpected_ids}")
            if any(run.get("model_name") != model.name for run in runs):
                cell_errors.append("trace model_name mismatch")
            if any(run.get("split") != split for run in runs):
                cell_errors.append("trace split mismatch")

            if summary:
                if summary.get("model_name") != model.name or summary.get("split") != split:
                    cell_errors.append("summary model or split mismatch")
                if int(summary.get("task_count", -1)) != len(runs):
                    cell_errors.append("summary task_count does not match trace count")
                if int(summary.get("expected_task_count", -1)) != expected_sizes[split]:
                    cell_errors.append("summary expected_task_count does not match frozen split")
                provenance = copy.deepcopy(summary.get("reproducibility", {}))
                stored_fingerprint = provenance.pop("run_provenance_fingerprint", None)
                summary_fingerprint = summary.get("run_provenance_fingerprint")
                recomputed = _provenance_fingerprint(provenance, model.name, split) if provenance else None
                if not stored_fingerprint or stored_fingerprint != summary_fingerprint or stored_fingerprint != recomputed:
                    cell_errors.append("summary provenance fingerprint is missing or invalid")
                if any(run.get("provenance_fingerprint") != stored_fingerprint for run in runs):
                    cell_errors.append("one or more trace records have stale provenance")
                if provenance.get("freeze_manifest_sha256") != freeze_sha:
                    cell_errors.append("freeze manifest SHA-256 mismatch")
                if provenance.get("freeze_source_commit") != freeze_manifest.get("source_commit"):
                    cell_errors.append("freeze source commit mismatch")
                if provenance.get("config_sha256") != config_sha:
                    cell_errors.append("config SHA-256 mismatch")
                if provenance.get("task_manifest_sha256") != task_sha:
                    cell_errors.append("task manifest SHA-256 mismatch")
                if provenance.get("split_manifest_sha256") != split_sha:
                    cell_errors.append("split manifest SHA-256 mismatch")
                if provenance.get("protocol_version") != config["protocol"]["version"]:
                    cell_errors.append("protocol version mismatch")
                if provenance.get("protocol_stage") != "formal_shadow_collection" or not provenance.get(
                    "protocol_locked"
                ):
                    cell_errors.append("trace was not collected under a locked formal protocol")

                run_commit = str(provenance.get("run_source_commit") or "")
                if not run_commit:
                    cell_errors.append("run source commit is missing")
                else:
                    run_commits.add(run_commit)
                artifact = provenance.get("model_artifact") or {}
                if artifact.get("model_name") != model.name:
                    cell_errors.append("model artifact name mismatch")
                if not artifact.get("base_blob_sha256") or not artifact.get("modelfile_sha256"):
                    cell_errors.append("model blob or Modelfile hash is missing")
                else:
                    model_artifacts[model.name].add(json.dumps(artifact, sort_keys=True))

                expected_images = {
                    f"{config['environment']['image_prefix']}{int(task['filesystem_version'])}"
                    for task in tasks
                    if str(task["task_id"]) in expected
                }
                recorded_images = provenance.get("docker_image_digests") or {}
                if set(recorded_images) != expected_images:
                    cell_errors.append("Docker image set does not match the frozen split")
                for image, digest in recorded_images.items():
                    if not str(digest).startswith("sha256:"):
                        cell_errors.append(f"invalid Docker image digest for {image}")
                    docker_digests[image].add(str(digest))

            total_runs += len(runs)
            errors.extend(f"{model.slug}/{split}: {error}" for error in cell_errors)
            cell_results.append(
                {
                    "model_name": model.name,
                    "model_slug": model.slug,
                    "split": split,
                    "expected_runs": expected_sizes[split],
                    "observed_runs": len(runs),
                    "trace_path": str(trace_path),
                    "trace_sha256": _sha256(trace_path) if trace_path.is_file() else None,
                    "summary_sha256": _sha256(summary_path) if summary_path.is_file() else None,
                    "passed": not cell_errors,
                    "errors": cell_errors,
                }
            )

    if len(run_commits) != 1:
        errors.append(f"formal cells do not share exactly one run source commit: {sorted(run_commits)}")
    for model_name, artifacts in model_artifacts.items():
        if len(artifacts) != 1:
            errors.append(f"model artifact changed across cells for {model_name}")
    for image, digests in docker_digests.items():
        if len(digests) != 1:
            errors.append(f"Docker image changed across formal cells for {image}")
    if total_runs != expected_total:
        errors.append(f"formal matrix expected exactly {expected_total} runs, found {total_runs}")

    return {
        "phase": "phase1_5_shadow",
        "audit": "formal_matrix_completeness_and_provenance",
        "passed": not errors,
        "expected_models": [model.name for model in models],
        "expected_splits": list(FORMAL_SPLITS),
        "expected_cells": len(models) * len(FORMAL_SPLITS),
        "expected_total_runs": expected_total,
        "observed_total_runs": total_runs,
        "run_source_commits": sorted(run_commits),
        "freeze_manifest_sha256": freeze_sha,
        "cells": cell_results,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the complete frozen Phase 1.5 formal matrix.")
    parser.add_argument("--config", default="config/phase1_5_shadow.yaml")
    parser.add_argument("--results-root", default="results/phase1_5_shadow/full")
    parser.add_argument("--freeze-manifest", default="data/phase1_5/protocol_freeze_manifest.json")
    parser.add_argument("--output", default="results/phase1_5_shadow/full/formal_collection_audit.json")
    args = parser.parse_args()

    result = audit_formal_collection(
        load_simple_yaml(args.config),
        args.config,
        args.results_root,
        args.freeze_manifest,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit("Formal matrix audit failed; annotation packet generation remains locked")


if __name__ == "__main__":
    main()
