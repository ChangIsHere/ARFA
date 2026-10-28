"""Reject incomplete, stale, or infrastructure-damaged Phase 3 policy runs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from src.common.config import load_simple_yaml
from src.common.utils import read_jsonl


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _expected_ids(config: dict[str, Any], split: str) -> set[str]:
    payload = json.loads(Path(config["data"]["split_path"]).read_text(encoding="utf-8"))
    split_map = payload["splits"]
    if split == "development":
        names = ("shadow_development", "shadow_validation", "shadow_test")
    elif split == "final":
        names = ("phase3_final",)
    else:
        raise ValueError(f"Unknown split: {split}")
    return set().union(*(set(map(str, split_map[name])) for name in names))


def audit_policy_run(
    config_path: Path,
    result_dir: Path,
    split: str,
    pair: str,
    policy: str,
) -> dict[str, Any]:
    config = load_simple_yaml(config_path)
    expected_ids = _expected_ids(config, split)
    summary_path = result_dir / "summary.json"
    metrics_path = result_dir / "metrics.csv"
    traces_path = result_dir / "traces.jsonl"
    for path in (summary_path, metrics_path, traces_path):
        if not path.is_file():
            raise ValueError(f"Missing policy artifact: {path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with metrics_path.open(encoding="utf-8", newline="") as handle:
        metrics = list(csv.DictReader(handle))
    traces = read_jsonl(traces_path)
    metric_ids = [str(row["task_id"]) for row in metrics]
    trace_ids = [str(row["task_id"]) for row in traces]
    fingerprint = summary.get("provenance", {}).get("fingerprint")

    checks = {
        "summary_task_count": int(summary.get("task_count", -1)) == len(expected_ids),
        "summary_expected_task_count": int(summary.get("expected_task_count", -1)) == len(expected_ids),
        "zero_model_request_errors": int(summary.get("model_request_error_count", -1)) == 0,
        "metrics_exact_task_set": len(metric_ids) == len(set(metric_ids))
        and set(metric_ids) == expected_ids,
        "traces_exact_task_set": len(trace_ids) == len(set(trace_ids))
        and set(trace_ids) == expected_ids,
        "summary_identity": summary.get("experiment_split") == split
        and summary.get("pair_name") == pair
        and summary.get("policy") == policy,
        "row_identity": all(
            row.get("experiment_split") == split
            and row.get("pair_name") == pair
            and row.get("policy") == policy
            for row in metrics
        ),
        "trace_fingerprints_match": bool(fingerprint)
        and all(row.get("provenance_fingerprint") == fingerprint for row in traces),
        "config_hash_matches": summary.get("provenance", {}).get("config_sha256")
        == _sha256(config_path),
    }
    report = {
        "valid": all(checks.values()),
        "split": split,
        "pair": pair,
        "policy": policy,
        "expected_tasks": len(expected_ids),
        "checks": checks,
    }
    (result_dir / "audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "final"), required=True)
    parser.add_argument("--pair", required=True)
    parser.add_argument("--policy", required=True)
    args = parser.parse_args()
    report = audit_policy_run(args.config, args.result_dir, args.split, args.pair, args.policy)
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
