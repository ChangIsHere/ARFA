import csv
import hashlib
import json
from pathlib import Path

from src.phase3_arfa.audit_policy_run import audit_policy_run


def _write_fixture(tmp_path: Path, request_errors: int = 0) -> tuple[Path, Path]:
    split = tmp_path / "splits.json"
    split.write_text(json.dumps({"splits": {
        "shadow_development": ["a"], "shadow_validation": ["b"],
        "shadow_test": [], "phase3_final": ["z"],
    }}), encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(f"data:\n  split_path: {split}\n", encoding="utf-8")
    fingerprint = "frozen"
    result = tmp_path / "result"
    result.mkdir()
    summary = {
        "task_count": 2, "expected_task_count": 2,
        "model_request_error_count": request_errors,
        "experiment_split": "development", "pair_name": "gemma", "policy": "residual",
        "provenance": {
            "fingerprint": fingerprint,
            "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        },
    }
    (result / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with (result / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("task_id", "experiment_split", "pair_name", "policy"))
        writer.writeheader()
        for task_id in ("a", "b"):
            writer.writerow({"task_id": task_id, "experiment_split": "development", "pair_name": "gemma", "policy": "residual"})
    with (result / "traces.jsonl").open("w", encoding="utf-8") as handle:
        for task_id in ("a", "b"):
            handle.write(json.dumps({"task_id": task_id, "provenance_fingerprint": fingerprint}) + "\n")
    return config, result


def test_policy_audit_accepts_complete_error_free_run(tmp_path: Path) -> None:
    config, result = _write_fixture(tmp_path)
    assert audit_policy_run(config, result, "development", "gemma", "residual")["valid"]


def test_policy_audit_rejects_model_request_errors(tmp_path: Path) -> None:
    config, result = _write_fixture(tmp_path, request_errors=1)
    report = audit_policy_run(config, result, "development", "gemma", "residual")
    assert not report["valid"]
    assert not report["checks"]["zero_model_request_errors"]
