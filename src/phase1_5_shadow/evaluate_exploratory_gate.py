from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.common.config import load_simple_yaml


def _find_metric(analysis: dict[str, Any], method: str) -> dict[str, Any]:
    for row in analysis.get("test_metrics", []):
        if row.get("method") == method:
            return row
    raise ValueError(f"Missing test metric: {method}")


def evaluate_engineering_gate(
    analysis: dict[str, Any],
    collection_audit: dict[str, Any],
    phase2_completeness: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    gate_config = config["gate"]
    review_config = config["review"]
    full = _find_metric(analysis, "full_residual_score")
    gain = analysis["incremental_value"]["full_vs_expectation_gate_only"]
    readiness = analysis["readiness"]

    checks = {
        "formal_collection_complete": bool(collection_audit.get("passed"))
        and int(collection_audit.get("observed_total_runs", 0))
        == int(collection_audit.get("expected_total_runs", -1)),
        "blind_labels_complete": bool(readiness.get("ready"))
        and int(readiness.get("annotation_items", 0))
        >= int(gate_config["minimum_annotation_items"]),
        "embedding_backend_available": str(analysis.get("embedding_backend", "")).startswith(
            "sentence-transformers/"
        ),
        "full_residual_auc_above_engineering_floor": full.get("roc_auc") is not None
        and float(full["roc_auc"]) >= float(gate_config["minimum_full_residual_auc"]),
        "gain_over_expectation_gate_only": gain.get("observed_auc_gain") is not None
        and float(gain["observed_auc_gain"])
        >= float(gate_config["minimum_auc_gain_over_expectation_gate_only"]),
        "gain_interval_excludes_zero": not bool(gate_config["require_gain_ci_excludes_zero"])
        or (
            gain.get("bootstrap_ci95_low") is not None
            and float(gain["bootstrap_ci95_low"]) > 0
        ),
        "phase2_baseline_complete": bool(phase2_completeness.get("complete"))
        and int(phase2_completeness.get("observed_runs", 0)) == 600,
    }
    passed = all(checks.values())
    return {
        "gate_version": gate_config["version"],
        "gate_scope": gate_config["scope"],
        "interpretation": (
            "Engineering readiness for a guarded exploratory Phase 3 pilot. "
            "This is not a deployment-safety or confirmatory residual-evidence gate."
        ),
        "review_provenance": {
            "mode": review_config["mode"],
            "reviewer_count_reported_by_project_owner": int(
                review_config["reviewer_count_reported_by_project_owner"]
            ),
            "independently_partitioned_review_reported": bool(
                review_config["independently_partitioned_review_reported"]
            ),
            "per_reviewer_raw_files_available": bool(
                review_config["per_reviewer_raw_files_available"]
            ),
        },
        "observed": {
            "formal_shadow_runs": int(collection_audit.get("observed_total_runs", 0)),
            "annotation_items": int(readiness.get("annotation_items", 0)),
            "full_residual_roc_auc": full.get("roc_auc"),
            "full_residual_auc_ci95": full.get("confidence_intervals", {}).get("roc_auc"),
            "auc_gain_over_expectation_gate_only": gain.get("observed_auc_gain"),
            "auc_gain_ci95": [gain.get("bootstrap_ci95_low"), gain.get("bootstrap_ci95_high")],
            "phase2_runs": int(phase2_completeness.get("observed_runs", 0)),
        },
        "checks": checks,
        "passed": passed,
        "phase3_p_exploratory_unlocked": passed and bool(
            config["unlock"]["phase3_exploratory_pilot"]
        ),
        "phase3_formal_unlocked": passed and bool(
            config["unlock"]["phase3_formal_evaluation"]
        ),
        "formal_evidence_gate_passed": bool(
            analysis.get("acceptance_gate", {}).get("passed", False)
        ),
    }


def _report(result: dict[str, Any]) -> str:
    status = "PASS" if result["passed"] else "FAIL"
    observed = result["observed"]
    lines = [
        "# Phase 1.5 Exploratory Engineering Gate",
        "",
        f"Engineering gate: **{status}**.",
        "",
        result["interpretation"],
        "",
        "## Evidence Summary",
        "",
        f"- Formal shadow collection: {observed['formal_shadow_runs']}/300 runs",
        f"- Complete blind annotation items: {observed['annotation_items']}",
        f"- Full residual held-out ROC-AUC: {observed['full_residual_roc_auc']:.3f}",
        "- Full residual gain over expectation-gate-only: "
        f"{observed['auc_gain_over_expectation_gate_only']:.3f}",
        f"- Phase 2 always-reason baseline: {observed['phase2_runs']}/600 runs",
        "",
        "## Review Provenance",
        "",
        "The project owner reports that six people independently divided and reviewed the "
        "AI-assisted labels. The current repository preserves the AI judge audit and the "
        "owner-reported human-review status. Separate raw reviewer files are not present, "
        "so the reported agreement statistics remain inter-model rather than six-reviewer "
        "inter-annotator agreement.",
        "",
        "## Interpretation",
        "",
        "The engineering gate is intentionally scoped to deciding whether the data, signal, "
        "and baseline infrastructure justify a guarded online controller pilot. It does not "
        "require the offline shadow router to satisfy deployment-level precision and coverage "
        "simultaneously, because Phase 3-P directly measures end-to-end task success, recovery, "
        "and reasoning-call savings under a fail-open controller.",
        "",
        "The stricter formal evidence gate remains recorded as a diagnostic result. Its failure "
        "is interpreted as evidence that the first offline threshold is too coarse for direct "
        "deployment, not as an engineering blocker for a conservative exploratory pilot.",
        "",
        "## Unlock Status",
        "",
        f"- Phase 3-P exploratory pilot: **{'UNLOCKED' if result['phase3_p_exploratory_unlocked'] else 'LOCKED'}**",
        f"- Phase 3 formal evaluation: **{'UNLOCKED' if result['phase3_formal_unlocked'] else 'LOCKED'}**",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Phase 1.5 exploratory engineering gate.")
    parser.add_argument("--config", default="config/phase1_5_exploratory_gate.yaml")
    parser.add_argument("--analysis", default="results/phase1_5_shadow/analysis_ai_blind/analysis.json")
    parser.add_argument(
        "--collection-audit",
        default="results/phase1_5_shadow/full/formal_collection_audit.json",
    )
    parser.add_argument(
        "--phase2-completeness",
        default="results/phase2_baseline/paper/analysis/completeness.json",
    )
    parser.add_argument("--output-dir", default="evidence/phase1_5_exploratory")
    args = parser.parse_args()

    result = evaluate_engineering_gate(
        json.loads(Path(args.analysis).read_text(encoding="utf-8")),
        json.loads(Path(args.collection_audit).read_text(encoding="utf-8")),
        json.loads(Path(args.phase2_completeness).read_text(encoding="utf-8")),
        load_simple_yaml(args.config),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "exploratory_gate.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "README.md").write_text(_report(result), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit("Exploratory engineering gate did not pass")


if __name__ == "__main__":
    main()

