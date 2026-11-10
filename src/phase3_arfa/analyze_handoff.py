"""Paired analysis for the Phase 3 handoff ablation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.common.utils import read_csv, write_csv
from src.phase3_arfa.analyze_results import _paired_comparison, _policy_summary


MODES = ("full_visible", "structured_state", "observation_only")


def analyze(root: Path, expected: int = 100) -> dict[str, Any]:
    rows: dict[str, list[dict[str, str]]] = {}
    provenances: dict[str, dict[str, Any]] = {}
    for mode in MODES:
        run_dir = root / "development" / mode / "residual"
        rows[mode] = read_csv(run_dir / "metrics.csv")
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        provenances[mode] = summary["provenance"]
    task_sets = [{row["task_id"] for row in rows[mode]} for mode in MODES]
    shared_keys = (
        "config_sha256", "task_manifest_sha256", "split_manifest_sha256",
        "selected_task_ids_sha256", "source_sha256", "fast_model", "slow_model",
        "docker_image_digests", "trajectory_workers",
    )
    complete = all(len(rows[mode]) == expected for mode in MODES)
    task_sets_match = all(ids == task_sets[0] for ids in task_sets[1:])
    provenance_matches = all(
        provenances[mode].get(key) == provenances[MODES[0]].get(key)
        for mode in MODES[1:] for key in shared_keys
    )
    if not (complete and task_sets_match and provenance_matches):
        raise ValueError("Handoff cells are incomplete or do not share tasks and provenance")
    return {
        "phase": "phase3_handoff_ablation",
        "split": "development",
        "expected_tasks_per_mode": expected,
        "audit": {
            "complete": complete,
            "task_sets_match": task_sets_match,
            "provenance_matches": provenance_matches,
        },
        "mode_summaries": [
            {**_policy_summary(mode, rows[mode], expected), "handoff_mode": mode}
            for mode in MODES
        ],
        "paired_comparisons": {
            "structured_state_vs_full_visible": _paired_comparison(
                rows["structured_state"], rows["full_visible"], 10_000, 1507
            ),
            "observation_only_vs_full_visible": _paired_comparison(
                rows["observation_only"], rows["full_visible"], 10_000, 1507
            ),
            "structured_state_vs_observation_only": _paired_comparison(
                rows["structured_state"], rows["observation_only"], 10_000, 1507
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.result_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "analysis.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_csv(args.output_dir / "mode_summary.csv", result["mode_summaries"])
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
