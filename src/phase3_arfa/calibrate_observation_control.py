"""Choose the online observation-only threshold on Phase 1.5 validation labels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from src.common.utils import read_jsonl
from src.phase3_arfa.learned_observation import LearnedObservationScorer


def calibrate(annotation_path: Path, source_map_path: Path, seed: int = 1507) -> dict:
    scorer = LearnedObservationScorer(str(annotation_path), str(source_map_path), seed)
    sources = {row["annotation_id"]: row for row in read_jsonl(source_map_path)}
    rows = [
        row for row in read_jsonl(annotation_path)
        if sources.get(row["annotation_id"], {}).get("split") == "shadow_validation"
        and row.get("slow_reasoning_needed") in {"yes", "no"}
    ]
    labels = [int(row["slow_reasoning_needed"] == "yes") for row in rows]
    if len(set(labels)) != 2:
        raise ValueError("Validation requires both binary label classes")
    scores = [
        scorer.score(
            str(row.get("actual_observation") or ""),
            int(row.get("exit_code") or 0),
        )
        for row in rows
    ]
    candidates = sorted(set(scores) | {0.0, 1.0})
    threshold = max(
        candidates,
        key=lambda value: (
            balanced_accuracy_score(labels, [int(score >= value) for score in scores]),
            value,
        ),
    )
    return {
        "training": scorer.training_summary,
        "validation_split": "shadow_validation",
        "validation_binary_items": len(rows),
        "validation_positive_items": sum(labels),
        "selection_rule": "maximize balanced accuracy; prefer higher threshold on ties",
        "threshold": float(threshold),
        "validation_balanced_accuracy": float(
            balanced_accuracy_score(labels, [int(score >= threshold) for score in scores])
        ),
        "validation_roc_auc": float(roc_auc_score(labels, scores)),
        "annotation_sha256": hashlib.sha256(annotation_path.read_bytes()).hexdigest(),
        "source_map_sha256": hashlib.sha256(source_map_path.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--source-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = calibrate(args.annotations, args.source_map)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
