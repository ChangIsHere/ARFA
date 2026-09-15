from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.common.config import load_simple_yaml
from src.common.utils import read_jsonl, write_jsonl
from src.phase1_5_shadow.analyze_annotations import (
    AMBIGUOUS_LABEL,
    LABELS,
    _annotation_agreement,
    _annotation_metadata_issues,
    _raw_label,
    _validate_label_values,
)


def review(
    primary: list[dict[str, Any]], secondary: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _validate_label_values(primary)
    _validate_label_values(secondary)
    agreement = _annotation_agreement(
        primary,
        secondary,
        float(config["annotation"]["secondary_fraction"]),
    )
    incomplete = {
        "primary": {
            target: sum(not _raw_label(row, target) for row in primary)
            for target in LABELS
        },
        "secondary": {
            target: sum(not _raw_label(row, target) for row in secondary)
            for target in LABELS
        },
    }
    ambiguous = {
        "primary": {
            target: sum(_raw_label(row, target) == AMBIGUOUS_LABEL for row in primary)
            for target in LABELS
        },
        "secondary": {
            target: sum(_raw_label(row, target) == AMBIGUOUS_LABEL for row in secondary)
            for target in LABELS
        },
    }
    metadata_issues = {
        "primary": _annotation_metadata_issues(primary),
        "secondary": _annotation_metadata_issues(secondary),
    }
    primary_by_id = {str(row["annotation_id"]): row for row in primary}
    disagreements = []
    for row in secondary:
        other = primary_by_id.get(str(row["annotation_id"]))
        if not other:
            continue
        differing = [target for target in LABELS if _raw_label(other, target) != _raw_label(row, target)]
        if differing:
            disagreements.append(
                {
                    "annotation_id": row["annotation_id"],
                    "differing_labels": differing,
                    "primary": {target: _raw_label(other, target) for target in LABELS},
                    "secondary": {target: _raw_label(row, target) for target in LABELS},
                }
            )
    minimum_kappa = float(config["annotation"]["minimum_cohen_kappa"])
    kappa_gate = all(
        agreement[target]["multiclass_cohen_kappa"] is not None
        and agreement[target]["multiclass_cohen_kappa"] >= minimum_kappa
        for target in LABELS
    )
    complete = not any(value for packet in incomplete.values() for value in packet.values())
    complete = complete and not metadata_issues["primary"] and not metadata_issues["secondary"]
    readiness = {
        "ready": complete and agreement["complete"] and kappa_gate,
        "scope": "pilot_guideline_review_only",
        "paper_agreement_result": False,
        "primary_items": len(primary),
        "secondary_items": len(secondary),
        "incomplete_labels": incomplete,
        "ambiguous_labels": ambiguous,
        "metadata_issues": metadata_issues,
        "agreement": agreement,
        "minimum_multiclass_kappa": minimum_kappa,
        "kappa_gate_passed": kappa_gate,
        "disagreement_count": len(disagreements),
    }
    return {"readiness": readiness}, disagreements


def main() -> None:
    parser = argparse.ArgumentParser(description="Review pilot annotation clarity without residual scoring.")
    parser.add_argument("--config", default="config/phase1_5_shadow.yaml")
    parser.add_argument("--primary", default="results/phase1_5_shadow/pilot_annotation/primary_labeled.jsonl")
    parser.add_argument("--secondary", default="results/phase1_5_shadow/pilot_annotation/secondary_labeled.jsonl")
    parser.add_argument("--output-dir", default="results/phase1_5_shadow/pilot_review")
    args = parser.parse_args()

    result, disagreements = review(
        read_jsonl(args.primary),
        read_jsonl(args.secondary),
        load_simple_yaml(args.config),
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "pilot_review.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_jsonl(output / "disagreements.jsonl", disagreements)
    print(json.dumps(result, indent=2))
    if not result["readiness"]["ready"]:
        raise SystemExit("Pilot annotation review did not pass; formal protocol remains unlocked")


if __name__ == "__main__":
    main()
