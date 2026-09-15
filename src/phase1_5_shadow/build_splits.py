from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.common.utils import read_jsonl


DEFAULT_SIZES = {
    "shadow_development": 40,
    "shadow_validation": 20,
    "shadow_test": 40,
    "phase3_final": 100,
}


def _rank(seed: int, *parts: str) -> str:
    return hashlib.sha256(f"{seed}|{'|'.join(parts)}".encode("utf-8")).hexdigest()


def build_split(tasks: list[dict[str, Any]], seed: int = 1507) -> dict[str, Any]:
    if len(tasks) != sum(DEFAULT_SIZES.values()):
        raise ValueError(f"Expected {sum(DEFAULT_SIZES.values())} tasks, found {len(tasks)}")
    if len({str(task["task_id"]) for task in tasks}) != len(tasks):
        raise ValueError("Task IDs must be unique")

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        by_category[str(task["task_category"])].append(task)

    assignments: dict[str, list[str]] = {name: [] for name in DEFAULT_SIZES}
    category_counts: dict[str, Counter[str]] = {name: Counter() for name in DEFAULT_SIZES}
    total = len(tasks)

    for category in sorted(by_category, key=lambda item: (len(by_category[item]), item)):
        rows = sorted(by_category[category], key=lambda task: _rank(seed, str(task["task_id"])))
        for task in rows:
            candidates = [name for name, size in DEFAULT_SIZES.items() if len(assignments[name]) < size]
            if not candidates:
                raise RuntimeError("No split capacity remains")

            def score(name: str) -> tuple[float, float, str]:
                desired_category = len(rows) * DEFAULT_SIZES[name] / total
                category_deficit = desired_category - category_counts[name][category]
                capacity_ratio = (DEFAULT_SIZES[name] - len(assignments[name])) / DEFAULT_SIZES[name]
                return category_deficit, capacity_ratio, _rank(seed, category, str(task["task_id"]), name)

            selected = max(candidates, key=score)
            assignments[selected].append(str(task["task_id"]))
            category_counts[selected][category] += 1

    task_lookup = {str(task["task_id"]): task for task in tasks}
    distributions: dict[str, Any] = {}
    for split, task_ids in assignments.items():
        task_ids.sort()
        rows = [task_lookup[task_id] for task_id in task_ids]
        distributions[split] = {
            "tasks": len(rows),
            "by_category": dict(sorted(Counter(str(row["task_category"]) for row in rows).items())),
            "by_filesystem": dict(sorted(Counter(str(row["filesystem_version"]) for row in rows).items())),
        }

    return {
        "phase": "phase1_5_shadow",
        "seed": seed,
        "split_unit": "task_id",
        "policy": "deterministic category-balanced assignment with exact split capacities",
        "locked_before_shadow_collection": True,
        "allowed_phase1_5_splits": ["shadow_development", "shadow_validation", "shadow_test"],
        "reserved_split": "phase3_final",
        "sizes": DEFAULT_SIZES,
        "splits": assignments,
        "distributions": distributions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze task-level Phase 1.5 and Phase 3 splits.")
    parser.add_argument("--tasks", default="data/phase2/intercode_nl2bash_official_200.jsonl")
    parser.add_argument("--output", default="data/phase1_5/task_splits.json")
    parser.add_argument("--seed", type=int, default=1507)
    args = parser.parse_args()

    task_path = Path(args.tasks)
    payload = build_split(read_jsonl(task_path), seed=args.seed)
    payload["source_task_manifest"] = str(task_path)
    payload["source_task_manifest_sha256"] = hashlib.sha256(task_path.read_bytes()).hexdigest()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "sizes": payload["sizes"], "distributions": payload["distributions"]}, indent=2))


if __name__ == "__main__":
    main()
