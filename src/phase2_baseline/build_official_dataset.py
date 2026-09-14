from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.common.utils import ensure_parent, write_jsonl


SOURCE_COMMIT = "c3e46d827cfc9d4c704ec078f7abf9f41e3191d8"


def _category(instruction: str) -> str:
    first = instruction.strip().split(maxsplit=1)[0].lower().rstrip(".,:")
    aliases = {
        "calculate": "compute",
        "compute": "compute",
        "count": "count",
        "counts": "count",
        "create": "create_or_modify",
        "copies": "copy_or_move",
        "copy": "copy_or_move",
        "delete": "delete",
        "display": "inspect",
        "find": "search",
        "list": "inspect",
        "print": "inspect",
        "prints": "inspect",
        "recursively": "search",
        "remove": "delete",
        "search": "search",
    }
    return aliases.get(first, "other")


def build_records(source_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for fs_version in range(1, 5):
        source_path = source_dir / f"nl2bash_fs_{fs_version}.json"
        raw = json.loads(source_path.read_text(encoding="utf-8"))
        for source_index, item in enumerate(raw):
            records.append(
                {
                    "task_id": f"intercode-nl2bash-fs{fs_version}-{source_index:03d}",
                    "benchmark": "intercode_nl2bash",
                    "split": "arfa_phase2_evaluation",
                    "source_partition": "full_intercode_nl2bash_suite",
                    "source_repo": "princeton-nlp/intercode",
                    "source_commit": SOURCE_COMMIT,
                    "source_path": str(source_path),
                    "filesystem_version": fs_version,
                    "source_index": source_index,
                    "task_category": _category(str(item["query"])),
                    "instruction": str(item["query"]),
                    "gold_command": item["gold"],
                    "evaluator": {"kind": "intercode_official_reward_reimplementation"},
                }
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the full official InterCode NL2Bash Phase 2 manifest.")
    parser.add_argument(
        "--source-dir",
        default="data/phase1/external_intercode/data/nl2bash",
    )
    parser.add_argument(
        "--output",
        default="data/phase2/intercode_nl2bash_official_200.jsonl",
    )
    parser.add_argument(
        "--summary",
        default="data/phase2/intercode_nl2bash_official_200_summary.json",
    )
    args = parser.parse_args()

    records = build_records(Path(args.source_dir))
    if len(records) != 200:
        raise SystemExit(f"Expected 200 official tasks, found {len(records)}")
    write_jsonl(args.output, records)

    summary = {
        "benchmark": "intercode_nl2bash",
        "source_repo": "princeton-nlp/intercode",
        "source_commit": SOURCE_COMMIT,
        "task_count": len(records),
        "tasks_by_filesystem": dict(Counter(str(row["filesystem_version"]) for row in records)),
        "tasks_by_category": dict(Counter(str(row["task_category"]) for row in records)),
        "split": "arfa_phase2_evaluation",
        "development_set": "data/phase2/intercode_bash_local_tasks.jsonl",
    }
    summary_path = ensure_parent(args.summary)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
