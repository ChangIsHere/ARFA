from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.common.utils import read_jsonl, write_csv, write_jsonl
from src.phase2_baseline.intercode_docker_environment import InterCodeDockerEnvironment


def validate_task(task: dict[str, Any], image_prefix: str, network_mode: str) -> dict[str, Any]:
    env = InterCodeDockerEnvironment(task=task, image_prefix=image_prefix, network_mode=network_mode)
    try:
        command_result = env.run(str(task["gold_command"]))
        evaluator = env.evaluate(task, trace_text=command_result.observation, final_answer="")
    finally:
        env.close()
    return {
        "task_id": task["task_id"],
        "filesystem_version": task["filesystem_version"],
        "task_category": task["task_category"],
        "gold_exit_code": command_result.exit_code,
        "self_reward": evaluator["reward"],
        "self_success": evaluator["success"],
        "answer_similarity": evaluator["answer_similarity"],
        "diff_miss": evaluator["diff_miss"],
        "diff_extra": evaluator["diff_extra"],
        "valid": command_result.exit_code == 0 and evaluator["success"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run official gold commands against the Phase 2 InterCode images.")
    parser.add_argument("--tasks", default="data/phase2/intercode_nl2bash_official_200.jsonl")
    parser.add_argument("--output-dir", default="results/phase2_baseline/environment_validation")
    parser.add_argument("--image-prefix", default="arfa/intercode-nl2bash:fs")
    parser.add_argument("--task-limit", type=int, default=0)
    parser.add_argument("--network-mode", default="bridge")
    args = parser.parse_args()

    tasks = read_jsonl(args.tasks)
    if args.task_limit:
        tasks = tasks[: args.task_limit]
    rows: list[dict[str, Any]] = []
    output_dir = Path(args.output_dir)
    for index, task in enumerate(tasks, start=1):
        row = validate_task(task, args.image_prefix, args.network_mode)
        rows.append(row)
        write_jsonl(output_dir / "gold_validation.jsonl", rows)
        print(f"[{index}/{len(tasks)}] {row['task_id']} reward={row['self_reward']:.2f} exit={row['gold_exit_code']}")

    write_csv(output_dir / "gold_validation.csv", rows)
    summary = {
        "task_count": len(rows),
        "valid_count": sum(bool(row["valid"]) for row in rows),
        "self_success_count": sum(bool(row["self_success"]) for row in rows),
        "nonzero_gold_exit_count": sum(int(row["gold_exit_code"]) != 0 for row in rows),
        "complete": len(rows) == 200,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
