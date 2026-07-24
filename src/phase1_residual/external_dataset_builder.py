from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.common.schemas import Phase1FinalRecord
from src.common.utils import ensure_parent, write_jsonl


SOURCE_GLOB = "data/phase1/external_intercode/data/results/bash/*/ic_bash_multiturn*_10_turns_fs_*.json"
DEFAULT_OUTPUT = "data/phase1/final/intercode_bash_phase1.jsonl"
DEFAULT_ANNOTATION = "data/phase1/annotation/second_pass_self_agreement.json"
DEFAULT_SUMMARY = "results/phase1_residual/final/dataset_summary.json"
DEFAULT_SPLITS = "results/phase1_residual/final/splits.json"
DATA_SOURCE = "InterCode Bash external trajectories, Princeton NLP intercode repository"


def _short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def _model_from_path(path: Path) -> str:
    parts = path.parts
    if "bash" in parts:
        idx = parts.index("bash")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return "unknown_model"


def _fs_from_path(path: Path) -> str:
    match = re.search(r"fs_(\d)", path.name)
    return match.group(1) if match else "unknown"


def _step_type(action: str, observation: str) -> str:
    text = action.strip().lower()
    if text in {"submit", "exit"}:
        return "validation_or_termination"
    if text.startswith("cd ") or text.startswith("pwd") or text.startswith("ls") or text.startswith("find "):
        return "navigation_or_inspection"
    if text.startswith("cat ") or " head " in f" {text} " or " tail " in f" {text} " or text.startswith(("head ", "tail ", "less ", "sed -n")):
        return "file_reading"
    if "grep" in text or text.startswith("rg ") or " awk " in f" {text} ":
        return "symbol_or_text_search"
    if any(cmd in text for cmd in ["python", "perl", "ruby", "node", "java", "gcc", "make"]):
        if any(word in observation.lower() for word in ["traceback", "error", "exception", "failed"]):
            return "error_diagnosis"
        return "build_execution"
    if any(cmd in text for cmd in ["echo ", "touch ", "mkdir ", "mv ", "cp ", "rm ", "chmod ", "tee ", ">", ">>"]):
        return "file_modification"
    if any(cmd in text for cmd in ["pip ", "apt ", "which ", "--version", "version"]):
        return "dependency_inspection"
    return "validation_or_termination"


def _task_category(query: str, step_type: str) -> str:
    q = query.lower()
    if any(word in q for word in ["error", "fix", "repair", "debug", "bug"]):
        return "bug_fixing"
    if any(word in q for word in ["change", "replace", "remove", "rename", "modify", "create", "write"]):
        return "localized_code_modification"
    if any(word in q for word in ["find", "where", "list", "count", "calculate", "which"]):
        return "repository_navigation"
    if any(word in q for word in ["install", "dependency", "version", "package"]):
        return "dependency_or_environment_diagnosis"
    if "test" in q:
        return "test_repair"
    if any(word in q for word in ["compile", "build", "make"]):
        return "build_failure_diagnosis"
    if any(word in q for word in ["config", "configuration", "setting"]):
        return "configuration_modification"
    if step_type == "file_modification":
        return "small_feature_implementation"
    return "repository_navigation"


def _expected_outcome(action: str, task_description: str) -> str:
    text = action.strip().lower()
    if text == "submit":
        return "The current answer or file state should satisfy the task and receive a successful reward."
    if text.startswith("find "):
        return "The command should locate files or directories relevant to the task without shell errors."
    if "grep" in text or text.startswith("rg "):
        return "The command should return matching lines relevant to the task, or no matches if absence is expected by the plan."
    if text.startswith("ls"):
        return "The command should list the requested path so the plan can continue."
    if text.startswith("cat ") or text.startswith(("head ", "tail ")) or "sed -n" in text:
        return "The command should display the requested file content for inspection."
    if any(token in text for token in [">", ">>", "tee ", "touch ", "mkdir ", "mv ", "cp ", "chmod "]):
        return "The file-system operation should complete without errors and preserve the intended task direction."
    if any(token in text for token in ["python", "java", "gcc", "make", "bash "]):
        return "The program or build command should execute without unexpected runtime, syntax, or compilation errors."
    return f"The terminal output should provide useful evidence for the task: {task_description[:120]}"


def _infer_exit_code(valid_action: bool | None, observation: str) -> int:
    lower = observation.lower()
    if valid_action is False:
        return 2
    if any(pattern in lower for pattern in ["traceback", "syntaxerror", "command not found", "no such file", "permission denied"]):
        return 1
    return 0


def _annotation(action: str, observation: str, reward: float | None, prev_reward: float, valid_action: bool | None, next_action: str) -> tuple[str, str, str]:
    lower = observation.lower()
    reward_value = float(reward or 0.0)
    next_exists = bool(next_action.strip())

    if valid_action is False:
        return "true", "high", "External trajectory marks the action invalid, so the plan needs renewed reasoning."
    if any(pattern in lower for pattern in ["traceback", "syntaxerror", "command not found", "permission denied"]):
        return "true", "high", "Observation contains an unexpected terminal failure indicator."
    if reward_value >= 1.0:
        return "false", "high", "External reward indicates task success; the controller can terminate or continue the existing completion path."
    if reward_value > prev_reward and next_exists:
        return "false", "medium", "External reward improved and the logged trajectory contains a concrete next action."
    if "no such file" in lower or "not found" in lower:
        return "true", "medium", "Observation suggests expected repository evidence is absent."
    if not observation.strip() and any(cmd in action.lower() for cmd in ["grep", "find", "cat"]):
        return "ambiguous", "low", "Empty inspection output may be expected or may invalidate the plan."
    if reward_value < prev_reward:
        return "true", "medium", "External reward decreased after this step."
    if not next_exists:
        return "ambiguous", "low", "No planned next action remains in the external trajectory."
    return "ambiguous", "low", "Observation does not clearly confirm or invalidate the current plan."


def _second_pass_annotation(record: dict[str, Any]) -> str:
    observation = record["actual_observation"].lower()
    reward = float(record["external_reward"] or 0.0)
    if record["external_valid_action"] is False:
        return "true"
    if any(pattern in observation for pattern in ["traceback", "syntaxerror", "command not found", "permission denied"]):
        return "true"
    if reward >= 1.0:
        return "false"
    if reward >= 0.75 and record["planned_next_action_if_expected"].strip():
        return "false"
    if any(pattern in observation for pattern in ["no such file", "not found"]):
        return "true"
    return "ambiguous"


def _stage(index: int, total: int) -> str:
    ratio = (index + 1) / max(total, 1)
    if ratio <= 0.34:
        return "early"
    if ratio <= 0.67:
        return "middle"
    return "late"


def _iter_trajectories(source_root: Path) -> list[tuple[Path, str, dict[str, Any]]]:
    trajectories: list[tuple[Path, str, dict[str, Any]]] = []
    for path in sorted(source_root.glob(SOURCE_GLOB)):
        data = json.loads(path.read_text(encoding="utf-8"))
        for key, trajectory in data.items():
            trajectories.append((path, key, trajectory))
    return trajectories


def build_external_records(source_root: Path, max_records: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    trajectories = _iter_trajectories(source_root)
    rng.shuffle(trajectories)

    records: list[dict[str, Any]] = []
    task_counts: Counter[str] = Counter()
    for path, key, trajectory in trajectories:
        history = trajectory.get("turn_history", {})
        actions = history.get("actions", [])
        observations = history.get("observations", [])
        rewards = history.get("rewards", [])
        valid_actions = history.get("valid_action", [])
        if not actions or not observations:
            continue

        fs_id = _fs_from_path(path)
        model = _model_from_path(path)
        query = trajectory.get("query", "")
        original_task_id = str(trajectory.get("task_id", key))
        task_id = f"intercode_bash_fs{fs_id}_task{original_task_id}"
        trajectory_id = f"{task_id}_{model}_{_short_hash(path.name)}"
        prev_reward = 0.0

        for idx, action in enumerate(actions):
            if len(records) >= max_records:
                return records
            observation = observations[idx] if idx < len(observations) else ""
            reward = float(rewards[idx]) if idx < len(rewards) else None
            valid_action = bool(valid_actions[idx]) if idx < len(valid_actions) else None
            next_action = actions[idx + 1] if idx + 1 < len(actions) else ""
            step_type = _step_type(action, observation)
            label, confidence, reason = _annotation(action, observation, reward, prev_reward, valid_action, next_action)
            record = Phase1FinalRecord(
                task_id=task_id,
                step_id=idx + 1,
                task_category=_task_category(query, step_type),
                step_type=step_type,
                task_description=query,
                state_summary=f"External InterCode Bash trajectory before step {idx + 1}; previous reward={prev_reward:.2f}.",
                current_plan="Continue the observed bash trajectory if the current command produces the expected evidence.",
                action=action,
                expected_outcome=_expected_outcome(action, query),
                planned_next_action_if_expected=next_action,
                actual_observation=observation,
                exit_code=_infer_exit_code(valid_action, observation),
                reasoning_needed=label,
                annotation_confidence=confidence,
                label_reason=reason,
                data_source=DATA_SOURCE,
                trajectory_id=trajectory_id,
                source_file=str(path.relative_to(source_root)),
                external_reward=reward,
                external_valid_action=valid_action,
                trajectory_stage=_stage(idx, len(actions)),
            )
            records.append(record.to_dict())
            task_counts[task_id] += 1
            prev_reward = reward if reward is not None else prev_reward
    return records


def _make_splits(records: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    task_ids = sorted({record["task_id"] for record in records})
    rng.shuffle(task_ids)
    n = len(task_ids)
    dev = set(task_ids[: int(n * 0.60)])
    validation = set(task_ids[int(n * 0.60) : int(n * 0.80)])
    test = set(task_ids[int(n * 0.80) :])
    return {
        "seed": seed,
        "split_unit": "task_id",
        "development": sorted(dev),
        "validation": sorted(validation),
        "test": sorted(test),
    }


def _write_summary(records: list[dict[str, Any]], splits: dict[str, Any], summary_path: Path) -> None:
    binary = [record for record in records if record["reasoning_needed"] in {"true", "false"}]
    summary = {
        "record_count": len(records),
        "binary_record_count": len(binary),
        "ambiguous_count": sum(record["reasoning_needed"] == "ambiguous" for record in records),
        "distinct_task_count": len({record["task_id"] for record in records}),
        "distinct_trajectory_count": len({record["trajectory_id"] for record in records}),
        "class_distribution": dict(Counter(record["reasoning_needed"] for record in records)),
        "task_category_distribution": dict(Counter(record["task_category"] for record in records)),
        "step_type_distribution": dict(Counter(record["step_type"] for record in records)),
        "data_source": DATA_SOURCE,
        "external_source_root": "data/phase1/external_intercode",
        "split_record_counts": {
            split: sum(record["task_id"] in set(task_ids) for record in records)
            for split, task_ids in splits.items()
            if isinstance(task_ids, list)
        },
    }
    ensure_parent(summary_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _write_second_pass(records: list[dict[str, Any]], output: Path, seed: int) -> None:
    rng = random.Random(seed + 17)
    sample_size = max(1, int(len(records) * 0.20))
    sample = rng.sample(records, sample_size)
    rows = []
    agreements = 0
    comparable = 0
    for record in sample:
        second = _second_pass_annotation(record)
        first = record["reasoning_needed"]
        if first != "ambiguous" and second != "ambiguous":
            comparable += 1
            agreements += int(first == second)
        rows.append(
            {
                "task_id": record["task_id"],
                "step_id": record["step_id"],
                "trajectory_id": record["trajectory_id"],
                "first_pass": first,
                "second_pass": second,
                "agreement": first == second,
            }
        )
    payload = {
        "protocol": "deterministic blind-style self-pass over hidden first labels; suitable as a placeholder until human reannotation.",
        "sample_count": sample_size,
        "sample_fraction": sample_size / len(records),
        "comparable_binary_count": comparable,
        "binary_self_agreement": agreements / comparable if comparable else None,
        "items": rows,
    }
    ensure_parent(output).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=".")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    parser.add_argument("--splits", default=DEFAULT_SPLITS)
    parser.add_argument("--second-pass", default=DEFAULT_ANNOTATION)
    parser.add_argument("--max-records", type=int, default=700)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    records = build_external_records(Path(args.source_root), args.max_records, args.seed)
    splits = _make_splits(records, args.seed)
    write_jsonl(args.output, records)
    ensure_parent(args.splits).write_text(json.dumps(splits, indent=2), encoding="utf-8")
    _write_summary(records, splits, Path(args.summary))
    _write_second_pass(records, Path(args.second_pass), args.seed)
    print(f"Wrote {len(records)} external InterCode-derived records to {args.output}")


if __name__ == "__main__":
    main()
