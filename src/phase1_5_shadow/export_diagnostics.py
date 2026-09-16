"""Descriptive, label-free residual/model diagnostics from preserved shadow traces."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean

from sklearn.metrics import roc_auc_score

from src.common.config import load_simple_yaml
from src.common.utils import read_jsonl, write_csv
from src.phase1_5_shadow.analyze_annotations import _apply_scores
from src.phase1_5_shadow.residual import continuation_is_self_contained, structured_residual
from src.phase1_residual.residual_calculator import embedding_residuals


def collect_steps(paths: list[Path]) -> tuple[list[dict], list[dict]]:
    steps, runs, seen = [], [], set()
    for path in paths:
        for trace in read_jsonl(path):
            key = (trace["model_name"], trace["split"], trace["task_id"])
            if key in seen:
                raise ValueError(f"Duplicate run: {key}")
            seen.add(key)
            run = dict(zip(("model", "split", "task_id"), key))
            run.update(success=bool(trace["success"]), model_calls=trace["model_calls"],
                       total_tokens=trace["total_tokens"], trace_path=str(path))
            runs.append(run)
            step_ids = set()
            for step in trace["steps"]:
                if not step.get("action_executed"):
                    continue
                if step["step_id"] in step_ids:
                    raise ValueError(f"Duplicate step in {key}: {step['step_id']}")
                step_ids.add(step["step_id"])
                score, checks = structured_residual(
                    step["expected_signals"], step["expected_outcome"], step["exit_code"],
                    step["stdout"], step["stderr"], step["observation"],
                )
                if not math.isclose(score, step["structured_residual"], abs_tol=1e-9):
                    raise ValueError(f"Recorded residual differs from formula: {key}, {step['step_id']}")
                action, continuation = step["action"].strip(), step["next_action_if_expected"].strip()
                action_gate = not action or not continuation_is_self_contained(action)
                expectation_gate = (action_gate or not continuation or continuation.upper() == "<REASON>"
                                    or not continuation_is_self_contained(continuation))
                steps.append({
                    **run, "step_id": step["step_id"], "exit_code": step["exit_code"],
                    "expected_outcome": step["expected_outcome"], "observation": step["observation"],
                    "structured_expectation_score": score,
                    "observation_only_raw_score": step["observation_only_score"],
                    "expectation_quality": step["expectation_quality"],
                    "action_safety_gate": float(action_gate), "expectation_safety_gate": float(expectation_gate),
                    "shadow_decision": step["shadow_decision"],
                    **{name: int(value) for name, value in checks.items()},
                })
    return steps, runs


def summarize_runs(steps: list[dict], runs: list[dict]) -> None:
    by_run = defaultdict(list)
    for step in steps:
        by_run[(step["model"], step["split"], step["task_id"])].append(step)
    for run in runs:
        selected = sorted(by_run[(run["model"], run["split"], run["task_id"])], key=lambda s: s["step_id"])
        run["executed_steps"] = len(selected)
        run["first_structured_residual"] = selected[0]["structured_expectation_score"] if selected else None
        for name, field in (("structured", "structured_expectation_score"),
                            ("semantic", "semantic_residual_score"),
                            ("observation_only", "observation_only_raw_score"),
                            ("full", "full_residual_score")):
            run[f"mean_{name}_residual"] = mean(s[field] for s in selected) if selected else None


def failure_auc(runs: list[dict], field: str) -> float | None:
    usable = [r for r in runs if r[field] is not None]
    labels = [int(not r["success"]) for r in usable]
    return float(roc_auc_score(labels, [r[field] for r in usable])) if len(set(labels)) == 2 else None


def model_summary(runs: list[dict]) -> list[dict]:
    summaries = []
    for model in sorted({r["model"] for r in runs}):
        for split in ["all"] + sorted({r["split"] for r in runs}):
            selected = [r for r in runs if r["model"] == model and (split == "all" or r["split"] == split)]
            if not selected:
                continue
            row = {"model": model, "split": split, "runs": len(selected),
                   "executed_steps": sum(r["executed_steps"] for r in selected),
                   "runs_with_residual": sum(r["executed_steps"] > 0 for r in selected),
                   "success_rate": mean(r["success"] for r in selected),
                   "mean_model_calls": mean(r["model_calls"] for r in selected)}
            for name in ("structured", "semantic", "observation_only", "full"):
                field = f"mean_{name}_residual"
                values = [r[field] for r in selected if r[field] is not None]
                row[field] = mean(values) if values else None
                row[f"{name}_failure_auc"] = failure_auc(selected, field)
            row["first_step_failure_auc"] = failure_auc(selected, "first_structured_residual")
            summaries.append(row)
    return summaries


def paired_comparisons(runs: list[dict], threshold: float) -> list[dict]:
    """Compare independent from-start runs, NOT actual mid-trajectory model switches."""
    rows = []
    models = sorted({r["model"] for r in runs})
    for split in sorted({r["split"] for r in runs}):
        lookup = {m: {r["task_id"]: r for r in runs if r["model"] == m and r["split"] == split} for m in models}
        for base, peer in itertools.permutations(models, 2):
            if set(lookup[base]) != set(lookup[peer]):
                raise ValueError(f"Unpaired task set: {split}, {base}, {peer}")
            for group in ("all", "low_first_residual", "high_first_residual", "no_executed_step"):
                tasks = []
                for task_id, run in lookup[base].items():
                    residual = run["first_structured_residual"]
                    category = "no_executed_step" if residual is None else (
                        "high_first_residual" if residual >= threshold else "low_first_residual")
                    if group == "all" or category == group:
                        tasks.append(task_id)
                if not tasks:
                    continue
                base_success = mean(lookup[base][t]["success"] for t in tasks)
                peer_success = mean(lookup[peer][t]["success"] for t in tasks)
                rows.append({"split": split, "base_model": base, "peer_model": peer, "group": group,
                             "threshold": threshold, "paired_tasks": len(tasks),
                             "base_success_rate": base_success, "peer_success_rate": peer_success,
                             "peer_minus_base_success": peer_success - base_success})
    return rows


def export(trace_root: Path, analysis_path: Path, config_path: Path, output: Path) -> dict:
    paths = sorted(trace_root.glob("*/shadow_*/traces.jsonl"))
    if not paths:
        raise ValueError(f"No shadow traces under {trace_root}")
    config = load_simple_yaml(config_path)
    analysis = json.loads(analysis_path.read_text())
    weight = float(analysis["selected_semantic_weight"])
    steps, runs = collect_steps(paths)
    if not steps:
        raise ValueError("No executed steps")
    semantic, backend = embedding_residuals(steps, config["residual"]["embedding_model"], require_model=True)
    _apply_scores(steps, semantic, weight, config["shadow"]["minimum_expectation_quality"])
    summarize_runs(steps, runs)
    summary = {
        "analysis_kind": "post_collection_descriptive_model_association",
        "label_use": "No annotation labels used for outcomes or model comparisons. Fusion weight is inherited from the historical label-based analysis, not refitted.",
        "runs": len(runs), "executed_steps": len(steps), "models": len({r["model"] for r in runs}),
        "unique_tasks": len({r["task_id"] for r in runs}), "semantic_backend": backend,
        "historical_semantic_weight": weight,
        "association_unit": "task within model and split; trajectory means weight tasks equally",
        "interpretation": "Failure AUC is retrospective association with evaluator task failure, not reasoning-necessity accuracy. First-step residual precedes the outcome but is not a validated router. Paired runs do not estimate the effect of switching a model mid-trajectory.",
        "by_model": model_summary(runs),
    }
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "step_scores.csv", [{k: v for k, v in s.items() if k not in {"expected_outcome", "observation"}} for s in steps])
    write_csv(output / "task_scores.csv", runs)
    write_csv(output / "model_comparisons.csv", paired_comparisons(runs, config["shadow"]["provisional_threshold"]))
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    annotation_snapshot = {
        "source": str(analysis_path),
        "source_sha256": hashlib.sha256(analysis_path.read_bytes()).hexdigest(),
        "target": "slow_reasoning_needed",
        "scope": "historical held-out annotation association, not online reliability",
        "label_provenance": "Owner reports six-person review of AI-assisted labels; separate per-reviewer records unavailable.",
        "qualification": "Original judge prompt omitted compact-code meanings. Its impact on reviewed labels is unmeasured; label-based findings are provisional.",
        "methods": [{"method": row["method"], "binary_items": row["n"],
                     "roc_auc": row["roc_auc"],
                     "roc_auc_ci95": row.get("confidence_intervals", {}).get("roc_auc"),
                     "cluster_count": row.get("confidence_intervals", {}).get("cluster_count")}
                    for row in analysis["test_metrics"]],
        "incremental_value": analysis["incremental_value"],
    }
    (output / "annotation_association.json").write_text(json.dumps(annotation_snapshot, indent=2) + "\n")
    inputs = paths + [analysis_path, config_path, Path(__file__),
                     Path("src/phase1_5_shadow/residual.py"),
                     Path("src/phase1_5_shadow/analyze_annotations.py"),
                     Path("src/phase1_residual/residual_calculator.py")]
    cache_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    cache_ref = cache_home / "hub/models--sentence-transformers--all-MiniLM-L6-v2/refs/main"
    manifest = {"inputs": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                "cached_embedding_revision": cache_ref.read_text().strip() if cache_ref.exists() else None}
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path, default=Path("results/phase1_5_shadow/full"))
    parser.add_argument("--analysis", type=Path, default=Path("results/phase1_5_shadow/analysis_ai_blind/analysis.json"))
    parser.add_argument("--config", type=Path, default=Path("config/phase1_5_shadow.yaml"))
    parser.add_argument("--output", type=Path, default=Path("evidence/residual_diagnostics"))
    args = parser.parse_args()
    result = export(args.trace_root, args.analysis, args.config, args.output)
    print(json.dumps({key: result[key] for key in ("runs", "executed_steps", "models", "historical_semantic_weight")}, indent=2))


if __name__ == "__main__":
    main()
