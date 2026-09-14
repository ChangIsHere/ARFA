# Phase 2 Baseline Agent

Phase 2 builds the standard terminal-based ReAct baseline. It does not contain ARFA fast/slow routing and does not use the residual gate for control.

## Purpose

The baseline answers the question:

```text
How many model calls, tokens, and seconds does a normal always-reason ReAct agent need on the same terminal tasks?
```

This creates the comparison point for Phase 3 ARFA-Min.

## Current Implementation

Phase 2 now includes:

- an OpenAI-compatible chat client for local model servers
- a standard ReAct loop that calls the model after every observation
- structured JSON agent outputs with `action` and `expected_outcome`
- an isolated local development sandbox and Docker-backed official environment
- the complete 200-task released InterCode NL2Bash suite
- per-task traces, summary metrics, and CSV metrics
- checkpoint/resume, gold-environment validation, subgroup analysis, and completeness gates
- a scripted dry-run validator for harness testing only

The 8-task development fixture is:

```text
data/phase2/intercode_bash_local_tasks.jsonl
```

The task instructions are derived from external InterCode Bash queries, while the local fixture makes them reproducibly executable without depending on Docker during harness development.

The frozen paper test manifest is:

```text
data/phase2/intercode_nl2bash_official_200.jsonl
```

It contains all 200 released tasks at InterCode commit `c3e46d827cfc9d4c704ec078f7abf9f41e3191d8`: 60/53/60/27 tasks from filesystem versions 1/2/3/4. ARFA uses the complete suite as its frozen Phase 2 evaluation set; it is not described as an upstream official test split and was not used for prompt tuning.

## Running The Harness Validator

```bash
bash scripts/phase2_run_baseline.sh --dry-run
```

Dry-run output is not paper evidence. It only checks that the runner, sandbox, evaluator, and logging work.

## Running A Real Local Model

Start an OpenAI-compatible local model server, then run:

```bash
bash scripts/phase2_run_baseline.sh
```

The default config targets Ollama's OpenAI-compatible endpoint:

```text
base_url: http://localhost:11434/v1
model_name: arfa-qwen2.5-coder:7b-8k
```

Frozen paper model matrix:

```text
arfa-qwen2.5-coder:7b-8k
arfa-llama3.1:8b-8k
arfa-qwen2.5-coder:14b-8k
```

Paper-usable Phase 2 results must come from a real model endpoint, not from `--dry-run`.

## Paper Protocol

The Phase 2 paper run is a standard always-reason ReAct baseline. It has no residual score, routing threshold, fast path, small-model handoff, or ARFA control decision. Each model uses temperature 0, at most 12 reasoning calls per task, the same structured prompt, and a fresh pair of network-disabled containers per task. One container runs the agent and the other runs the official gold command.

The reward reimplements the three released InterCode Bash components: filesystem diff, changed-file content, and final observation TF-IDF similarity. A task succeeds at reward 1.00. Runs are checkpointed after every task and can be resumed without rerunning completed task IDs. The main analysis retains all 200 tasks and also reports a strict gold-health sensitivity subset based on the environment validation output.

Build and validate the four filesystem images before model evaluation:

```bash
bash scripts/phase2_build_intercode_images.sh
bash scripts/phase2_validate_official_environment.sh
```

Run the full serial model matrix:

```bash
bash scripts/phase2_run_paper_matrix.sh
```

The matrix is paper-complete only when all three model directories contain exactly 200 traces and `analysis/completeness.json` reports `complete: true`.

## Final Status

The matrix is complete: 3 models x 200 tasks = 600 runs, and the completeness gate passes. Qwen2.5-Coder 7B achieves 26.0% task success, Llama 3.1 8B achieves 25.5%, and Qwen2.5-Coder 14B achieves 40.0%. The compact tracked evidence package for paper writing is:

```text
docs/phase2_writing_bundle/README_FOR_WRITING.md
```

## Outputs

Generated outputs are ignored by git:

```text
results/phase2_baseline/local_react/traces.jsonl
results/phase2_baseline/local_react/summary.json
results/phase2_baseline/local_react/metrics.csv
```

Paper-run outputs are written under:

```text
results/phase2_baseline/paper/
```
