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

Original completed model matrix:

```text
arfa-qwen2.5-coder:7b-8k
arfa-llama3.1:8b-8k
arfa-qwen2.5-coder:14b-8k
```

The expansion adds five single-model baselines on the **same 200 task IDs**:

```text
arfa-llama3.2:3b-8k
arfa-gemma3:4b-8k
arfa-gemma3:12b-8k
arfa-phi4-mini:3.8b-8k
arfa-phi4:14b-8k
```

The planned pairs are Qwen2.5-Coder 7B/14B, Llama 3.2 3B with Llama 3.1 8B,
Gemma 3 4B/12B, and Phi-4 Mini 3.8B/Phi-4 14B. The Llama pair crosses model
releases; the other three pairs stay within named model families.
This is a model-comparison extension, not yet a Phase 3 routing result.

Paper-usable Phase 2 results must come from a real model endpoint, not from `--dry-run`.

## Paper Protocol

The Phase 2 paper run is a standard always-reason ReAct baseline. It has no residual score, routing threshold, fast path, small-model handoff, or ARFA control decision. Each model uses temperature 0, at most 12 reasoning calls per task, the same structured prompt, and a fresh pair of containers per task. The recorded network mode is `bridge`. One container runs the agent and the other runs the official gold command.

The reward reimplements the three released InterCode Bash components: filesystem diff, changed-file content, and final observation TF-IDF similarity. A task succeeds at reward >= 0.99. Runs are checkpointed after every task and can be resumed without rerunning completed task IDs. The main analysis retains all 200 tasks and also reports a strict gold-health sensitivity subset based on the environment validation output.

Build and validate the four filesystem images before model evaluation:

```bash
bash scripts/phase2_build_intercode_images.sh
bash scripts/phase2_validate_official_environment.sh
```

Download the five new base models with `ollama pull`, then create their 8K
aliases with `bash scripts/phase2_create_ollama_models.sh --expansion-only`. Run the new
baselines serially, leaving the three completed runs untouched:

```bash
bash scripts/phase2_run_paper_matrix.sh --expansion-only
```

The expansion has finished and its detached `screen` session has exited normally.
Each model's completed-task count is the line count of its
`results/phase2_baseline/paper/<model>/traces.jsonl` file. If the computer
restarts, run the same script again; `--resume` checks the model, task IDs,
and recorded source hashes before continuing. Do not launch two copies of
the matrix at once.

The expanded matrix is paper-complete only when all eight model directories
contain the **same 200 unique task IDs**, the recorded prompt/agent/environment
hashes and protocol fields agree, and `analysis/completeness.json` reports
`complete: true`. The original runs used Ollama 0.34.0; the expansion uses
0.34.1. Their saved config-file hashes differ, so that difference is retained
as a provenance note even when the recorded protocol fields match.

## Final Status

The full matrix is complete: 8 models x 200 tasks = 1,600 attempts, with the same
200 unique task IDs per model. Completion does not mean every task was solved.
Success rates range from 12.5% to 40.0%; failed tasks remain part of the results.
All recorded protocol checks pass. Gold validation identifies 195 healthy tasks,
reported separately alongside the full 200-task analysis.

The [writing package](phase2_writing_bundle/README_FOR_WRITING.md) contains all
eight results, four figures, paired confidence intervals, failure diagnostics,
and source hashes. One run per model/task is available. The reported McNemar
p-values are unadjusted; latency and tokens do not constitute measured energy.

Preflight is stored separately under `results/phase2_baseline/preflight/` and
must not be counted as paper results. Llama 3.2 3B and Gemma 3 4B/12B completed
one task each. Gemma replies required recoverable JSON extraction. DeepSeek-R1
was rejected during preflight because explicit reasoning exhausted the fixed
700-token response budget. DeepSeek-Coder 6.7B was then excluded after its first
20 formal-prefix tasks all failed and most repeatedly emitted non-JSON responses,
leading to its replacement for runtime and protocol reasons after evaluation
had begun. This is a post-observation model-selection decision, not a
preregistered exclusion. Those traces remain under
`results/phase2_baseline/preflight/`; the formal fourth pair uses Phi-4 models.
The extra model tags and sizes are listed by
[Ollama](https://ollama.com/library/llama3.2),
[Gemma 3](https://ollama.com/library/gemma3), and
[Phi-4 Mini](https://ollama.com/library/phi4-mini), and
[Phi-4](https://ollama.com/library/phi4).

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
