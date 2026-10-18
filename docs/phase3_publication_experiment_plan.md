# ARFA Publication Experiment Plan

## Goal

Build evidence that execution residual improves fast/slow model selection, not
merely that two models can be mixed. Development data is used for design and
calibration. The reserved final split is evaluated once after the protocol is
frozen.

## Current Position

- Phase 1.5 contains the offline residual study.
- Phase 2 contains 1,600 standalone model runs.
- Phase 3 development contains matched 100-task results for Qwen, Llama, Gemma,
  and Phi.
- Qwen and Gemma show an efficiency benefit. Llama and Phi reduce slow-model
  calls without reducing total trajectory cost.
- The 100-task final split has not been used.

The current material is sufficient for a workshop draft. The stages below are
needed for a stronger confirmatory paper.

## Stage 1: Mixed-Pair Qualification

Purpose: determine whether the Llama and Phi outcomes are caused by residual
routing or by the original fast models.

Candidates:

- Gemma 3 4B -> Llama 3.1 8B;
- Gemma 3 4B -> Phi-4 14B.

Each candidate uses the same 20 development tasks under residual, slow-only,
and fast-only policies. A candidate advances only when:

- all three cells contain 20 unique tasks;
- model request errors are zero;
- fast-model parse errors are at most 5% of calls;
- fast-model latency per call is at least 25% below the slow model;
- residual success is within one task of slow-only;
- residual total calls are no more than 110% of slow-only.

The criteria are implemented in `src/phase3_arfa/qualify_pair.py`. Qualification
does not select a residual threshold and does not access final tasks.

## Stage 2: Residual-Specific Controls

First run Qwen on the shared 100 development tasks with matched controls, then
extend to Gemma and any qualified mixed pair. The
[control-extension protocol](phase3_control_extension.md) contains the exact
first-round setup and its current status.

- `always_slow`;
- `always_fast`;
- `residual`;
- `observation_only`;
- `learned_observation_only`;
- `expectation_gate_only`;
- `random_budget`;
- `fixed_budget`.

The learned observation-only controller is especially important because its
offline AUC is not lower than the full residual's AUC. Random and fixed
policies use development-selected target budgets; their *achieved* fast-call
rates must be checked before claiming budget matching. These comparisons test
whether expectation-observation residual adds value beyond observation-only
scoring, expectation-quality gating, or generic use of a smaller model.

## Stage 3: Freeze

Freeze model hashes, prompts, parser, residual formula, threshold, fast-call
budget, random seeds, task IDs, maximum steps, Docker images, GPU concurrency,
analysis code, and acceptance criteria. Store their hashes in a manifest.

Primary comparisons:

- residual success versus matched slow-only;
- residual wall time and energy versus slow-only;
- residual success versus random and observation-only at matched fast-call
  budgets.

## Stage 4: InterCode Final

Run the frozen policies once on all 100 reserved tasks. Every policy uses the
same task IDs, hardware class, concurrency, prompt, context, and evaluator.
Report paired bootstrap intervals and exact McNemar tests. Do not retune after
viewing final results.

## Stage 5: External Benchmark

Evaluate the two strongest pairs on a second coding-agent benchmark with its
official harness and a declared task count. Terminal-Bench 2.0 has 89 tasks,
so a complete run is more informative than imposing an arbitrary 100-task
minimum. This is the main external-validity step for a regular paper. The
second benchmark does not reuse InterCode thresholds unless threshold
transfer is itself the stated test.

## Stage 6: Ablations And Stability

On development or separate validation data, measure:

- residual without expectation text;
- semantic-only and structured-only residual;
- threshold and fast-call-budget sensitivity;
- task-category effects;
- repeated runs on a fixed 30-50 task subset;
- measured GPU energy per task and per successful task.

## Claim Boundary

The intended claim is conditional:

> Execution residual can improve fast/slow model selection when the fast model
> is genuinely faster, follows the action schema, and avoids recovery work.

Negative model pairs remain part of the evidence because they identify the
conditions under which reduced slow-model use fails to produce end-to-end
savings.
