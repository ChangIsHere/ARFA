# ARFA Journal Experiment Plan

## Current Evidence

The project now has four layers of evidence:

1. Offline diagnostics on 300 shadow trajectories show that execution residual
   is associated with reviewed reasoning-demand labels, but does not outperform
   learned observation-only scoring offline.
2. Phase 2 characterizes eight standalone models on the same 200 InterCode
   NL2Bash tasks.
3. Four native fast/slow development pairs show that reducing slow calls does
   not always reduce total cost. Qwen and Gemma benefit; Llama and Phi generate
   enough extra work to offset part of the saving.
4. Frozen Qwen and Gemma final matrices compare six policies on 100 reserved
   tasks. Gemma provides the strongest result; Qwen mainly improves efficiency
   against always-slow and does not separate from the stronger routing controls.
5. Two cross-family pairs complete the 2x2 Qwen/Gemma model-selection matrix.
   Qwen 7B/Gemma 12B reduces time, tokens, and slow calls; Gemma 4B/Qwen 14B
   shows that parameter count alone does not define a useful fast track.
6. A three-mode handoff ablation shows that structured state preserves the
   measured success level while reducing context cost relative to full visible
   history. Observation-only handoff requires more slow-model calls.

The journal extension should explain when residual helps, demonstrate transfer,
and replace run-level infrastructure monitoring with policy-level energy data.

## Stage 1: Mixed-Pair Factorial

Cross the two useful fast models and two useful slow models:

| Fast model | Slow model | Status |
| --- | --- | --- |
| Qwen 7B | Qwen 14B | complete native pair |
| Gemma 4B | Gemma 12B | complete native pair |
| Gemma 4B | Qwen 14B | complete cross-family pair |
| Qwen 7B | Gemma 12B | complete cross-family pair |

Run residual, learned observation-only, budget-matched random, always-slow, and
always-fast. Use development tasks only for qualification and threshold choice.
Freeze each accepted pair before evaluation. This 2x2 design tests whether the
gain follows the fast model, the slow model, or their interaction.

Primary outputs:

- success and paired success difference;
- slow-call rate, tokens, wall time, and wall time per successful task;
- schema failures, repeated actions, and trajectory length;
- interaction analysis across fast and slow model identity.

## Stage 2: Handoff Robustness

Status: complete on the 100-task development split. Full visible history,
structured state, and observation-only handoff use the same Qwen 7B/Gemma 12B
pair and residual controller. The structured state condition retains explicit
plan and expectation information without assuming access to hidden reasoning.

## Stage 3: External Benchmark Transfer

First add a pinned Terminal-Bench release because it changes both tasks and
environment while preserving terminal interaction. Run the Qwen 7B/Gemma 12B
pair with the same five policies. Report direct threshold transfer before any
benchmark-specific calibration.

Then add a software-repair benchmark using its official harness. Start with a
fixed SWE-bench Verified pilot to validate the adapter, followed by the complete
feasible evaluation set. Keep benchmark-specific engineering separate from the
routing rule so that residual is the experimental variable.

## Stage 4: Stability And Energy

Repeat a stratified fixed subset at least three times for the representative
pair and four main policies: residual, learned observation-only, random, and
always-slow. Keep GPU type, concurrency, model residency, and task order fixed.

Measure per policy:

- GPU joules and average power after subtracting an idle baseline;
- energy per task and per successful task;
- wall time, tokens, calls, and peak memory;
- between-run variance and confidence intervals.

Current Qwen and Gemma GPU summaries cover whole matrices and are operational
records, not policy-level energy evidence.

## Stage 5: Residual Ablations

Use the representative pair on development or a separate validation set. Compare:

- structured residual only;
- semantic residual only;
- full residual without the commitment gate;
- full residual;
- learned observation-only with the same training budget.

Match achieved fast-call rates where possible. The purpose is to identify which
part of residual contributes online value, not merely which variant routes more
often to the slow model.

## Stage 6: Final Analysis

Use task-level paired bootstrap intervals and exact McNemar tests. Report task
categories, failure modes, fast-call calibration, Pareto frontiers, and cost per
successful task. Do not pool absolute latency across hardware classes.

The target claim remains conditional:

> Execution residual improves fast/slow model selection when the fast model is
> sufficiently capable and genuinely cheaper over the full trajectory. The
> benefit depends on both model identities and transfers imperfectly across
> tasks and benchmarks.

## Recommended Order

1. Complete the two cross-family pairs. Done.
2. Complete the handoff ablation. Done.
3. Freeze and run a pinned Terminal-Bench transfer.
4. Add repeated policy-level energy measurements.
5. Run the residual-component ablations.
6. Validate the SWE-bench adapter and scale the external evaluation.

The first three items are the minimum high-value extension beyond the current
workshop-level evidence. The full sequence is the journal target.
