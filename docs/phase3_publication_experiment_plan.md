# ARFA Publication Experiment Plan

## Goal

Test whether execution residual improves fast/slow model selection, identify
which model pairs benefit, and check whether the result transfers beyond the
development benchmark.

## Existing Evidence

- Phase 1.5 contains the offline residual analysis.
- Phase 2 characterizes eight standalone models on the same 200 tasks.
- Phase 3 compares four native fast/slow pairs under residual, always-slow, and
  always-fast routing on 100 shared development tasks.
- Gemma 4B/12B has the clearest development tradeoff. Qwen, Llama, and Phi show
  that reduced slow-model use does not always reduce total trajectory cost.
- The reserved 100-task Phase 3 final split has not been used.

## Stage 1: Within-Family Routing Controls (complete)

The Gemma 4B/12B and Qwen 7B/14B development matrices are complete:

- residual;
- learned observation-only;
- expectation-gate-only;
- random routing at a matched fast-call budget;
- the completed always-slow and always-fast endpoints.

All online controls use the same tasks, model artifacts, prompt, evaluator,
context, concurrency, and step cap within a pair. Residual beat the two simple
controls in both pairs, but did not clearly beat learned observation-only
routing.

## Stage 2: Cost-Reversal Analysis (complete)

Trace-level analysis of Llama and Phi measures calls per task, tokens per call,
repeated actions, parse errors, and max-step rates. A nominally smaller model
can cost more by extending the trajectory. Fast-model action quality is
therefore part of the model-pair selection problem.

## Stage 3: Mixed Pairs

Hold one side of the pair fixed while changing the other. Measure how
fast-model latency, schema adherence, success, and recovery work affect the
residual tradeoff. This stage explains the pair dependence observed in the
four-family study.

Short development qualifications may reject pairs that cannot keep both models
resident, produce excessive parse errors, or make the nominal fast model slower
than the slow model. Pair qualification does not use final tasks.

## Stage 4: Freeze

Freeze model hashes, prompts, parser, residual formula, threshold, fast-call
budget, random seed, task IDs, step cap, Docker images, GPU concurrency, and
analysis code. Store their hashes in a manifest.

Primary comparisons are:

- residual success and cost versus always-slow;
- residual success and cost versus learned observation-only;
- residual success versus random routing at a matched achieved fast-call rate.

## Stage 5: InterCode Final

Evaluate the frozen policies once on all 100 reserved tasks. Use identical task
IDs, hardware class, concurrency, prompt, context, and evaluator. Report paired
bootstrap intervals and exact McNemar tests. Do not retune after viewing the
final results.

## Stage 6: External Benchmark

Evaluate the frozen representative pair on a second terminal or coding-agent
benchmark using its official harness. A complete benchmark run is preferable
to an arbitrary sample-size target. Report both direct threshold transfer and,
if needed, a separately labeled benchmark-specific calibration.

## Stage 7: Ablations And Stability

Use development or separate validation data for residual-component ablations,
threshold sensitivity, task-category analysis, repeated runs on a fixed subset,
and measured GPU energy per task and per successful task.

## Intended Claim

The intended claim is conditional: execution residual can improve fast/slow
model selection when the fast model is genuinely faster and capable enough to
avoid extra retries and recovery work. The four-pair study establishes this
dependence; the Gemma controls isolate the residual signal; mixed pairs explain
the model-selection mechanism; and the second benchmark tests transfer.
