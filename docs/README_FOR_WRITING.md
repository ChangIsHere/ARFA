# Writing Guide

## Main Idea

Residual compares an agent's expected command result with the actual output. It
can serve as a reference for fast/slow reasoning decisions. The current studies
examine that signal; Phase 3 will test its effect on task success and cost.

## What To Use

1. **Phase 1/1.5:** start with [residual formulas and results](residual_diagnostics.md).
   There are 300 shadow runs and 790 executed steps. Expectations were written
   before execution, but no reasoning calls were skipped.
2. **Label comparison:** full residual AUC is 0.743, raw residual AUC is 0.739,
   and learned observation-only AUC is 0.749 on 313 binary test items from 40 tasks.
   These use human-reviewed, AI-assisted labels and remain provisional; see the
   [study notes](residual_constraints.md). The numeric snapshot is in
   `evidence/residual_diagnostics/annotation_association.json`.
3. **Phase 2:** use the [baseline package](phase2_writing_bundle/README_FOR_WRITING.md)
   for its 1,600-run results (eight models, 200 shared tasks), tables, and figures. Its task set and prompt differ
   from the shadow study.
4. **Phase 3:** use the [implementation plan](04_phase3_arfa_dual_track.md) for the
   proposed experiment. Results have not been collected yet.

## How The Studies Fit Together

Phase 1 used existing external trajectories to explore residual scoring. It is
preserved as supplementary history in `archive/phase1_writing_bundle/`.
Phase 1.5 collected live shadow trajectories: the agent wrote an expectation
before executing each action, then the system recorded the observed output and
residual. The agent continued to call its model at every step, so this study
measures associations rather than the benefit of online routing.

Phase 2 measures eight single-model ReAct agents on the same 200 terminal tasks.
The loop is instruction, model response, shell action, observation, and another
model response, until completion or the 12-step limit. The evaluator compares
the resulting state and output with the released gold command. All 1,600
attempts are recorded, including unsuccessful attempts. The shared budget is
8K context, temperature zero, and 700 output tokens per call.

Phase 3 will use residual as one input to selecting a fast or slow model. The
four available baseline pairs are Qwen 7B/14B, Llama 3B/8B, Gemma 4B/12B, and
Phi Mini 3.8B/Phi 14B. The existing Phase 3 design is Qwen-first; the other pairs
are available for extensions, not already collected routing experiments. If
Phase 3 changes prompts, execution rules, or budgets, it needs matched single-
model baselines under those settings. The historical Phase 2 results alone
cannot isolate the effect of residual routing.

## Latest Phase 2 Update

The baseline matrix grew from three models and 600 attempts to eight models and
1,600 attempts. Qwen 14B has the highest observed success rate, 40%; Gemma 4B
averages 22.98 seconds per task at 26% success. Pair differences and their
uncertainty are in the baseline package. These results establish comparison
points for routing; they do not establish that ARFA saves time or energy.

DeepSeek-R1 and DeepSeek-Coder were considered before the Phi pair. R1 exhausted
the response budget, and the Coder 6.7B run was stopped after 20 attempted tasks
with protocol failures and long runtimes. The replacement followed observed
results and must remain described as such. The
[Phase 2 protocol](03_phase2_baseline_agent.md) records this selection history.

The current bundle adds five model summaries, all eight model configurations,
updated figures and tables, and `metrics/task_outcomes.jsonl` with one compact
record per attempt. It removes an excluded DeepSeek summary from the formal
bundle. Formatting recovery, hard JSON failure, and request errors are reported
separately. Two request-error records lack complete cost accounting; see the
bundle's scope notes before interpreting token or call totals.

## What Is Published

GitHub includes the implementation, prompts, configs, model definitions, the
200-task manifest, per-task compact outcomes, aggregate statistics, diagnostic
examples, source hashes, and PNG/PDF figures plus Markdown/LaTeX tables. These
support drafting and reanalysis of task-level outcomes.

Full conversations, downloaded external repositories, annotation working files,
model weights, caches, and terminal logs stay local. A trace hash identifies a
local source file; it does not make that file publicly downloadable. Rebuilding
all trace-level diagnostics requires those traces or a new model run. Historical
reports and frozen protocols are retained for provenance and should not be
mistaken for the current result set.

Use `main` for code and evidence and `paper` for manuscript drafts. Record the
`main` commit used by a draft. The manuscript branch is not automatically
updated when evidence changes.

## Draft Wording

> We study execution residuals as a reference for fast/slow reasoning decisions.
> In the shadow study, residual distributions differed across models and showed
> modest associations with task failure. The full score reached AUC 0.743
> (95% CI 0.681-0.811) against the existing human-reviewed, AI-assisted labels.
> This label comparison is preliminary and did not show an advantage over the
> learned observation-only baseline. We next test whether using residual in an
> online agent improves the balance between task success and reasoning cost.

The [study notes](residual_constraints.md) collect annotation details, previous
selection criteria, and the scope of these results. Keep those details with the
experiment description rather than repeating them throughout the draft.

Current numeric exports are in `evidence/residual_diagnostics/`. Earlier Phase 1
figures are supplementary and stored in `docs/archive/phase1_writing_bundle/`.
Full raw data remains local. Record the code commit used for each paper version.
