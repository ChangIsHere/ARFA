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
