# ARFA Writing Handoff

## Central Statement

Execution residuals can be used as an auxiliary decision signal for fast/slow
reasoning-model invocation. The offline diagnostics identify model-conditioned
patterns and preliminary associations with task outcomes and reviewed
reasoning-necessity labels, motivating an online evaluation in Phase 3.

This is the current supported positioning, not a claim that ARFA already controls
an agent reliably, reduces costs, or selects the best model.

## Evidence Chain

1. **Combined Phase 1/1.5: residual diagnostics.** Three models, 100 shared tasks,
   300 shadow runs, 790 executed steps. Expectations precede execution. Use
   [formulas and model results](residual_diagnostics.md) and the numeric artifacts
   in `evidence/residual_diagnostics/`. The original downloaded-trajectory Phase 1
   is supplementary and lives in `docs/archive/phase1_writing_bundle/`.
2. **Reviewed-label association.** The historical full-score AUC is 0.743 with
   task-bootstrap 95% CI 0.681-0.811 on 313 binary test items from 40 tasks. Raw
   residual AUC is 0.739. Cite these as associations with owner-reported
   human-reviewed AI-assisted labels, with the prompt/provenance qualification.
   They are not independent-human gold validation. The tracked snapshot includes
   observation-only and expectation-gate-only comparisons to avoid selective claims.
3. **Phase 2: reference cost and task success.** The unchanged
   [600-run baseline package](phase2_writing_bundle/README_FOR_WRITING.md) provides
   methods, paired comparisons, tables, and figures. Its task set and prompt differ
   from the shadow collection; do not directly attribute cross-phase differences
   to residual control.
4. **Phase 3: online evaluation to perform.** The implementation is still a
   scaffold. Test task success and total reasoning cost under matched controls.
   Current fast continuation is not an already deployed small fast model.

## Suggested Preliminary-Study Text

> Execution residuals provide a usable auxiliary signal for investigating
> fast/slow reasoning decisions. In our shadow study, residual distributions
> differed across model conditions and were modestly associated with task failure.
> Against the existing human-reviewed, AI-assisted reasoning-necessity labels, the
> full residual score achieved a held-out ROC-AUC of 0.743 (task-bootstrap 95% CI:
> 0.681-0.811). These findings motivate online evaluation rather than establish
> safe routing: the reviewed-label evidence remains provisional, and residual did
> not show a clear advantage over the learned observation-only baseline.

## Constraints To Include

Link [constraints](residual_constraints.md), retaining these points in the draft:
label provenance and the corrected AI prompt defect; limited transfer across
models; possible incorrect expectations/continuations; zero selected semantic
weight; no consistent advantage over strong observation-only controls; one
benchmark and no completed online ARFA experiment. Retired strict-gate outcomes
remain historical operating constraints, not rewritten successful experiments.

The lighter gate permits development. It is not a statistical proof of partial
reliability. A manuscript can use the completed methods, diagnostics, and baseline
now, but its Phase 3 results section must remain pending.

## Artifact Use

Use only current artifacts linked above as main results. Archived figures are
supplementary. Full traces, models, and raw labels remain local; compact evidence,
input hashes, source code, and the Phase 2 writing package are tracked on GitHub.
Record the code commit used by each manuscript version. Do not present the
repository as a public release of the full raw dataset.
