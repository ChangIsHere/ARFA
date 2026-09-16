# Phase 1.5 Exploratory Engineering Gate

This is a historical, label-dependent engineering-readiness snapshot. A compact-code
AI-annotation prompt defect was subsequently found; its impact on reviewed labels
is unmeasured. Numeric label-based results remain provisional. See
[current constraints](../../docs/residual_constraints.md) and
[task-outcome diagnostics](../../docs/residual_diagnostics.md). This qualification
does not rewrite the stored result as a different experiment.

Engineering gate: **PASS**.

Engineering readiness for a guarded exploratory Phase 3 pilot. This is not a deployment-safety or confirmatory residual-evidence gate.

## Evidence Summary

- Formal shadow collection: 300/300 runs
- Complete blind annotation items: 751
- Full residual held-out ROC-AUC: 0.743
- Full residual gain over expectation-gate-only: 0.124
- Phase 2 always-reason baseline: 600/600 runs

## Review Provenance

The project owner reports that six people independently divided and reviewed the AI-assisted labels. The current repository preserves the AI judge audit and the owner-reported human-review status. Separate raw reviewer files are not present, so the reported agreement statistics remain inter-model rather than six-reviewer inter-annotator agreement.

## Interpretation

The engineering gate is intentionally scoped to deciding whether the data, signal, and baseline infrastructure justify a guarded online controller pilot. It does not require the offline shadow router to satisfy deployment-level precision and coverage simultaneously, because Phase 3-P directly measures end-to-end task success, recovery, and reasoning-call savings under a fail-open controller.

The stricter formal evidence gate remains recorded as a diagnostic result. Its failure is interpreted as evidence that the first offline threshold is too coarse for direct deployment, not as an engineering blocker for a conservative exploratory pilot.

## Unlock Status

- Phase 3-P exploratory pilot: **UNLOCKED**
- Phase 3 formal evaluation: **LOCKED**
