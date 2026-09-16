# Constraints And Evidence Boundaries

Residual is usable as an auxiliary decision signal in a Phase 3 experiment. The
limitations below constrain the strength of conclusions, not the ability to build
and test an agent that consults this signal. Partial offline association should
not be equated with validated online routing reliability.

## Current Gate Policy

After inspecting the initial offline results, the project now treats Phase 1/1.5
as a diagnostic study rather than a deployment qualification test. The original
strict acceptance thresholds are retired as a development prerequisite. The
already-defined lighter engineering gate remains the permission to build a guarded
Phase 3 controller. Its stored pass is engineering readiness, not a new positive
scientific result or a retroactive preregistration.

The reason for this distinction is scope: offline classifier precision and coverage
do not determine the end-to-end cost of skipped reasoning, recovery, and task
success. Those quantities require an online comparison. This does not establish
that high precision is impossible, or explain away individual failures.

Historical strict thresholds and failed outcomes remain reproducible in the frozen
`config/phase1_5_shadow.yaml`, `docs/phase1_5_protocol.md`, and local
`results/phase1_5_shadow/analysis_ai_blind/analysis.json`. Do not change the original
freeze manifest or claim the old criterion passed. Current configuration is
`config/phase3_arfa.yaml`; it does not read the strict acceptance result.

## Signal Limitations

- The legacy external-trajectory Phase 1 constructed expectations after collection.
  Its heuristic labels and residual features may overlap. It is supplementary.
- Shadow expectations were committed before execution, but shadow candidates were
  never actually used to skip reasoning. A matched expectation can still encode a
  wrong plan, miss a hidden error, or provide an unusable continuation.
- The historical strict selection produced no feasible fast operating point under
  its simultaneous requirements. That is a limitation of the evaluated combination
  of score, labels, and operating constraints, not proof of safe routing.
- Historical label-based full AUC was 0.743, compared with 0.749 for learned
  observation-only. A gain over expectation-gate-only does not prove incremental
  value over a strong observation baseline. Semantic fusion selected weight zero.
- New task-outcome associations are modest and model-dependent. Whole-trajectory
  residual averages include information obtained after the first decision. They
  are not online predictors available at the beginning of the task.

## Annotation Provenance And Prompt Defect

The project owner reports that six people independently divided and reviewed the
AI-assisted labels. Record this as owner-reported human-reviewed AI-assisted data.
Separate per-reviewer original files are not available. Divided review is not
automatically overlapping independent labeling; stored formal agreement statistics
measure AI-judge agreement, not human inter-annotator agreement. The earlier 23+6
human pilot remains separate and cannot certify the full packet.

Code inspection found that the AI annotator's compact output fields `i/e/r/c/v`
and value codes were not explained to the judge in its original prompt/schema.
The decoder knew their meaning, but that does not show the judge interpreted them
correctly. The prompt now explicitly maps all fields and codes and future batch
audits record prompt/schema hashes. No old label or audit was overwritten.

The extent of this defect's effect on the reviewed packet is unmeasured. Therefore
existing label-dependent AUC, agreement, and the lighter gate's numeric inputs are
provisional, even though their stored calculation is retained. The new structured,
semantic, and observation-only task-outcome associations use no subjective labels.
The historical fusion weight is retained separately and is label-derived.

Before relying on label-based conclusions, use an identifiable human adjudicated
subset or a fresh corrected-judge sensitivity analysis. This is not a blocker for
building an online experiment with objective task outcomes.

## Online Scope

The current continuation validity check is a coarse string heuristic, not a safety
proof. For example, a pipeline can still have missing stdin. Phase 3 needs actual
continuation execution, recovery, cost accounting, and paired controls. Planned
one-fast-step/fallback settings do not mean those mechanisms already exist.

Phase 2 is complete, not universal validation: three locally quantized models,
one run per task/model, one benchmark. Baseline task uncertainty is not independent
run-to-run uncertainty. The reserved Phase 3 tasks were excluded from residual
fitting but were already evaluated in Phase 2. Keep that distinction in the paper.
