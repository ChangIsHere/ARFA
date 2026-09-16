# Residual Evidence

Current evidence for the combined Phase 1/1.5 diagnostic study.

- `step_scores.csv`: 790 executed steps; numeric residuals, gates, checks, and trace keys.
- `task_scores.csv`: 300 runs; task-level means, first-step residual, evaluator outcome.
- `summary.json`: model/split summaries, denominators, and descriptive failure AUCs.
- `annotation_association.json`: existing reviewed-label associations, baseline
  comparisons, intervals, denominators, and provenance qualifications.
- `model_comparisons.csv`: same-task model comparisons, with fixed first-residual groups.
- `source_manifest.json`: input hashes and locally cached MiniLM revision.

Regenerate from repository root: `bash scripts/phase1_export_diagnostics.sh`.
Full text stays in the original local traces referenced in the CSV, avoiding a
second copy of the dataset. Do not mix step, deduplicated-item, and task denominators.

[Formulas and interpretation](../../docs/residual_diagnostics.md) explain the
distinction between raw mismatch, safety gates, and hypothetical shadow decisions.
[Constraints](../../docs/residual_constraints.md) describe label limitations and
the retired strict gate. These exports are diagnostics, not new agent runs.
