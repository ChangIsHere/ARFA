# Phase 1 Leakage Audit

The residual methods must not access:

- `reasoning_needed`
- `label_reason`
- `annotation_confidence`
- manually assigned post-hoc failure categories
- any field created after annotation

Allowed controller-visible inputs:

- pre-execution expected outcome
- action
- raw terminal observation
- exit code or valid-action status observable through execution
- observable file, test, or environment state

Automated leakage findings are written to `results/phase1_residual/final/leakage_audit.md`.
