# Phase 1 Experiment Protocol

## Objective

Validate whether execution residual has predictive utility for deciding whether a terminal-based coding agent requires renewed reasoning after an execution step.

## Data

The final Phase 1 dataset is derived from external InterCode Bash trajectories downloaded from the official Princeton NLP InterCode repository.

The initial 50-record dataset is preserved only as `pilot_50`.

## Splits

Split by `task_id` using a fixed seed:

- 60% development
- 20% validation
- 20% held-out test

Thresholds are selected on validation only. Held-out test metrics must not be used for retuning.

## Threshold Policy

The primary threshold maximizes predicted fast-path rate subject to validation false-fast rate no greater than 5%.

Secondary operating points are also reported at 1%, 5%, and 10%.

## Metrics

Report accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, PR-AUC, false-fast count/rate, false-slow count/rate, fast-path rate, safe-fast precision, and confusion matrix.

Bootstrap confidence intervals are computed at the task level.

## Acceptance Gate

Phase 2 remains locked unless the selected residual method passes the provisional evidence gate described in the final report.
