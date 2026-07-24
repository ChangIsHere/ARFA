# Phase 1 Method

Phase 1 evaluates residual methods on external InterCode Bash trajectories.

Residual methods:

- always-reason baseline
- never-reason baseline
- keyword baseline
- TF-IDF lexical residual
- embedding residual
- structured residual
- hybrid residual

Threshold selection uses validation data only. The primary operating point maximizes fast-path rate subject to validation false-fast rate no greater than 5%.

Held-out test data is evaluated only after thresholds are frozen.
