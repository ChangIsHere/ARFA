# Results Summary

## Phase 1 Full InterCode Bash Run

Current status: complete enough for a workshop preliminary-study section, but not enough for final ARFA efficiency claims.

Data:

```text
execution steps: 10,500
binary-labeled steps: 6,087
ambiguous steps retained: 4,413
distinct tasks: 200
distinct trajectories: 1,648
source result files: 33
embedding backend: sentence-transformers/all-MiniLM-L6-v2
```

Hybrid residual held-out test:

```text
F1: 0.849
PR-AUC: 0.919
false-fast rate: 0.032
fast-path rate: 0.070
safe-fast precision: 0.671
evidence gate: not passed
```

Interpretation:

Residual appears useful as a reasoning-necessity signal on external terminal trajectories, but the current policy is too conservative or insufficiently precise for strong fast-path deployment. The next work should improve annotation quality, expectation quality, and routing policy before making Phase 3 claims.

Writing bundle:

```text
docs/phase1_writing_bundle/README_FOR_WRITING.md
```
