# Phase 1 Leakage Audit

Residual feature extraction uses only expected outcome, action, raw observation, inferred exit code, and observable command text.

Forbidden label columns referenced by method score columns: `[]`

Known risks:

- Expectations are heuristic annotations derived from action text because InterCode trajectories do not store pre-execution expectations.
- Exit codes are inferred from InterCode valid-action metadata and terminal text, not directly recorded shell exit statuses.
- Thresholds are selected on validation only, then frozen for held-out test metrics.
- The dataset uses external InterCode trajectories, but labels remain an ARFA annotation layer and need human review.

Data sources: `{'InterCode Bash external trajectories, Princeton NLP intercode repository': 10500}`
