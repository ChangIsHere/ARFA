# Phase 3 ARFA Dual Track

The guarded Phase 3-P engineering pilot is unlocked after the Phase 1.5 exploratory engineering gate and the complete Phase 2 baseline review. The separate formal Phase 3 evaluation remains locked.

Planned ARFA design:

```text
if safety_trigger:
    slow_track
elif residual > threshold:
    slow_track
elif valid_next_plan_exists:
    fast_track
else:
    slow_track
```

Phase 3-P uses development-only calibration, permits at most one consecutive fast step, and fails open to slow reasoning. It must not consume the reserved `phase3_final` tasks. The pilot exists to measure end-to-end behavior that an offline threshold cannot fully establish: task success, recovery after an incorrect fast decision, reasoning-call reduction, token use, and latency.

```bash
bash scripts/phase1_5_evaluate_exploratory_gate.sh
bash scripts/phase3_run_exploratory.sh
```

The strict Phase 1.5 evidence gate remains useful as a diagnostic: the first offline threshold was too coarse to satisfy deployment-level safety and coverage simultaneously. The exploratory gate instead establishes that the pipeline is complete and that the residual has sufficient ranking signal to justify a protected online pilot.
