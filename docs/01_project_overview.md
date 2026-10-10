# Project Overview

ARFA asks whether an execution expectation-observation mismatch can inform the
decision to invoke a terminal agent's reasoning model again.

The current project has three connected stages:

1. Residual diagnostics, combining the original Phase 1 and Phase 1.5 collections.
   Document the signal and its model dependence without assuming a safe router.
2. Completed Phase 2 single-model characterization. Measure eight models'
   standalone success and cost, then identify four useful fast/slow pairs.
3. Phase 3 online control and primary comparison. Run residual, always-slow, and
   always-fast under the same prompt, tasks, hardware, and execution loop to test
   whether residual-guided selection improves the quality-efficiency tradeoff.

All eight Phase 2 model runs and all four Phase 3 development pairs are complete.
The reserved Phase 3 final split has not been opened. Current evidence supports a
workshop draft; residual-specific controls, confirmatory evaluation, and external
validation remain future work.

Phase 2 describes the available models; it does not isolate the effect of routing.
The matched Phase 3 controls are the baselines for causal claims about residual.

See [the current residual study](residual_diagnostics.md) and
[the Phase 3 study](04_phase3_arfa_dual_track.md). Old phase-numbered source paths
remain unchanged where they are needed for reproducibility.
