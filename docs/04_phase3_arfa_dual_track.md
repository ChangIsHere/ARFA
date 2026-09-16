# Phase 3 Online Control

Controller development is permitted under the existing lighter engineering gate.
The historical strict offline gate is no longer a prerequisite. This policy
change permits an experiment; it does not establish residual effectiveness.

Residual is an auxiliary reference for deciding whether expensive reasoning should
be invoked. The current fast-path design executes a precommitted continuation; it
does not yet call a separate small model. If Phase 3 uses a small fast model and a
large slow model, that is a distinct implementation choice requiring its own
cost accounting and matched controls. Neither design is an existing online result.

## Implementation Status

`arfa_agent.py`, `router.py`, `fast_track.py`, and `slow_track.py` are placeholders.
`bash scripts/phase3_run_arfa.sh` only reports readiness and planned configuration.
Safety fallback and fast-step limits are configuration intentions, not yet tested
online behavior. No Phase 3 model run has started.

## Next Implementation

1. Produce an action, expectation, and executable continuation before observation.
   Implement continuation validity checks, including missing stdin across separate
   commands; the current string heuristic is not a safety guarantee.
2. Add a controller that can execute one precommitted continuation, then return to
   reasoning. Missing commitments, errors, or uncertain state require reasoning.
3. Calibrate on development tasks only. Start with structured residual; keep the
   semantic component as an ablation rather than assuming it improves the signal.
4. Compare always-reason, observation-only control, expectation-gate-only control,
   and residual control under the same continuation protocol and budget. Include
   an always-reason arm with the same prompt to isolate commitment overhead from
   routing. Phase 2 remains the original reference, not a substitute for that arm.
5. Record paired task success, model calls, tokens, wall time, and recovery after
   a wrong continuation. Count expectation, scoring, and recovery costs.

Use the same three models where practical; model-specific calibration is a
hypothesis to test. Model switching between trajectories is not the same as
reasoning on/off control and is not implemented or demonstrated by the diagnostics.

The reserved `phase3_final` tasks stay outside development. They were excluded
from residual fitting but were already run in Phase 2, so they are not globally
unseen. Final evaluation requires an implemented controller and a fixed comparison
protocol, not a retrospective declaration that an old gate passed.

## Configuration

`config/phase3_arfa.yaml` is the single current configuration. It retains the
lighter engineering-readiness result, permits development, and disallows using
reserved final tasks for the pilot. Do not run the old strict analyzer as a launch
requirement. Historical outcomes and remaining limitations are documented in
[constraints](residual_constraints.md).
