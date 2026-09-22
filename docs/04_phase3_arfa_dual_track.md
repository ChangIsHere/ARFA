# Phase 3 Plan

The proposed Qwen 14B/7B comparison, success margin, and cost-per-success analysis are
recorded in [the pre-run study protocol](phase3_study_protocol.md). Verify task
coverage and the historical baselines with `bash scripts/phase3_study_preflight.sh`.

Phase 3 will test whether using residual to decide when to reason again improves
the balance between task success and cost. Development can proceed under the
lighter engineering check; the previous strict check is described in
[study notes](residual_constraints.md).

## Current Code

The agent, router, fast track, and slow track are placeholders.
`bash scripts/phase3_run_arfa.sh` reports settings and readiness, not a live run.

The planned fast path executes a command committed before seeing the observation.
Calling a small fast model instead is another design choice. That version would
need to count the small model's calls and tokens as well.

## Next Steps

1. Generate an action, expected result, and possible next command before execution.
   Check that the next command has its required inputs.
2. Allow one fast step before returning to reasoning. Use reasoning when the
   command is missing, the state is uncertain, or an error occurs.
3. Calibrate on development tasks. Start with structured residual and test the
   semantic score separately.
4. Compare always-reason, observation-only, commitment-check-only, and residual
   policies using the same prompt and budget. Keep the original Phase 2 results
   as a reference, and run a matching always-reason arm for this experiment.
5. Record task success, calls, tokens, time, and recovery after wrong fast decisions.
   Include the cost of generating expectations and checking residuals.

Keep the reserved `phase3_final` tasks out of development. Fix the controller and
comparison settings before evaluating them. `config/phase3_arfa.yaml` is the
single current configuration.
