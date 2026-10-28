# Phase 3 Final Protocol

The Qwen and Gemma protocols are frozen at source commit `2673f89`. Each
manifest hashes the model configuration, task manifest and split, routing code,
prompt, evaluator, analysis, and bounded NRP runner.

The final matrix contains six policies on the reserved 100-task split:

- residual;
- learned observation-only;
- expectation gate-only;
- budget-matched random;
- always slow;
- always fast.

Run one pair with:

```bash
bash deployment/nrp/run_phase3_final.sh qwen
bash deployment/nrp/run_phase3_final.sh gemma
```

The runner verifies the manifest before requesting a GPU, rejects task limits
and threshold overrides, checks all 100 task IDs after every policy, and removes
the Kubernetes Job on exit.
