# Phase 3 Final Evidence

The Qwen and Gemma protocols are frozen at source commit `eecee51`. Each
manifest hashes the model configuration, task manifest and split, routing code,
prompt, evaluator, analysis, and bounded NRP runner.

The completed final matrix contains six policies on the reserved 100-task split:

- residual;
- learned observation-only;
- expectation gate-only;
- budget-matched random;
- always slow;
- always fast.

Both Qwen and Gemma completed all 600 policy-task runs with zero model-request
failures. Compact paper-facing outputs are stored in `qwen/` and `gemma/`:

- `policy_summary.csv`: policy-level success, calls, tokens, and wall time;
- `analysis.json`: paired bootstrap intervals and exact McNemar tests;
- `gpu_summary.txt`: run-level GPU monitoring;
- `runtime_preflight.json` and `runtime_provenance.txt`: environment records.

Key results against always-slow are:

| Pair | Success change | Wall-time reduction | Slow-call reduction | Token reduction | McNemar p |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen 7B/14B | +1 pp | 14.4% | 41.3% | 6.3% | 1.0000 |
| Gemma 4B/12B | +9 pp | 21.5% | 43.5% | 12.7% | 0.0225 |

Run one frozen pair with:

```bash
bash deployment/nrp/run_phase3_final.sh qwen
bash deployment/nrp/run_phase3_final.sh gemma
```

The runner verifies the manifest before requesting a GPU, rejects task limits
and threshold overrides, checks all 100 task IDs after every policy, and removes
the Kubernetes Job on exit.

GPU summaries cover the full matrix rather than isolated policies. They should
not be interpreted as causal per-policy energy measurements.
