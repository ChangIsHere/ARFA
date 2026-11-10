# Gemma Routing Controls

This directory contains the compact evidence from the Gemma 4B/12B control
study on the 100-task InterCode development split. All four policies used the
same task IDs, model artifacts, prompt, evaluator, context length, and step
limit.

| Policy | Success | Tokens | Model calls | Fast-call rate |
| --- | ---: | ---: | ---: | ---: |
| Residual | 28/100 | 390,911 | 321 | 37.7% |
| Learned observation-only | 31/100 | 394,414 | 315 | 39.4% |
| Expectation-gate-only | 25/100 | 440,179 | 347 | 45.0% |
| Random-budget | 25/100 | 427,674 | 340 | 35.6% |

Residual routing used 11.2% fewer tokens than expectation-gate-only and 8.6%
fewer than random-budget, while completing three additional tasks in each
comparison. Learned observation-only completed three more tasks than residual
and used 0.9% more tokens. None of the paired success differences is
statistically significant on 100 tasks.

The observation-only run resumed after an infrastructure interruption. Its
success, token, and call counts cover all 100 tasks, but its wall time is not a
clean comparison with the other policies. Expectation-gate-only recorded one
model request error. These are development results. The later reserved-split
Gemma matrix reran all policies under a frozen, matched protocol and is the
primary paper result; see [`evidence/phase3_final/gemma/`](../../../phase3_final/gemma/).

- `policy_summary.csv`: compact paper table.
- `analysis.json`: paired estimates, bootstrap intervals, and McNemar tests.
- `*_summary.json` and `*_metrics.csv`: policy-level and task-level results.
- `runtime_preflight.json`, `runtime_provenance.txt`, and
  `gpu_environment.csv`: execution provenance.

Raw traces remain local under
`results/phase3_arfa/nrp/nrp-gemma-controls-v2/` and are not committed.
