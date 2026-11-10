# Phase 3 Handoff Ablation

This development experiment measures the information lost when routing switches
between Qwen 7B and Gemma 12B. It runs residual routing on the same 100 tasks in
three conditions:

- `full_visible`: the recent visible model responses and terminal observations;
- `structured_state`: the current plan, action, expected outcome, expected
  signals, committed continuation, and terminal observation;
- `observation_only`: the task and latest terminal observation, without the
  previous model response or structured commitment.

The handoff condition changes only when the selected model role changes. A model
that retains the same role receives the normal bounded visible history. Models,
router threshold, prompts, task split, step cap, concurrency, and evaluator are
otherwise fixed. This is a development ablation and does not replace the frozen
mixed-pair final results.

## Results

All three modes completed the same 100 development tasks and passed task-set,
provenance, config-hash, and request-error audits.

| Handoff mode | Success | Wall time/task | Tokens/task | Slow calls/task |
| --- | ---: | ---: | ---: | ---: |
| Full visible history | 29% | 481.7 s | 4,642 | 2.29 |
| Structured state | 28% | 379.6 s | 3,644 | 2.43 |
| Observation only | 31% | 576.2 s | 4,767 | 3.39 |

Structured state differed from full visible history by -1 success point
(95% task-bootstrap CI: -6 to +4 points) while reducing wall time by 21.2% and
tokens by 21.5%. The 95% confidence intervals for both cost reductions remained
positive. Compared with observation-only handoff, structured state reduced wall
time by 34.1%, tokens by 23.6%, and slow-model calls by 28.3%, with a -3 point
success difference whose confidence interval included zero.

These development results show that cross-model routing does not require hidden
reasoning tokens. A compact explicit state can preserve the measured task
success while reducing context and inference cost. Passing only the latest
observation is not a free simplification: it required more model calls and more
slow-model use in this study.

`analysis.json` contains paired bootstrap intervals and exact McNemar tests.
Full task-level traces remain under `results/phase3_arfa/handoff/v1/`.
