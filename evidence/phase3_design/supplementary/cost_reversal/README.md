# Fast-Model Cost Reversal

This analysis uses the selected development traces. It tests whether a
nominally smaller model can increase total cost by taking more reasoning
steps or emitting more tokens per call.

| Pair | Fast calls/task vs slow | Fast tokens/call vs slow | Fast max-step | Slow max-step | Residual calls/task vs slow |
| --- | ---: | ---: | ---: | ---: | ---: |
| llama | +26.6% | +23.4% | 12.0% | 2.0% | +13.0% |
| phi | +37.5% | +3.3% | 3.0% | 3.0% | +4.0% |

Paired task differences (fast-only minus slow-only):

| Pair | Metric | Mean difference | 95% bootstrap CI |
| --- | --- | ---: | ---: |
| llama | model_calls | +1.27 | [+0.52, +2.02] |
| llama | tokens_per_call | +235.96 | [+60.94, +427.27] |
| llama | repeated_actions | +0.14 | [-0.06, +0.34] |
| llama | parse_errors | +0.01 | [-0.03, +0.05] |
| llama | max_steps | +0.10 | [+0.03, +0.17] |
| phi | model_calls | +1.12 | [+0.52, +1.72] |
| phi | tokens_per_call | -1.47 | [-78.99, +77.59] |
| phi | repeated_actions | +0.42 | [+0.25, +0.59] |
| phi | parse_errors | +0.31 | [+0.06, +0.65] |
| phi | max_steps | +0.00 | [-0.05, +0.05] |

Positive values mean more work than always-slow. Failed-task calls and
prompt/completion token splits are retained in `policy_metrics.csv` and
`role_metrics.csv`.
