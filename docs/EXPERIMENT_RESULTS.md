# ARFA Experiment Results

This is the numerical source for the workshop draft. Percentage changes are
reported from the residual policy's perspective. For cost metrics, a negative
change means residual used less time, fewer tokens, or fewer slow-model calls.
Success changes are absolute percentage points.

Absolute latency is comparable only within a matched matrix. Development pairs
ran on different hardware, so their raw seconds must not be pooled.

## Experiment Inventory

| Study | Design | Task-level runs |
| --- | --- | ---: |
| Shadow diagnostics | 3 models x 100 tasks | 300 |
| Standalone models | 8 models x 200 tasks | 1,600 |
| Native-pair development | 4 pairs x 3 policies x 100 tasks | 1,200 |
| Reserved Qwen/Gemma evaluation | 2 pairs x 6 policies x 100 tasks | 1,200 |
| Cross-family evaluation | 2 pairs x 5 policies x 100 tasks | 1,000 |
| Handoff ablation | 3 modes x 100 tasks | 300 |
| **Total** |  | **5,600** |

## Offline Residual Diagnostics

The shadow study contains 300 trajectories and 790 executed terminal steps.
Against 313 non-ambiguous reviewed reasoning-demand labels:

| Score | ROC-AUC | Task-bootstrap 95% CI |
| --- | ---: | ---: |
| Expectation gate only | 0.619 | [0.564, 0.684] |
| Observation heuristic | 0.732 | [0.671, 0.792] |
| Raw residual | 0.739 | [0.685, 0.791] |
| Full residual | 0.743 | [0.681, 0.811] |
| Learned observation only | 0.749 | [0.670, 0.821] |

Full residual is associated with reviewed reasoning demand and improves over the
expectation gate by 0.124 AUC. It does not outperform learned observation-only;
their observed difference is -0.006 with 95% CI [-0.087, 0.082]. This motivates
the online controls rather than proving the router offline.

Source: [`evidence/residual_diagnostics/`](../evidence/residual_diagnostics/).

## Standalone Model Characterization

Each model ran the same 200 InterCode NL2Bash tasks.

| Model | Success | Tokens/task | Wall time/task |
| --- | ---: | ---: | ---: |
| Qwen 7B | 26.0% | 6,429 | 47.3 s |
| Qwen 14B | 40.0% | 3,305 | 85.7 s |
| Llama 3B | 18.0% | 7,610 | 24.4 s |
| Llama 8B | 25.5% | 3,708 | 58.7 s |
| Gemma 4B | 26.0% | 3,955 | 23.0 s |
| Gemma 12B | 31.5% | 5,424 | 86.6 s |
| Phi Mini | 12.5% | 5,535 | 26.5 s |
| Phi 14B | 25.5% | 4,357 | 94.8 s |

These runs characterize candidate models. They are not the causal routing
baseline because the matched Phase 3 policies use a different prompt and loop.

Source: [`docs/phase2_writing_bundle/`](phase2_writing_bundle/README_FOR_WRITING.md).

## Native-Pair Development Study

Each pair ran residual, always-slow, and always-fast on the same 100 development
tasks. Cost changes below compare residual with always-slow.

| Pair | Residual success | Slow success | Fast success | Wall time | Tokens | Slow calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen 7B/14B | 41% | 41% | 31% | -4.9% | -1.4% | -40.9% |
| Gemma 4B/12B | 29% | 26% | 21% | -30.6% | -30.6% | -48.1% |
| Llama 3B/8B | 19% | 16% | 8% | +13.8% | +24.8% | -22.6% |
| Phi Mini/14B | 33% | 30% | 23% | +9.8% | +2.5% | -28.4% |

Residual reduced slow-model calls in every pair. Only Qwen and Gemma reduced
end-to-end cost. This result motivated the frozen Qwen/Gemma evaluation and the
cost-reversal analysis.

Source: [`evidence/phase3_design/native_pairs/`](../evidence/phase3_design/native_pairs/).

## Reserved Qwen Evaluation

All six policies completed the same 100 reserved tasks. Residual success was 42%.

| Residual compared with | Success change | Wall time | Tokens | Slow calls | McNemar p |
| --- | ---: | ---: | ---: | ---: | ---: |
| Always slow | +1 pp | -14.4% | -6.3% | -41.3% | 1.0000 |
| Always fast | +8 pp | +42.5% | +0.5% | n/a | 0.1153 |
| Learned observation only | +1 pp | +0.3% | -1.4% | +2.9% | 1.0000 |
| Expectation gate only | 0 pp | +3.1% | +7.7% | +17.1% | 1.0000 |
| Budget-matched random | +3 pp | -6.2% | -4.1% | +0.6% | 0.5078 |

Qwen supplies stable efficiency evidence against always-slow. It does not show a
clear advantage over learned observation-only or expectation-gate-only.

## Reserved Gemma Evaluation

All six policies completed the same 100 reserved tasks. Residual success was 37%.

| Residual compared with | Success change | Wall time | Tokens | Slow calls | McNemar p |
| --- | ---: | ---: | ---: | ---: | ---: |
| Always slow | +9 pp | -21.5% | -12.7% | -43.5% | 0.0225 |
| Always fast | +14 pp | +70.3% | +8.3% | n/a | 0.0094 |
| Learned observation only | +7 pp | +22.3% | +1.7% | +4.3% | 0.0654 |
| Expectation gate only | +11 pp | +1.7% | -6.1% | +8.3% | 0.0074 |
| Budget-matched random | +7 pp | -7.2% | -8.1% | -6.7% | 0.0923 |

Gemma is the strongest native result. Residual improves success and cost over
always-slow and random routing. Against learned observation-only, the residual
success advantage is larger but costs are similar or higher.

Sources: [`evidence/phase3_final/`](../evidence/phase3_final/).

## Cross-Family Model Pairs

### Qwen 7B / Gemma 12B

Residual success was 37%.

| Residual compared with | Success change | Wall time | Tokens | Slow calls |
| --- | ---: | ---: | ---: | ---: |
| Always slow | +4 pp | -33.5% | -25.8% | -49.5% |
| Learned observation only | -1 pp | -11.0% | -12.8% | -11.3% |
| Budget-matched random | +3 pp | -15.4% | -17.4% | -13.8% |

This is the strongest cost result. Residual keeps success near the best policy
while reducing time, tokens, and slow calls.

### Gemma 4B / Qwen 14B

Residual success was 39%.

| Residual compared with | Success change | Wall time | Tokens | Slow calls |
| --- | ---: | ---: | ---: | ---: |
| Always slow | -3 pp | -29.8% | -7.3% | -43.3% |
| Learned observation only | -4 pp | -3.6% | +0.1% | +4.9% |
| Budget-matched random | -2 pp | -4.3% | -8.8% | 0.0% |

This pair saves time but loses success. Together with the stronger reverse pair,
it shows that parameter count alone does not define a useful fast model.

Source: [`evidence/phase3_mixed_pairs/`](../evidence/phase3_mixed_pairs/).

## Fast-Model Cost Reversal

Relative to always-slow, Llama fast-only used 26.6% more calls per task and
23.4% more tokens per call. Its max-step rate increased from 2% to 12%.

Phi fast-only used 37.5% more calls per task. Repeated actions increased from
0.10 to 0.52 per task and parse errors from 0 to 0.31 per task.

The smaller model can therefore increase total cost by generating extra work.
This explains why reducing slow calls did not guarantee lower time or tokens.

Source: [`evidence/phase3_design/supplementary/cost_reversal/`](../evidence/phase3_design/supplementary/cost_reversal/).

## Cross-Model Handoff Ablation

The Qwen 7B/Gemma 12B residual agent ran three handoff modes on the same 100
development tasks.

| Handoff mode | Success | Calls/task | Slow calls/task |
| --- | ---: | ---: | ---: |
| Full visible history | 29% | 3.56 | 2.29 |
| Structured state | 28% | 3.67 | 2.43 |
| Observation only | 31% | 5.11 | 3.39 |

Structured state versus full visible history:

- success: -1 pp, 95% CI [-6, +4];
- wall time: -21.2%;
- tokens: -21.5%;
- slow calls: +6.1%.

Structured state versus observation only:

- success: -3 pp, with a confidence interval that includes zero;
- wall time: -34.1%;
- tokens: -23.6%;
- model calls: -28.2%;
- slow calls: -28.3%.

Full visible history versus observation only:

- success: -2 pp;
- wall time: -16.4%;
- tokens: -2.6%;
- model calls: -30.3%;
- slow calls: -32.4%.

The compact structured state preserves the measured success level while reducing
context cost. Observation alone is insufficient: the receiving model performs
more reasoning and invokes the slow model more often.

Source: [`evidence/phase3_handoff/`](../evidence/phase3_handoff/).

## Supported Result

Execution residual can guide fast/slow model selection, but the benefit depends
on both model identities. The strongest configurations preserve or improve task
success while reducing slow-model calls by 41-50%, wall time by 14-34%, and
tokens by 6-26% relative to always-slow.

The current evidence does not show that residual universally beats learned
observation-only routing. All completed task experiments use one benchmark, so
cross-benchmark transfer remains untested.
