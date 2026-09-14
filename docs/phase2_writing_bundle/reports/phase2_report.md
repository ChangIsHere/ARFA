# Phase 2 Paper Baseline Report

Standard ReAct baseline only. No residual gate or fast/slow routing is enabled.

## Main Results

| Model | Tasks | Success | 95% CI | Mean reward | Calls/task | Tokens/task | Model latency/task (s) | Wall/task (s) | Max-step rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| arfa-qwen2.5-coder:7b-8k | 200 | 0.260 | [0.204, 0.325] | 0.749 | 5.28 | 6429.4 | 45.54 | 47.29 | 0.260 |
| arfa-llama3.1:8b-8k | 200 | 0.255 | [0.200, 0.320] | 0.745 | 3.98 | 3707.7 | 54.76 | 58.69 | 0.055 |
| arfa-qwen2.5-coder:14b-8k | 200 | 0.400 | [0.335, 0.469] | 0.810 | 3.29 | 3305.1 | 83.14 | 85.70 | 0.090 |

## Paired Model Comparisons

Differences use the same 200 tasks, with a deterministic 10,000-sample paired bootstrap CI and a two-sided exact McNemar test.

| Model A | Model B | A-B success | 95% paired CI | A-only | B-only | McNemar p |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| arfa-qwen2.5-coder:7b-8k | arfa-llama3.1:8b-8k | 0.005 | [-0.060, 0.065] | 22 | 21 | 1 |
| arfa-qwen2.5-coder:7b-8k | arfa-qwen2.5-coder:14b-8k | -0.140 | [-0.205, -0.075] | 10 | 38 | 6.17e-05 |
| arfa-llama3.1:8b-8k | arfa-qwen2.5-coder:14b-8k | -0.145 | [-0.210, -0.080] | 10 | 39 | 3.846e-05 |

## Protocol Diagnostics

Recovered format steps contain extractable JSON wrapped in extra text; hard parse failures contain no usable JSON object.

| Model | Recovered format | Hard parse | Nonzero exits | Repeated actions |
| --- | ---: | ---: | ---: | ---: |
| arfa-qwen2.5-coder:7b-8k | 39 | 92 | 128 | 425 |
| arfa-llama3.1:8b-8k | 0 | 110 | 122 | 120 |
| arfa-qwen2.5-coder:14b-8k | 600 | 41 | 50 | 138 |

## Gold-Healthy Sensitivity Analysis (195 tasks)

This subset requires the released gold command to exit zero and achieve self-reward 1.00 in the pinned environment.

| Model | Tasks | Success | 95% CI | Mean reward | Calls/task | Tokens/task | Model latency/task (s) | Wall/task (s) | Max-step rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| arfa-qwen2.5-coder:7b-8k | 195 | 0.267 | [0.210, 0.333] | 0.755 | 5.24 | 6341.7 | 45.19 | 46.69 | 0.256 |
| arfa-llama3.1:8b-8k | 195 | 0.262 | [0.205, 0.327] | 0.748 | 3.98 | 3719.4 | 54.89 | 58.78 | 0.056 |
| arfa-qwen2.5-coder:14b-8k | 195 | 0.410 | [0.344, 0.480] | 0.813 | 3.25 | 3238.5 | 81.51 | 84.00 | 0.087 |

## Completeness Gate

Every model must contain exactly 200 tasks from the released InterCode NL2Bash suite before these results are treated as paper-ready.

- `arfa-qwen2.5-coder:7b-8k`: PASS (200/200)
- `arfa-llama3.1:8b-8k`: PASS (200/200)
- `arfa-qwen2.5-coder:14b-8k`: PASS (200/200)

Overall matrix: **PASS** (3/3 models).
