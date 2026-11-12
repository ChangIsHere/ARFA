# Phase 2 Paper Baseline Report

Standard ReAct baseline only. No residual gate or fast/slow routing is enabled.

## Main Results

| Model | Tasks | Success | 95% CI | Mean reward | Calls/task | Tokens/task | Model latency/task (s) | Wall/task (s) | Max-step rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| arfa-qwen2.5-coder:7b-8k | 200 | 0.260 | [0.204, 0.325] | 0.749 | 5.28 | 6429.4 | 45.54 | 47.29 | 0.260 |
| arfa-llama3.1:8b-8k | 200 | 0.255 | [0.200, 0.320] | 0.745 | 3.98 | 3707.7 | 54.76 | 58.69 | 0.055 |
| arfa-qwen2.5-coder:14b-8k | 200 | 0.400 | [0.335, 0.469] | 0.810 | 3.29 | 3305.1 | 83.14 | 85.70 | 0.090 |
| arfa-llama3.2:3b-8k | 200 | 0.180 | [0.133, 0.239] | 0.705 | 6.12 | 7610.0 | 22.25 | 24.37 | 0.160 |
| arfa-gemma3:4b-8k | 200 | 0.260 | [0.204, 0.325] | 0.759 | 3.02 | 3955.1 | 20.37 | 22.98 | 0.050 |
| arfa-gemma3:12b-8k | 200 | 0.315 | [0.255, 0.382] | 0.782 | 4.11 | 5424.0 | 83.91 | 86.61 | 0.100 |
| arfa-phi4-mini:3.8b-8k | 200 | 0.125 | [0.086, 0.178] | 0.693 | 3.72 | 5535.1 | 25.15 | 26.46 | 0.195 |
| arfa-phi4:14b-8k | 200 | 0.255 | [0.200, 0.320] | 0.753 | 3.14 | 4356.9 | 93.09 | 94.80 | 0.090 |

## Paired Model Comparisons

Differences use the same 200 tasks, with a deterministic 10,000-sample paired bootstrap CI and a two-sided exact McNemar test.

| Model A | Model B | A-B success | 95% paired CI | A-only | B-only | McNemar p |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| arfa-qwen2.5-coder:7b-8k | arfa-llama3.1:8b-8k | 0.005 | [-0.060, 0.065] | 22 | 21 | 1 |
| arfa-qwen2.5-coder:7b-8k | arfa-qwen2.5-coder:14b-8k | -0.140 | [-0.205, -0.075] | 10 | 38 | 6.17e-05 |
| arfa-qwen2.5-coder:7b-8k | arfa-llama3.2:3b-8k | 0.080 | [0.010, 0.150] | 34 | 18 | 0.03648 |
| arfa-qwen2.5-coder:7b-8k | arfa-gemma3:4b-8k | 0.000 | [-0.065, 0.065] | 22 | 22 | 1 |
| arfa-qwen2.5-coder:7b-8k | arfa-gemma3:12b-8k | -0.055 | [-0.120, 0.010] | 17 | 28 | 0.1352 |
| arfa-qwen2.5-coder:7b-8k | arfa-phi4-mini:3.8b-8k | 0.135 | [0.065, 0.205] | 40 | 13 | 0.0002685 |
| arfa-qwen2.5-coder:7b-8k | arfa-phi4:14b-8k | 0.005 | [-0.065, 0.075] | 26 | 25 | 1 |
| arfa-llama3.1:8b-8k | arfa-qwen2.5-coder:14b-8k | -0.145 | [-0.210, -0.080] | 10 | 39 | 3.846e-05 |
| arfa-llama3.1:8b-8k | arfa-llama3.2:3b-8k | 0.075 | [0.010, 0.140] | 30 | 15 | 0.0357 |
| arfa-llama3.1:8b-8k | arfa-gemma3:4b-8k | -0.005 | [-0.075, 0.065] | 25 | 26 | 1 |
| arfa-llama3.1:8b-8k | arfa-gemma3:12b-8k | -0.060 | [-0.125, 0.005] | 17 | 29 | 0.1038 |
| arfa-llama3.1:8b-8k | arfa-phi4-mini:3.8b-8k | 0.130 | [0.065, 0.195] | 39 | 13 | 0.0004095 |
| arfa-llama3.1:8b-8k | arfa-phi4:14b-8k | 0.000 | [-0.070, 0.070] | 26 | 26 | 1 |
| arfa-qwen2.5-coder:14b-8k | arfa-llama3.2:3b-8k | 0.220 | [0.155, 0.290] | 51 | 7 | 2.402e-09 |
| arfa-qwen2.5-coder:14b-8k | arfa-gemma3:4b-8k | 0.140 | [0.065, 0.210] | 43 | 15 | 0.0003069 |
| arfa-qwen2.5-coder:14b-8k | arfa-gemma3:12b-8k | 0.085 | [0.025, 0.145] | 28 | 11 | 0.009475 |
| arfa-qwen2.5-coder:14b-8k | arfa-phi4-mini:3.8b-8k | 0.275 | [0.205, 0.345] | 62 | 7 | 4.103e-12 |
| arfa-qwen2.5-coder:14b-8k | arfa-phi4:14b-8k | 0.145 | [0.080, 0.210] | 39 | 10 | 3.846e-05 |
| arfa-llama3.2:3b-8k | arfa-gemma3:4b-8k | -0.080 | [-0.145, -0.015] | 15 | 31 | 0.0259 |
| arfa-llama3.2:3b-8k | arfa-gemma3:12b-8k | -0.135 | [-0.200, -0.070] | 11 | 38 | 0.000142 |
| arfa-llama3.2:3b-8k | arfa-phi4-mini:3.8b-8k | 0.055 | [0.000, 0.115] | 23 | 12 | 0.08953 |
| arfa-llama3.2:3b-8k | arfa-phi4:14b-8k | -0.075 | [-0.145, -0.010] | 17 | 32 | 0.04438 |
| arfa-gemma3:4b-8k | arfa-gemma3:12b-8k | -0.055 | [-0.125, 0.015] | 20 | 31 | 0.1608 |
| arfa-gemma3:4b-8k | arfa-phi4-mini:3.8b-8k | 0.135 | [0.080, 0.190] | 32 | 5 | 7.428e-06 |
| arfa-gemma3:4b-8k | arfa-phi4:14b-8k | 0.005 | [-0.065, 0.075] | 26 | 25 | 1 |
| arfa-gemma3:12b-8k | arfa-phi4-mini:3.8b-8k | 0.190 | [0.125, 0.255] | 44 | 6 | 3.244e-08 |
| arfa-gemma3:12b-8k | arfa-phi4:14b-8k | 0.060 | [-0.005, 0.130] | 30 | 18 | 0.1114 |
| arfa-phi4-mini:3.8b-8k | arfa-phi4:14b-8k | -0.130 | [-0.190, -0.070] | 8 | 34 | 6.877e-05 |

## Protocol Diagnostics

Recovered format steps contain extractable JSON wrapped in extra text; hard parse failures contain no usable JSON object.

| Model | Recovered format | Hard parse | Request failures | Nonzero exits | Repeated actions |
| --- | ---: | ---: | ---: | ---: | ---: |
| arfa-qwen2.5-coder:7b-8k | 39 | 92 | 0 | 128 | 425 |
| arfa-llama3.1:8b-8k | 0 | 110 | 0 | 122 | 120 |
| arfa-qwen2.5-coder:14b-8k | 600 | 41 | 0 | 50 | 138 |
| arfa-llama3.2:3b-8k | 0 | 237 | 0 | 347 | 239 |
| arfa-gemma3:4b-8k | 565 | 39 | 0 | 129 | 88 |
| arfa-gemma3:12b-8k | 684 | 136 | 1 | 92 | 103 |
| arfa-phi4-mini:3.8b-8k | 650 | 93 | 1 | 176 | 314 |
| arfa-phi4:14b-8k | 517 | 111 | 0 | 73 | 97 |

## Gold-Healthy Sensitivity Analysis (195 tasks)

This subset requires the released gold command to exit zero and achieve self-reward 1.00 in the pinned environment.

| Model | Tasks | Success | 95% CI | Mean reward | Calls/task | Tokens/task | Model latency/task (s) | Wall/task (s) | Max-step rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| arfa-qwen2.5-coder:7b-8k | 195 | 0.267 | [0.210, 0.333] | 0.755 | 5.24 | 6341.7 | 45.19 | 46.69 | 0.256 |
| arfa-llama3.1:8b-8k | 195 | 0.262 | [0.205, 0.327] | 0.748 | 3.98 | 3719.4 | 54.89 | 58.78 | 0.056 |
| arfa-qwen2.5-coder:14b-8k | 195 | 0.410 | [0.344, 0.480] | 0.813 | 3.25 | 3238.5 | 81.51 | 84.00 | 0.087 |
| arfa-llama3.2:3b-8k | 195 | 0.185 | [0.136, 0.245] | 0.707 | 6.02 | 7468.9 | 22.11 | 24.12 | 0.149 |
| arfa-gemma3:4b-8k | 195 | 0.267 | [0.210, 0.333] | 0.761 | 3.02 | 3942.0 | 20.35 | 22.77 | 0.051 |
| arfa-gemma3:12b-8k | 195 | 0.323 | [0.261, 0.392] | 0.786 | 4.08 | 5359.4 | 83.22 | 85.83 | 0.097 |
| arfa-phi4-mini:3.8b-8k | 195 | 0.128 | [0.088, 0.182] | 0.697 | 3.77 | 5657.2 | 25.57 | 26.79 | 0.200 |
| arfa-phi4:14b-8k | 195 | 0.262 | [0.205, 0.327] | 0.755 | 3.18 | 4444.1 | 94.35 | 95.97 | 0.092 |

## Completeness Gate

Every model must contain the exact same 200 task IDs and matching protocol hashes before these results are treated as paper-ready.

- `arfa-qwen2.5-coder:7b-8k`: PASS (200/200; issues: none)
- `arfa-llama3.1:8b-8k`: PASS (200/200; issues: none)
- `arfa-qwen2.5-coder:14b-8k`: PASS (200/200; issues: none)
- `arfa-llama3.2:3b-8k`: PASS (200/200; issues: none)
- `arfa-gemma3:4b-8k`: PASS (200/200; issues: none)
- `arfa-gemma3:12b-8k`: PASS (200/200; issues: none)
- `arfa-phi4-mini:3.8b-8k`: PASS (200/200; issues: none)
- `arfa-phi4:14b-8k`: PASS (200/200; issues: none)

Overall matrix: **PASS** (8/8 models).
