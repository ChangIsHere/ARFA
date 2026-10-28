# Native Model-Pair Results

These are the four selected Phase 3 development results. Every policy used the
same 100 InterCode NL2Bash tasks within a pair. Changes below compare residual
routing with always-slow.

| Pair | Residual success | Slow success | Fast success | Slow calls | Wall time | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen 7B/14B | 41% | 41% | 31% | -40.9% | -4.9% | -1.4% |
| Gemma 4B/12B | 29% | 26% | 21% | -48.1% | -30.6% | -30.6% |
| Llama 3B/8B | 19% | 16% | 8% | -22.6% | +13.8% | +24.8% |
| Phi Mini/14B | 33% | 30% | 23% | -28.4% | +9.8% | +2.5% |

Qwen and Gemma are the positive cases. Llama and Phi are the contrasting cases:
both routed fewer calls to the slow model and improved the success point
estimate, but total cost increased. This is the main evidence that routing
outcomes depend on the chosen fast/slow model pair.

Gemma threshold `0.30` was selected on the development tasks. Phi threshold
`0.15` is the highest-success diagnostic setting and did not pass the saved
efficiency selection rule. The reserved final split remains unopened.

`results.csv` contains the machine-readable table. Each pair directory contains
the selected summaries, 100 task-level records per policy, paired analysis, and
runtime metadata.
