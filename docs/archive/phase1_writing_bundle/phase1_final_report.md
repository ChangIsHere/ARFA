# Phase 1 Final Report

## Status

This is the external-data Phase 1 experiment. It tests whether residual is useful as a reasoning-trigger signal; it does not prove residual is necessary.

## Data Source

The final dataset is derived from external InterCode Bash trajectories downloaded from the official Princeton NLP InterCode repository.

- Records: 10500
- Binary records: 6087
- Ambiguous records retained: 4413
- Distinct tasks: 200
- Distinct trajectories: 1648
- Class distribution: `{'false': 1477, 'ambiguous': 4413, 'true': 4610}`
- Embedding backend: `sentence-transformers/all-MiniLM-L6-v2`

## Threshold Policy

Thresholds are selected on the validation split by maximizing predicted fast-path rate subject to a false-fast constraint. The primary constraint is 5%.

Hybrid primary validation threshold: `0.426432`

## Held-Out Test Metrics

| Method | Accuracy | Balanced Acc. | F1 | PR-AUC | False-fast | Fast-path | Safe-fast precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| always_reason | 0.727 | 0.500 | 0.842 | 0.863 | 0.000 | 0.000 | n/a |
| never_reason | 0.273 | 0.500 | 0.000 | 0.863 | 1.000 | 1.000 | 0.273 |
| keyword | 0.727 | 0.500 | 0.842 | 0.942 | 0.000 | 0.000 | n/a |
| tfidf | 0.708 | 0.487 | 0.829 | 0.702 | 0.026 | 0.019 | 0.000 |
| embedding | 0.700 | 0.483 | 0.823 | 0.623 | 0.038 | 0.028 | 0.030 |
| structured | 0.727 | 0.500 | 0.842 | 0.977 | 0.000 | 0.000 | n/a |
| hybrid | 0.750 | 0.570 | 0.849 | 0.919 | 0.032 | 0.070 | 0.671 |

## Evidence Gate

Gate passed: `False`

- Hybrid beats always/never/keyword/TF-IDF on F1 or PR-AUC: True
- Held-out false-fast rate <= 5%: True
- Fast-path rate >= 15%: False
- Safe-fast precision >= 95%: False
- No leakage source identified in the automated audit: review_required

## Limitations

- Expectations are generated from action text before reading each observation, because InterCode logs do not contain agent-written expectations.
- Labels are an ARFA annotation layer over external trajectories and need human review before manuscript claims.
- InterCode Bash is useful for terminal interaction, but it is not a full replacement for later coding-agent benchmarks such as SWE-bench Lite.

Phase 2 remains locked until this report, the leakage audit, and the acceptance gate are reviewed.
