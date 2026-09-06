# ARFA Phase 1 Writing Bundle

This bundle contains the compact Phase 1 evidence package for writing a workshop-style report or paper draft.

## What This Result Is Enough For

This result is enough to write a complete **Phase 1 experimental report / preliminary study section**:

- external data source
- dataset construction summary
- task-level splits
- residual methods
- validation-based threshold selection
- held-out test metrics
- ablation analysis
- subgroup analysis
- false-fast / false-slow discussion
- leakage and limitation discussion
- paper-ready tables and figures

This result is **not** enough to claim that ARFA already reduces token usage, latency, or large-model calls in a live coding agent, because Phase 2/3 agent execution has not been implemented or evaluated yet.

## Data Source

The Phase 1 full dataset is derived from external InterCode Bash trajectories from the official Princeton NLP InterCode repository.

Local external dataset path:

```text
data/phase1/external_intercode/
```

This path is ignored by git and is not included in this writing bundle.

Derived local JSONL dataset:

```text
data/phase1/final/intercode_bash_phase1.jsonl
```

This derived JSONL has 10,500 execution-step records and is also ignored by git to avoid committing generated data.

## Current Phase 1 Full-Data Summary

```text
Total execution steps: 10,500
Binary-labeled steps: 6,087
Ambiguous steps retained: 4,413
Distinct tasks: 200
Distinct trajectories: 1,648
InterCode source result files: 33
Embedding backend: sentence-transformers/all-MiniLM-L6-v2
```

Class distribution:

```text
reasoning_needed = true: 4,610
reasoning_needed = false: 1,477
ambiguous: 4,413
```

Task-level split record counts:

```text
development: 6,479
validation: 1,903
test: 2,118
```

Binary records used for primary metrics:

```text
development: 3,729
validation: 1,184
test: 1,174
```

## Main Held-Out Test Result

Hybrid residual:

```text
threshold: 0.426432
accuracy: 0.750
balanced accuracy: 0.570
F1: 0.849
PR-AUC: 0.919
false-fast rate: 0.032
fast-path rate: 0.070
safe-fast precision: 0.671
```

Evidence gate:

```text
Gate passed: false
```

Why the gate failed:

```text
false-fast <= 5%: passed
fast-path >= 15%: failed
safe-fast precision >= 95%: failed
```

## Safe Claims To Write

- Phase 1 evaluates whether execution residual is a useful reasoning-trigger signal.
- The experiment uses full external InterCode Bash trajectories available locally.
- Residual methods are evaluated under task-level development, validation, and held-out test splits.
- The hybrid residual achieves higher held-out F1 than simple baselines while keeping false-fast rate below 5%.
- The current method does not yet satisfy the full ARFA evidence gate because the fast-path rate and safe-fast precision remain insufficient.
- These findings motivate improved expectation generation, annotation review, and stronger routing criteria before implementing Phase 2/3.

## Claims Not Yet Supported

- Do not claim ARFA reduces token usage, latency, or large-model calls.
- Do not claim ARFA-Min has been evaluated as a live agent.
- Do not claim residual is proven necessary.
- Do not claim labels are human gold labels.
- Do not claim SWE-bench Lite results.

## Files In This Bundle

Report:

```text
phase1_final_report.md
```

Metrics and analysis:

```text
metrics/dataset_summary.json
metrics/test_metrics.csv
metrics/thresholds.json
metrics/ablation_results.csv
metrics/results_by_task_category.csv
metrics/results_by_step_type.csv
metrics/failure_case_analysis.md
metrics/leakage_audit.md
metrics/false_fast_cases.jsonl
```

Paper tables:

```text
tables/table1_dataset_summary.md
tables/table2_main_results.md
tables/table3_ablation.md
tables/table1_dataset_summary.tex
tables/table2_main_results.tex
tables/table3_ablation.tex
```

Paper figures:

```text
figures/fig1_dataset_overview.png
figures/fig2_main_metrics.png
figures/fig3_safety_tradeoff.png
figures/fig4_ablation.png
figures/fig5_subgroup_results.png
figures/fig6_operating_points.png
```

## Suggested Report Framing

Use this as a Phase 1 preliminary evidence section:

> We first evaluate whether execution residual provides a usable signal for reasoning necessity before deploying it inside a live agent. On full InterCode Bash trajectories, the hybrid residual improves held-out F1 over simple baselines and satisfies the false-fast safety constraint, but it does not yet meet the full evidence gate because fast-path coverage and safe-fast precision are insufficient. These results suggest that residual contains useful signal, while also showing that naive residual routing is not yet strong enough for full ARFA deployment without improved expectation quality and annotation review.
