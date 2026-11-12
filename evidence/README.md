# Evidence Index

This directory contains compact outputs used to verify the reported results.
Raw traces and checkpoints remain under the ignored `results/` tree.

## Primary Evidence

- `residual_diagnostics/`: 300 shadow runs, step- and task-level residual scores,
  annotation association, model comparisons, and source hashes.
- `phase2_baselines/`: standalone results for eight models on the same 200
  InterCode tasks, including paired comparisons, figures, and protocol records.
- `phase3_design/native_pairs/`: four native fast/slow development pairs with
  residual, always-slow, and always-fast policies.
- `phase3_final/`: reserved 100-task Qwen and Gemma matrices with six matched
  policies, task-level metrics, paired bootstrap intervals, exact McNemar tests,
  and frozen protocol manifests.
- `phase3_mixed_pairs/`: two cross-family final matrices with five matched
  policies, task-level metrics, and policy-level GPU records.
- `phase3_handoff/`: full-visible, structured-state, and observation-only
  cross-model handoff results with task-level metrics.

## Supporting Evidence

- `phase3_design/supplementary/cost_reversal/`: why nominally smaller models can
  increase calls, retries, and total cost.
- `phase3_design/supplementary/gemma_controls/`: Gemma development controls.
- `phase3_design/supplementary/qwen_controls/`: Qwen development controls.
- `phase1_5_pilot/`: pilot annotation and collection audit.
- `phase1_5_exploratory/`: historical exploratory gate result.

## Recommended Reading Order

1. [`docs/EXPERIMENT_RESULTS.md`](../docs/EXPERIMENT_RESULTS.md)
2. `phase3_final/README.md`
3. `phase3_mixed_pairs/README.md`
4. `phase3_handoff/README.md`
5. `phase3_design/supplementary/cost_reversal/README.md`

Every value in the consolidated results document points to one of these
locations. Absolute latency is compared only inside a matched matrix because the
development experiments used different hardware.
