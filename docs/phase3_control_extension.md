# Phase 3 Control Study

## Question

The four-pair development study shows that residual routing depends on the
models being combined. Gemma 4B/12B gave the clearest balance of task success,
slow-model calls, tokens, and wall time. It is the representative pair for the
full routing-control study.

This study asks whether expectation-observation residual provides useful model
selection information beyond simpler alternatives. Both the Gemma and Qwen
development matrices are complete.

## Policies

All policies use the same Gemma models, 100 InterCode development tasks,
prompt, evaluator, context length, step cap, concurrency, and model artifacts.

- `residual`: routes from expected-versus-observed execution mismatch;
- `learned_observation_only`: reads terminal output and exit code but not the
  expectation, action, plan, or continuation;
- `expectation_gate_only`: uses expectation and continuation safety gates but
  does not compare them with the terminal output;
- `random_budget`: randomly routes at the residual policy's achieved fast-call
  rate using a fixed seed;
- `always_slow` and `always_fast`: endpoints from the completed pair study.

The learned observation-only scorer is fitted on the Phase 1.5 development
annotations and uses the saved validation threshold in
[`learned_observation_validation.json`](../evidence/phase3_design/supplementary/learned_observation_validation.json).
The Gemma residual threshold starts at `0.30`, selected by the saved
development calibration. The matched random target is `0.35`, based on its
achieved fast-call rate. Ollama library tags changed after the original pair
study, so the control study freezes the current Gemma aliases and reruns its
residual reference in the same GPU Job as the controls. Results from different
model generations are not combined.

## Measurements

Report task success, tokens, model calls by role, retries, achieved fast-call
rate, task wall time, and sampled GPU energy. Compare policies by paired task
ID and report bootstrap intervals. Latency and energy are directly comparable
only when runs use the same GPU model and execution settings.

The controls are development experiments. Their purpose is method comparison,
not a final held-out claim. They were followed by frozen Qwen and Gemma final
matrices, documented in [`evidence/phase3_final/`](../evidence/phase3_final/).

## Gemma Development Results

| Policy | Success | Tokens | Calls | Fast-call rate |
| --- | ---: | ---: | ---: | ---: |
| Residual | 28% | 390,911 | 321 | 37.7% |
| Learned observation-only | 31% | 394,414 | 315 | 39.4% |
| Expectation-gate-only | 25% | 440,179 | 347 | 45.0% |
| Random-budget | 25% | 427,674 | 340 | 35.6% |

Residual outperformed the two simple controls on success and token use. It was
competitive with, but did not outperform, learned observation-only. Paired
success intervals include zero. Full compact evidence is in
[`evidence/phase3_design/supplementary/gemma_controls/`](../evidence/phase3_design/supplementary/gemma_controls/).

## Execution

Run a single policy with a bounded GPU Job:

```bash
bash deployment/nrp/run_gemma_controls.sh learned_observation_only full
```

Run residual and learned observation-only on one GPU:

```bash
bash deployment/nrp/run_gemma_controls_matrix.sh core
```

Run all four focused policies on one GPU:

```bash
bash deployment/nrp/run_gemma_controls_matrix.sh full
```

The bounded runner verifies that both models remain resident, records runtime
provenance, and deletes the GPU Job on exit. Local raw results are written below
`results/phase3_arfa/nrp/nrp-gemma-controls-v2/`.

## Final Follow-Up

The Qwen and Gemma reserved-split matrices, two cross-family pairs, and the
three-mode handoff ablation are complete. Their paper-facing results are in
[`evidence/phase3_final/`](../evidence/phase3_final/),
[`evidence/phase3_mixed_pairs/`](../evidence/phase3_mixed_pairs/), and
[`evidence/phase3_handoff/`](../evidence/phase3_handoff/).

The remaining extensions are external-benchmark transfer, repeated policy-level
energy measurement, and residual-component ablations. See
[phase3_publication_experiment_plan.md](phase3_publication_experiment_plan.md).
