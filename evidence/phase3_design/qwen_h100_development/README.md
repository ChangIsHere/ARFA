# Qwen H100 Phase 3 Development Core

This bundle records the matched Qwen Phase 3 development matrix run with both
models resident on one NVIDIA H100 80GB GPU. Each policy used the same 100
InterCode NL2Bash development tasks.

## Results

- Residual: 41/100 successful, 6.86 seconds per task, 165 slow-model calls.
- Always-slow: 41/100 successful, 7.21 seconds per task, 279 slow-model calls.
- Always-fast: 31/100 successful, 5.42 seconds per task, no slow-model calls.
- Residual routed 41.7% of calls to Qwen 7B.

Against always-slow, residual preserved the same 41% task success and reduced
slow-model calls by 40.86%. The task-bootstrap 95% interval for slow-call
reduction was [34.67%, 46.45%]. Wall time decreased by 4.87%, but its 95%
interval [-2.50%, 11.86%] crossed zero, so this run does not establish a
statistically stable wall-time improvement.

Against always-fast, residual improved success by 10 percentage points. The
bootstrap 95% interval was [3, 17] percentage points, and the paired exact
McNemar p-value was 0.01294. Residual was 26.54% slower per task than always-fast.

## Interpretation

This development result supports the intended quality-cost behavior: residual
routing matched the slow model's task success while substantially reducing slow
calls, and it outperformed the fast-only model on task success. It is not the
reserved final-split result, and no direct energy measurement was collected.

An earlier Mac run was used to debug the controller but was superseded because it
could not keep both models resident. Its timing is not pooled with this H100 run.
Raw H100 traces remain under the ignored
`results/phase3_arfa/nrp/nrp-qwen-h100-development/` directory and are linked to
this bundle by the hashes in `manifest.json`.

The three `*_metrics.csv` files contain the 100 task-level records for each
policy. `analysis.json` contains the paired bootstrap intervals and McNemar
tests used above.
