# Llama Phase 3 Development Core

This bundle records the matched Llama Phase 3 development matrix. Llama 3.2 3B
and Llama 3.1 8B remained resident together on one NVIDIA RTX 3090 24GB GPU.
Each policy used the same 100 InterCode NL2Bash development tasks.

## Results

- Residual: 19/100 successful, 12.70 seconds per task, 369 slow-model calls.
- Always-slow: 16/100 successful, 11.16 seconds per task, 477 slow-model calls.
- Always-fast: 8/100 successful, 8.85 seconds per task, no slow-model calls.
- Residual routed 31.5% of calls to Llama 3B.

Against always-slow, residual reduced slow-model calls by 22.64%. The paired
bootstrap 95% interval was [14.09%, 30.72%]. Residual gained 3 success-rate
points, but the interval [-5, 11] points and exact McNemar p-value 0.629 do not
establish a success difference. It used 13.78% more wall time and 24.82% more
tokens, so this development policy did not improve total efficiency over
always-slow for the Llama pair.

Against always-fast, residual gained 11 success-rate points. The bootstrap 95%
interval was [5, 18] points and the exact McNemar p-value was 0.00342. Residual
was 43.55% slower per task, while its token reduction point estimate was 20.12%
with an interval that crossed zero.

## Interpretation

The Llama run shows a different tradeoff from Qwen. Residual routing improved
task success over the fast-only agent and reduced 8B calls relative to the
slow-only agent, but the extra 3B calls increased total tokens and wall time.
This pair therefore needs development-set calibration before any held-out run.
It should not be pooled with the Qwen H100 timing result.

Two always-fast tasks damaged system libraries inside their isolated task
containers. They are retained as failed attempts with an explicit evaluator
environment-error record; the matrix continued and remained 100/100 complete.
Residual and always-slow had no evaluator environment errors.

The model server used Ollama 0.34.1 with both 8K aliases at `keep_alive=-1`.
At readiness, both models were 100% GPU resident and used 9,227 MiB of 24,576
MiB. Raw traces remain under the ignored
`results/phase3_arfa/nrp/nrp-llama-24gb-development/` directory and are linked
to this bundle by `manifest.json`.

The three `*_metrics.csv` files contain the 100 task-level records for each
policy. `analysis.json` contains the paired bootstrap intervals and McNemar
tests used above.
