# Phase 3 Mixed-Pair Extension

This extension crosses the two useful fast models and two useful slow models.
The native Qwen and Gemma cells are complete under `evidence/phase3_final/`.
The remaining cross-family cells are:

- Gemma 4B fast with Qwen 14B slow;
- Qwen 7B fast with Gemma 12B slow.

Each cross-family pair runs residual, learned observation-only, budget-matched
random, always-slow, and always-fast on the same 100 tasks. Model versions,
prompt, context, evaluator, step cap, concurrency, and random seed are fixed.
The two pairs run as consecutive bounded A6000 Jobs. Each Job keeps its two
models resident with the same inference and trajectory concurrency used by the
native v4 study, then terminates when the matrix ends or its heartbeat expires.

The earlier 20-task Gemma 4B/Qwen 14B qualification found that the nominal fast
model was not faster on its A6000 run. The full matrix retains that result as a
pair-dependence test rather than treating model size as a sufficient speed proxy.

## Final results

Both pairs completed all five policies on the same 100-task final split. Every
cell passed the task-set, provenance, configuration-hash, and request-error
audits. The random policy was budget matched to residual within five percentage
points.

| Pair | Comparison | Success difference | Wall-time reduction | Token reduction | Slow-call reduction |
| --- | --- | ---: | ---: | ---: | ---: |
| Gemma 4B / Qwen 14B | residual vs. always-slow | -3 pp | 29.8% | 7.3% | 43.3% |
| Gemma 4B / Qwen 14B | residual vs. observation-only | -4 pp | 3.6% | -0.1% | -4.9% |
| Qwen 7B / Gemma 12B | residual vs. always-slow | +4 pp | 33.5% | 25.8% | 49.5% |
| Qwen 7B / Gemma 12B | residual vs. observation-only | -1 pp | 11.0% | 12.8% | 11.3% |
| Qwen 7B / Gemma 12B | residual vs. random-budget | +3 pp | 15.4% | 17.4% | 13.8% |

For Qwen 7B / Gemma 12B, residual preserved the always-slow success level within
sampling uncertainty while reducing wall time, tokens, and slow calls. Its cost
reductions over budget-matched random routing also had positive task-bootstrap
95% confidence intervals. Gemma 4B / Qwen 14B was less favorable: the nominal
fast model was not meaningfully faster than the slow model, and observation-only
routing achieved the highest success count. Together, the cells show that
residual routing depends on the operational properties of the selected model
pair; parameter count alone does not identify a useful fast track.

Compact paper-facing artifacts are stored under `gemma4_qwen14/` and
`qwen7_gemma12/`. They include policy summaries, paired analysis, per-cell
audits, GPU summaries, runtime preflight, and provenance. Full task traces remain
under `results/phase3_arfa/mixed_pairs/v1/`.
