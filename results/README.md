# Local Results

Generated traces and checkpoints are excluded from Git. Compact evidence used
for writing is tracked under `evidence/` and `docs/phase2_writing_bundle/`.

Retained Phase 3 roots are the completed Qwen, Llama, Gemma, and Phi development
runs, the routing-control runs, and the successful mixed-pair qualification
under `phase3_arfa/nrp/`. Superseded thresholds, smoke tests, logs, and failed
qualification attempts have been removed.

The completed Gemma observation-only trace is stored as `traces.jsonl.gz`.
Compression is lossless and reduces an unusually repetitive raw trace from
several gigabytes to a manageable archive. Paper-facing summaries and per-task
metrics are tracked under `evidence/phase3_design/supplementary/gemma_controls/`.
