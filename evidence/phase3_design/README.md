# Phase 3 Evidence

- `development_summary.csv` is the compact four-pair result table.
- `qwen_h100_development/` contains the Qwen matched-policy summaries, paired
  analysis, runtime notes, and trace hashes.
- `llama_24gb_development/` contains the equivalent Llama evidence.
- `gemma_development/` contains the development-selected threshold result and
  records that selection in its manifest.
- `phi_development/` contains the best-success diagnostic result and records
  that no tested threshold passed the saved selection rule.
- `gemma_high_memory_parallel4_calibration/` and
  `phi_high_memory_parallel4_calibration/` retain the complete threshold tables.
- `runtime_preflight_nrp_qwen.json` and `runtime_preflight_nrp_llama.json`
  record accepted preflight checks.

Each pair bundle contains `residual_metrics.csv`, `always_slow_metrics.csv`, and
`always_fast_metrics.csv`. These paper-facing per-task records support paired
checks of success, calls, tokens, and latency without publishing the much larger
terminal traces. `analysis.json` contains the confidence intervals, while
`manifest.json` records run status, hardware, model identities, and trace hashes.

Raw traces remain under `results/phase3_arfa/nrp/` and are not committed. The
evidence manifests link them by SHA-256. All four studies used the same 100
development task IDs and three matched policies. Hardware differed, so absolute
latency is interpreted within each pair rather than pooled across pairs.

The reserved Phase 3 final split has not been opened. Decision rules are in
`docs/phase3_study_protocol.md`; the manuscript handoff is
`docs/README_FOR_WRITING.md`.
