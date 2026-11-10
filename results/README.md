# Local Results

This directory contains generated raw traces, checkpoints, and run logs. It is
excluded from Git. Compact paper-facing evidence is tracked in `evidence/` and
the Phase 2 writing bundle.

## Current Paper-Facing Raw Roots

- `phase1_5_shadow/full/`: 300 shadow runs used by the residual diagnostics.
- `phase2_baseline/paper/`: eight standalone model runs and Phase 2 analysis.
- `phase3_arfa/nrp/`: successful native-pair development and control runs.
- `phase3_arfa/final/qwen-v4/`: frozen Qwen reserved-split matrix.
- `phase3_arfa/final/gemma-v4/`: frozen Gemma reserved-split matrix.
- `phase3_arfa/mixed_pairs/v1/`: two cross-family matrices.
- `phase3_arfa/handoff/v1/`: three handoff modes.

The canonical result summaries are:

- `docs/EXPERIMENT_RESULTS.md`
- `evidence/phase3_design/`
- `evidence/phase3_final/`
- `evidence/phase3_mixed_pairs/`
- `evidence/phase3_handoff/`

Old partial final runs, failed infrastructure runs, and qualification-only runs
are not part of the paper evidence. They may be removed after confirming that
the compact evidence bundle and final raw root are intact.
