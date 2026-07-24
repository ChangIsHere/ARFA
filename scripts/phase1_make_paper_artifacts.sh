#!/usr/bin/env bash
set -euo pipefail

MPLCONFIGDIR="${TMPDIR:-/tmp}/arfa_matplotlib" python3 -m src.phase1_residual.paper_artifacts \
  --final-dir results/phase1_residual/final
