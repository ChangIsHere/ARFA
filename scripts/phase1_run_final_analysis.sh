#!/usr/bin/env bash
set -euo pipefail

MPLCONFIGDIR="${TMPDIR:-/tmp}/arfa_matplotlib" python3 -m src.phase1_residual.analyze_final \
  --config config/phase1_residual.yaml
