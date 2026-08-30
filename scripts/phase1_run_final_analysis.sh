#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

MPLCONFIGDIR="${TMPDIR:-/tmp}/arfa_matplotlib" "$PYTHON_BIN" -m src.phase1_residual.analyze_final \
  --config config/phase1_residual.yaml
