#!/usr/bin/env bash
set -euo pipefail

export MPLCONFIGDIR="${TMPDIR:-/tmp}/arfa_matplotlib"
.venv/bin/python -m src.phase2_baseline.paper_artifacts "$@"
