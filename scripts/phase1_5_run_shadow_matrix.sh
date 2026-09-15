#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
.venv/bin/python -m src.phase1_5_shadow.run_formal_matrix "$@"
