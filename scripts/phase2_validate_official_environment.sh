#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
.venv/bin/python -m src.phase2_baseline.validate_official_environment "$@"
