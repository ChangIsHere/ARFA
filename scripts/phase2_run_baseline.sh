#!/usr/bin/env bash
set -euo pipefail

.venv/bin/python -m src.phase2_baseline.run_baseline "$@"
