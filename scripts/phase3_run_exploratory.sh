#!/usr/bin/env bash
set -euo pipefail

.venv/bin/python -m src.phase3_arfa.run_arfa --config config/phase3_exploratory.yaml "$@"

