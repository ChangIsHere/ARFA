#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
.venv/bin/python -m src.phase3_arfa.runtime_preflight \
  --config config/phase3_arfa.yaml \
  --output evidence/phase3_design/runtime_preflight.json
