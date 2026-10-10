#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
pair="${1:-qwen}"

.venv/bin/python -m src.phase3_arfa.model_residency \
  --config config/phase3_arfa.yaml \
  --pair "$pair" \
  --output "evidence/phase3_design/model_residency_${pair}.json"
