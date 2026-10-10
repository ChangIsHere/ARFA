#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

pair="${1:-qwen}"
policy="${2:-residual}"
shift "$(( $# > 0 ? 1 : 0 ))"
shift "$(( $# > 0 ? 1 : 0 ))"

export PYTHONUNBUFFERED=1
.venv/bin/python -m src.phase3_arfa.run_arfa \
  --config config/phase3_arfa.yaml \
  --pair "$pair" \
  --policy "$policy" \
  --split development \
  --resume \
  "$@"
