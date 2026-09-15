#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
.venv/bin/python -m src.phase1_5_shadow.run_shadow \
  --split shadow_development \
  --task-limit 10 \
  --model-name arfa-qwen2.5-coder:7b-8k \
  --output-dir results/phase1_5_shadow/pilot/qwen2.5-coder-7b \
  --resume \
  "$@"
