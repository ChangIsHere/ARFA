#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
models=(
  "arfa-qwen2.5-coder:7b-8k|qwen2.5-coder-7b"
  "arfa-llama3.1:8b-8k|llama3.1-8b"
  "arfa-qwen2.5-coder:14b-8k|qwen2.5-coder-14b"
)
splits=(shadow_development shadow_validation shadow_test)

for entry in "${models[@]}"; do
  model_name="${entry%%|*}"
  model_slug="${entry##*|}"
  ollama show "${model_name}" >/dev/null
  for split in "${splits[@]}"; do
    .venv/bin/python -m src.phase1_5_shadow.run_shadow \
      --split "${split}" \
      --model-name "${model_name}" \
      --output-dir "results/phase1_5_shadow/full/${model_slug}/${split}" \
      --resume
  done
done

.venv/bin/python -m src.phase1_5_shadow.build_annotation_packet
