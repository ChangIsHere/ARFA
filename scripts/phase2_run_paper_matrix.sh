#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1

config_path="config/phase2_paper.yaml"
results_root="results/phase2_baseline/paper"

models=(
  "arfa-qwen2.5-coder:7b-8k|qwen2.5-coder-7b"
  "arfa-llama3.1:8b-8k|llama3.1-8b"
  "arfa-qwen2.5-coder:14b-8k|qwen2.5-coder-14b"
)

for entry in "${models[@]}"; do
  model_name="${entry%%|*}"
  model_slug="${entry##*|}"
  ollama show "${model_name}" >/dev/null
  .venv/bin/python -m src.phase2_baseline.run_baseline \
    --config "${config_path}" \
    --model-name "${model_name}" \
    --output-dir "${results_root}/${model_slug}" \
    --resume
done

.venv/bin/python -m src.phase2_baseline.analyze_paper_results --results-root "${results_root}"
bash scripts/phase2_make_paper_artifacts.sh --analysis-dir "${results_root}/analysis"
