#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1

config_path="config/phase2_paper.yaml"
results_root="results/phase2_baseline/paper"

models=(
  "arfa-qwen2.5-coder:7b-8k|qwen2.5-coder-7b"
  "arfa-llama3.1:8b-8k|llama3.1-8b"
  "arfa-qwen2.5-coder:14b-8k|qwen2.5-coder-14b"
  "arfa-llama3.2:3b-8k|llama3.2-3b"
  "arfa-gemma3:4b-8k|gemma3-4b"
  "arfa-gemma3:12b-8k|gemma3-12b"
  "arfa-phi4-mini:3.8b-8k|phi4-mini-3.8b"
  "arfa-phi4:14b-8k|phi4-14b"
)

expansion_only=false
if [[ "${1:-}" == "--expansion-only" ]]; then
  expansion_only=true
  shift
fi
if [[ "$#" -ne 0 ]]; then
  echo "Usage: $0 [--expansion-only]" >&2
  exit 2
fi

for entry in "${models[@]}"; do
  model_name="${entry%%|*}"
  model_slug="${entry##*|}"
  if [[ "${expansion_only}" == true && "${model_slug}" == qwen2.5-coder-* ]]; then
    continue
  fi
  if [[ "${expansion_only}" == true && "${model_slug}" == llama3.1-8b ]]; then
    continue
  fi
  ollama show "${model_name}" >/dev/null
  .venv/bin/python -m src.phase2_baseline.run_baseline \
    --config "${config_path}" \
    --model-name "${model_name}" \
    --output-dir "${results_root}/${model_slug}" \
    --resume
done

.venv/bin/python -m src.phase2_baseline.analyze_paper_results --results-root "${results_root}" --expected-models 8
bash scripts/phase2_make_paper_artifacts.sh --analysis-dir "${results_root}/analysis"
bash scripts/phase2_build_writing_bundle.sh --results-root "${results_root}"
