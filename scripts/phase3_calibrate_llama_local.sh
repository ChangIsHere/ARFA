#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

config="config/phase3_arfa.yaml"
result_root="results/phase3_arfa/local/llama_threshold_parallel2_development"
analysis_dir="${ARFA_CALIBRATION_OUTPUT_DIR:-${result_root}/analysis}"
workers=2
thresholds=(0.15 0.20 0.25 0.30 0.35)

.venv/bin/python -m src.phase3_arfa.run_arfa \
  --config "${config}" \
  --pair llama \
  --policy always_slow \
  --split development \
  --workers "${workers}" \
  --output-dir "${result_root}/development/llama/always_slow" \
  --resume

for threshold in "${thresholds[@]}"; do
  if [[ "${threshold}" == "0.35" ]]; then
    output_dir="${result_root}/development/llama/residual"
  else
    slug="threshold_${threshold/./_}"
    output_dir="${result_root}/calibration/${slug}/development/llama/residual"
  fi
  echo "[calibration] llama local development threshold=${threshold}"
  .venv/bin/python -m src.phase3_arfa.run_arfa \
    --config "${config}" \
    --pair llama \
    --policy residual \
    --split development \
    --threshold "${threshold}" \
    --workers "${workers}" \
    --output-dir "${output_dir}" \
    --resume
done

.venv/bin/python -m src.phase3_arfa.calibrate_threshold \
  --result-root "${result_root}" \
  --pair llama \
  --split development \
  --thresholds 0.15 0.20 0.25 0.30 0.35 \
  --output-dir "${analysis_dir}"
