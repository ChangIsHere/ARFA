#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

pair_selection="${1:-all}"
matrix_scope="${2:-residual}"
log_dir="results/phase3_arfa/logs"
mkdir -p "$log_dir"
log_path="$log_dir/development_${pair_selection}_${matrix_scope}_$(date +%Y%m%d_%H%M%S).log"

bash scripts/phase3_runtime_preflight.sh
echo "Phase 3 log: $log_path"
caffeinate -dimsu bash scripts/phase3_run_development_matrix.sh \
  "$pair_selection" "$matrix_scope" 2>&1 | tee "$log_path"
