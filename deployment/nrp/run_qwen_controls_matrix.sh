#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

mode="${1:-plan}"
case "${mode}" in
  plan|run) ;;
  *) echo "Usage: $0 [plan|run]" >&2; exit 2 ;;
esac

policies=(
  residual always_slow always_fast learned_observation_only
  observation_only expectation_gate_only random_budget fixed_budget
)
result_root="results/phase3_arfa/nrp/nrp-qwen-controls-v2"

if [[ "${mode}" == "plan" ]]; then
  for policy in "${policies[@]}"; do
    printf '%s: 12 smoke tasks, then 100 development tasks\n' "${policy}"
  done
  printf 'Results: %s\n' "${result_root}"
  exit 0
fi

mkdir -p "${result_root}/logs"
exec > >(tee -a "${result_root}/logs/matrix.log") 2>&1
printf 'Matrix started at %s; source commit %s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git rev-parse HEAD)"

for policy in "${policies[@]}"; do
  printf '\nStarting policy %s at %s\n' "${policy}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  bash deployment/nrp/run_qwen_controls.sh "${policy}" full
  printf 'Completed policy %s at %s\n' "${policy}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
done

.venv/bin/python -m src.phase3_arfa.analyze_results \
  --result-root "${result_root}" --split development --pair qwen --strict-controls
printf '\nMatrix completed at %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
