#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

mode="${1:-plan}"
case "${mode}" in
  plan) policies=() ;;
  core) policies=(residual learned_observation_only) ;;
  full) policies=(residual learned_observation_only expectation_gate_only random_budget) ;;
  *) echo "Usage: $0 [plan|core|full]" >&2; exit 2 ;;
esac

result_root="results/phase3_arfa/nrp/nrp-gemma-controls-v2"
if [[ "${mode}" == "plan" ]]; then
  printf '%s\n' 'core: residual + learned_observation_only, one bounded GPU Job per policy'
  printf '%s\n' 'full: residual + learned_observation_only + expectation_gate_only + random_budget, one bounded GPU Job per policy'
  printf 'Each policy: 12-task utilization smoke, then 100 development tasks\nResults: %s\n' "${result_root}"
  exit 0
fi

for policy in "${policies[@]}"; do
  deployment/nrp/run_gemma_controls.sh "${policy}" full
done

.venv/bin/python -m src.phase3_arfa.analyze_results \
  --result-root "${result_root}" --split development --pair gemma --strict-controls
