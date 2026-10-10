#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

scope="${1:-core}"
case "${scope}" in
  residual) policies=(residual) ;;
  core) policies=(residual always_slow always_fast) ;;
  full) policies=(residual always_slow always_fast observation_only fixed_switch) ;;
  *)
    echo "Usage: $0 [residual|core|full]" >&2
    exit 2
    ;;
esac

endpoint="http://127.0.0.1:11435"
workers="${ARFA_TRAJECTORY_WORKERS:-8}"
curl -fsS "${endpoint}/api/tags" >/dev/null
resident=$(curl -fsS "${endpoint}/api/ps")
grep -q 'arfa-qwen2.5-coder:7b-8k' <<< "${resident}"
grep -q 'arfa-qwen2.5-coder:14b-8k' <<< "${resident}"

.venv/bin/python -m src.phase3_arfa.runtime_preflight \
  --config config/phase3_arfa_nrp_qwen.yaml \
  --output evidence/phase3_design/runtime_preflight_nrp_qwen.json

for policy in "${policies[@]}"; do
  .venv/bin/python -m src.phase3_arfa.run_arfa \
    --config config/phase3_arfa_nrp_qwen.yaml \
    --pair qwen \
    --policy "${policy}" \
    --split development \
    --workers "${workers}" \
    --resume
done
