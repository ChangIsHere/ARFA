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

endpoint="http://127.0.0.1:11436"
workers="${ARFA_TRAJECTORY_WORKERS:-2}"
curl -fsS "${endpoint}/api/tags" >/dev/null
resident=$(curl -fsS "${endpoint}/api/ps")
grep -q 'arfa-llama3.2:3b-8k' <<< "${resident}"
grep -q 'arfa-llama3.1:8b-8k' <<< "${resident}"

OLLAMA_HOST="${endpoint}" .venv/bin/python -m src.phase3_arfa.runtime_preflight \
  --config config/phase3_arfa_nrp_llama.yaml \
  --output evidence/phase3_design/runtime_preflight_nrp_llama.json

for policy in "${policies[@]}"; do
  .venv/bin/python -m src.phase3_arfa.run_arfa \
    --config config/phase3_arfa_nrp_llama.yaml \
    --pair llama \
    --policy "${policy}" \
    --split development \
    --workers "${workers}" \
    --resume
done

.venv/bin/python -m src.phase3_arfa.analyze_results \
  --result-root results/phase3_arfa/nrp/nrp-llama-24gb-development \
  --pair llama \
  --split development
