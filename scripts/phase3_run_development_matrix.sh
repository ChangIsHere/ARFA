#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

pair_selection="${1:-qwen}"
matrix_scope="${2:-core}"

if [[ "$pair_selection" == "all" ]]; then
  pairs=(qwen llama gemma phi)
elif [[ "$pair_selection" =~ ^(qwen|llama|gemma|phi)$ ]]; then
  pairs=("$pair_selection")
else
  echo "Usage: $0 [qwen|llama|gemma|phi|all] [residual|core|full]" >&2
  exit 2
fi

case "$matrix_scope" in
  residual) policies=(residual) ;;
  core) policies=(residual always_slow always_fast) ;;
  full) policies=(residual always_slow always_fast observation_only fixed_switch) ;;
  *)
    echo "Usage: $0 [qwen|llama|gemma|phi|all] [residual|core|full]" >&2
    exit 2
    ;;
esac

for pair in "${pairs[@]}"; do
  for policy in "${policies[@]}"; do
    .venv/bin/python -m src.phase3_arfa.run_arfa \
      --config config/phase3_arfa.yaml \
      --pair "$pair" \
      --policy "$policy" \
      --split development \
      --resume
  done
done
