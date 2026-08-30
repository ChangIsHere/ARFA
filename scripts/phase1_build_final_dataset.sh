#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-$PWD/.cache/sentence_transformers}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

"$PYTHON_BIN" -m src.phase1_residual.external_dataset_builder \
  --source-root . \
  --output data/phase1/final/intercode_bash_phase1.jsonl \
  --summary results/phase1_residual/final/dataset_summary.json \
  --splits results/phase1_residual/final/splits.json \
  --second-pass data/phase1/annotation/second_pass_self_agreement.json \
  --max-records 700 \
  --seed 1729

"$PYTHON_BIN" -m src.phase1_residual.residual_calculator \
  --config config/phase1_residual.yaml \
  --dataset data/phase1/final/intercode_bash_phase1.jsonl \
  --output results/phase1_residual/final/residual_scores.csv
