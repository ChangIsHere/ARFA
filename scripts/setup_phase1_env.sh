#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

if [ ! -d ".venv" ]; then
  "$PYTHON_BIN" -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-$PWD/.cache/sentence_transformers}"

.venv/bin/python -c "from sentence_transformers import SentenceTransformer; model=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); print('ready:', model.get_sentence_embedding_dimension())"

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -c "from sentence_transformers import SentenceTransformer; model=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); print('offline_ready:', model.get_sentence_embedding_dimension())"
