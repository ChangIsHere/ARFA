#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-$PWD/.cache/sentence_transformers}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

"$PYTHON_BIN" -c "from sentence_transformers import SentenceTransformer; model=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); emb=model.encode(['expected terminal output', 'actual terminal output'], normalize_embeddings=True); print('embedding_backend=sentence-transformers/all-MiniLM-L6-v2'); print('shape=', emb.shape)"
