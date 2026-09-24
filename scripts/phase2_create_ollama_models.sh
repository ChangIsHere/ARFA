#!/usr/bin/env bash
set -euo pipefail

expansion_only=false
if [[ "${1:-}" == "--expansion-only" ]]; then
  expansion_only=true
  shift
fi
if [[ "$#" -ne 0 ]]; then
  echo "Usage: $0 [--expansion-only]" >&2
  exit 2
fi

if [[ "${expansion_only}" == false ]]; then
  ollama create arfa-qwen2.5-coder:7b-8k -f ollama/Modelfile.qwen2.5-coder-7b-8k
  ollama create arfa-llama3.1:8b-8k -f ollama/Modelfile.llama3.1-8b-8k
  ollama create arfa-qwen2.5-coder:14b-8k -f ollama/Modelfile.qwen2.5-coder-14b-8k
fi
ollama create arfa-llama3.2:3b-8k -f ollama/Modelfile.llama3.2-3b-8k
ollama create arfa-gemma3:4b-8k -f ollama/Modelfile.gemma3-4b-8k
ollama create arfa-gemma3:12b-8k -f ollama/Modelfile.gemma3-12b-8k
ollama create arfa-phi4-mini:3.8b-8k -f ollama/Modelfile.phi4-mini-3.8b-8k
ollama create arfa-phi4:14b-8k -f ollama/Modelfile.phi4-14b-8k
