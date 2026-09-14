#!/usr/bin/env bash
set -euo pipefail

ollama create arfa-qwen2.5-coder:7b-8k -f ollama/Modelfile.qwen2.5-coder-7b-8k
ollama create arfa-llama3.1:8b-8k -f ollama/Modelfile.llama3.1-8b-8k
ollama create arfa-qwen2.5-coder:14b-8k -f ollama/Modelfile.qwen2.5-coder-14b-8k
