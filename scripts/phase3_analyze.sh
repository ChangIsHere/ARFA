#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
pair="${1:-qwen}"
split="${2:-development}"

.venv/bin/python -m src.phase3_arfa.analyze_results --pair "$pair" --split "$split"
