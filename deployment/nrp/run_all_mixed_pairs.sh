#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

bash deployment/nrp/run_mixed_pair_final.sh gemma4_qwen14
bash deployment/nrp/run_mixed_pair_final.sh qwen7_gemma12
