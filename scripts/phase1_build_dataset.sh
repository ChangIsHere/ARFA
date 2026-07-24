#!/usr/bin/env bash
set -euo pipefail

python3 -m src.phase1_residual.dataset_builder --config config/phase1_residual.yaml
python3 -m src.phase1_residual.residual_calculator --config config/phase1_residual.yaml
