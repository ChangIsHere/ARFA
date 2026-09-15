#!/usr/bin/env bash
set -euo pipefail

.venv/bin/python -m src.phase1_5_shadow.build_annotation_packet "$@"
