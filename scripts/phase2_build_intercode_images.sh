#!/usr/bin/env bash
set -euo pipefail

for fs_version in 1 2 3 4; do
  docker build \
    --build-arg "FS_VERSION=${fs_version}" \
    --file docker/phase2_intercode.Dockerfile \
    --tag "arfa/intercode-nl2bash:fs${fs_version}" \
    data/phase1/external_intercode
done
