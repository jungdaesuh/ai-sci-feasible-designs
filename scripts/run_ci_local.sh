#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIP_CACHE_DIR_DEFAULT="${HOME}/.cache/pip"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-${PIP_CACHE_DIR_DEFAULT}}"

mkdir -p "${PIP_CACHE_DIR}"

DOCKER_IMAGE="python:3.10-bullseye"

docker run --rm -t \
  -v "${ROOT_DIR}:/work" \
  -v "${PIP_CACHE_DIR}:/root/.cache/pip" \
  -w /work \
  -e AI_SCIENTIST_ALLOW_PHYSICS_MOCKS=1 \
  -e PIP_CONSTRAINT=/work/constraints/ci.txt \
  -e PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu \
  "${DOCKER_IMAGE}" \
  bash -euo pipefail -c "\
    python -m pip install --upgrade pip; \
    python -m pip install -e '.[test,optimization,datasets]'; \
    pytest -m 'not integration and not slow' tests/; \
  "
