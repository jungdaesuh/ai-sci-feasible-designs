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
  "${DOCKER_IMAGE}" \
  bash -euo pipefail -c "\
    apt-get update; \
    apt-get install -y cmake ninja-build gfortran libopenblas-dev libnetcdf-dev git; \
    python -m pip install --upgrade pip; \
    python -m pip install -e '.[test,optimization]'; \
    python -m pip install 'git+https://github.com/proximafusion/constellaration.git'; \
    pytest tests/; \
  "
