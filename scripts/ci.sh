#!/usr/bin/env bash
set -euo pipefail

echo "[ci] upgrading pip and installing dependencies"
python -m pip install --upgrade pip
python -m pip install "scikit-learn>=1.3.0,<1.5" prometheus_client

echo "[ci] running surrogate and evaluation smoke tests"
pytest -q tests/optim/test_surrogate_bundle.py tests/tools/test_tools_reliability.py

echo "[ci] running RAG indexing tests"
pytest -q tests/rag/test_rag.py tests/rag/test_rag_indexing.py

if [[ "${CI_RF_PERF:-0}" == "1" ]]; then
  echo "[ci] running optional RF timing perf check"
  pytest -q tests/optim/test_surrogate_bundle.py -k "perf" || true
fi
