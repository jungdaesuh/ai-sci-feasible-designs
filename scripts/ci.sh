#!/usr/bin/env bash
set -euo pipefail

echo "[ci] upgrading pip and installing dependencies"
python -m pip install --upgrade pip
python -m pip install scikit-learn

echo "[ci] running surrogate and evaluation smoke tests"
pytest -q tests/test_surrogate_bundle.py tests/test_tools_reliability.py

echo "[ci] running RAG indexing tests"
pytest -q tests/test_rag.py tests/test_rag_indexing.py

if [[ "${CI_RF_PERF:-0}" == "1" ]]; then
  echo "[ci] running optional RF timing perf check"
  pytest -q tests/test_surrogate_bundle.py -k "perf" || true
fi
