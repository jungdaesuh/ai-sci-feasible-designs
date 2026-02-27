#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Pin codex-native-first defaults for canary runs without mutating baseline config.
export AI_SCIENTIST_MODEL_CONFIG_PATH="${AI_SCIENTIST_MODEL_CONFIG_PATH:-configs/model.codex_native_canary.yaml}"
export MODEL_PROVIDER="${MODEL_PROVIDER:-codex_native}"
export AI_SCIENTIST_INSTRUCT_MODEL="${AI_SCIENTIST_INSTRUCT_MODEL:-codex-native-short-loop}"
export AI_SCIENTIST_THINKING_MODEL="${AI_SCIENTIST_THINKING_MODEL:-codex-native-full}"
export AI_SCIENTIST_ROLE_PLANNING_MODEL="${AI_SCIENTIST_ROLE_PLANNING_MODEL:-codex-native-full}"
export AI_SCIENTIST_ROLE_LITERATURE_MODEL="${AI_SCIENTIST_ROLE_LITERATURE_MODEL:-codex-native-full}"
export AI_SCIENTIST_ROLE_ANALYSIS_MODEL="${AI_SCIENTIST_ROLE_ANALYSIS_MODEL:-codex-native-full}"

# Usage:
#   scripts/run_codex_native_canary.sh
#   AI_SCIENTIST_REMOTE_PROVIDER=1 CODEX_NATIVE_BEARER_TOKEN=... scripts/run_codex_native_canary.sh
python3 tools/ci_tools_smoke.py "$@"
