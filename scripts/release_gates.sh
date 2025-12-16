#!/usr/bin/env bash
# =============================================================================
# release_gates.sh - Environment validation gates for macOS release builds
#
# Usage: ./scripts/release_gates.sh [--all|--mock|--native|--smoke]
#
# This script validates that the local environment is correctly configured
# with constellaration from site-packages (not vendored/editable), and that
# the native vmecpp extension loads without dlopen errors.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PYTHON="$REPO_ROOT/.venv/bin/python"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_pass() { echo -e "${GREEN}✓ $1${NC}"; }
log_fail() { echo -e "${RED}✗ $1${NC}"; }
log_info() { echo -e "${YELLOW}→ $1${NC}"; }

# Check that .venv exists
if [[ ! -x "$VENV_PYTHON" ]]; then
    log_fail "No .venv found at $REPO_ROOT/.venv"
    echo "Create it with: python -m venv .venv && source .venv/bin/activate && pip install -e '.[full]' && pip install 'constellaration==0.2.3'"
    exit 1
fi

run_mock_gate() {
    log_info "Running mock/unit gate..."
    AI_SCIENTIST_PHYSICS_BACKEND=mock "$VENV_PYTHON" -m pytest -q "$REPO_ROOT/tests" \
        --ignore="$REPO_ROOT/tests/test_real_backend_required.py" \
        --ignore="$REPO_ROOT/tests/test_physics_integration.py"
    log_pass "Mock/unit gate passed"
}

run_native_gate() {
    log_info "Running real/native gate..."
    AI_SCIENTIST_PHYSICS_BACKEND=real \
    AI_SCIENTIST_REQUIRE_SITEPACKAGES_CONSTELLARATION=1 \
    "$VENV_PYTHON" -m pytest -q "$REPO_ROOT/tests/test_real_backend_required.py"
    log_pass "Real/native gate passed"
}

run_smoke_gate() {
    log_info "Running real smoke gate (VMEC integration)..."
    AI_SCIENTIST_PHYSICS_BACKEND=real \
    "$VENV_PYTHON" -m pytest -q "$REPO_ROOT/tests/test_physics_integration.py"
    log_pass "Real smoke gate passed"
}

run_all() {
    run_mock_gate
    echo ""
    run_native_gate
    echo ""
    run_smoke_gate
    echo ""
    log_pass "All release gates passed!"
}

# Parse arguments
case "${1:-all}" in
    --mock)   run_mock_gate ;;
    --native) run_native_gate ;;
    --smoke)  run_smoke_gate ;;
    --all|*)  run_all ;;
esac
