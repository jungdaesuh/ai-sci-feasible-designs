"""Verify the Phase 1 tools_api smoke script exercises every declared schema."""

from ai_scientist.tools_api_smoke import run_smoke


def test_tools_api_smoke_runs() -> None:
    run_smoke()
