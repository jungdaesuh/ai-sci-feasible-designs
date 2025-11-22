"""Lightweight orchestration stubs to keep runner imports satisfied.

These helpers are intentionally minimal and file-oriented so unit tests that
touch the runner can import without requiring the downstream adaptation stack.
"""

from . import adaptation

__all__ = ["adaptation"]
