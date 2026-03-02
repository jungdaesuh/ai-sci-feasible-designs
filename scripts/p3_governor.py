#!/usr/bin/env python
# ruff: noqa: E402,F401
"""Compatibility wrapper for legacy p3_governor entrypoint.

Canonical shared runtime now lives in scripts/governor.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.governor as _governor

# Re-export all non-dunder names for full backward compatibility.
for _name in dir(_governor):
    if _name.startswith("__"):
        continue
    globals()[_name] = getattr(_governor, _name)

main = _governor.main


if __name__ == "__main__":
    main()
