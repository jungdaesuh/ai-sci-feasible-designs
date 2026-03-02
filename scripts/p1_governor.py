#!/usr/bin/env python
# ruff: noqa: E402
"""Thin P1 wrapper over shared governor runtime."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.governor import main as governor_main


def main() -> None:
    argv = sys.argv[1:]
    if "--problem" not in argv:
        argv = ["--problem", "p1", *argv]
    governor_main(argv)


if __name__ == "__main__":
    main()
