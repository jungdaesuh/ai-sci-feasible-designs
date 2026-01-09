#!/usr/bin/env python
"""Extract a single boundary JSON object from a P3 submission list.

Many P3 files (including leaderboard seeds) are stored as:
- a JSON list of JSON-encoded strings (each string is a boundary object)

This helper writes a single boundary object to an output file so it can be used
as a parent for `scripts/p3_propose.py`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract one boundary from P3 list.")
    parser.add_argument("submission", type=Path, help="P3 submission JSON list file.")
    parser.add_argument("--index", type=int, required=True, help="0-based index.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path.")
    args = parser.parse_args()

    raw = json.loads(args.submission.read_text())
    if not isinstance(raw, list):
        raise SystemExit("Expected a JSON list.")
    if args.index < 0 or args.index >= len(raw):
        raise SystemExit(f"Index out of range: {args.index} (len={len(raw)})")

    item = raw[int(args.index)]
    boundary = json.loads(item) if isinstance(item, str) else item
    if not isinstance(boundary, dict):
        raise SystemExit("Selected item is not a JSON object boundary.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(boundary, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
