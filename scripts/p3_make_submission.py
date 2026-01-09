#!/usr/bin/env python
# ruff: noqa: E402
"""Build a P3 submission JSON from feasible candidates in the WorldModel DB.

Output format matches `scripts/score_candidates.py --problem p3`:
- a JSON list of boundary JSON *strings* (each string is a compact JSON object).
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pymoo.indicators import hv

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ai_scientist.memory.schema import init_db


@dataclass(frozen=True)
class Point:
    design_hash: str
    lgradb: float
    aspect: float
    boundary: dict


def _connect(db_path: Path) -> sqlite3.Connection:
    init_db(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _hv_value(points: list[Point]) -> float:
    if not points:
        return 0.0
    X = np.array([(-p.lgradb, p.aspect) for p in points], dtype=float)
    indicator = hv.Hypervolume(ref_point=np.array([1.0, 20.0], dtype=float))
    out = indicator(X)
    assert out is not None
    return float(out)


def _greedy_select(points: list[Point], *, max_points: int) -> list[Point]:
    remaining = list(points)
    selected: list[Point] = []
    current = 0.0
    while remaining and len(selected) < max_points:
        best_idx = -1
        best_val = current
        for i, cand in enumerate(remaining):
            trial = selected + [cand]
            val = _hv_value(trial)
            if val > best_val + 1e-12:
                best_val = val
                best_idx = i
        if best_idx < 0:
            break
        selected.append(remaining.pop(best_idx))
        current = best_val
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Make a P3 submission file.")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("reports/p3_world_model.sqlite"),
        help="SQLite DB path for P3 runs.",
    )
    parser.add_argument("--experiment-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-points", type=int, default=16)
    args = parser.parse_args()

    conn = _connect(args.db)
    try:
        rows = conn.execute(
            """
            SELECT c.params_json AS params_json, m.objective AS lgradb, m.raw_json AS raw
            FROM metrics m
            JOIN candidates c ON m.candidate_id = c.id
            WHERE c.experiment_id = ? AND m.is_feasible = 1
            """,
            (int(args.experiment_id),),
        ).fetchall()

        points: list[Point] = []
        for row in rows:
            payload = json.loads(str(row["raw"]))
            metrics = payload.get("metrics", {})
            aspect = metrics.get("aspect_ratio")
            lgradb = row["lgradb"]
            if aspect is None or lgradb is None:
                continue
            boundary = json.loads(str(row["params_json"]))
            design_hash = str(payload.get("design_hash", ""))
            points.append(
                Point(
                    design_hash=design_hash,
                    lgradb=float(lgradb),
                    aspect=float(aspect),
                    boundary=boundary,
                )
            )

        selected = _greedy_select(points, max_points=int(args.max_points))
        submission = [json.dumps(p.boundary, separators=(",", ":")) for p in selected]

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(submission, indent=2))

        with conn:
            conn.execute(
                "INSERT INTO artifacts (experiment_id, path, kind) VALUES (?, ?, ?)",
                (int(args.experiment_id), str(args.output), "submission_json"),
            )

        print(
            f"selected={len(selected)} hv={_hv_value(selected):.6f} output={args.output}"
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
