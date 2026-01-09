#!/usr/bin/env python
# ruff: noqa: E402
"""Lightweight P3 dashboard for a running SQLite-backed loop.

This script reads from the WorldModel schema and prints:
- queue status (pending/running/done/failed)
- feasible count
- current hypervolume estimate (from feasible points)
- most common failing constraint (recent window)

It never mutates the DB.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pymoo.indicators import hv

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ai_scientist.memory.schema import init_db


def _connect(db_path: Path) -> sqlite3.Connection:
    init_db(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


def _status_bucket(status: str) -> str:
    if not status:
        return "unknown"
    return status.split(":", 1)[0]


def _compute_hv(points: list[tuple[float, float]]) -> float:
    if not points:
        return 0.0
    X = np.array([(-lgradb, aspect) for lgradb, aspect in points], dtype=float)
    indicator = hv.Hypervolume(ref_point=np.array([1.0, 20.0], dtype=float))
    value = indicator(X)
    assert value is not None
    return float(value)


def _pareto(points: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    # points are (lgradB, aspect, design_hash)
    keep: list[tuple[float, float, str]] = []
    for lgradb, aspect, h in points:
        dominated = False
        for l2, a2, _h2 in points:
            if (l2 >= lgradb and a2 <= aspect) and (l2 > lgradb or a2 < aspect):
                dominated = True
                break
        if not dominated:
            keep.append((lgradb, aspect, h))
    keep.sort(key=lambda x: (x[1], -x[0]))
    return keep


def main() -> None:
    parser = argparse.ArgumentParser(description="P3 dashboard (read-only).")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("reports/p3_world_model.sqlite"),
        help="SQLite DB path for P3 runs.",
    )
    parser.add_argument("--experiment-id", type=int, required=True)
    parser.add_argument("--interval-sec", type=float, default=10.0)
    parser.add_argument("--recent-window", type=int, default=200)
    args = parser.parse_args()

    conn = _connect(args.db)
    try:
        while True:
            exp_id = int(args.experiment_id)

            rows = conn.execute(
                "SELECT status, COUNT(*) AS n FROM candidates WHERE experiment_id = ? GROUP BY status",
                (exp_id,),
            ).fetchall()
            status_counts: Counter[str] = Counter()
            for row in rows:
                status_counts[_status_bucket(str(row["status"]))] += int(row["n"])

            total_candidates = int(sum(status_counts.values()))

            feasible_rows = conn.execute(
                """
                SELECT m.objective AS lgradb, m.raw_json AS raw
                FROM metrics m
                JOIN candidates c ON m.candidate_id = c.id
                WHERE c.experiment_id = ? AND m.is_feasible = 1
                """,
                (exp_id,),
            ).fetchall()

            feasible_points: list[tuple[float, float]] = []
            feasible_points_full: list[tuple[float, float, str]] = []
            for row in feasible_rows:
                payload = json.loads(str(row["raw"]))
                m = payload.get("metrics", {})
                aspect = m.get("aspect_ratio")
                lgradb = row["lgradb"]
                if aspect is None or lgradb is None:
                    continue
                h = str(payload.get("design_hash", ""))
                feasible_points.append((float(lgradb), float(aspect)))
                feasible_points_full.append((float(lgradb), float(aspect), h))

            hv_value = _compute_hv(feasible_points)
            pareto = _pareto(feasible_points_full)

            recent_fail = conn.execute(
                """
                SELECT m.raw_json AS raw
                FROM metrics m
                JOIN candidates c ON m.candidate_id = c.id
                WHERE c.experiment_id = ? AND m.is_feasible = 0
                ORDER BY m.id DESC
                LIMIT ?
                """,
                (exp_id, int(args.recent_window)),
            ).fetchall()

            worst_counter: Counter[str] = Counter()
            family_stats: dict[str, Counter[str]] = defaultdict(Counter)
            for row in recent_fail:
                payload = json.loads(str(row["raw"]))
                violations = payload.get("violations", {})
                worst_name = None
                worst_val = 0.0
                if isinstance(violations, dict):
                    for name, val in violations.items():
                        try:
                            v = float(val)
                        except (TypeError, ValueError):
                            continue
                        if v > worst_val:
                            worst_val = v
                            worst_name = str(name)
                if worst_name is not None:
                    worst_counter[worst_name] += 1

                meta = payload.get("meta", {})
                family = ""
                if isinstance(meta, dict):
                    family = str(meta.get("move_family", ""))
                if family:
                    family_stats[family]["fail"] += 1

            recent_ok = conn.execute(
                """
                SELECT m.raw_json AS raw
                FROM metrics m
                JOIN candidates c ON m.candidate_id = c.id
                WHERE c.experiment_id = ? AND m.is_feasible = 1
                ORDER BY m.id DESC
                LIMIT ?
                """,
                (exp_id, int(args.recent_window)),
            ).fetchall()
            for row in recent_ok:
                payload = json.loads(str(row["raw"]))
                meta = payload.get("meta", {})
                family = ""
                if isinstance(meta, dict):
                    family = str(meta.get("move_family", ""))
                if family:
                    family_stats[family]["ok"] += 1

            best_compact = min(feasible_points_full, key=lambda p: p[1], default=None)
            best_lgradb = max(feasible_points_full, key=lambda p: p[0], default=None)

            print(f"[{_utc_stamp()}] exp={exp_id} hv={hv_value:.6f}")
            print(
                "queue total=%d pending=%d running=%d done=%d failed=%d"
                % (
                    total_candidates,
                    status_counts.get("pending", 0),
                    status_counts.get("running", 0),
                    status_counts.get("done", 0),
                    status_counts.get("failed", 0),
                )
            )
            print("feasible=%d pareto=%d" % (len(feasible_points_full), len(pareto)))
            if best_compact is not None:
                print(
                    "best_compact aspect=%.6f lgradB=%.6f hash=%s"
                    % (best_compact[1], best_compact[0], best_compact[2][:10])
                )
            if best_lgradb is not None:
                print(
                    "best_lgradB lgradB=%.6f aspect=%.6f hash=%s"
                    % (best_lgradb[0], best_lgradb[1], best_lgradb[2][:10])
                )
            if worst_counter:
                worst_name, worst_n = worst_counter.most_common(1)[0]
                print(f"recent worst_constraint={worst_name} count={worst_n}")
            if family_stats:
                parts = []
                for fam, counts in sorted(family_stats.items()):
                    ok = int(counts.get("ok", 0))
                    fail = int(counts.get("fail", 0))
                    parts.append(f"{fam}: ok={ok} fail={fail}")
                print("recent by_family: " + "; ".join(parts))
            print("")

            time.sleep(float(args.interval_sec))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
