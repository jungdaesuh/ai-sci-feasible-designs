"""Query DB for cycle snapshot and diverse parent selection.

Reads DB state at the start of a harness cycle:
- Queue counts (pending, running, done)
- Frontier value (best feasible objective, direction-normalized)
- Near-feasible candidate count
- Up to n diverse parent paths (frontier best, near-feasible best, stepping stone)
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

from harness.problem_adapter import ProblemAdapter
from harness.types import CycleSnapshot

# Candidates with feasibility <= this threshold and is_feasible=0 are "near-feasible".
_NEAR_FEASIBLE_MAX: float = 0.02


def _flatten_boundary(params_json: str) -> np.ndarray:
    """Flatten r_cos + z_sin coefficients into a 1-D float vector."""
    boundary = json.loads(params_json)
    r = np.array(boundary["r_cos"], dtype=float).ravel()
    z = np.array(boundary["z_sin"], dtype=float).ravel()
    return np.concatenate([r, z])


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [-1, 1]; returns 0.0 when either vector has zero norm."""
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _count_status(
    conn: sqlite3.Connection,
    experiment_id: int,
    problem: str,
    status: str,
) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM candidates "
        "WHERE experiment_id = ? AND problem = ? AND status = ?",
        (experiment_id, problem, status),
    ).fetchone()
    return int(row[0]) if row else 0


def _count_running(
    conn: sqlite3.Connection,
    experiment_id: int,
    problem: str,
) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM candidates "
        "WHERE experiment_id = ? AND problem = ? AND status LIKE 'running:%'",
        (experiment_id, problem),
    ).fetchone()
    return int(row[0]) if row else 0


def _count_near_feasible(
    conn: sqlite3.Connection,
    experiment_id: int,
    problem: str,
    threshold: float,
) -> int:
    row = conn.execute(
        """
        SELECT COUNT(*) FROM candidates c
        JOIN metrics m ON m.candidate_id = c.id
        WHERE c.experiment_id = ? AND c.problem = ?
          AND c.status = 'done'
          AND m.is_feasible = 0
          AND m.feasibility <= ?
        """,
        (experiment_id, problem, threshold),
    ).fetchone()
    return int(row[0]) if row else 0


def _query_frontier(
    conn: sqlite3.Connection,
    experiment_id: int,
    problem: str,
    problem_adapter: ProblemAdapter,
) -> tuple[float | None, int | None]:
    """Return (frontier_value, candidate_id) for the best feasible candidate.

    frontier_value is direction-normalized (higher is always better).
    """
    rows = conn.execute(
        """
        SELECT c.id, m.raw_json
        FROM candidates c
        JOIN metrics m ON m.candidate_id = c.id
        WHERE c.experiment_id = ? AND c.problem = ?
          AND c.status = 'done' AND m.is_feasible = 1
        """,
        (experiment_id, problem),
    ).fetchall()

    if not rows:
        return None, None

    best_value: float | None = None
    best_id: int | None = None

    for row in rows:
        raw = json.loads(row[1])
        # raw_json may be {"metrics": {...}} or a flat metrics dict.
        metrics = raw.get("metrics", raw)
        obj_val = problem_adapter.objective_value(metrics)
        if best_value is None or obj_val > best_value:
            best_value = obj_val
            best_id = int(row[0])

    return best_value, best_id


def select_diverse_parents(
    conn: sqlite3.Connection,
    frontier_id: int | None,
    n: int = 3,
    *,
    run_dir: Path,
    experiment_id: int,
    problem: str,
    near_feasible_threshold: float = _NEAR_FEASIBLE_MAX,
) -> list[Path]:
    """Select n diverse parent candidates and return their boundary file paths.

    Three-slot strategy (deduplicated):

    - **Slot 0 — Frontier best**: The candidate identified by ``frontier_id``
      (best feasible by direction-normalized objective).

    - **Slot 1 — Near-feasible best**: The infeasible candidate with the lowest
      feasibility score (closest to the feasibility boundary, i.e., closest to
      being feasible). Falls back to the first feasible candidate (by insertion
      order) that is not the frontier when no near-feasible exists.

    - **Slot 2 — Stepping stone**: A feasible candidate (not already in slots 0–1)
      that maximises cosine distance from the frontier, increasing design-space
      diversity.

    If fewer than ``n`` candidates fill the three slots, the remainder are padded
    with other done candidates in insertion order.

    Paths point to ``run_dir/candidates/{design_hash}.json``.  The governor is
    responsible for materialising those files for the sandbox.
    """
    rows = conn.execute(
        """
        SELECT c.id, c.design_hash, c.params_json, m.feasibility, m.is_feasible
        FROM candidates c
        JOIN metrics m ON m.candidate_id = c.id
        WHERE c.experiment_id = ? AND c.problem = ? AND c.status = 'done'
        ORDER BY c.id ASC
        """,
        (experiment_id, problem),
    ).fetchall()

    if not rows:
        return []

    # -- Slot 0: frontier best --------------------------------------------------
    frontier_row = None
    if frontier_id is not None:
        for r in rows:
            if int(r[0]) == frontier_id:
                frontier_row = r
                break
    if frontier_row is None:
        # Fallback: first feasible candidate in insertion order.
        for r in rows:
            if r[4]:
                frontier_row = r
                break

    # -- Slot 1: near-feasible best ---------------------------------------------
    near_feasible_rows = [
        r for r in rows if not r[4] and r[3] <= near_feasible_threshold
    ]
    near_feasible_rows.sort(key=lambda r: r[3])  # ascending feasibility (closest first)
    nearfeasible_row = near_feasible_rows[0] if near_feasible_rows else None

    if nearfeasible_row is None:
        # Fallback: second-best feasible (first feasible that is not the frontier).
        frontier_id_local = int(frontier_row[0]) if frontier_row is not None else -1
        for r in rows:
            if r[4] and int(r[0]) != frontier_id_local:
                nearfeasible_row = r
                break

    # -- Slot 2: stepping stone (max cosine distance from frontier) -------------
    stepping_row = None
    if frontier_row is not None:
        frontier_vec = _flatten_boundary(frontier_row[2])
        exclude_ids: set[int] = {int(frontier_row[0])}
        if nearfeasible_row is not None:
            exclude_ids.add(int(nearfeasible_row[0]))

        best_distance = -1.0
        for r in rows:
            if int(r[0]) in exclude_ids:
                continue
            if r[4] != 1:
                # Only consider feasible candidates for the stepping stone.
                continue
            vec = _flatten_boundary(r[2])
            dist = 1.0 - _cosine_similarity(frontier_vec, vec)
            if dist > best_distance:
                best_distance = dist
                stepping_row = r

    # -- Assemble paths (deduplicate by candidate id) --------------------------
    candidates_dir = run_dir / "candidates"
    selected: list[Path] = []
    seen_ids: set[int] = set()

    for row in (frontier_row, nearfeasible_row, stepping_row):
        if row is None:
            continue
        rid = int(row[0])
        if rid in seen_ids:
            continue
        seen_ids.add(rid)
        selected.append(candidates_dir / f"{row[1]}.json")
        if len(selected) >= n:
            break

    # Pad with remaining candidates in insertion order if we have fewer than n.
    if len(selected) < n:
        for row in rows:
            if len(selected) >= n:
                break
            rid = int(row[0])
            if rid in seen_ids:
                continue
            seen_ids.add(rid)
            selected.append(candidates_dir / f"{row[1]}.json")

    return selected[:n]


def read_snapshot(
    conn: sqlite3.Connection,
    problem_adapter: ProblemAdapter,
    run_dir: Path,
    *,
    experiment_id: int,
    near_feasible_threshold: float = _NEAR_FEASIBLE_MAX,
    n_parents: int = 3,
) -> CycleSnapshot:
    """Query DB and return a CycleSnapshot for the current harness cycle.

    Args:
        conn: SQLite connection.
        problem_adapter: Wraps the problem profile for direction-normalised objectives.
        run_dir: Run directory root; parent files live at
            ``run_dir/candidates/{design_hash}.json``.
        experiment_id: Experiment to scope all queries.
        near_feasible_threshold: Feasibility upper bound for the "near-feasible" label.
        n_parents: Number of diverse parent paths to select (default 3).
    """
    problem = problem_adapter.problem

    pending_count = _count_status(conn, experiment_id, problem, "pending")
    running_count = _count_running(conn, experiment_id, problem)
    done_count = _count_status(conn, experiment_id, problem, "done")
    near_feasible_count = _count_near_feasible(
        conn, experiment_id, problem, near_feasible_threshold
    )

    frontier_value, frontier_id = _query_frontier(
        conn, experiment_id, problem, problem_adapter
    )

    parent_paths = select_diverse_parents(
        conn,
        frontier_id,
        n_parents,
        run_dir=run_dir,
        experiment_id=experiment_id,
        problem=problem,
        near_feasible_threshold=near_feasible_threshold,
    )

    return CycleSnapshot(
        frontier_value=frontier_value,
        pending_count=pending_count,
        running_count=running_count,
        done_count=done_count,
        near_feasible_count=near_feasible_count,
        parent_paths=tuple(parent_paths),
    )
