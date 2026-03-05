"""Shared fixtures for harness tests."""

from __future__ import annotations

import json
import sqlite3

import pytest

from ai_scientist.memory import hash_payload
from ai_scientist.memory.schema import init_db
from harness.recorder import ensure_harness_table


@pytest.fixture()
def harness_db(tmp_path):
    """In-memory-like SQLite DB with full schema + harness_cycles table."""
    db_path = tmp_path / "test_harness.sqlite"
    init_db(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        INSERT INTO experiments
            (started_at, config_json, git_sha, constellaration_sha, notes)
        VALUES
            ('2026-01-01T00:00:00+00:00', '{}', 'test_sha', 'const_sha', NULL)
        """
    )
    conn.commit()
    ensure_harness_table(conn)
    yield conn
    conn.close()


def insert_candidate_with_metric(
    conn: sqlite3.Connection,
    *,
    experiment_id: int = 1,
    problem: str = "p2",
    boundary: dict,
    feasibility: float,
    is_feasible: int,
    objective: float,
    constraint_margins: dict[str, float] | None = None,
) -> int:
    """Insert a candidate + metric row, return candidate_id."""
    design_hash = hash_payload(boundary)
    cursor = conn.execute(
        """
        INSERT INTO candidates
            (experiment_id, problem, params_json, seed, status,
             design_hash, operator_family, model_route)
        VALUES (?, ?, ?, 1, 'done', ?, 'test', 'test')
        """,
        (experiment_id, problem, json.dumps(boundary), design_hash),
    )
    candidate_id = cursor.lastrowid
    assert candidate_id is not None

    raw = {"metrics": {"lgradB": objective, "aspect_ratio": 8.0}}
    if constraint_margins is not None:
        raw["constraint_margins"] = constraint_margins

    conn.execute(
        """
        INSERT INTO metrics
            (candidate_id, raw_json, feasibility, objective, hv, is_feasible)
        VALUES (?, ?, ?, ?, NULL, ?)
        """,
        (candidate_id, json.dumps(raw), feasibility, objective, is_feasible),
    )
    conn.commit()
    return int(candidate_id)
