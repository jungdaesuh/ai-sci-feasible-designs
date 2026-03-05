"""Persist cycle audit records to the harness_cycles table.

Additive only: creates one new table via CREATE TABLE IF NOT EXISTS.
Does not modify any existing tables in schema.py.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from harness.types import EnqueueResult


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS harness_cycles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id INTEGER NOT NULL,
    experiment_id INTEGER NOT NULL,
    snapshot_hash TEXT NOT NULL,
    proposal_source TEXT,
    script_hash TEXT,
    model TEXT,
    latency_ms INTEGER,
    candidate_count INTEGER NOT NULL DEFAULT 0,
    inserted INTEGER NOT NULL DEFAULT 0,
    skipped INTEGER NOT NULL DEFAULT 0,
    frontier_pre REAL,
    frontier_post REAL,
    frontier_delta REAL,
    exploration_mode TEXT,
    experience_summary TEXT,
    error_type TEXT,
    error_context TEXT,
    stop_decision INTEGER NOT NULL DEFAULT 0,
    stop_reason TEXT,
    created_at TEXT NOT NULL
)
"""


def ensure_harness_table(conn: sqlite3.Connection) -> None:
    """Create the harness_cycles table if it does not exist.

    Idempotent — safe to call on every governor startup.
    """
    conn.executescript(_CREATE_TABLE)


def record_cycle(
    conn: sqlite3.Connection,
    *,
    cycle_id: int,
    experiment_id: int,
    snapshot_hash: str,
    proposal_source: str | None = None,
    script_hash: str | None = None,
    model: str | None = None,
    latency_ms: int | None = None,
    candidate_count: int = 0,
    enqueue_result: EnqueueResult | None = None,
    frontier_pre: float | None = None,
    frontier_post: float | None = None,
    frontier_delta: float | None = None,
    exploration_mode: str | None = None,
    experience_summary: str | None = None,
    error_type: str | None = None,
    error_context: str | None = None,
    stop_decision: bool = False,
    stop_reason: str | None = None,
) -> None:
    """Insert one audit row into harness_cycles.

    Does NOT commit — caller controls the transaction boundary,
    consistent with enqueue_candidate() in p3_enqueue.py.
    """
    inserted = enqueue_result.inserted if enqueue_result else 0
    skipped = enqueue_result.skipped if enqueue_result else 0
    created_at = datetime.now(timezone.utc).isoformat()

    conn.execute(
        """
        INSERT INTO harness_cycles (
            cycle_id, experiment_id, snapshot_hash,
            proposal_source, script_hash, model, latency_ms,
            candidate_count, inserted, skipped,
            frontier_pre, frontier_post, frontier_delta,
            exploration_mode, experience_summary,
            error_type, error_context,
            stop_decision, stop_reason,
            created_at
        ) VALUES (
            ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?,
            ?, ?,
            ?, ?,
            ?, ?,
            ?
        )
        """,
        (
            int(cycle_id),
            int(experiment_id),
            str(snapshot_hash),
            proposal_source,
            script_hash,
            model,
            latency_ms,
            int(candidate_count),
            int(inserted),
            int(skipped),
            frontier_pre,
            frontier_post,
            frontier_delta,
            exploration_mode,
            experience_summary,
            error_type,
            error_context,
            int(stop_decision),
            stop_reason,
            created_at,
        ),
    )
