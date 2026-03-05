"""Tests for harness.recorder — M1 acceptance."""

from __future__ import annotations

from harness.recorder import ensure_harness_table, record_cycle
from harness.types import EnqueueResult


def test_ensure_table_idempotent(harness_db):
    """Calling ensure_harness_table twice does not raise."""
    ensure_harness_table(harness_db)
    ensure_harness_table(harness_db)
    # Table exists and is queryable.
    row = harness_db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='harness_cycles'"
    ).fetchone()
    assert row is not None


def test_record_cycle_persists(harness_db):
    record_cycle(
        harness_db,
        cycle_id=1,
        experiment_id=1,
        snapshot_hash="abc123",
        proposal_source="print('hello')",
        script_hash="def456",
        model="test-model",
        latency_ms=500,
        candidate_count=3,
        enqueue_result=EnqueueResult(inserted=2, skipped=1),
        frontier_pre=8.0,
        frontier_post=8.5,
        frontier_delta=0.5,
        exploration_mode="exploit",
    )

    row = harness_db.execute(
        "SELECT * FROM harness_cycles WHERE cycle_id = 1"
    ).fetchone()
    assert row is not None
    assert row["cycle_id"] == 1
    assert row["experiment_id"] == 1
    assert row["snapshot_hash"] == "abc123"
    assert row["proposal_source"] == "print('hello')"
    assert row["model"] == "test-model"
    assert row["candidate_count"] == 3
    assert row["inserted"] == 2
    assert row["skipped"] == 1
    assert row["frontier_delta"] == 0.5
    assert row["exploration_mode"] == "exploit"
    assert row["stop_decision"] == 0
    assert row["created_at"] is not None


def test_record_cycle_with_error_context(harness_db):
    record_cycle(
        harness_db,
        cycle_id=2,
        experiment_id=1,
        snapshot_hash="err_hash",
        error_type="SandboxViolation",
        error_context="import os detected in proposal",
        stop_decision=True,
        stop_reason="circuit_break",
    )

    row = harness_db.execute(
        "SELECT * FROM harness_cycles WHERE cycle_id = 2"
    ).fetchone()
    assert row is not None
    assert row["error_type"] == "SandboxViolation"
    assert row["error_context"] == "import os detected in proposal"
    assert row["stop_decision"] == 1
    assert row["stop_reason"] == "circuit_break"
    assert row["candidate_count"] == 0
    assert row["inserted"] == 0
