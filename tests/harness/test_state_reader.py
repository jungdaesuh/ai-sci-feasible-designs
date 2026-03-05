"""Tests for harness.state_reader — M2 acceptance.

Acceptance criteria:
- test_read_snapshot_from_seeded_db: returns valid CycleSnapshot from existing DB
- test_diverse_parents_returns_three: parent set has 3 distinct candidates
- test_stepping_stone_is_distant: stepping stone is maximally distant from frontier
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_scientist.memory import hash_payload
from ai_scientist.problem_profiles import P2_PROFILE
from harness.problem_adapter import ProblemAdapter
from harness.state_reader import read_snapshot, select_diverse_parents


def _insert_candidate(
    conn,
    *,
    boundary: dict,
    experiment_id: int = 1,
    problem: str = "p2",
    feasibility: float,
    is_feasible: int,
    objective: float,
) -> int:
    """Insert a candidate + metric row into the DB, return candidate_id."""
    design_hash = hash_payload(boundary)
    cursor = conn.execute(
        "INSERT INTO candidates "
        "(experiment_id, problem, params_json, seed, status, design_hash, operator_family, model_route) "
        "VALUES (?, ?, ?, 1, 'done', ?, 'test', 'test')",
        (experiment_id, problem, json.dumps(boundary), design_hash),
    )
    candidate_id = cursor.lastrowid
    assert candidate_id is not None
    raw = {"metrics": {"lgradB": objective, "aspect_ratio": 8.0}}
    conn.execute(
        "INSERT INTO metrics (candidate_id, raw_json, feasibility, objective, hv, is_feasible) "
        "VALUES (?, ?, ?, ?, NULL, ?)",
        (candidate_id, json.dumps(raw), feasibility, objective, is_feasible),
    )
    conn.commit()
    return int(candidate_id)


def _make_boundary(r00: float, z11: float) -> dict:
    """Create a minimal 2×3 boundary (m=0,1; ntor=1) with distinguishable vectors."""
    return {
        "r_cos": [[r00, 0.0, 0.0], [0.0, 0.05, 0.0]],
        "z_sin": [[0.0, 0.0, 0.0], [0.0, z11, 0.0]],
    }


def _seed_files(run_dir: Path, *boundaries: dict) -> None:
    """Write boundary JSON files to run_dir/candidates/ so paths resolve."""
    candidates_dir = run_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    for boundary in boundaries:
        design_hash = hash_payload(boundary)
        (candidates_dir / f"{design_hash}.json").write_text(json.dumps(boundary))


class TestReadSnapshot:
    def test_read_snapshot_from_seeded_db(self, harness_db, tmp_path):
        """read_snapshot returns a valid CycleSnapshot from an existing DB."""
        adapter = ProblemAdapter(P2_PROFILE)
        run_dir = tmp_path / "run"

        b_low = _make_boundary(1.0, 0.1)  # feasible, lower objective
        b_best = _make_boundary(1.0, 0.2)  # feasible, best objective
        b_nf = _make_boundary(0.5, 0.3)  # near-feasible (infeasible, low feasibility)
        _seed_files(run_dir, b_low, b_best, b_nf)

        _insert_candidate(
            harness_db, boundary=b_low, feasibility=0.0, is_feasible=1, objective=7.5
        )
        _insert_candidate(
            harness_db, boundary=b_best, feasibility=0.0, is_feasible=1, objective=9.0
        )
        _insert_candidate(
            harness_db, boundary=b_nf, feasibility=0.01, is_feasible=0, objective=5.0
        )

        snapshot = read_snapshot(harness_db, adapter, run_dir, experiment_id=1)

        assert snapshot.frontier_value == pytest.approx(9.0)
        assert snapshot.done_count == 3
        assert snapshot.near_feasible_count == 1
        assert snapshot.pending_count == 0
        assert snapshot.running_count == 0
        assert len(snapshot.parent_paths) == 3
        assert len(set(snapshot.parent_paths)) == len(snapshot.parent_paths)

    def test_read_snapshot_empty_db_returns_none_frontier(self, harness_db, tmp_path):
        """read_snapshot with no candidates returns None frontier and zero counts."""
        adapter = ProblemAdapter(P2_PROFILE)
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        snapshot = read_snapshot(harness_db, adapter, run_dir, experiment_id=1)

        assert snapshot.frontier_value is None
        assert snapshot.done_count == 0
        assert snapshot.pending_count == 0
        assert snapshot.running_count == 0
        assert snapshot.near_feasible_count == 0
        assert snapshot.parent_paths == ()

    def test_read_snapshot_counts_pending(self, harness_db, tmp_path):
        """pending_count reflects candidates with status='pending'."""
        adapter = ProblemAdapter(P2_PROFILE)
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        b = _make_boundary(1.0, 0.1)
        h = hash_payload(b)
        harness_db.execute(
            "INSERT INTO candidates "
            "(experiment_id, problem, params_json, seed, status, design_hash, operator_family, model_route) "
            "VALUES (1, 'p2', ?, 1, 'pending', ?, 'test', 'test')",
            (json.dumps(b), h),
        )
        harness_db.commit()

        snapshot = read_snapshot(harness_db, adapter, run_dir, experiment_id=1)

        assert snapshot.pending_count == 1
        assert snapshot.running_count == 0
        assert snapshot.done_count == 0

    def test_read_snapshot_counts_running(self, harness_db, tmp_path):
        """running_count reflects candidates with status like 'running:*'."""
        adapter = ProblemAdapter(P2_PROFILE)
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        b = _make_boundary(1.0, 0.1)
        h = hash_payload(b)
        harness_db.execute(
            "INSERT INTO candidates "
            "(experiment_id, problem, params_json, seed, status, design_hash, operator_family, model_route) "
            "VALUES (1, 'p2', ?, 1, 'running:1:20260304T120000', ?, 'test', 'test')",
            (json.dumps(b), h),
        )
        harness_db.commit()

        snapshot = read_snapshot(harness_db, adapter, run_dir, experiment_id=1)

        assert snapshot.running_count == 1
        assert snapshot.pending_count == 0

    def test_read_snapshot_near_feasible_threshold(self, harness_db, tmp_path):
        """Only candidates at or below near_feasible_threshold are counted."""
        adapter = ProblemAdapter(P2_PROFILE)
        run_dir = tmp_path / "run"

        b_inside = _make_boundary(1.0, 0.1)  # feasibility=0.015 <= 0.02 → near-feasible
        b_outside = _make_boundary(
            0.9, 0.2
        )  # feasibility=0.05 > 0.02 → NOT near-feasible
        _seed_files(run_dir, b_inside, b_outside)

        _insert_candidate(
            harness_db,
            boundary=b_inside,
            feasibility=0.015,
            is_feasible=0,
            objective=5.0,
        )
        _insert_candidate(
            harness_db,
            boundary=b_outside,
            feasibility=0.05,
            is_feasible=0,
            objective=4.0,
        )

        snapshot = read_snapshot(harness_db, adapter, run_dir, experiment_id=1)

        assert snapshot.near_feasible_count == 1


class TestDiverseParents:
    def test_diverse_parents_returns_three(self, harness_db, tmp_path):
        """select_diverse_parents returns exactly 3 distinct paths when ≥3 done."""
        run_dir = tmp_path / "run"

        b1 = _make_boundary(1.0, 0.1)
        b2 = _make_boundary(0.9, 0.2)
        b3 = _make_boundary(0.5, 0.3)
        b4 = _make_boundary(0.1, 0.9)
        _seed_files(run_dir, b1, b2, b3, b4)

        cid1 = _insert_candidate(
            harness_db, boundary=b1, feasibility=0.0, is_feasible=1, objective=9.0
        )
        _insert_candidate(
            harness_db, boundary=b2, feasibility=0.0, is_feasible=1, objective=7.0
        )
        _insert_candidate(
            harness_db, boundary=b3, feasibility=0.01, is_feasible=0, objective=5.0
        )
        _insert_candidate(
            harness_db, boundary=b4, feasibility=0.0, is_feasible=1, objective=6.0
        )

        parents = select_diverse_parents(
            harness_db,
            cid1,
            n=3,
            run_dir=run_dir,
            experiment_id=1,
            problem="p2",
        )

        assert len(parents) == 3
        assert len(set(parents)) == 3  # all distinct paths

    def test_stepping_stone_is_distant(self, harness_db, tmp_path):
        """Stepping stone is the feasible candidate with max cosine distance from frontier."""
        run_dir = tmp_path / "run"

        # Frontier vector lives in r_cos[0][0] direction → [1,0,0,0,0,0, 0,0,0,0,0,0]
        frontier_b = {
            "r_cos": [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            "z_sin": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        }
        # Near-parallel to frontier (cosine distance ≈ 0)
        close_b = {
            "r_cos": [[0.99, 0.01, 0.0], [0.0, 0.0, 0.0]],
            "z_sin": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        }
        # Orthogonal to frontier in z_sin dimension (cosine distance = 1.0)
        distant_b = {
            "r_cos": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            "z_sin": [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        }
        _seed_files(run_dir, frontier_b, close_b, distant_b)

        frontier_cid = _insert_candidate(
            harness_db,
            boundary=frontier_b,
            feasibility=0.0,
            is_feasible=1,
            objective=9.0,
        )
        _insert_candidate(
            harness_db, boundary=close_b, feasibility=0.0, is_feasible=1, objective=7.0
        )
        _insert_candidate(
            harness_db,
            boundary=distant_b,
            feasibility=0.0,
            is_feasible=1,
            objective=6.0,
        )

        parents = select_diverse_parents(
            harness_db,
            frontier_cid,
            n=3,
            run_dir=run_dir,
            experiment_id=1,
            problem="p2",
        )

        distant_hash = hash_payload(distant_b)
        distant_path = run_dir / "candidates" / f"{distant_hash}.json"
        assert distant_path in parents

    def test_diverse_parents_fewer_than_n(self, harness_db, tmp_path):
        """Returns fewer than n paths when DB has fewer done candidates."""
        run_dir = tmp_path / "run"
        b = _make_boundary(1.0, 0.1)
        _seed_files(run_dir, b)

        cid = _insert_candidate(
            harness_db, boundary=b, feasibility=0.0, is_feasible=1, objective=8.0
        )

        parents = select_diverse_parents(
            harness_db,
            cid,
            n=3,
            run_dir=run_dir,
            experiment_id=1,
            problem="p2",
        )

        assert len(parents) == 1
        assert len(set(parents)) == len(parents)  # no duplicates

    def test_diverse_parents_empty_db(self, harness_db, tmp_path):
        """Returns empty list when no done candidates exist."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        parents = select_diverse_parents(
            harness_db,
            None,
            n=3,
            run_dir=run_dir,
            experiment_id=1,
            problem="p2",
        )

        assert parents == []

    def test_diverse_parents_frontier_is_first(self, harness_db, tmp_path):
        """Slot 0 path always corresponds to the frontier candidate."""
        run_dir = tmp_path / "run"

        b_frontier = _make_boundary(1.0, 0.1)
        b_other = _make_boundary(0.5, 0.5)
        _seed_files(run_dir, b_frontier, b_other)

        frontier_cid = _insert_candidate(
            harness_db,
            boundary=b_frontier,
            feasibility=0.0,
            is_feasible=1,
            objective=9.0,
        )
        _insert_candidate(
            harness_db, boundary=b_other, feasibility=0.0, is_feasible=1, objective=6.0
        )

        parents = select_diverse_parents(
            harness_db,
            frontier_cid,
            n=3,
            run_dir=run_dir,
            experiment_id=1,
            problem="p2",
        )

        frontier_hash = hash_payload(b_frontier)
        frontier_path = run_dir / "candidates" / f"{frontier_hash}.json"
        assert frontier_path in parents
        assert parents[0] == frontier_path
