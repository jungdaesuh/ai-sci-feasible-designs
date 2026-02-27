# ruff: noqa: E402
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

from ai_scientist.memory.schema import init_db

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import p3_governor_ab as ab


def _create_experiment(conn: sqlite3.Connection, *, notes: str) -> int:
    cur = conn.execute(
        """
        INSERT INTO experiments (started_at, config_json, git_sha, constellaration_sha, notes)
        VALUES (?, ?, ?, ?, ?)
        """,
        ("2026-02-25T00:00:00+00:00", "{}", "deadbeef", "const-sha", notes),
    )
    experiment_id = cur.lastrowid
    assert experiment_id is not None
    return int(experiment_id)


def _insert_eval(
    conn: sqlite3.Connection,
    *,
    experiment_id: int,
    design_hash: str,
    route: str,
    is_feasible: bool,
    feasibility: float,
    objective: float | None,
    aspect: float | None,
    error: str | None = None,
) -> None:
    candidate_cur = conn.execute(
        """
        INSERT INTO candidates
        (experiment_id, problem, params_json, seed, status, design_hash, lineage_parent_hashes_json, novelty_score, operator_family, model_route)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            int(experiment_id),
            "p3",
            json.dumps({"r_cos": [[1.0]], "z_sin": [[0.0]]}),
            1,
            "done",
            design_hash,
            "[]",
            0.2,
            "blend",
            route,
        ),
    )
    candidate_id = candidate_cur.lastrowid
    assert candidate_id is not None
    raw = {"metrics": {"aspect_ratio": aspect}, "error": error}
    conn.execute(
        """
        INSERT INTO metrics (candidate_id, raw_json, feasibility, objective, hv, is_feasible)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            int(candidate_id),
            json.dumps(raw),
            float(feasibility),
            objective,
            None,
            1 if is_feasible else 0,
        ),
    )


def test_compute_hv_2d_matches_expected_rectangle_union() -> None:
    hv_value = ab._compute_hv(
        [(8.0, 10.0), (5.0, 8.0)],
        ref_x=1.0,
        ref_y=20.0,
    )
    assert hv_value == 102.0


def test_build_report_prefers_adaptive_arm_at_fixed_budget(tmp_path: Path) -> None:
    db_path = tmp_path / "wm.sqlite"
    init_db(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        static_id = _create_experiment(conn, notes="static")
        adaptive_id = _create_experiment(conn, notes="adaptive")

        _insert_eval(
            conn,
            experiment_id=static_id,
            design_hash="s1",
            route="governor_static_recipe/mirror",
            is_feasible=False,
            feasibility=1.0,
            objective=None,
            aspect=12.0,
        )
        _insert_eval(
            conn,
            experiment_id=static_id,
            design_hash="s2",
            route="governor_static_recipe/mirror",
            is_feasible=True,
            feasibility=0.0,
            objective=4.0,
            aspect=12.0,
        )
        _insert_eval(
            conn,
            experiment_id=static_id,
            design_hash="s3",
            route="governor_static_recipe/mirror",
            is_feasible=False,
            feasibility=0.5,
            objective=None,
            aspect=11.0,
        )
        _insert_eval(
            conn,
            experiment_id=adaptive_id,
            design_hash="a1",
            route="governor_adaptive/near_feasible/log10_qi",
            is_feasible=True,
            feasibility=0.0,
            objective=5.0,
            aspect=12.0,
        )
        _insert_eval(
            conn,
            experiment_id=adaptive_id,
            design_hash="a2",
            route="governor_adaptive/near_feasible/log10_qi",
            is_feasible=False,
            feasibility=0.7,
            objective=None,
            aspect=12.0,
        )
        _insert_eval(
            conn,
            experiment_id=adaptive_id,
            design_hash="a3",
            route="governor_adaptive/near_feasible/log10_qi",
            is_feasible=True,
            feasibility=0.0,
            objective=6.0,
            aspect=11.0,
        )
        conn.commit()
    finally:
        conn.close()

    db_conn = ab._connect(db_path)
    try:
        static_rows = ab._fetch_eval_rows(db_conn, experiment_id=static_id)
        adaptive_rows = ab._fetch_eval_rows(db_conn, experiment_id=adaptive_id)
    finally:
        db_conn.close()

    budget = ab._resolve_budget(
        budget_requested=3,
        static_count=len(static_rows),
        adaptive_count=len(adaptive_rows),
    )
    static_summary = ab._build_arm_summary(
        experiment_id=static_id,
        rows=static_rows,
        budget=budget,
        ref_x=1.0,
        ref_y=20.0,
    )
    adaptive_summary = ab._build_arm_summary(
        experiment_id=adaptive_id,
        rows=adaptive_rows,
        budget=budget,
        ref_x=1.0,
        ref_y=20.0,
    )
    report = ab._build_report(
        static_summary=static_summary,
        adaptive_summary=adaptive_summary,
        budget_requested=3,
        budget_used=budget,
        ref_x=1.0,
        ref_y=20.0,
    )

    assert report["budget_used"] == 3
    assert report["gates"]["pass"] is True
    assert report["gates"]["m23_contract_pass"] is True
    assert report["gates"]["budget_meets_m24_minimum"] is False
    assert report["gates"]["non_trivial_feasible_evidence"] is True
    assert report["gates"]["m24_performance_evidence_pass"] is False
    assert report["deltas"]["hv_at_budget"] > 0.0
    assert report["deltas"]["feasible_yield_per_100"] > 0.0
    assert report["static"]["route_summary_at_budget"]["static_path_rows"] == 3
    assert report["adaptive"]["route_summary_at_budget"]["adaptive_path_rows"] == 3


def test_m24_gate_passes_with_budget_and_feasible_evidence() -> None:
    static_summary = ab.ArmSummary(
        experiment_id=1,
        eval_count_total=20,
        eval_count_budget=20,
        feasible_count_budget=1,
        feasible_yield_per_100=5.0,
        hv_at_budget=10.0,
        hv_final=11.0,
        best_lgradb_feasible_at_budget=6.0,
        route_summary_at_budget={"static_path_rows": 20, "adaptive_path_rows": 0},
    )
    adaptive_summary = ab.ArmSummary(
        experiment_id=2,
        eval_count_total=20,
        eval_count_budget=20,
        feasible_count_budget=2,
        feasible_yield_per_100=10.0,
        hv_at_budget=12.0,
        hv_final=13.0,
        best_lgradb_feasible_at_budget=7.0,
        route_summary_at_budget={
            "static_path_rows": 0,
            "adaptive_path_rows": 20,
            "fallback_static_delegate_rows": 0,
        },
    )
    report = ab._build_report(
        static_summary=static_summary,
        adaptive_summary=adaptive_summary,
        budget_requested=20,
        budget_used=20,
        ref_x=1.0,
        ref_y=20.0,
    )

    assert report["gates"]["m23_contract_pass"] is True
    assert report["gates"]["budget_meets_m24_minimum"] is True
    assert report["gates"]["non_trivial_feasible_evidence"] is True
    assert report["gates"]["m24_performance_evidence_pass"] is True


def test_m24_gate_fails_without_feasible_evidence() -> None:
    static_summary = ab.ArmSummary(
        experiment_id=1,
        eval_count_total=20,
        eval_count_budget=20,
        feasible_count_budget=0,
        feasible_yield_per_100=0.0,
        hv_at_budget=0.0,
        hv_final=0.0,
        best_lgradb_feasible_at_budget=None,
        route_summary_at_budget={"static_path_rows": 20, "adaptive_path_rows": 0},
    )
    adaptive_summary = ab.ArmSummary(
        experiment_id=2,
        eval_count_total=20,
        eval_count_budget=20,
        feasible_count_budget=0,
        feasible_yield_per_100=0.0,
        hv_at_budget=0.0,
        hv_final=0.0,
        best_lgradb_feasible_at_budget=None,
        route_summary_at_budget={
            "static_path_rows": 0,
            "adaptive_path_rows": 20,
            "fallback_static_delegate_rows": 0,
        },
    )
    report = ab._build_report(
        static_summary=static_summary,
        adaptive_summary=adaptive_summary,
        budget_requested=20,
        budget_used=20,
        ref_x=1.0,
        ref_y=20.0,
    )

    assert report["gates"]["m23_contract_pass"] is True
    assert report["gates"]["budget_meets_m24_minimum"] is True
    assert report["gates"]["non_trivial_feasible_evidence"] is False
    assert report["gates"]["m24_performance_evidence_pass"] is False


def test_fetch_eval_rows_flags_runtime_error_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "wm.sqlite"
    init_db(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        exp_id = _create_experiment(conn, notes="runtime-error-check")
        _insert_eval(
            conn,
            experiment_id=exp_id,
            design_hash="e1",
            route="governor_static_recipe/mirror",
            is_feasible=False,
            feasibility=float("inf"),
            objective=None,
            aspect=None,
            error="runtime failure",
        )
        _insert_eval(
            conn,
            experiment_id=exp_id,
            design_hash="e2",
            route="governor_static_recipe/mirror",
            is_feasible=False,
            feasibility=1.0,
            objective=None,
            aspect=12.0,
            error=None,
        )
        conn.commit()
    finally:
        conn.close()

    db_conn = ab._connect(db_path)
    try:
        rows = ab._fetch_eval_rows(db_conn, experiment_id=exp_id)
    finally:
        db_conn.close()

    assert len(rows) == 2
    assert rows[0].has_eval_error is True
    assert rows[1].has_eval_error is False


def test_resolve_budget_auto_uses_max_shared() -> None:
    assert (
        ab._resolve_budget(
            budget_requested=0,
            static_count=7,
            adaptive_count=5,
        )
        == 5
    )


def test_resolve_budget_rejects_negative_budget() -> None:
    with pytest.raises(ValueError, match="--budget must be >= 0"):
        ab._resolve_budget(
            budget_requested=-1,
            static_count=7,
            adaptive_count=5,
        )


def test_resolve_budget_rejects_missing_eval_rows() -> None:
    with pytest.raises(
        ValueError,
        match="Both experiments must have at least one evaluated metric row",
    ):
        ab._resolve_budget(
            budget_requested=0,
            static_count=0,
            adaptive_count=5,
        )


def test_resolve_budget_clips_positive_request_to_max_shared() -> None:
    assert (
        ab._resolve_budget(
            budget_requested=12,
            static_count=7,
            adaptive_count=5,
        )
        == 5
    )


def test_validate_m24_min_budget_rejects_values_below_default() -> None:
    with pytest.raises(ValueError, match="--m24-min-budget must be >="):
        ab._validate_m24_min_budget(19)


def test_validate_contract_rejects_same_experiment_id() -> None:
    summary = ab.ArmSummary(
        experiment_id=1,
        eval_count_total=10,
        eval_count_budget=5,
        feasible_count_budget=1,
        feasible_yield_per_100=20.0,
        hv_at_budget=1.0,
        hv_final=2.0,
        best_lgradb_feasible_at_budget=3.0,
        route_summary_at_budget={"static_path_rows": 5, "adaptive_path_rows": 5},
    )
    with pytest.raises(ValueError, match="must be distinct"):
        ab._validate_ab_contract(
            static_summary=summary,
            adaptive_summary=summary,
            allow_legacy_route_metadata=False,
        )


def test_validate_contract_rejects_missing_route_labels() -> None:
    static_summary = ab.ArmSummary(
        experiment_id=1,
        eval_count_total=10,
        eval_count_budget=5,
        feasible_count_budget=1,
        feasible_yield_per_100=20.0,
        hv_at_budget=1.0,
        hv_final=2.0,
        best_lgradb_feasible_at_budget=3.0,
        route_summary_at_budget={"static_path_rows": 0, "adaptive_path_rows": 0},
    )
    adaptive_summary = ab.ArmSummary(
        experiment_id=2,
        eval_count_total=10,
        eval_count_budget=5,
        feasible_count_budget=1,
        feasible_yield_per_100=20.0,
        hv_at_budget=1.0,
        hv_final=2.0,
        best_lgradb_feasible_at_budget=3.0,
        route_summary_at_budget={"static_path_rows": 0, "adaptive_path_rows": 0},
    )
    with pytest.raises(ValueError, match="Static arm has zero static-route rows"):
        ab._validate_ab_contract(
            static_summary=static_summary,
            adaptive_summary=adaptive_summary,
            allow_legacy_route_metadata=False,
        )


def test_validate_contract_allows_legacy_route_labels_with_flag() -> None:
    static_summary = ab.ArmSummary(
        experiment_id=1,
        eval_count_total=10,
        eval_count_budget=5,
        feasible_count_budget=1,
        feasible_yield_per_100=20.0,
        hv_at_budget=1.0,
        hv_final=2.0,
        best_lgradb_feasible_at_budget=3.0,
        route_summary_at_budget={"static_path_rows": 0, "adaptive_path_rows": 0},
    )
    adaptive_summary = ab.ArmSummary(
        experiment_id=2,
        eval_count_total=10,
        eval_count_budget=5,
        feasible_count_budget=1,
        feasible_yield_per_100=20.0,
        hv_at_budget=1.0,
        hv_final=2.0,
        best_lgradb_feasible_at_budget=3.0,
        route_summary_at_budget={"static_path_rows": 0, "adaptive_path_rows": 0},
    )
    ab._validate_ab_contract(
        static_summary=static_summary,
        adaptive_summary=adaptive_summary,
        allow_legacy_route_metadata=True,
    )


def test_validate_contract_rejects_mixed_route_contamination() -> None:
    static_summary = ab.ArmSummary(
        experiment_id=1,
        eval_count_total=10,
        eval_count_budget=5,
        feasible_count_budget=1,
        feasible_yield_per_100=20.0,
        hv_at_budget=1.0,
        hv_final=2.0,
        best_lgradb_feasible_at_budget=3.0,
        route_summary_at_budget={"static_path_rows": 4, "adaptive_path_rows": 1},
    )
    adaptive_summary = ab.ArmSummary(
        experiment_id=2,
        eval_count_total=10,
        eval_count_budget=5,
        feasible_count_budget=1,
        feasible_yield_per_100=20.0,
        hv_at_budget=1.0,
        hv_final=2.0,
        best_lgradb_feasible_at_budget=3.0,
        route_summary_at_budget={
            "static_path_rows": 0,
            "adaptive_path_rows": 5,
            "fallback_static_delegate_rows": 0,
        },
    )
    with pytest.raises(ValueError, match="Static arm route contamination detected"):
        ab._validate_ab_contract(
            static_summary=static_summary,
            adaptive_summary=adaptive_summary,
            allow_legacy_route_metadata=False,
        )


def test_validate_contract_rejects_fallback_only_adaptive() -> None:
    static_summary = ab.ArmSummary(
        experiment_id=1,
        eval_count_total=10,
        eval_count_budget=5,
        feasible_count_budget=1,
        feasible_yield_per_100=20.0,
        hv_at_budget=1.0,
        hv_final=2.0,
        best_lgradb_feasible_at_budget=3.0,
        route_summary_at_budget={"static_path_rows": 5, "adaptive_path_rows": 0},
    )
    adaptive_summary = ab.ArmSummary(
        experiment_id=2,
        eval_count_total=10,
        eval_count_budget=5,
        feasible_count_budget=1,
        feasible_yield_per_100=20.0,
        hv_at_budget=1.0,
        hv_final=2.0,
        best_lgradb_feasible_at_budget=3.0,
        route_summary_at_budget={
            "static_path_rows": 0,
            "adaptive_path_rows": 5,
            "fallback_static_delegate_rows": 5,
        },
    )
    with pytest.raises(ValueError, match="contains only static-delegate fallback"):
        ab._validate_ab_contract(
            static_summary=static_summary,
            adaptive_summary=adaptive_summary,
            allow_legacy_route_metadata=False,
        )


def test_validate_contract_rejects_adaptive_route_contamination() -> None:
    static_summary = ab.ArmSummary(
        experiment_id=1,
        eval_count_total=10,
        eval_count_budget=5,
        feasible_count_budget=1,
        feasible_yield_per_100=20.0,
        hv_at_budget=1.0,
        hv_final=2.0,
        best_lgradb_feasible_at_budget=3.0,
        route_summary_at_budget={"static_path_rows": 5, "adaptive_path_rows": 0},
    )
    adaptive_summary = ab.ArmSummary(
        experiment_id=2,
        eval_count_total=10,
        eval_count_budget=5,
        feasible_count_budget=1,
        feasible_yield_per_100=20.0,
        hv_at_budget=1.0,
        hv_final=2.0,
        best_lgradb_feasible_at_budget=3.0,
        route_summary_at_budget={
            "static_path_rows": 1,
            "adaptive_path_rows": 4,
            "fallback_static_delegate_rows": 0,
        },
    )
    with pytest.raises(ValueError, match="Adaptive arm route contamination detected"):
        ab._validate_ab_contract(
            static_summary=static_summary,
            adaptive_summary=adaptive_summary,
            allow_legacy_route_metadata=False,
        )


def test_validate_contract_rejects_eval_errors_at_budget() -> None:
    static_summary = ab.ArmSummary(
        experiment_id=1,
        eval_count_total=10,
        eval_count_budget=5,
        feasible_count_budget=1,
        feasible_yield_per_100=20.0,
        hv_at_budget=1.0,
        hv_final=2.0,
        best_lgradb_feasible_at_budget=3.0,
        route_summary_at_budget={"static_path_rows": 5, "adaptive_path_rows": 0},
        error_count_budget=1,
    )
    adaptive_summary = ab.ArmSummary(
        experiment_id=2,
        eval_count_total=10,
        eval_count_budget=5,
        feasible_count_budget=1,
        feasible_yield_per_100=20.0,
        hv_at_budget=1.0,
        hv_final=2.0,
        best_lgradb_feasible_at_budget=3.0,
        route_summary_at_budget={
            "static_path_rows": 0,
            "adaptive_path_rows": 5,
            "fallback_static_delegate_rows": 0,
        },
        error_count_budget=0,
    )
    with pytest.raises(ValueError, match="Static arm exceeded allowed eval-error rows"):
        ab._validate_ab_contract(
            static_summary=static_summary,
            adaptive_summary=adaptive_summary,
            allow_legacy_route_metadata=False,
            max_error_rows_budget=0,
        )


def test_validate_contract_allows_eval_errors_when_threshold_relaxed() -> None:
    static_summary = ab.ArmSummary(
        experiment_id=1,
        eval_count_total=10,
        eval_count_budget=5,
        feasible_count_budget=1,
        feasible_yield_per_100=20.0,
        hv_at_budget=1.0,
        hv_final=2.0,
        best_lgradb_feasible_at_budget=3.0,
        route_summary_at_budget={"static_path_rows": 5, "adaptive_path_rows": 0},
        error_count_budget=1,
    )
    adaptive_summary = ab.ArmSummary(
        experiment_id=2,
        eval_count_total=10,
        eval_count_budget=5,
        feasible_count_budget=1,
        feasible_yield_per_100=20.0,
        hv_at_budget=1.0,
        hv_final=2.0,
        best_lgradb_feasible_at_budget=3.0,
        route_summary_at_budget={
            "static_path_rows": 0,
            "adaptive_path_rows": 5,
            "fallback_static_delegate_rows": 0,
        },
        error_count_budget=1,
    )
    ab._validate_ab_contract(
        static_summary=static_summary,
        adaptive_summary=adaptive_summary,
        allow_legacy_route_metadata=False,
        max_error_rows_budget=1,
    )


def test_enforce_required_gates_accepts_m24_pass_report() -> None:
    report = {
        "gates": {
            "m23_contract_pass": True,
            "m24_performance_evidence_pass": True,
        }
    }
    ab._enforce_required_gates(
        report,
        require_m23_pass=True,
        require_m24_pass=True,
    )


def test_enforce_required_gates_rejects_m24_failure_when_required() -> None:
    report = {
        "gates": {
            "m23_contract_pass": True,
            "m24_performance_evidence_pass": False,
        }
    }
    with pytest.raises(ValueError, match="M2\\.4 performance-evidence gate failed"):
        ab._enforce_required_gates(
            report,
            require_m23_pass=False,
            require_m24_pass=True,
        )
