# ruff: noqa: E402
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

from ai_scientist.memory.schema import init_db

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import m3_policy_hardening_validate as m36


def _p1_report(*, budget_used: int = 20, m33_pass: bool = True) -> dict:
    return {
        "generated_at": "2026-02-26T00:00:00+00:00",
        "problem": "p1",
        "budget_used": budget_used,
        "gates": {
            "m33_contract_pass": m33_pass,
            "non_trivial_feasible_evidence": True,
            "best_feasible_metric_non_regression": True,
            "feasible_yield_non_regression": True,
        },
    }


def _p2_report(*, budget_used: int = 20, m33_pass: bool = True) -> dict:
    return {
        "generated_at": "2026-02-26T00:00:00+00:00",
        "problem": "p2",
        "budget_used": budget_used,
        "gates": {
            "m33_contract_pass": m33_pass,
            "non_trivial_feasible_evidence": True,
            "best_feasible_metric_non_regression": True,
            "feasible_yield_non_regression": True,
        },
    }


def _p3_report(
    *,
    static_experiment_id: int = 1,
    adaptive_experiment_id: int = 2,
    budget_used: int = 20,
    m24_pass: bool = True,
) -> dict:
    return {
        "generated_at": "2026-02-26T00:00:00+00:00",
        "budget_used": budget_used,
        "gates": {
            "m24_performance_evidence_pass": m24_pass,
        },
        "static": {
            "experiment_id": static_experiment_id,
            "route_summary_at_budget": {
                "novelty_reject_rate": 0.15,
                "model_routes": {"governor_static_recipe/mirror": 20},
            },
        },
        "adaptive": {
            "experiment_id": adaptive_experiment_id,
            "route_summary_at_budget": {
                "novelty_reject_rate": 0.1,
                "model_routes": {"governor_adaptive/near_feasible/mirror": 20},
            },
        },
    }


def _insert_experiment(conn: sqlite3.Connection, experiment_id: int) -> None:
    conn.execute(
        """
        INSERT INTO experiments
        (id, started_at, config_json, git_sha, constellaration_sha, notes)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            int(experiment_id),
            "2026-02-26T00:00:00+00:00",
            "{}",
            "deadbeef",
            "const-sha",
            None,
        ),
    )


def _insert_reward_event(
    conn: sqlite3.Connection,
    *,
    experiment_id: int,
    model_route: str,
    reward: float,
) -> None:
    conn.execute(
        """
        INSERT INTO model_router_reward_events
        (experiment_id, problem, model_route, window_size, previous_feasible_yield,
         current_feasible_yield, previous_hv, current_hv, reward, reward_components_json,
         created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            int(experiment_id),
            "p3",
            str(model_route),
            20,
            10.0,
            11.0,
            1.0,
            1.1,
            float(reward),
            "{}",
            "2026-02-26T00:00:00+00:00",
        ),
    )


def _reward_summaries(
    db_path: Path, *, static_id: int, adaptive_id: int
) -> tuple[dict, dict]:
    conn = m36._connect(db_path)
    try:
        return (
            m36._router_reward_summary(conn, experiment_id=static_id),
            m36._router_reward_summary(conn, experiment_id=adaptive_id),
        )
    finally:
        conn.close()


def test_m36_report_passes_with_reward_telemetry(tmp_path: Path) -> None:
    db_path = tmp_path / "wm.sqlite"
    init_db(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        _insert_experiment(conn, 1)
        _insert_experiment(conn, 2)
        _insert_reward_event(
            conn,
            experiment_id=1,
            model_route="governor_static_recipe/mirror",
            reward=0.05,
        )
        _insert_reward_event(
            conn,
            experiment_id=2,
            model_route="governor_adaptive/near_feasible/mirror",
            reward=0.15,
        )
        conn.commit()
    finally:
        conn.close()

    p1_summary = m36._extract_p1_p2_summary(
        report=_p1_report(),
        expected_problem="p1",
        min_budget=20,
    )
    p2_summary = m36._extract_p1_p2_summary(
        report=_p2_report(),
        expected_problem="p2",
        min_budget=20,
    )
    p3_summary = m36._extract_p3_summary(report=_p3_report(), min_budget=20)
    static_reward, adaptive_reward = _reward_summaries(
        db_path,
        static_id=1,
        adaptive_id=2,
    )
    report = m36._build_report(
        p1_summary=p1_summary,
        p2_summary=p2_summary,
        p3_summary=p3_summary,
        static_reward=static_reward,
        adaptive_reward=adaptive_reward,
        min_budget=20,
        require_reward_telemetry=True,
    )

    assert report["gates"]["p3_reward_delta_available"] is True
    assert report["router_reward"]["deltas"]["avg_reward"] == pytest.approx(0.10)
    assert report["gates"]["m36_contract_pass"] is True


def test_m36_report_strict_mode_fails_without_reward_telemetry(tmp_path: Path) -> None:
    db_path = tmp_path / "wm.sqlite"
    init_db(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        _insert_experiment(conn, 1)
        _insert_experiment(conn, 2)
        conn.commit()
    finally:
        conn.close()

    p1_summary = m36._extract_p1_p2_summary(
        report=_p1_report(),
        expected_problem="p1",
        min_budget=20,
    )
    p2_summary = m36._extract_p1_p2_summary(
        report=_p2_report(),
        expected_problem="p2",
        min_budget=20,
    )
    p3_summary = m36._extract_p3_summary(report=_p3_report(), min_budget=20)
    static_reward, adaptive_reward = _reward_summaries(
        db_path,
        static_id=1,
        adaptive_id=2,
    )
    report = m36._build_report(
        p1_summary=p1_summary,
        p2_summary=p2_summary,
        p3_summary=p3_summary,
        static_reward=static_reward,
        adaptive_reward=adaptive_reward,
        min_budget=20,
        require_reward_telemetry=True,
    )

    assert report["gates"]["p3_reward_delta_available"] is False
    assert report["gates"]["m36_contract_pass"] is False
    with pytest.raises(ValueError, match="M3\\.6 fixed-budget gate failed"):
        m36._enforce_required_gates(report, require_m36_pass=True)


def test_m36_report_allows_missing_reward_telemetry_with_override(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "wm.sqlite"
    init_db(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        _insert_experiment(conn, 1)
        _insert_experiment(conn, 2)
        conn.commit()
    finally:
        conn.close()

    p1_summary = m36._extract_p1_p2_summary(
        report=_p1_report(),
        expected_problem="p1",
        min_budget=20,
    )
    p2_summary = m36._extract_p1_p2_summary(
        report=_p2_report(),
        expected_problem="p2",
        min_budget=20,
    )
    p3_summary = m36._extract_p3_summary(report=_p3_report(), min_budget=20)
    static_reward, adaptive_reward = _reward_summaries(
        db_path,
        static_id=1,
        adaptive_id=2,
    )
    report = m36._build_report(
        p1_summary=p1_summary,
        p2_summary=p2_summary,
        p3_summary=p3_summary,
        static_reward=static_reward,
        adaptive_reward=adaptive_reward,
        min_budget=20,
        require_reward_telemetry=False,
    )

    assert report["gates"]["p3_reward_delta_available"] is False
    assert report["gates"]["m36_contract_pass"] is True


def test_extract_p1_p2_summary_rejects_problem_mismatch() -> None:
    with pytest.raises(ValueError, match="Expected P1 report"):
        m36._extract_p1_p2_summary(
            report=_p2_report(),
            expected_problem="p1",
            min_budget=20,
        )


def test_validate_min_budget_rejects_values_below_meaningful_threshold() -> None:
    with pytest.raises(ValueError, match="--min-budget must be >="):
        m36._validate_min_budget(19)


def test_extract_p1_p2_summary_rejects_non_boolean_gate_values() -> None:
    report = _p1_report()
    report["gates"]["m33_contract_pass"] = "false"
    with pytest.raises(ValueError, match="Field must be boolean when provided"):
        m36._extract_p1_p2_summary(
            report=report,
            expected_problem="p1",
            min_budget=20,
        )


def test_extract_p3_summary_rejects_non_boolean_gate_values() -> None:
    report = _p3_report()
    report["gates"]["m24_performance_evidence_pass"] = "false"
    with pytest.raises(ValueError, match="Field must be boolean when provided"):
        m36._extract_p3_summary(report=report, min_budget=20)


def test_extract_p3_summary_requires_positive_router_route_counts() -> None:
    report = _p3_report()
    report["static"]["route_summary_at_budget"]["model_routes"] = {
        "governor_static_recipe/mirror": 0
    }
    report["adaptive"]["route_summary_at_budget"]["model_routes"] = {
        "governor_adaptive/near_feasible/mirror": 0
    }

    summary = m36._extract_p3_summary(report=report, min_budget=20)

    assert summary["static_model_routes"] == {}
    assert summary["adaptive_model_routes"] == {}
    assert summary["router_decisions_available"] is False
