# ruff: noqa: E402
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import p1_p2_fixed_budget_compare as compare


def _write_history(run_dir: Path, rows: list[dict]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row) for row in rows) + "\n"
    (run_dir / "history.jsonl").write_text(payload, encoding="utf-8")


def _arm_summary(
    *,
    run_dir: str,
    feasible_count_budget: int,
    feasible_yield_per_100: float,
    best_feasible_metric_at_budget: float | None,
    error_count_budget: int,
    restart_seed_labeled_rows_budget: int,
    eval_count_total: int = 3,
    eval_count_budget: int = 3,
) -> compare.ArmSummary:
    return compare.ArmSummary(
        run_dir=run_dir,
        eval_count_total=eval_count_total,
        eval_count_budget=eval_count_budget,
        feasible_count_budget=feasible_count_budget,
        feasible_yield_per_100=feasible_yield_per_100,
        best_feasible_metric_at_budget=best_feasible_metric_at_budget,
        error_count_budget=error_count_budget,
        restart_seed_labeled_rows_budget=restart_seed_labeled_rows_budget,
    )


def test_build_report_p1_passes_when_adaptive_is_non_regressing(tmp_path: Path) -> None:
    static_dir = tmp_path / "p1_static"
    adaptive_dir = tmp_path / "p1_adaptive"
    _write_history(
        static_dir,
        [
            {"objective": 3.1, "constraint_violation_inf": 0.0, "error": None},
            {"objective": 3.4, "constraint_violation_inf": 0.5, "error": None},
            {"objective": 3.0, "constraint_violation_inf": 0.2, "error": None},
        ],
    )
    _write_history(
        adaptive_dir,
        [
            {
                "objective": 3.0,
                "constraint_violation_inf": 0.0,
                "error": None,
                "restart_seed": "state",
            },
            {
                "objective": 2.9,
                "constraint_violation_inf": 0.0,
                "error": None,
                "restart_seed": "best_low",
            },
            {
                "objective": 3.3,
                "constraint_violation_inf": 0.1,
                "error": None,
                "restart_seed": "best_violation",
            },
        ],
    )

    static_rows = compare._load_history_rows(static_dir, problem="p1")
    adaptive_rows = compare._load_history_rows(adaptive_dir, problem="p1")
    budget = compare._resolve_budget(
        budget_requested=3,
        static_count=len(static_rows),
        adaptive_count=len(adaptive_rows),
    )
    static_summary = compare._build_arm_summary(
        run_dir=static_dir,
        rows=static_rows,
        budget=budget,
        problem="p1",
    )
    adaptive_summary = compare._build_arm_summary(
        run_dir=adaptive_dir,
        rows=adaptive_rows,
        budget=budget,
        problem="p1",
    )
    report = compare._build_report(
        problem="p1",
        static_summary=static_summary,
        adaptive_summary=adaptive_summary,
        budget_requested=3,
        budget_used=budget,
    )

    assert report["gates"]["best_feasible_metric_non_regression"] is True
    assert report["gates"]["feasible_yield_non_regression"] is True
    assert report["gates"]["m33_contract_pass"] is True
    assert report["gates"]["pass"] is True
    assert report["static"]["best_feasible_metric_at_budget"] == 3.1
    assert report["adaptive"]["best_feasible_metric_at_budget"] == 2.9
    assert report["adaptive"]["restart_seed_labeled_rows_budget"] == 3


def test_build_report_p2_fails_when_adaptive_best_metric_regresses(
    tmp_path: Path,
) -> None:
    static_dir = tmp_path / "p2_static"
    adaptive_dir = tmp_path / "p2_adaptive"
    _write_history(
        static_dir,
        [
            {"lgradb": 6.0, "constraint_violation_inf": 0.0, "error": None},
            {"lgradb": 5.0, "constraint_violation_inf": 0.0, "error": None},
        ],
    )
    _write_history(
        adaptive_dir,
        [
            {"lgradb": 5.5, "constraint_violation_inf": 0.0, "error": None},
            {"lgradb": 5.0, "constraint_violation_inf": 0.0, "error": None},
        ],
    )

    static_rows = compare._load_history_rows(static_dir, problem="p2")
    adaptive_rows = compare._load_history_rows(adaptive_dir, problem="p2")
    budget = compare._resolve_budget(
        budget_requested=2,
        static_count=len(static_rows),
        adaptive_count=len(adaptive_rows),
    )
    static_summary = compare._build_arm_summary(
        run_dir=static_dir,
        rows=static_rows,
        budget=budget,
        problem="p2",
    )
    adaptive_summary = compare._build_arm_summary(
        run_dir=adaptive_dir,
        rows=adaptive_rows,
        budget=budget,
        problem="p2",
    )
    report = compare._build_report(
        problem="p2",
        static_summary=static_summary,
        adaptive_summary=adaptive_summary,
        budget_requested=2,
        budget_used=budget,
    )

    assert report["gates"]["best_feasible_metric_non_regression"] is False
    assert report["gates"]["feasible_yield_non_regression"] is True
    assert report["gates"]["non_trivial_feasible_evidence"] is True
    assert report["gates"]["m33_contract_pass"] is False


def test_build_report_fails_without_any_feasible_evidence(tmp_path: Path) -> None:
    static_dir = tmp_path / "p1_static_infeasible"
    adaptive_dir = tmp_path / "p1_adaptive_infeasible"
    _write_history(
        static_dir,
        [
            {"objective": 3.1, "constraint_violation_inf": 1.0, "error": None},
            {"objective": 3.2, "constraint_violation_inf": 0.8, "error": None},
        ],
    )
    _write_history(
        adaptive_dir,
        [
            {
                "objective": 3.0,
                "constraint_violation_inf": 1.2,
                "error": None,
                "restart_seed": "state",
            },
            {
                "objective": 2.9,
                "constraint_violation_inf": 0.7,
                "error": None,
                "restart_seed": "best_low",
            },
        ],
    )
    static_rows = compare._load_history_rows(static_dir, problem="p1")
    adaptive_rows = compare._load_history_rows(adaptive_dir, problem="p1")
    report = compare._build_report(
        problem="p1",
        static_summary=compare._build_arm_summary(
            run_dir=static_dir,
            rows=static_rows,
            budget=2,
            problem="p1",
        ),
        adaptive_summary=compare._build_arm_summary(
            run_dir=adaptive_dir,
            rows=adaptive_rows,
            budget=2,
            problem="p1",
        ),
        budget_requested=2,
        budget_used=2,
    )
    assert report["gates"]["best_feasible_metric_non_regression"] is True
    assert report["gates"]["feasible_yield_non_regression"] is True
    assert report["gates"]["non_trivial_feasible_evidence"] is False
    assert report["gates"]["m33_contract_pass"] is False


def test_load_rows_supports_p2_objective_fallback(tmp_path: Path) -> None:
    run_dir = tmp_path / "p2_run"
    _write_history(
        run_dir,
        [
            {
                "objective": -7.2,
                "constraint_violation_inf": 0.0,
                "error": None,
            }
        ],
    )
    rows = compare._load_history_rows(run_dir, problem="p2")
    assert len(rows) == 1
    assert rows[0].metric_value == 7.2


def test_load_rows_supports_legacy_violation_key(tmp_path: Path) -> None:
    run_dir = tmp_path / "p1_legacy"
    _write_history(
        run_dir,
        [
            {
                "objective": 2.5,
                "constraint_violation_l2": 0.0,
                "error": None,
            }
        ],
    )
    rows = compare._load_history_rows(run_dir, problem="p1")
    summary = compare._build_arm_summary(
        run_dir=run_dir,
        rows=rows,
        budget=1,
        problem="p1",
    )
    assert summary.feasible_count_budget == 1
    assert summary.best_feasible_metric_at_budget == 2.5


def test_load_rows_prefers_official_feasibility_over_legacy_keys(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "p1_precedence"
    _write_history(
        run_dir,
        [
            {
                "objective": 2.5,
                "feasibility_official": 0.3,
                "constraint_violation_inf": 0.0,
                "error": None,
            }
        ],
    )
    rows = compare._load_history_rows(run_dir, problem="p1")
    summary = compare._build_arm_summary(
        run_dir=run_dir,
        rows=rows,
        budget=1,
        problem="p1",
    )
    assert summary.feasible_count_budget == 0
    assert summary.best_feasible_metric_at_budget is None


def test_resolve_budget_auto_and_clip_behavior() -> None:
    assert (
        compare._resolve_budget(
            budget_requested=0,
            static_count=8,
            adaptive_count=5,
        )
        == 5
    )
    assert (
        compare._resolve_budget(
            budget_requested=12,
            static_count=8,
            adaptive_count=5,
        )
        == 5
    )


def test_resolve_budget_rejects_negative_requested_budget() -> None:
    with pytest.raises(ValueError, match="--budget must be >= 0"):
        compare._resolve_budget(
            budget_requested=-1,
            static_count=1,
            adaptive_count=1,
        )


def test_validate_run_contract_rejects_same_dirs() -> None:
    summary = _arm_summary(
        run_dir="same",
        feasible_count_budget=1,
        feasible_yield_per_100=33.3,
        best_feasible_metric_at_budget=2.5,
        error_count_budget=0,
        restart_seed_labeled_rows_budget=0,
    )
    with pytest.raises(ValueError, match="must be distinct"):
        compare._validate_run_contract(
            static_run_dir=Path("same"),
            adaptive_run_dir=Path("same"),
            static_summary=summary,
            adaptive_summary=summary,
            max_error_rows_budget=0,
            allow_legacy_restart_metadata=False,
        )


def test_validate_run_contract_rejects_error_rows_at_budget() -> None:
    static_summary = _arm_summary(
        run_dir="static",
        feasible_count_budget=1,
        feasible_yield_per_100=33.3,
        best_feasible_metric_at_budget=2.5,
        error_count_budget=1,
        restart_seed_labeled_rows_budget=0,
    )
    adaptive_summary = _arm_summary(
        run_dir="adaptive",
        feasible_count_budget=2,
        feasible_yield_per_100=66.6,
        best_feasible_metric_at_budget=2.4,
        error_count_budget=0,
        restart_seed_labeled_rows_budget=3,
    )
    with pytest.raises(ValueError, match="Static arm exceeded allowed eval-error rows"):
        compare._validate_run_contract(
            static_run_dir=Path("static"),
            adaptive_run_dir=Path("adaptive"),
            static_summary=static_summary,
            adaptive_summary=adaptive_summary,
            max_error_rows_budget=0,
            allow_legacy_restart_metadata=False,
        )


def test_validate_run_contract_enforces_restart_seed_metadata() -> None:
    static_summary = _arm_summary(
        run_dir="static",
        feasible_count_budget=1,
        feasible_yield_per_100=50.0,
        best_feasible_metric_at_budget=2.5,
        error_count_budget=0,
        restart_seed_labeled_rows_budget=1,
        eval_count_total=2,
        eval_count_budget=2,
    )
    adaptive_summary = _arm_summary(
        run_dir="adaptive",
        feasible_count_budget=1,
        feasible_yield_per_100=50.0,
        best_feasible_metric_at_budget=2.4,
        error_count_budget=0,
        restart_seed_labeled_rows_budget=0,
        eval_count_total=2,
        eval_count_budget=2,
    )
    with pytest.raises(
        ValueError, match="Static arm contains restart-seed-labeled rows"
    ):
        compare._validate_run_contract(
            static_run_dir=Path("static"),
            adaptive_run_dir=Path("adaptive"),
            static_summary=static_summary,
            adaptive_summary=adaptive_summary,
            max_error_rows_budget=0,
            allow_legacy_restart_metadata=False,
        )


def test_validate_run_contract_requires_adaptive_restart_seed_metadata() -> None:
    static_summary = _arm_summary(
        run_dir="static",
        feasible_count_budget=1,
        feasible_yield_per_100=50.0,
        best_feasible_metric_at_budget=2.5,
        error_count_budget=0,
        restart_seed_labeled_rows_budget=0,
        eval_count_total=2,
        eval_count_budget=2,
    )
    adaptive_summary = _arm_summary(
        run_dir="adaptive",
        feasible_count_budget=1,
        feasible_yield_per_100=50.0,
        best_feasible_metric_at_budget=2.4,
        error_count_budget=0,
        restart_seed_labeled_rows_budget=0,
        eval_count_total=2,
        eval_count_budget=2,
    )
    with pytest.raises(
        ValueError, match="Adaptive arm has zero restart-seed-labeled rows"
    ):
        compare._validate_run_contract(
            static_run_dir=Path("static"),
            adaptive_run_dir=Path("adaptive"),
            static_summary=static_summary,
            adaptive_summary=adaptive_summary,
            max_error_rows_budget=0,
            allow_legacy_restart_metadata=False,
        )


def test_validate_run_contract_allows_legacy_restart_metadata_override() -> None:
    static_summary = _arm_summary(
        run_dir="static",
        feasible_count_budget=1,
        feasible_yield_per_100=50.0,
        best_feasible_metric_at_budget=2.5,
        error_count_budget=0,
        restart_seed_labeled_rows_budget=1,
        eval_count_total=2,
        eval_count_budget=2,
    )
    adaptive_summary = _arm_summary(
        run_dir="adaptive",
        feasible_count_budget=1,
        feasible_yield_per_100=50.0,
        best_feasible_metric_at_budget=2.4,
        error_count_budget=0,
        restart_seed_labeled_rows_budget=0,
        eval_count_total=2,
        eval_count_budget=2,
    )
    compare._validate_run_contract(
        static_run_dir=Path("static"),
        adaptive_run_dir=Path("adaptive"),
        static_summary=static_summary,
        adaptive_summary=adaptive_summary,
        max_error_rows_budget=0,
        allow_legacy_restart_metadata=True,
    )


def test_build_report_serializes_run_contract_provenance() -> None:
    static_summary = _arm_summary(
        run_dir="static",
        feasible_count_budget=1,
        feasible_yield_per_100=50.0,
        best_feasible_metric_at_budget=2.5,
        error_count_budget=0,
        restart_seed_labeled_rows_budget=0,
    )
    adaptive_summary = _arm_summary(
        run_dir="adaptive",
        feasible_count_budget=2,
        feasible_yield_per_100=100.0,
        best_feasible_metric_at_budget=2.4,
        error_count_budget=0,
        restart_seed_labeled_rows_budget=0,
    )
    report = compare._build_report(
        problem="p1",
        static_summary=static_summary,
        adaptive_summary=adaptive_summary,
        budget_requested=20,
        budget_used=20,
        allow_legacy_restart_metadata=True,
        max_error_rows_budget=3,
    )
    run_contract = report["run_contract"]
    assert run_contract["allow_legacy_restart_metadata"] is True
    assert run_contract["strict_restart_seed_metadata_enforced"] is False
    assert run_contract["max_error_rows_budget"] == 3


def test_enforce_required_gates_rejects_failed_report() -> None:
    report = {"gates": {"m33_contract_pass": False}}
    with pytest.raises(ValueError, match="M3\\.3 fixed-budget gate failed"):
        compare._enforce_required_gates(report, require_m33_pass=True)
