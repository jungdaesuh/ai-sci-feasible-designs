from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path

from ai_scientist.memory import hash_payload
from ai_scientist.memory.schema import init_db
from ai_scientist.problem_profiles import get_problem_profile
from scripts.p3_governor import (
    RunSurgeryState,
    _apply_operator_shift_lock,
    _maybe_run_surgery,
    _prune_backlog,
    _select_rebootstrap_pair,
)


def _make_db(tmp_path: Path) -> Path:
    db = tmp_path / "wm.sqlite"
    init_db(db)
    conn = sqlite3.connect(str(db))
    try:
        conn.execute(
            """
            INSERT INTO experiments (started_at, config_json, git_sha, constellaration_sha, notes)
            VALUES ('2026-01-01T00:00:00+00:00', '{}', 'sha', 'const_sha', NULL)
            """
        )
        conn.commit()
    finally:
        conn.close()
    return db


def _boundary(seed: int) -> dict:
    return {
        "r_cos": [
            [1.0 + 0.001 * float(seed), 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        "z_sin": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "n_field_periods": 3,
        "is_stellarator_symmetric": True,
    }


def _insert_candidate_metric(
    conn: sqlite3.Connection,
    *,
    boundary: dict,
    status: str,
    operator_family: str,
    model_route: str,
    feasibility: float,
    objective: float,
    is_feasible: int,
) -> None:
    design_hash = hash_payload(boundary)
    cursor = conn.execute(
        """
        INSERT INTO candidates
        (experiment_id, problem, params_json, seed, status, design_hash, operator_family, model_route)
        VALUES (1, 'p3', ?, 1, ?, ?, ?, ?)
        """,
        (
            json.dumps(boundary),
            str(status),
            str(design_hash),
            str(operator_family),
            str(model_route),
        ),
    )
    assert cursor.lastrowid is not None
    candidate_id = int(cursor.lastrowid)
    payload = {
        "metrics": {"aspect_ratio": 8.0, "lgradB": float(objective)},
        "constraint_margins": {"log10_qi": max(float(feasibility), 0.0)},
    }
    conn.execute(
        """
        INSERT INTO metrics (candidate_id, raw_json, feasibility, objective, hv, is_feasible)
        VALUES (?, ?, ?, ?, NULL, ?)
        """,
        (
            int(candidate_id),
            json.dumps(payload),
            float(feasibility),
            float(objective),
            int(is_feasible),
        ),
    )


def _insert_candidate_only(
    conn: sqlite3.Connection,
    *,
    boundary: dict,
    status: str,
    operator_family: str,
    model_route: str,
) -> None:
    design_hash = hash_payload(boundary)
    conn.execute(
        """
        INSERT INTO candidates
        (experiment_id, problem, params_json, seed, status, design_hash, operator_family, model_route)
        VALUES (1, 'p3', ?, 1, ?, ?, ?, ?)
        """,
        (
            json.dumps(boundary),
            str(status),
            str(design_hash),
            str(operator_family),
            str(model_route),
        ),
    )


def test_stall_detection_triggers_rebootstrap(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        _insert_candidate_metric(
            conn,
            boundary=_boundary(1),
            status="done",
            operator_family="blend",
            model_route="governor_llm/bridge",
            feasibility=0.03,
            objective=5.0,
            is_feasible=0,
        )
        _insert_candidate_metric(
            conn,
            boundary=_boundary(2),
            status="done",
            operator_family="scale_groups",
            model_route="governor_llm/repair",
            feasibility=0.005,
            objective=4.8,
            is_feasible=1,
        )
        conn.commit()
        profile = get_problem_profile("p3")
        policy = profile.autonomy_policy.run_surgery
        state = RunSurgeryState(
            no_objective_progress_windows=int(policy.objective_stall_windows) - 1,
            no_feasibility_progress_windows=int(policy.feasibility_stall_windows) - 1,
        )
        result = _maybe_run_surgery(
            conn=conn,
            experiment_id=1,
            profile=profile,
            run_dir=run_dir,
            state=state,
            progress={
                "metric_delta": int(policy.eval_window),
                "objective_delta": 0.0,
                "feasibility_delta": 0.0,
                "window_ready": True,
            },
            pending_n=0,
            target_queue=8,
            current_workers=policy.autoscale_min_workers,
            invalid_parent_failure_streak=0,
            partner_available=True,
            disable_run_surgery=False,
            disable_autoscale=True,
        )
        assert result is not None
        assert result["action"] == "rebootstrap_pair"
        assert result["reason"] == "stall_no_frontier_progress"
        assert result["forced_action"] == "global_restart"
        assert result["post_restart_forced_action"] in {"bridge", "jump"}
        parent_pair = result["parent_pair"]
        assert isinstance(parent_pair, dict)
        assert isinstance(parent_pair.get("hash"), str)
    finally:
        conn.close()


def test_stall_prefers_rebootstrap_before_backlog_prune(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        _insert_candidate_metric(
            conn,
            boundary=_boundary(1),
            status="done",
            operator_family="blend",
            model_route="governor_llm/bridge",
            feasibility=0.03,
            objective=5.0,
            is_feasible=0,
        )
        _insert_candidate_metric(
            conn,
            boundary=_boundary(2),
            status="done",
            operator_family="scale_groups",
            model_route="governor_llm/repair",
            feasibility=0.005,
            objective=4.8,
            is_feasible=1,
        )
        for i in range(8):
            _insert_candidate_only(
                conn,
                boundary=_boundary(200 + i),
                status="pending",
                operator_family="blend",
                model_route="governor_llm/bridge",
            )
        conn.commit()
        profile = get_problem_profile("p3")
        policy = replace(
            profile.autonomy_policy.run_surgery,
            backlog_prune_min_pending=4,
            backlog_prune_fraction=0.5,
        )
        profile = replace(
            profile,
            autonomy_policy=replace(profile.autonomy_policy, run_surgery=policy),
        )
        state = RunSurgeryState(
            no_objective_progress_windows=int(policy.objective_stall_windows) - 1,
            no_feasibility_progress_windows=int(policy.feasibility_stall_windows) - 1,
        )
        result = _maybe_run_surgery(
            conn=conn,
            experiment_id=1,
            profile=profile,
            run_dir=run_dir,
            state=state,
            progress={
                "metric_delta": int(policy.eval_window),
                "objective_delta": 0.0,
                "feasibility_delta": 0.0,
                "window_ready": True,
            },
            pending_n=8,
            target_queue=8,
            current_workers=policy.autoscale_min_workers,
            invalid_parent_failure_streak=0,
            partner_available=True,
            disable_run_surgery=False,
            disable_autoscale=True,
        )
        assert result is not None
        assert result["action"] == "rebootstrap_pair"
        assert int(result.get("backlog_pruned_count", 0)) == 0
    finally:
        conn.close()


def test_rebootstrap_pair_is_deterministic(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        _insert_candidate_metric(
            conn,
            boundary=_boundary(10),
            status="done",
            operator_family="blend",
            model_route="route_a",
            feasibility=0.025,
            objective=5.5,
            is_feasible=0,
        )
        _insert_candidate_metric(
            conn,
            boundary=_boundary(11),
            status="done",
            operator_family="blend",
            model_route="route_b",
            feasibility=0.004,
            objective=5.4,
            is_feasible=1,
        )
        _insert_candidate_metric(
            conn,
            boundary=_boundary(12),
            status="done",
            operator_family="blend",
            model_route="route_c",
            feasibility=0.006,
            objective=5.3,
            is_feasible=1,
        )
        conn.commit()
        policy = get_problem_profile("p3").autonomy_policy.run_surgery
        first = _select_rebootstrap_pair(
            conn,
            experiment_id=1,
            policy=policy,
            run_dir=run_dir,
        )
        second = _select_rebootstrap_pair(
            conn,
            experiment_id=1,
            policy=policy,
            run_dir=run_dir,
        )
        assert first[0] == second[0]
        assert first[1] == second[1]
        assert first[2] == second[2]
        assert first[3] == second[3]
    finally:
        conn.close()


def test_backlog_prune_only_pending_and_route_filtered(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        for i in range(6):
            _insert_candidate_only(
                conn,
                boundary=_boundary(100 + i),
                status="pending",
                operator_family="blend",
                model_route="governor_llm/bridge",
            )
        for i in range(2):
            _insert_candidate_only(
                conn,
                boundary=_boundary(200 + i),
                status="pending",
                operator_family="scale_groups",
                model_route="governor_llm/repair",
            )
        for i in range(2):
            _insert_candidate_only(
                conn,
                boundary=_boundary(300 + i),
                status="running",
                operator_family="blend",
                model_route="governor_llm/bridge",
            )
        conn.commit()
        policy = replace(
            get_problem_profile("p3").autonomy_policy.run_surgery,
            backlog_prune_min_pending=4,
            backlog_prune_fraction=0.5,
        )
        pruned = _prune_backlog(
            conn,
            experiment_id=1,
            policy=policy,
            reason="stall",
        )
        assert pruned == 3
        deferred_count = conn.execute(
            """
            SELECT COUNT(*) AS n
            FROM candidates
            WHERE experiment_id = 1
              AND status LIKE 'deferred:auto_prune:%'
              AND model_route = 'governor_llm/bridge'
              AND operator_family = 'blend'
            """
        ).fetchone()
        assert deferred_count is not None
        assert int(deferred_count["n"]) == 3
        untouched_other = conn.execute(
            """
            SELECT COUNT(*) AS n
            FROM candidates
            WHERE experiment_id = 1
              AND status = 'pending'
              AND model_route = 'governor_llm/repair'
              AND operator_family = 'scale_groups'
            """
        ).fetchone()
        assert untouched_other is not None
        assert int(untouched_other["n"]) == 2
        running_untouched = conn.execute(
            """
            SELECT COUNT(*) AS n
            FROM candidates
            WHERE experiment_id = 1
              AND status = 'running'
              AND model_route = 'governor_llm/bridge'
              AND operator_family = 'blend'
            """
        ).fetchone()
        assert running_untouched is not None
        assert int(running_untouched["n"]) == 2
    finally:
        conn.close()


def test_operator_shift_lock_prevents_repeated_family() -> None:
    profile = get_problem_profile("p3")
    state = RunSurgeryState(operator_shift_lock_remaining=2, operator_shift_index=0)
    first = _apply_operator_shift_lock(
        state=state,
        profile=profile,
        partner_available=True,
    )
    second = _apply_operator_shift_lock(
        state=state,
        profile=profile,
        partner_available=True,
    )
    assert first in {"repair", "bridge"}
    assert second in {"repair", "bridge"}
    assert first != second
    assert state.operator_shift_lock_remaining == 0


def test_authority_hierarchy_hard_trigger_not_overridden(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        profile = get_problem_profile("p3")
        state = RunSurgeryState()
        result = _maybe_run_surgery(
            conn=conn,
            experiment_id=1,
            profile=profile,
            run_dir=run_dir,
            state=state,
            progress={
                "metric_delta": 0,
                "objective_delta": 0.0,
                "feasibility_delta": 0.0,
            },
            pending_n=0,
            target_queue=8,
            current_workers=profile.autonomy_policy.run_surgery.autoscale_min_workers,
            invalid_parent_failure_streak=profile.autonomy_policy.run_surgery.invalid_basin_failure_limit,
            partner_available=True,
            disable_run_surgery=True,
            disable_autoscale=True,
        )
        assert result is None
    finally:
        conn.close()


def test_invalid_basin_escape_gracefully_handles_missing_rebootstrap_parents(
    tmp_path: Path,
) -> None:
    db = _make_db(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        profile = get_problem_profile("p3")
        state = RunSurgeryState()
        result = _maybe_run_surgery(
            conn=conn,
            experiment_id=1,
            profile=profile,
            run_dir=run_dir,
            state=state,
            progress={
                "metric_delta": 0,
                "objective_delta": 0.0,
                "feasibility_delta": 0.0,
            },
            pending_n=0,
            target_queue=8,
            current_workers=profile.autonomy_policy.run_surgery.autoscale_min_workers,
            invalid_parent_failure_streak=profile.autonomy_policy.run_surgery.invalid_basin_failure_limit,
            partner_available=True,
            disable_run_surgery=False,
            disable_autoscale=True,
        )
        assert result is not None
        assert result["action"] == "invalid_basin_escape"
        assert result["reason"] == "invalid_basin_escape_no_rebootstrap_pair"
        assert result["forced_action"] == "global_restart"
    finally:
        conn.close()


def test_autoscale_up_down_respects_bounds_and_cooldown(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        profile = get_problem_profile("p3")
        policy = profile.autonomy_policy.run_surgery
        state = RunSurgeryState(autoscale_cooldown_remaining=0)
        up_result = _maybe_run_surgery(
            conn=conn,
            experiment_id=1,
            profile=profile,
            run_dir=run_dir,
            state=state,
            progress={
                "metric_delta": 0,
                "objective_delta": 0.0,
                "feasibility_delta": 0.0,
            },
            pending_n=int(policy.autoscale_up_pending_ratio * 10.0),
            target_queue=10,
            current_workers=policy.autoscale_min_workers,
            invalid_parent_failure_streak=0,
            partner_available=True,
            disable_run_surgery=True,
            disable_autoscale=False,
        )
        assert up_result is not None
        assert up_result["action"] == "autoscale_up"
        assert int(up_result["desired_workers"]) <= int(policy.autoscale_max_workers)
        assert int(state.autoscale_cooldown_remaining) == int(
            policy.autoscale_cooldown_cycles
        )
        cooldown_result = _maybe_run_surgery(
            conn=conn,
            experiment_id=1,
            profile=profile,
            run_dir=run_dir,
            state=state,
            progress={
                "metric_delta": 0,
                "objective_delta": 0.0,
                "feasibility_delta": 0.0,
            },
            pending_n=int(policy.autoscale_up_pending_ratio * 10.0),
            target_queue=10,
            current_workers=policy.autoscale_max_workers,
            invalid_parent_failure_streak=0,
            partner_available=True,
            disable_run_surgery=True,
            disable_autoscale=False,
        )
        assert cooldown_result is None
        state.autoscale_cooldown_remaining = 0
        down_result = _maybe_run_surgery(
            conn=conn,
            experiment_id=1,
            profile=profile,
            run_dir=run_dir,
            state=state,
            progress={
                "metric_delta": 0,
                "objective_delta": 0.0,
                "feasibility_delta": 0.0,
            },
            pending_n=int(policy.autoscale_down_pending_ratio * 20.0),
            target_queue=20,
            current_workers=policy.autoscale_max_workers,
            invalid_parent_failure_streak=0,
            partner_available=True,
            disable_run_surgery=True,
            disable_autoscale=False,
        )
        assert down_result is not None
        assert down_result["action"] == "autoscale_down"
        assert int(down_result["desired_workers"]) >= int(policy.autoscale_min_workers)
    finally:
        conn.close()
