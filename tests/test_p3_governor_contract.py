from __future__ import annotations

import json
import math
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Mapping

import pytest

import ai_scientist.llm_controller as llm_controller
from ai_scientist.llm_controller import (
    load_optional_decision_command,
    load_optional_decision_file,
    resolve_decision_payload,
)
from ai_scientist.memory import hash_payload
from ai_scientist.memory.schema import init_db
from ai_scientist.p3_enqueue import candidate_seed
from ai_scientist.problem_profiles import get_problem_profile
from scripts import p3_governor
from scripts.p3_governor import (
    CandidateRow,
    _build_blend_cmd,
    _build_jump_recipe_cmds,
    _build_llm_mutation_cmds,
    _build_scale_cmd,
    _choose_partner,
    _consecutive_transient_failures,
    _deterministic_restart_action,
    _dominant_violation_rate,
    _fetch_candidates,
    _frontier_integrity_ok,
    _is_invalid_llm_override_reason,
    _remaining_run_budget,
    _record_cycle_manifest,
    _queue_desync_events_last20,
    _startup_replay_commands,
    _stagnation_cycles,
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


def _insert_candidate_with_metric(
    db: Path,
    *,
    problem: str = "p3",
    boundary: dict,
    feasibility: float,
    is_feasible: int,
    objective: float,
    aspect: float,
    violations: dict[str, float],
) -> None:
    design_hash = hash_payload(boundary)
    conn = sqlite3.connect(str(db))
    try:
        cursor = conn.execute(
            """
            INSERT INTO candidates
            (experiment_id, problem, params_json, seed, status, design_hash, operator_family, model_route)
            VALUES (1, ?, ?, 1, 'done', ?, 'scale_groups', 'seed')
            """,
            (str(problem), json.dumps(boundary), design_hash),
        )
        assert cursor.lastrowid is not None
        candidate_id = int(cursor.lastrowid)
        payload = {
            "metrics": {
                "aspect_ratio": float(aspect),
                "lgradB": float(objective),
                "log10_qi": -3.0,
                "iota_edge": 0.2,
            },
            "constraint_margins": violations,
        }
        conn.execute(
            """
            INSERT INTO metrics (candidate_id, raw_json, feasibility, objective, hv, is_feasible)
            VALUES (?, ?, ?, ?, NULL, ?)
            """,
            (
                candidate_id,
                json.dumps(payload),
                float(feasibility),
                float(objective),
                int(is_feasible),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _insert_malformed_candidate_with_metric(db: Path, *, design_hash: str) -> None:
    conn = sqlite3.connect(str(db))
    try:
        cursor = conn.execute(
            """
            INSERT INTO candidates
            (experiment_id, problem, params_json, seed, status, design_hash, operator_family, model_route)
            VALUES (1, 'p3', ?, 1, 'done', ?, 'scale_groups', 'seed')
            """,
            ("{invalid_json", design_hash),
        )
        assert cursor.lastrowid is not None
        candidate_id = int(cursor.lastrowid)
        payload = {
            "metrics": {"aspect_ratio": 8.0, "lgradB": 1.0},
            "constraint_margins": {"log10_qi": 0.2},
        }
        conn.execute(
            """
            INSERT INTO metrics (candidate_id, raw_json, feasibility, objective, hv, is_feasible)
            VALUES (?, ?, ?, ?, NULL, ?)
            """,
            (candidate_id, json.dumps(payload), 0.2, 1.0, 0),
        )
        conn.commit()
    finally:
        conn.close()


def test_load_optional_decision_file_none() -> None:
    assert load_optional_decision_file(None) is None


def test_load_optional_decision_file_valid_payload(tmp_path: Path) -> None:
    path = tmp_path / "decision.json"
    path.write_text(
        json.dumps(
            {
                "action": "repair",
                "target_constraint": "log10_qi",
                "mutations": [],
                "expected_effect": "improve",
                "restart_plan": None,
            }
        )
    )
    payload = load_optional_decision_file(path)
    assert isinstance(payload, dict)
    assert payload["action"] == "repair"


def test_load_optional_decision_file_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "decision.json"
    path.write_text("{invalid")
    with pytest.raises(ValueError):
        load_optional_decision_file(path)


def test_load_optional_decision_file_rejects_non_object(tmp_path: Path) -> None:
    path = tmp_path / "decision.json"
    path.write_text(json.dumps(["repair"]))
    with pytest.raises(ValueError):
        load_optional_decision_file(path)


def test_load_optional_decision_file_missing_file_raises_value_error(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(ValueError):
        load_optional_decision_file(missing)


def test_load_optional_decision_command_parses_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_run(
        args: list[str],
        input: str,
        text: bool,
        capture_output: bool,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert args == ["fake-llm"]
        assert '"observation"' in input
        assert text is True
        assert capture_output is True
        assert check is False
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout=json.dumps(
                {
                    "action": "repair",
                    "target_constraint": "log10_qi",
                    "mutations": [],
                    "expected_effect": "improve",
                    "restart_plan": None,
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)
    payload = load_optional_decision_command(
        command="fake-llm",
        observation={"x": 1},
        model="codex",
        session_id="sess_1",
    )
    assert isinstance(payload, dict)
    assert payload["action"] == "repair"


def test_resolve_decision_payload_falls_back_to_command_on_file_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_file_error(path: Path | None) -> Mapping[str, object] | None:
        raise ValueError("bad file payload")

    def _command_payload(
        *,
        command: str | None,
        observation: Mapping[str, object],
        model: str,
        session_id: str | None,
    ) -> Mapping[str, object] | None:
        assert command == "fake-llm"
        assert observation["sample"] == 1
        assert model == "codex"
        assert session_id == "sess"
        return {
            "action": "repair",
            "target_constraint": "log10_qi",
            "mutations": [],
            "expected_effect": "ok",
            "restart_plan": None,
        }

    monkeypatch.setattr(
        llm_controller, "load_optional_decision_file", _raise_file_error
    )
    monkeypatch.setattr(
        llm_controller,
        "load_optional_decision_command",
        _command_payload,
    )
    payload, input_source = resolve_decision_payload(
        decision_file=Path("decision.json"),
        decision_command="fake-llm",
        observation={"sample": 1},
        model="codex",
        session_id="sess",
    )
    assert input_source == "codex_command"
    assert isinstance(payload, dict)
    assert payload["action"] == "repair"


def test_invalid_llm_override_reason_classification() -> None:
    assert _is_invalid_llm_override_reason("invalid_llm_output:bad_schema")
    assert _is_invalid_llm_override_reason("invalid_llm_output")
    assert not _is_invalid_llm_override_reason("policy_fallback:bridge")
    assert not _is_invalid_llm_override_reason(None)


def test_remaining_run_budget_uses_stop_policy_caps() -> None:
    uncapped = _remaining_run_budget(
        proposal_cycles_emitted=0,
        max_cycles=0,
        elapsed_runtime_sec=0.0,
        max_runtime_sec=0.0,
    )
    assert math.isinf(uncapped)

    cycle_capped = _remaining_run_budget(
        proposal_cycles_emitted=2,
        max_cycles=5,
        elapsed_runtime_sec=1.0,
        max_runtime_sec=0.0,
    )
    assert cycle_capped == 3.0

    runtime_capped = _remaining_run_budget(
        proposal_cycles_emitted=0,
        max_cycles=0,
        elapsed_runtime_sec=7.5,
        max_runtime_sec=10.0,
    )
    assert runtime_capped == 2.5

    both_capped = _remaining_run_budget(
        proposal_cycles_emitted=2,
        max_cycles=5,
        elapsed_runtime_sec=9.5,
        max_runtime_sec=10.0,
    )
    assert both_capped == 0.5


def test_frontier_integrity_ok_accepts_valid_snapshot() -> None:
    assert _frontier_integrity_ok(
        snapshot={"hv": 1.0, "best_feasibility": 0.1, "feasible_count": 2},
        hv_value=1.0,
    )


def test_frontier_integrity_ok_rejects_invalid_snapshot() -> None:
    assert not _frontier_integrity_ok(
        snapshot={"hv": 1.0, "best_feasibility": 0.1, "feasible_count": -1},
        hv_value=1.0,
    )
    assert not _frontier_integrity_ok(
        snapshot={"hv": -0.1, "best_feasibility": 0.1, "feasible_count": 1},
        hv_value=-0.1,
    )


def test_deterministic_restart_action_escalates_by_profile() -> None:
    assert (
        _deterministic_restart_action(
            profile_problem="p3",
            policy_restart_plan="degraded_restart",
            partner_available=True,
        )
        == "bridge"
    )
    assert (
        _deterministic_restart_action(
            profile_problem="p3",
            policy_restart_plan="degraded_restart",
            partner_available=False,
        )
        == "global_restart"
    )
    assert (
        _deterministic_restart_action(
            profile_problem="p1",
            policy_restart_plan="degraded_restart",
            partner_available=False,
        )
        == "jump"
    )
    assert (
        _deterministic_restart_action(
            profile_problem="p3",
            policy_restart_plan="continue",
            partner_available=True,
        )
        is None
    )


def test_consecutive_transient_failures_from_recent_statuses(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        for status in ["running:1:20260101T000000", "failed:vmec", "failed:vmec"]:
            conn.execute(
                """
                INSERT INTO candidates
                (experiment_id, problem, params_json, seed, status, design_hash)
                VALUES (1, 'p3', '{}', 1, ?, ?)
                """,
                (status, f"h_{status}"),
            )
        conn.commit()
        assert _consecutive_transient_failures(conn, experiment_id=1) == 2
    finally:
        conn.close()


def test_queue_desync_events_detects_stale_running_rows(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(
            """
            INSERT INTO candidates
            (experiment_id, problem, params_json, seed, status, design_hash)
            VALUES (1, 'p3', '{}', 1, 'running:1:20000101T000000', 'h_stale')
            """
        )
        conn.commit()
        assert _queue_desync_events_last20(conn, experiment_id=1, stale_minutes=60) == 1
    finally:
        conn.close()


def test_stagnation_cycles_counts_until_recent_progress_delta(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        deltas = [0.0, 0.0, 0.2, 0.0]
        for cycle, hv_delta in enumerate(deltas, start=1):
            conn.execute(
                """
                INSERT INTO scratchpad_events (
                    experiment_id,
                    cycle,
                    step,
                    planner_intent_json,
                    aso_action,
                    intent_agreement,
                    override_reason,
                    diagnostics_json,
                    outcome_json,
                    created_at
                ) VALUES (?, ?, 1, '{}', 'reflection', 'n/a', NULL, '{}', ?, '2026-01-01T00:00:00+00:00')
                """,
                (
                    1,
                    cycle,
                    json.dumps({"realized_hv_delta": hv_delta}),
                ),
            )
        conn.commit()
        assert _stagnation_cycles(conn, experiment_id=1) == 1
    finally:
        conn.close()


def test_build_llm_mutation_cmds_translates_supported_groups(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    run_dir = tmp_path / "run"
    boundary = {
        "r_cos": [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "z_sin": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "n_field_periods": 3,
        "is_stellarator_symmetric": True,
    }
    design_hash = hash_payload(boundary)
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(
            """
            INSERT INTO candidates
            (experiment_id, problem, params_json, seed, status, design_hash)
            VALUES (1, 'p3', ?, 1, 'done', ?)
            """,
            (json.dumps(boundary), design_hash),
        )
        conn.commit()
    finally:
        conn.close()

    focus = CandidateRow(
        candidate_id=1,
        design_hash=design_hash,
        seed=1,
        feasibility=0.1,
        is_feasible=False,
        lgradb=1.0,
        aspect=8.0,
        violations={"log10_qi": 0.2},
        metrics={},
        meta={},
        lineage_parent_hashes=[],
        novelty_score=None,
        operator_family="scale_groups",
        model_route="test",
        params=boundary,
    )
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        cmds, diagnostics = _build_llm_mutation_cmds(
            proposal_script="scripts/p3_propose.py",
            problem="p3",
            db=db,
            conn=conn,
            experiment_id=1,
            run_dir=run_dir,
            batch_id=1,
            seed_base=100,
            focus=focus,
            llm_mutations=[
                {"parameter_group": "axisym_z", "normalized_delta": -0.1},
                {"parameter_group": "m_ge_3", "normalized_delta": 0.2},
                {"parameter_group": "unsupported", "normalized_delta": 0.1},
            ],
            route_prefix="governor_llm/repair",
        )
    finally:
        conn.close()
    assert len(cmds) == 2
    assert len(diagnostics["applied"]) == 2
    assert len(diagnostics["rejected"]) == 1


def test_build_jump_recipe_cmds_emits_non_local_batch(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    run_dir = tmp_path / "run"
    boundary = {
        "r_cos": [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "z_sin": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "n_field_periods": 3,
        "is_stellarator_symmetric": True,
    }
    design_hash = hash_payload(boundary)
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        for seed in [1, 2]:
            conn.execute(
                """
                INSERT INTO candidates
                (experiment_id, problem, params_json, seed, status, design_hash)
                VALUES (1, 'p3', ?, ?, 'done', ?)
                """,
                (json.dumps(boundary), seed, f"{design_hash}_{seed}"),
            )
        conn.commit()
    finally:
        conn.close()

    focus = CandidateRow(
        candidate_id=1,
        design_hash=f"{design_hash}_1",
        seed=1,
        feasibility=0.2,
        is_feasible=False,
        lgradb=1.0,
        aspect=8.0,
        violations={"log10_qi": 0.1},
        metrics={},
        meta={},
        lineage_parent_hashes=[],
        novelty_score=None,
        operator_family="scale_groups",
        model_route="seed",
        params=boundary,
    )
    partner = CandidateRow(
        candidate_id=2,
        design_hash=f"{design_hash}_2",
        seed=2,
        feasibility=0.1,
        is_feasible=True,
        lgradb=1.2,
        aspect=7.9,
        violations={},
        metrics={},
        meta={},
        lineage_parent_hashes=[],
        novelty_score=None,
        operator_family="blend",
        model_route="seed",
        params=boundary,
    )
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        cmds, diagnostics = _build_jump_recipe_cmds(
            proposal_script="scripts/p3_propose.py",
            problem="p3",
            db=db,
            conn=conn,
            jump_delta_cap=0.01,
            experiment_id=1,
            run_dir=run_dir,
            batch_id=1,
            seed_base=1000,
            focus=focus,
            partner=partner,
            route_prefix="governor_llm/jump",
        )
    finally:
        conn.close()
    assert len(cmds) >= 7
    assert "0.997300" in cmds[0].argv
    assert diagnostics["mode"] == "jump_non_local"


def test_choose_partner_supports_p1_constraint_metrics() -> None:
    candidates = [
        CandidateRow(
            candidate_id=1,
            design_hash="a",
            seed=1,
            feasibility=0.2,
            is_feasible=True,
            lgradb=1.0,
            aspect=9.0,
            violations={},
            metrics={"aspect_ratio": 9.0, "iota_edge": 0.25},
            meta={},
            lineage_parent_hashes=[],
            novelty_score=None,
            operator_family="seed",
            model_route="seed",
            params=None,
        ),
        CandidateRow(
            candidate_id=2,
            design_hash="b",
            seed=2,
            feasibility=0.2,
            is_feasible=True,
            lgradb=1.1,
            aspect=7.0,
            violations={},
            metrics={"aspect_ratio": 7.0, "iota_edge": 0.35},
            meta={},
            lineage_parent_hashes=[],
            novelty_score=None,
            operator_family="seed",
            model_route="seed",
            params=None,
        ),
    ]
    aspect_partner = _choose_partner(candidates, worst_constraint="aspect_ratio")
    iota_partner = _choose_partner(candidates, worst_constraint="iota_edge")
    assert aspect_partner is not None
    assert iota_partner is not None
    assert aspect_partner.design_hash == "b"
    assert iota_partner.design_hash == "b"


def test_fetch_candidates_applies_p1_minimize_objective_direction(
    tmp_path: Path,
) -> None:
    db = _make_db(tmp_path)
    boundary = {
        "r_cos": [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "z_sin": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "n_field_periods": 3,
        "is_stellarator_symmetric": True,
    }
    _insert_candidate_with_metric(
        db,
        problem="p1",
        boundary=boundary,
        feasibility=0.4,
        is_feasible=0,
        objective=2.0,
        aspect=6.0,
        violations={"aspect_ratio": 0.1},
    )
    _insert_candidate_with_metric(
        db,
        problem="p1",
        boundary=boundary,
        feasibility=0.4,
        is_feasible=0,
        objective=4.0,
        aspect=6.0,
        violations={"aspect_ratio": 0.1},
    )

    profile = get_problem_profile("p1")
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        rows = _fetch_candidates(conn, profile=profile, experiment_id=1, limit=10)
    finally:
        conn.close()

    utility_by_raw = {
        float(row.metrics["lgradB"]): float(row.lgradb)
        for row in rows
        if row.lgradb is not None and "lgradB" in row.metrics
    }
    assert utility_by_raw[2.0] > utility_by_raw[4.0]


def test_governor_global_restart_without_bootstrap_uses_auto_parents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = _make_db(tmp_path)
    run_dir = tmp_path / "run"
    boundary = {
        "r_cos": [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "z_sin": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "n_field_periods": 3,
        "is_stellarator_symmetric": True,
    }
    _insert_candidate_with_metric(
        db,
        boundary=boundary,
        feasibility=0.2,
        is_feasible=0,
        objective=1.0,
        aspect=8.0,
        violations={"log10_qi": 0.2},
    )

    decision_file = tmp_path / "decision.json"
    decision_file.write_text(
        json.dumps(
            {
                "action": "global_restart",
                "target_constraint": "log10_qi",
                "mutations": [],
                "expected_effect": "restart",
                "restart_plan": "global_restart",
            }
        )
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "p3_governor.py",
            "--problem",
            "p3",
            "--db",
            str(db),
            "--experiment-id",
            "1",
            "--run-dir",
            str(run_dir),
            "--workers",
            "1",
            "--queue-multiplier",
            "1",
            "--llm-enabled",
            "--llm-fallback",
            "--llm-allow-decision-file",
            "--llm-decision-file",
            str(decision_file),
        ],
    )
    p3_governor.main()

    artifacts = sorted((run_dir / "governor").glob("governor_batch_*.json"))
    assert artifacts
    payload = json.loads(artifacts[-1].read_text())
    assert payload["commands"]
    assert payload["recipe_mode"] == "llm_global_restart"
    assert payload["restart_parent_source"] == "auto_archive"
    assert payload["llm"]["effective_action"] == "global_restart"


def test_governor_circuit_break_emits_empty_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = _make_db(tmp_path)
    run_dir = tmp_path / "run"
    boundary = {
        "r_cos": [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "z_sin": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "n_field_periods": 3,
        "is_stellarator_symmetric": True,
    }
    _insert_candidate_with_metric(
        db,
        boundary=boundary,
        feasibility=0.2,
        is_feasible=0,
        objective=1.0,
        aspect=8.0,
        violations={"log10_qi": 0.2},
    )
    decision_file = tmp_path / "decision.json"
    decision_file.write_text(
        json.dumps(
            {
                "action": "repair",
                "target_constraint": "log10_qi",
                "mutations": [],
                "expected_effect": "repair",
                "restart_plan": None,
            }
        )
    )

    conn = sqlite3.connect(str(db))
    try:
        for cycle in [1, 2, 3]:
            conn.execute(
                """
                INSERT INTO scratchpad_events
                (experiment_id, cycle, step, planner_intent_json, aso_action, intent_agreement,
                 override_reason, diagnostics_json, outcome_json, created_at)
                VALUES (1, ?, 0, '{}', 'repair', 'fallback', 'invalid_llm_output', '{}', '{}',
                        '2026-01-01T00:00:00+00:00')
                """,
                (cycle,),
            )
        conn.commit()
    finally:
        conn.close()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "p3_governor.py",
            "--problem",
            "p3",
            "--db",
            str(db),
            "--experiment-id",
            "1",
            "--run-dir",
            str(run_dir),
            "--workers",
            "1",
            "--queue-multiplier",
            "1",
            "--llm-enabled",
            "--llm-fallback",
            "--llm-allow-decision-file",
            "--llm-decision-file",
            str(decision_file),
        ],
    )
    p3_governor.main()

    artifacts = sorted((run_dir / "governor").glob("governor_batch_*.json"))
    assert artifacts
    payload = json.loads(artifacts[-1].read_text())
    assert payload["recipe_mode"] == "policy_circuit_break"
    assert payload["commands"] == []


def test_governor_hard_restart_trigger_blocks_llm_restart_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = _make_db(tmp_path)
    run_dir = tmp_path / "run_hard_restart_lock"
    boundary = {
        "r_cos": [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "z_sin": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "n_field_periods": 3,
        "is_stellarator_symmetric": True,
    }
    _insert_candidate_with_metric(
        db,
        boundary=boundary,
        feasibility=0.2,
        is_feasible=0,
        objective=1.0,
        aspect=8.0,
        violations={"log10_qi": 0.2},
    )
    decision_file = tmp_path / "decision_restart_override.json"
    decision_file.write_text(
        json.dumps(
            {
                "action": "repair",
                "target_constraint": "log10_qi",
                "mutations": [],
                "expected_effect": "retry softly",
                "restart_plan": "soft_retry",
            }
        )
    )

    conn = sqlite3.connect(str(db))
    try:
        for cycle in [1, 2, 3]:
            conn.execute(
                """
                INSERT INTO scratchpad_events
                (experiment_id, cycle, step, planner_intent_json, aso_action, intent_agreement,
                 override_reason, diagnostics_json, outcome_json, created_at)
                VALUES (1, ?, 0, '{}', 'repair', 'fallback', 'invalid_llm_output', '{}', '{}',
                        '2026-01-01T00:00:00+00:00')
                """,
                (cycle,),
            )
        conn.commit()
    finally:
        conn.close()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "p3_governor.py",
            "--problem",
            "p3",
            "--db",
            str(db),
            "--experiment-id",
            "1",
            "--run-dir",
            str(run_dir),
            "--workers",
            "1",
            "--queue-multiplier",
            "1",
            "--llm-enabled",
            "--llm-fallback",
            "--llm-allow-decision-file",
            "--llm-decision-file",
            str(decision_file),
        ],
    )
    p3_governor.main()

    artifacts = sorted((run_dir / "governor").glob("governor_batch_*.json"))
    assert artifacts
    payload = json.loads(artifacts[-1].read_text())
    assert payload["restart_policy"]["selected"] == "circuit_break"
    assert payload["recipe_mode"] == "policy_circuit_break"
    assert payload["commands"] == []
    assert payload["llm"]["restart_plan"] == "soft_retry"
    assert (
        payload["llm"]["fallback_reason"]
        == "policy_override_blocked:hard_restart_trigger"
    )


def test_governor_hard_restart_trigger_blocks_llm_action_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = _make_db(tmp_path)
    run_dir = tmp_path / "run_hard_restart_action_lock"
    boundary = {
        "r_cos": [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "z_sin": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "n_field_periods": 3,
        "is_stellarator_symmetric": True,
    }
    _insert_candidate_with_metric(
        db,
        boundary=boundary,
        feasibility=0.2,
        is_feasible=0,
        objective=1.0,
        aspect=8.0,
        violations={"log10_qi": 0.2},
    )
    decision_file = tmp_path / "decision_action_override.json"
    decision_file.write_text(
        json.dumps(
            {
                "action": "repair",
                "target_constraint": "log10_qi",
                "mutations": [],
                "expected_effect": "local move",
            }
        )
    )
    conn = sqlite3.connect(str(db))
    try:
        for cycle in range(1, 9):
            conn.execute(
                """
                INSERT INTO scratchpad_events
                (experiment_id, cycle, step, planner_intent_json, aso_action, intent_agreement,
                 override_reason, diagnostics_json, outcome_json, created_at)
                VALUES (1, ?, 1, '{}', 'reflection', 'n/a', NULL, '{}', ?,
                        '2026-01-01T00:00:00+00:00')
                """,
                (
                    cycle,
                    json.dumps(
                        {
                            "realized_feasibility_delta": 0.0,
                            "realized_hv_delta": 0.0,
                        }
                    ),
                ),
            )
        conn.commit()
    finally:
        conn.close()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "p3_governor.py",
            "--problem",
            "p3",
            "--db",
            str(db),
            "--experiment-id",
            "1",
            "--run-dir",
            str(run_dir),
            "--workers",
            "1",
            "--queue-multiplier",
            "1",
            "--llm-enabled",
            "--llm-fallback",
            "--llm-allow-decision-file",
            "--llm-decision-file",
            str(decision_file),
            "--max-stagnation-cycles",
            "999",
        ],
    )
    p3_governor.main()

    artifacts = sorted((run_dir / "governor").glob("governor_batch_*.json"))
    assert artifacts
    payload = json.loads(artifacts[-1].read_text())
    assert payload["restart_policy"]["selected"] == "global_restart"
    assert payload["llm"]["selected_action"] == "repair"
    assert payload["llm"]["effective_action"] == "global_restart"
    assert (
        payload["llm"]["fallback_reason"]
        == "policy_override_blocked:hard_restart_trigger"
    )
    assert payload["commands"]
    assert "--family blend" in payload["commands"][0]


def test_governor_llm_restart_override_does_not_append_mutation_cmds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = _make_db(tmp_path)
    run_dir = tmp_path / "run_restart_mutation_guard"
    boundary = {
        "r_cos": [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "z_sin": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "n_field_periods": 3,
        "is_stellarator_symmetric": True,
    }
    _insert_candidate_with_metric(
        db,
        boundary=boundary,
        feasibility=0.2,
        is_feasible=0,
        objective=1.0,
        aspect=8.0,
        violations={"log10_qi": 0.2},
    )
    decision_file = tmp_path / "decision_restart_with_mutations.json"
    decision_file.write_text(
        json.dumps(
            {
                "action": "repair",
                "target_constraint": "log10_qi",
                "mutations": [{"parameter_group": "axisym_z", "normalized_delta": 0.1}],
                "expected_effect": "attempt repair",
                "restart_plan": "global_restart",
            }
        )
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "p3_governor.py",
            "--problem",
            "p3",
            "--db",
            str(db),
            "--experiment-id",
            "1",
            "--run-dir",
            str(run_dir),
            "--workers",
            "1",
            "--queue-multiplier",
            "1",
            "--llm-enabled",
            "--llm-fallback",
            "--llm-allow-decision-file",
            "--llm-decision-file",
            str(decision_file),
        ],
    )
    p3_governor.main()

    artifacts = sorted((run_dir / "governor").glob("governor_batch_*.json"))
    assert artifacts
    payload = json.loads(artifacts[-1].read_text())
    assert payload["restart_policy"]["selected"] == "global_restart"
    assert payload["llm"]["selected_action"] == "repair"
    assert payload["llm"]["effective_action"] == "global_restart"
    assert (
        payload["llm"]["fallback_reason"]
        == "policy_override_blocked:hard_restart_trigger"
    )
    assert payload["recipe_mode"] == "llm_global_restart"
    assert len(payload["commands"]) == 1
    assert "--family blend" in payload["commands"][0]
    assert "mutation_diagnostics" not in payload["llm"]


def test_governor_default_stagnation_threshold_allows_global_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = _make_db(tmp_path)
    run_dir = tmp_path / "run_default_stagnation_restart"
    boundary = {
        "r_cos": [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "z_sin": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "n_field_periods": 3,
        "is_stellarator_symmetric": True,
    }
    _insert_candidate_with_metric(
        db,
        boundary=boundary,
        feasibility=0.2,
        is_feasible=0,
        objective=1.0,
        aspect=8.0,
        violations={"log10_qi": 0.2},
    )
    conn = sqlite3.connect(str(db))
    try:
        for cycle in range(1, 9):
            conn.execute(
                """
                INSERT INTO scratchpad_events
                (experiment_id, cycle, step, planner_intent_json, aso_action, intent_agreement,
                 override_reason, diagnostics_json, outcome_json, created_at)
                VALUES (1, ?, 1, '{}', 'reflection', 'n/a', NULL, '{}', ?,
                        '2026-01-01T00:00:00+00:00')
                """,
                (
                    cycle,
                    json.dumps(
                        {
                            "realized_feasibility_delta": 0.0,
                            "realized_hv_delta": 0.0,
                        }
                    ),
                ),
            )
        conn.commit()
    finally:
        conn.close()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "p3_governor.py",
            "--problem",
            "p3",
            "--db",
            str(db),
            "--experiment-id",
            "1",
            "--run-dir",
            str(run_dir),
            "--workers",
            "1",
            "--queue-multiplier",
            "1",
        ],
    )
    p3_governor.main()

    artifacts = sorted((run_dir / "governor").glob("governor_batch_*.json"))
    assert artifacts
    payload = json.loads(artifacts[-1].read_text())
    assert payload["restart_policy"]["selected"] == "global_restart"
    assert payload["commands"]
    assert "--family blend" in payload["commands"][0]


def test_governor_applies_llm_target_constraint_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = _make_db(tmp_path)
    run_dir = tmp_path / "run"
    boundary = {
        "r_cos": [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "z_sin": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "n_field_periods": 3,
        "is_stellarator_symmetric": True,
    }
    _insert_candidate_with_metric(
        db,
        boundary=boundary,
        feasibility=0.2,
        is_feasible=0,
        objective=1.0,
        aspect=8.0,
        violations={"log10_qi": 0.2},
    )

    decision_file = tmp_path / "decision.json"
    decision_file.write_text(
        json.dumps(
            {
                "action": "repair",
                "target_constraint": "vacuum",
                "mutations": [],
                "expected_effect": "improve vacuum",
                "restart_plan": None,
            }
        )
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "p3_governor.py",
            "--problem",
            "p3",
            "--db",
            str(db),
            "--experiment-id",
            "1",
            "--run-dir",
            str(run_dir),
            "--workers",
            "1",
            "--queue-multiplier",
            "1",
            "--llm-enabled",
            "--llm-fallback",
            "--llm-allow-decision-file",
            "--llm-decision-file",
            str(decision_file),
        ],
    )
    p3_governor.main()
    artifacts = sorted((run_dir / "governor").glob("governor_batch_*.json"))
    assert artifacts
    payload = json.loads(artifacts[-1].read_text())
    assert payload["focus"]["worst_constraint"] == "vacuum"
    context = payload["llm_observation"]["state"]["context"]
    assert "remaining_budget" in context
    assert "consecutive_failures" in context
    assert "queue_desync_events_last20" in context
    assert "stagnation_cycles" in context
    assert "policy_restart_plan" in context


def test_llm_transport_rejects_non_codex_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = _make_db(tmp_path)
    run_dir = tmp_path / "run"
    boundary = {
        "r_cos": [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "z_sin": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "n_field_periods": 3,
        "is_stellarator_symmetric": True,
    }
    _insert_candidate_with_metric(
        db,
        boundary=boundary,
        feasibility=0.2,
        is_feasible=0,
        objective=1.0,
        aspect=8.0,
        violations={"log10_qi": 0.2},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "p3_governor.py",
            "--problem",
            "p3",
            "--db",
            str(db),
            "--experiment-id",
            "1",
            "--run-dir",
            str(run_dir),
            "--workers",
            "1",
            "--queue-multiplier",
            "1",
            "--llm-enabled",
            "--llm-fallback",
            "--llm-codex-command",
            "python fake_llm.py",
        ],
    )
    with pytest.raises(ValueError, match="codex-only transport"):
        p3_governor.main()


def test_llm_transport_rejects_codex_prefixed_non_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = _make_db(tmp_path)
    run_dir = tmp_path / "run"
    boundary = {
        "r_cos": [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "z_sin": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "n_field_periods": 3,
        "is_stellarator_symmetric": True,
    }
    _insert_candidate_with_metric(
        db,
        boundary=boundary,
        feasibility=0.2,
        is_feasible=0,
        objective=1.0,
        aspect=8.0,
        violations={"log10_qi": 0.2},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "p3_governor.py",
            "--problem",
            "p3",
            "--db",
            str(db),
            "--experiment-id",
            "1",
            "--run-dir",
            str(run_dir),
            "--workers",
            "1",
            "--queue-multiplier",
            "1",
            "--llm-enabled",
            "--llm-fallback",
            "--llm-codex-command",
            "codex_malicious_wrapper run --json",
        ],
    )
    with pytest.raises(ValueError, match="codex-only transport"):
        p3_governor.main()


def test_governor_invalid_llm_output_falls_back_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = _make_db(tmp_path)
    run_dir = tmp_path / "run_invalid_llm_no_fallback"
    boundary = {
        "r_cos": [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "z_sin": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "n_field_periods": 3,
        "is_stellarator_symmetric": True,
    }
    _insert_candidate_with_metric(
        db,
        boundary=boundary,
        feasibility=0.2,
        is_feasible=0,
        objective=1.0,
        aspect=8.0,
        violations={"log10_qi": 0.2},
    )
    decision_file = tmp_path / "decision_invalid.json"
    decision_file.write_text(
        json.dumps(
            {
                "action": "jump",
                "target_constraint": "log10_qi",
                "mutations": [],
                "expected_effect": "invalid for p3",
                "restart_plan": None,
            }
        )
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "p3_governor.py",
            "--problem",
            "p3",
            "--db",
            str(db),
            "--experiment-id",
            "1",
            "--run-dir",
            str(run_dir),
            "--workers",
            "1",
            "--queue-multiplier",
            "1",
            "--llm-enabled",
            "--llm-fallback",
            "--llm-allow-decision-file",
            "--llm-decision-file",
            str(decision_file),
        ],
    )
    p3_governor.main()

    artifacts = sorted((run_dir / "governor").glob("governor_batch_*.json"))
    assert artifacts
    payload = json.loads(artifacts[-1].read_text())
    assert payload["llm"]["fallback_reason"] is not None
    assert str(payload["llm"]["fallback_reason"]).startswith("invalid_llm_output:")


def test_governor_invalid_llm_output_raises_without_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = _make_db(tmp_path)
    run_dir = tmp_path / "run_invalid_llm_no_fallback_raise"
    boundary = {
        "r_cos": [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "z_sin": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "n_field_periods": 3,
        "is_stellarator_symmetric": True,
    }
    _insert_candidate_with_metric(
        db,
        boundary=boundary,
        feasibility=0.2,
        is_feasible=0,
        objective=1.0,
        aspect=8.0,
        violations={"log10_qi": 0.2},
    )
    decision_file = tmp_path / "decision_invalid_raise.json"
    decision_file.write_text(
        json.dumps(
            {
                "action": "jump",
                "target_constraint": "log10_qi",
                "mutations": [],
                "expected_effect": "invalid for p3",
                "restart_plan": None,
            }
        )
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "p3_governor.py",
            "--problem",
            "p3",
            "--db",
            str(db),
            "--experiment-id",
            "1",
            "--run-dir",
            str(run_dir),
            "--workers",
            "1",
            "--queue-multiplier",
            "1",
            "--llm-enabled",
            "--llm-allow-decision-file",
            "--llm-decision-file",
            str(decision_file),
        ],
    )
    with pytest.raises(ValueError, match="llm_decision_unresolved"):
        p3_governor.main()


def test_replay_manifest_identity_check_is_deterministic(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    run_dir = tmp_path / "run"
    parent = run_dir / "candidates" / "parent.json"
    parent.parent.mkdir(parents=True, exist_ok=True)
    parent.write_text("{}")
    manifest_path = run_dir / "governor" / "resume_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": 1,
        "experiment_id": 1,
        "problem": "p3",
        "proposal_script": "scripts/p3_propose.py",
        "run_seed": 1000,
        "last_cycle": 0,
        "cycles": {},
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }
    command = _build_scale_cmd(
        proposal_script="scripts/p3_propose.py",
        problem="p3",
        db=db,
        experiment_id=1,
        run_dir=run_dir,
        batch_id=1,
        seed_base=2000,
        parent=parent,
        axisym_z=0.98,
        model_route="test_route",
    )
    _record_cycle_manifest(
        manifest=manifest,
        path=manifest_path,
        batch_id=1,
        seed_base=2000,
        cmds=[command],
        model_route="test_route",
    )
    _record_cycle_manifest(
        manifest=manifest,
        path=manifest_path,
        batch_id=1,
        seed_base=2000,
        cmds=[command],
        model_route="test_route",
    )
    mutated_command = _build_scale_cmd(
        proposal_script="scripts/p3_propose.py",
        problem="p3",
        db=db,
        experiment_id=1,
        run_dir=run_dir,
        batch_id=1,
        seed_base=2001,
        parent=parent,
        axisym_z=0.98,
        model_route="test_route",
    )
    with pytest.raises(ValueError, match="Replay identity mismatch"):
        _record_cycle_manifest(
            manifest=manifest,
            path=manifest_path,
            batch_id=1,
            seed_base=2001,
            cmds=[mutated_command],
            model_route="test_route",
        )


def test_resume_replay_seed_identity_continuity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = _make_db(tmp_path)
    run_dir = tmp_path / "run_resume_replay"
    governor_dir = run_dir / "governor"
    candidates_dir = run_dir / "candidates"
    governor_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir.mkdir(parents=True, exist_ok=True)

    parent_a = candidates_dir / "parent_a.json"
    parent_b = candidates_dir / "parent_b.json"
    parent_a.write_text(
        json.dumps(
            {
                "r_cos": [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                "z_sin": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                "n_field_periods": 3,
                "is_stellarator_symmetric": True,
            }
        )
    )
    parent_b.write_text(
        json.dumps(
            {
                "r_cos": [[1.0, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.0]],
                "z_sin": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                "n_field_periods": 3,
                "is_stellarator_symmetric": True,
            }
        )
    )

    manifest_path = governor_dir / "resume_manifest.json"
    manifest = {
        "version": 1,
        "experiment_id": 1,
        "problem": "p3",
        "proposal_script": "scripts/p3_propose.py",
        "run_seed": 1000,
        "phase": "feasibility_recovery",
        "last_cycle": 0,
        "cycles": {},
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }
    batch_id = 5
    seed_base = 17000
    manifest_cmd = _build_blend_cmd(
        proposal_script="scripts/p3_propose.py",
        problem="p3",
        db=db,
        experiment_id=1,
        run_dir=run_dir,
        batch_id=batch_id,
        seed_base=seed_base,
        parent_a=parent_a,
        parent_b=parent_b,
        t_min=0.2,
        t_max=0.4,
        t_step=0.2,
        model_route="resume/replay",
    )
    _record_cycle_manifest(
        manifest=manifest,
        path=manifest_path,
        batch_id=batch_id,
        seed_base=seed_base,
        cmds=[manifest_cmd],
        model_route="resume/replay",
    )

    existing_seed = candidate_seed(seed_base=seed_base, batch_id=batch_id, index=0)
    existing_hash = "resume_existing_seed_idx0"
    (candidates_dir / f"{existing_hash}_meta.json").write_text(
        json.dumps(
            {
                "experiment_id": 1,
                "batch_id": batch_id,
                "seed": existing_seed,
                "move_family": "blend",
                "parents": [],
                "knobs": {"t": 0.2},
                "created_at": "2026-01-01T00:00:00+00:00",
            }
        )
    )
    conn = sqlite3.connect(str(db))
    try:
        conn.execute(
            """
            INSERT INTO candidates
            (experiment_id, problem, params_json, seed, status, design_hash, operator_family, model_route)
            VALUES (1, 'p3', '{}', ?, 'done', ?, 'blend', 'resume')
            """,
            (existing_seed, existing_hash),
        )
        conn.commit()
    finally:
        conn.close()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "p3_governor.py",
            "--problem",
            "p3",
            "--db",
            str(db),
            "--experiment-id",
            "1",
            "--run-dir",
            str(run_dir),
            "--workers",
            "1",
            "--queue-multiplier",
            "1",
            "--execute",
        ],
    )
    p3_governor.main()

    expected_replay_seed = candidate_seed(
        seed_base=seed_base + 1,
        batch_id=batch_id,
        index=0,
    )
    conn = sqlite3.connect(str(db))
    try:
        rows = conn.execute(
            "SELECT seed FROM candidates WHERE experiment_id = 1 ORDER BY seed ASC"
        ).fetchall()
    finally:
        conn.close()
    observed = [int(row[0]) for row in rows]
    assert observed == [existing_seed, expected_replay_seed]

    p3_governor.main()
    conn = sqlite3.connect(str(db))
    try:
        rerun_rows = conn.execute(
            "SELECT seed FROM candidates WHERE experiment_id = 1 ORDER BY seed ASC"
        ).fetchall()
    finally:
        conn.close()
    rerun_observed = [int(row[0]) for row in rerun_rows]
    assert rerun_observed == observed

    persisted_manifest = json.loads(manifest_path.read_text())
    cycle_entry = persisted_manifest["cycles"][str(batch_id)]
    assert "--t-min" in cycle_entry["command_argvs"][0]
    assert "--t-step" in cycle_entry["command_argvs"][0]


def test_startup_replay_commands_only_for_missing_batches(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "governor").mkdir(parents=True, exist_ok=True)
    manifest = {
        "cycles": {
            "1": {
                "batch_id": 1,
                "command_argvs": [
                    ["python", "scripts/p3_propose.py", "--family", "scale_groups"]
                ],
            },
            "2": {
                "batch_id": 2,
                "command_argvs": [
                    ["python", "scripts/p3_propose.py", "--family", "blend"]
                ],
            },
        }
    }
    candidates_dir = run_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    (candidates_dir / "abc_meta.json").write_text(
        json.dumps(
            {
                "experiment_id": 1,
                "batch_id": 1,
                "seed": 123,
                "move_family": "seed",
                "parents": [],
                "knobs": {},
                "created_at": "2026-01-01T00:00:00+00:00",
            }
        )
    )
    cmds, diagnostics = _startup_replay_commands(manifest=manifest, run_dir=run_dir)
    assert len(cmds) == 1
    assert diagnostics["replay_batches"] == [2]


def test_startup_replay_commands_replays_partial_batch_remainder_when_candidate_count_missing(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "governor").mkdir(parents=True, exist_ok=True)
    manifest = {
        "cycles": {
            "1": {
                "batch_id": 7,
                "expected_candidates": 2,
                "command_argvs": [
                    [
                        "python",
                        "scripts/p3_propose.py",
                        "--family",
                        "blend",
                        "--seed-base",
                        "17000",
                        "--t-min",
                        "0.2",
                        "--t-max",
                        "0.4",
                        "--t-step",
                        "0.2",
                    ],
                ],
            }
        }
    }
    candidates_dir = run_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    (candidates_dir / "partial_meta.json").write_text(
        json.dumps(
            {
                "experiment_id": 1,
                "batch_id": 7,
                "seed": 17001,
                "move_family": "seed",
                "parents": [],
                "knobs": {},
                "created_at": "2026-01-01T00:00:00+00:00",
            }
        )
    )

    cmds, diagnostics = _startup_replay_commands(manifest=manifest, run_dir=run_dir)

    assert len(cmds) == 1
    assert "--seed-base" in cmds[0].argv
    assert "17001" in cmds[0].argv
    assert "--t" in cmds[0].argv
    assert "0.400000" in cmds[0].argv
    assert diagnostics["replay_batches"] == []
    assert diagnostics["partial_replay_batches"] == [7]
    assert diagnostics["partial_skipped_batches"] == []


def test_startup_replay_commands_legacy_manifest_blend_replays_partial_batch_remainder(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "governor").mkdir(parents=True, exist_ok=True)
    manifest = {
        "cycles": {
            "1": {
                "batch_id": 8,
                "command_argvs": [
                    [
                        "python",
                        "scripts/p3_propose.py",
                        "--family",
                        "blend",
                        "--seed-base",
                        "18000",
                        "--t-min",
                        "0.2",
                        "--t-max",
                        "0.4",
                        "--t-step",
                        "0.2",
                    ],
                ],
            }
        }
    }
    candidates_dir = run_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    (candidates_dir / "legacy_partial_meta.json").write_text(
        json.dumps(
            {
                "experiment_id": 1,
                "batch_id": 8,
                "seed": 18001,
                "move_family": "blend",
                "parents": [],
                "knobs": {},
                "created_at": "2026-01-01T00:00:00+00:00",
            }
        )
    )

    cmds, diagnostics = _startup_replay_commands(manifest=manifest, run_dir=run_dir)

    assert len(cmds) == 1
    assert "--seed-base" in cmds[0].argv
    assert "18001" in cmds[0].argv
    assert "--t" in cmds[0].argv
    assert "0.400000" in cmds[0].argv
    assert diagnostics["replay_batches"] == []
    assert diagnostics["partial_replay_batches"] == [8]
    assert diagnostics["partial_skipped_batches"] == []


def test_startup_replay_commands_skips_batches_with_pending_candidates(
    tmp_path: Path,
) -> None:
    db = _make_db(tmp_path)
    run_dir = tmp_path / "run"
    (run_dir / "governor").mkdir(parents=True, exist_ok=True)
    pending_hash = "pending_design_hash"
    manifest = {
        "experiment_id": 1,
        "cycles": {
            "1": {
                "batch_id": 9,
                "expected_candidates": 2,
                "command_argvs": [
                    [
                        "python",
                        "scripts/p3_propose.py",
                        "--family",
                        "blend",
                        "--seed-base",
                        "19000",
                        "--t-min",
                        "0.1",
                        "--t-max",
                        "0.3",
                        "--t-step",
                        "0.2",
                    ]
                ],
            }
        },
    }
    candidates_dir = run_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    (candidates_dir / f"{pending_hash}_meta.json").write_text(
        json.dumps(
            {
                "experiment_id": 1,
                "batch_id": 9,
                "seed": 19001,
                "move_family": "blend",
                "parents": [],
                "knobs": {},
                "created_at": "2026-01-01T00:00:00+00:00",
            }
        )
    )

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(
            """
            INSERT INTO candidates
            (experiment_id, problem, params_json, seed, status, design_hash, operator_family, model_route)
            VALUES (1, 'p3', '{}', 1, 'pending', ?, 'seed', 'test')
            """,
            (pending_hash,),
        )
        conn.commit()
        cmds, diagnostics = _startup_replay_commands(
            manifest=manifest,
            run_dir=run_dir,
            conn=conn,
        )
    finally:
        conn.close()

    assert cmds == []
    assert diagnostics["pending_skipped_batches"] == [9]


def test_dominant_violation_rate_uses_dominant_constraint_frequency() -> None:
    rows = [
        CandidateRow(
            candidate_id=1,
            design_hash="a",
            seed=1,
            feasibility=0.5,
            is_feasible=False,
            lgradb=None,
            aspect=None,
            violations={"mirror": 0.10},
            metrics={},
            meta={},
            lineage_parent_hashes=[],
            novelty_score=None,
            operator_family="seed",
            model_route="test",
        ),
        CandidateRow(
            candidate_id=2,
            design_hash="b",
            seed=2,
            feasibility=0.5,
            is_feasible=False,
            lgradb=None,
            aspect=None,
            violations={"mirror": 0.20},
            metrics={},
            meta={},
            lineage_parent_hashes=[],
            novelty_score=None,
            operator_family="seed",
            model_route="test",
        ),
        CandidateRow(
            candidate_id=3,
            design_hash="c",
            seed=3,
            feasibility=0.5,
            is_feasible=False,
            lgradb=None,
            aspect=None,
            violations={"mirror": 0.30, "log10_qi": 0.05},
            metrics={},
            meta={},
            lineage_parent_hashes=[],
            novelty_score=None,
            operator_family="seed",
            model_route="test",
        ),
        CandidateRow(
            candidate_id=4,
            design_hash="d",
            seed=4,
            feasibility=0.5,
            is_feasible=False,
            lgradb=None,
            aspect=None,
            violations={"log10_qi": 0.20},
            metrics={},
            meta={},
            lineage_parent_hashes=[],
            novelty_score=None,
            operator_family="seed",
            model_route="test",
        ),
        CandidateRow(
            candidate_id=5,
            design_hash="e",
            seed=5,
            feasibility=0.5,
            is_feasible=False,
            lgradb=None,
            aspect=None,
            violations={},
            metrics={},
            meta={},
            lineage_parent_hashes=[],
            novelty_score=None,
            operator_family="seed",
            model_route="test",
        ),
    ]
    assert _dominant_violation_rate(rows, limit=5) == pytest.approx(0.6)


def test_governor_enforces_global_command_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = _make_db(tmp_path)
    run_dir = tmp_path / "run"
    boundary = {
        "r_cos": [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "z_sin": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "n_field_periods": 3,
        "is_stellarator_symmetric": True,
    }
    _insert_candidate_with_metric(
        db,
        boundary=boundary,
        feasibility=0.2,
        is_feasible=0,
        objective=1.0,
        aspect=8.0,
        violations={"mirror": 0.2},
    )
    decision_file = tmp_path / "decision.json"
    decision_file.write_text(
        json.dumps(
            {
                "action": "repair",
                "target_constraint": "mirror",
                "mutations": [],
                "expected_effect": "repair mirror",
                "restart_plan": None,
            }
        )
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "p3_governor.py",
            "--problem",
            "p3",
            "--db",
            str(db),
            "--experiment-id",
            "1",
            "--run-dir",
            str(run_dir),
            "--workers",
            "1",
            "--queue-multiplier",
            "1",
            "--llm-enabled",
            "--llm-fallback",
            "--llm-allow-decision-file",
            "--llm-decision-file",
            str(decision_file),
        ],
    )
    p3_governor.main()
    artifacts = sorted((run_dir / "governor").glob("governor_batch_*.json"))
    assert artifacts
    payload = json.loads(artifacts[-1].read_text())
    assert len(payload["commands"]) == 8
    assert payload["command_cap"]["applied"] is True


def test_governor_handles_malformed_candidate_params_json_without_crashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = _make_db(tmp_path)
    run_dir = tmp_path / "run_malformed_candidate"
    _insert_malformed_candidate_with_metric(db, design_hash="malformed_design")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "p3_governor.py",
            "--problem",
            "p3",
            "--db",
            str(db),
            "--experiment-id",
            "1",
            "--run-dir",
            str(run_dir),
            "--workers",
            "1",
            "--queue-multiplier",
            "1",
        ],
    )
    p3_governor.main()
    artifacts = sorted((run_dir / "governor").glob("governor_batch_*.json"))
    assert artifacts == []


def test_governor_persists_phase_in_resume_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = _make_db(tmp_path)
    run_dir = tmp_path / "run"
    boundary = {
        "r_cos": [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "z_sin": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "n_field_periods": 3,
        "is_stellarator_symmetric": True,
    }
    _insert_candidate_with_metric(
        db,
        boundary=boundary,
        feasibility=0.2,
        is_feasible=0,
        objective=1.0,
        aspect=8.0,
        violations={"log10_qi": 0.2},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "p3_governor.py",
            "--problem",
            "p3",
            "--db",
            str(db),
            "--experiment-id",
            "1",
            "--run-dir",
            str(run_dir),
            "--workers",
            "1",
            "--queue-multiplier",
            "1",
        ],
    )
    p3_governor.main()
    manifest_path = run_dir / "governor" / "resume_manifest.json"
    payload = json.loads(manifest_path.read_text())
    assert payload["phase"] in {"feasibility_recovery", "frontier_improvement"}


def test_governor_stop_policy_max_cycles_cap_exits_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = _make_db(tmp_path)
    run_dir = tmp_path / "run_loop_cap"
    boundary = {
        "r_cos": [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "z_sin": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "n_field_periods": 3,
        "is_stellarator_symmetric": True,
    }
    _insert_candidate_with_metric(
        db,
        boundary=boundary,
        feasibility=0.2,
        is_feasible=0,
        objective=1.0,
        aspect=8.0,
        violations={"log10_qi": 0.2},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "p3_governor.py",
            "--problem",
            "p3",
            "--db",
            str(db),
            "--experiment-id",
            "1",
            "--run-dir",
            str(run_dir),
            "--workers",
            "1",
            "--queue-multiplier",
            "1",
            "--loop",
            "--max-cycles",
            "1",
        ],
    )
    p3_governor.main()
    artifacts = sorted((run_dir / "governor").glob("governor_batch_*.json"))
    assert len(artifacts) == 1


def test_governor_stop_policy_stagnation_cap_exits_before_enqueue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = _make_db(tmp_path)
    run_dir = tmp_path / "run_stagnation_cap"
    boundary = {
        "r_cos": [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "z_sin": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "n_field_periods": 3,
        "is_stellarator_symmetric": True,
    }
    _insert_candidate_with_metric(
        db,
        boundary=boundary,
        feasibility=0.2,
        is_feasible=0,
        objective=1.0,
        aspect=8.0,
        violations={"log10_qi": 0.2},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "p3_governor.py",
            "--problem",
            "p3",
            "--db",
            str(db),
            "--experiment-id",
            "1",
            "--run-dir",
            str(run_dir),
            "--workers",
            "1",
            "--queue-multiplier",
            "1",
            "--loop",
            "--max-stagnation-cycles",
            "1",
        ],
    )
    p3_governor.main()
    artifacts = sorted((run_dir / "governor").glob("governor_batch_*.json"))
    assert artifacts == []


def test_governor_stop_policy_runtime_cap_exits_before_enqueue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = _make_db(tmp_path)
    run_dir = tmp_path / "run_runtime_cap"
    boundary = {
        "r_cos": [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "z_sin": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "n_field_periods": 3,
        "is_stellarator_symmetric": True,
    }
    _insert_candidate_with_metric(
        db,
        boundary=boundary,
        feasibility=0.2,
        is_feasible=0,
        objective=1.0,
        aspect=8.0,
        violations={"log10_qi": 0.2},
    )

    call_count = {"n": 0}

    def _fake_monotonic() -> float:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return 0.0
        return 10.0

    monkeypatch.setattr(p3_governor.time, "monotonic", _fake_monotonic)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "p3_governor.py",
            "--problem",
            "p3",
            "--db",
            str(db),
            "--experiment-id",
            "1",
            "--run-dir",
            str(run_dir),
            "--workers",
            "1",
            "--queue-multiplier",
            "1",
            "--loop",
            "--max-runtime-sec",
            "1.0",
        ],
    )
    p3_governor.main()
    artifacts = sorted((run_dir / "governor").glob("governor_batch_*.json"))
    assert artifacts == []


@pytest.mark.parametrize(
    "problem,target_constraint",
    [("p1", "aspect_ratio"), ("p2", "log10_qi"), ("p3", "log10_qi")],
)
@pytest.mark.parametrize("llm_enabled", [False, True])
def test_governor_dry_run_profiles_llm_toggle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    problem: str,
    target_constraint: str,
    llm_enabled: bool,
) -> None:
    db = _make_db(tmp_path)
    run_dir = tmp_path / f"run_{problem}_{'llm' if llm_enabled else 'det'}"
    boundary = {
        "r_cos": [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "z_sin": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "n_field_periods": 3,
        "is_stellarator_symmetric": True,
    }
    _insert_candidate_with_metric(
        db,
        boundary=boundary,
        feasibility=0.2,
        is_feasible=0,
        objective=1.0,
        aspect=8.0,
        violations={"log10_qi": 0.2},
    )
    argv = [
        "p3_governor.py",
        "--problem",
        problem,
        "--db",
        str(db),
        "--experiment-id",
        "1",
        "--run-dir",
        str(run_dir),
        "--workers",
        "1",
        "--queue-multiplier",
        "1",
    ]
    if llm_enabled:
        decision_file = tmp_path / f"decision_{problem}.json"
        decision_file.write_text(
            json.dumps(
                {
                    "action": "repair",
                    "target_constraint": target_constraint,
                    "mutations": [],
                    "expected_effect": "repair",
                    "restart_plan": None,
                }
            )
        )
        argv.extend(
            [
                "--llm-enabled",
                "--llm-fallback",
                "--llm-allow-decision-file",
                "--llm-decision-file",
                str(decision_file),
            ]
        )

    monkeypatch.setattr(sys, "argv", argv)
    p3_governor.main()
    artifacts = sorted((run_dir / "governor").glob("governor_batch_*.json"))
    assert artifacts
