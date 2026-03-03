from __future__ import annotations

import json
import sqlite3
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

from ai_scientist.memory import hash_payload
from ai_scientist.memory.schema import init_db
from ai_scientist.problem_profiles import get_problem_profile
from scripts import governor as governor_runtime


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


def _insert_candidate_with_payload(
    db: Path,
    *,
    boundary: dict,
    raw_payload: dict,
    feasibility: float,
    is_feasible: int,
    objective: float,
    seed: int = 1,
) -> None:
    design_hash = hash_payload(boundary)
    conn = sqlite3.connect(str(db))
    try:
        cursor = conn.execute(
            """
            INSERT INTO candidates
            (experiment_id, problem, params_json, seed, status, design_hash, operator_family, model_route)
            VALUES (1, 'p3', ?, ?, 'done', ?, 'scale_groups', 'seed')
            """,
            (json.dumps(boundary), int(seed), design_hash),
        )
        assert cursor.lastrowid is not None
        candidate_id = int(cursor.lastrowid)
        conn.execute(
            """
            INSERT INTO metrics (candidate_id, raw_json, feasibility, objective, hv, is_feasible)
            VALUES (?, ?, ?, ?, NULL, ?)
            """,
            (
                candidate_id,
                json.dumps(raw_payload),
                float(feasibility),
                float(objective),
                int(is_feasible),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _boundary() -> dict:
    return {
        "r_cos": [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "z_sin": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "n_field_periods": 3,
        "is_stellarator_symmetric": True,
    }


def test_fetch_candidates_uses_violations_when_constraint_margins_missing(
    tmp_path: Path,
) -> None:
    db = _make_db(tmp_path)
    boundary = _boundary()
    _insert_candidate_with_payload(
        db,
        boundary=boundary,
        raw_payload={
            "metrics": {"aspect_ratio": 8.1, "lgradB": 1.2},
            "violations": {"log10_qi": 0.22, "mirror": 0.05},
        },
        feasibility=0.22,
        is_feasible=0,
        objective=1.2,
    )
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    profile = get_problem_profile("p3")
    try:
        rows = governor_runtime._fetch_candidates(
            conn, profile=profile, experiment_id=1, limit=10
        )
    finally:
        conn.close()
    assert rows
    assert rows[0].violations.get("log10_qi") == pytest.approx(0.22)
    assert governor_runtime._dominant_violation(rows, limit=20) == "log10_qi"


def test_next_batch_id_from_manifest_is_monotonic(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "candidates").mkdir(parents=True, exist_ok=True)
    manifest = {"last_cycle": 5}
    assert (
        governor_runtime._next_batch_id_from_manifest(
            manifest=manifest, run_dir=run_dir
        )
        == 6
    )

    (run_dir / "candidates" / "seen_meta.json").write_text(
        json.dumps(
            {
                "experiment_id": 1,
                "batch_id": 12,
                "seed": 1234,
                "move_family": "seed",
                "parents": [],
                "knobs": {},
                "created_at": "2026-01-01T00:00:00+00:00",
            }
        )
    )
    assert (
        governor_runtime._next_batch_id_from_manifest(
            manifest=manifest, run_dir=run_dir
        )
        == 13
    )


def test_restart_blend_schedule_is_deterministically_jittered() -> None:
    first = governor_runtime._restart_blend_schedule(
        run_seed=170303003,
        batch_id=2,
        t_min=0.85,
        t_max=0.95,
        t_step=0.005,
    )
    second = governor_runtime._restart_blend_schedule(
        run_seed=170303003,
        batch_id=3,
        t_min=0.85,
        t_max=0.95,
        t_step=0.005,
    )
    assert first != second
    for t_min, t_max, t_step in [first, second]:
        assert 0.0 <= t_min < t_max <= 1.0
        assert t_step > 0.0


def test_bootstrap_cycle_logs_step0_scratchpad_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = _make_db(tmp_path)
    run_dir = tmp_path / "run"
    parent_a = tmp_path / "parent_a.json"
    parent_b = tmp_path / "parent_b.json"
    parent_a.write_text(json.dumps(_boundary()))
    b = _boundary()
    b["r_cos"][1][1] = 0.2
    parent_b.write_text(json.dumps(b))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "governor.py",
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
            "--bootstrap-parent-a",
            str(parent_a),
            "--bootstrap-parent-b",
            str(parent_b),
        ],
    )
    governor_runtime.main()

    conn = sqlite3.connect(str(db))
    try:
        row = conn.execute(
            """
            SELECT COUNT(*) FROM scratchpad_events
            WHERE experiment_id = 1 AND cycle = 1 AND step = 0 AND aso_action = 'bootstrap'
            """
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    assert int(row[0]) == 1


def test_zero_yield_cycles_advance_batch_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = _make_db(tmp_path)
    run_dir = tmp_path / "run"
    boundary = _boundary()
    _insert_candidate_with_payload(
        db,
        boundary=boundary,
        raw_payload={
            "metrics": {"aspect_ratio": 8.0, "lgradB": 1.0},
            "constraint_margins": {"log10_qi": 0.2},
        },
        feasibility=0.2,
        is_feasible=0,
        objective=1.0,
    )

    def _fake_run_cmds(cmds: object) -> dict[str, int | bool]:
        assert cmds
        return {"inserted": 0, "skipped": 8, "parsed": True}

    monkeypatch.setattr(governor_runtime, "_run_cmds", _fake_run_cmds)
    monkeypatch.setattr(governor_runtime.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "governor.py",
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
            "2",
            "--execute",
            "--loop",
            "--max-cycles",
            "2",
        ],
    )
    governor_runtime.main()

    artifacts = sorted((run_dir / "governor").glob("governor_batch_*.json"))
    assert len(artifacts) == 2
    payloads = [json.loads(path.read_text()) for path in artifacts]
    assert payloads[0]["batch_id"] == 1
    assert payloads[1]["batch_id"] == 2


def test_run_cmds_parses_inserted_and_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_subprocess_run(
        argv: list[str],
        *,
        check: bool,
        text: bool,
        capture_output: bool,
    ) -> SimpleNamespace:
        assert check is True
        assert text is True
        assert capture_output is True
        assert argv
        return SimpleNamespace(
            stdout="problem=p3 batch_id=7 family=scale_groups inserted=2 skipped=5\n",
            stderr="",
        )

    monkeypatch.setattr(governor_runtime.subprocess, "run", _fake_subprocess_run)
    cmd = governor_runtime.ProposalCommand(argv=["python", "scripts/p3_propose.py"])
    summary = governor_runtime._run_cmds([cmd, cmd])
    assert summary == {"inserted": 4, "skipped": 10, "parsed": True}


def test_startup_replay_ignores_malformed_cycle_keys(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "cycles": {
            "bad-key": {"batch_id": 1, "command_argvs": [["python", "bad.py"]]},
            "2": {"batch_id": 2, "command_argvs": [["python", "ok.py"]]},
        }
    }
    cmds, diagnostics = governor_runtime._startup_replay_commands(
        manifest=manifest,
        run_dir=run_dir,
        conn=None,
    )
    assert [cmd.argv for cmd in cmds] == [["python", "ok.py"]]
    assert diagnostics["replay_command_count"] == 1


def test_replay_cycles_are_counted_toward_max_cycles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = _make_db(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    replay_cmd = governor_runtime.ProposalCommand(
        argv=["python", "scripts/p3_propose.py"]
    )

    monkeypatch.setattr(
        governor_runtime,
        "_startup_replay_commands",
        lambda **_: (
            [replay_cmd],
            {
                "replay_batches": [1],
                "partial_replay_batches": [],
                "partial_skipped_batches": [],
                "pending_skipped_batches": [],
                "replay_command_count": 1,
            },
        ),
    )
    monkeypatch.setattr(governor_runtime.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "governor.py",
            "--problem",
            "p3",
            "--db",
            str(db),
            "--experiment-id",
            "1",
            "--run-dir",
            str(run_dir),
            "--loop",
            "--max-cycles",
            "1",
        ],
    )
    governor_runtime.main()
    artifacts = sorted((run_dir / "governor").glob("governor_batch_*.json"))
    assert artifacts == []
