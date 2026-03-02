from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

from ai_scientist.memory import hash_payload
from ai_scientist.memory.schema import init_db
from scripts import p3_governor


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
            VALUES (1, 'p3', ?, 1, 'done', ?, 'scale_groups', 'seed')
            """,
            (json.dumps(boundary), design_hash),
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
