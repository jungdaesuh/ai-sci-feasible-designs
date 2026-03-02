from __future__ import annotations

import json
import sqlite3
import sys
import types
from pathlib import Path

from ai_scientist.memory.schema import init_db

try:
    from scripts import p3_worker
except ImportError:
    constellaration_module = types.ModuleType("constellaration")
    forward_model_module = types.ModuleType("constellaration.forward_model")
    problems_module = types.ModuleType("constellaration.problems")
    geometry_module = types.ModuleType("constellaration.geometry")
    surface_module = types.ModuleType("constellaration.geometry.surface_rz_fourier")
    geometry_module.surface_rz_fourier = surface_module
    constellaration_module.forward_model = forward_model_module
    constellaration_module.problems = problems_module
    constellaration_module.geometry = geometry_module
    sys.modules["constellaration"] = constellaration_module
    sys.modules["constellaration.forward_model"] = forward_model_module
    sys.modules["constellaration.problems"] = problems_module
    sys.modules["constellaration.geometry"] = geometry_module
    sys.modules["constellaration.geometry.surface_rz_fourier"] = surface_module
    from scripts import p3_worker


def _setup_worker_db(tmp_path: Path) -> tuple[Path, str]:
    db_path = tmp_path / "wm.sqlite"
    init_db(db_path)
    boundary = {
        "r_cos": [[1.0, 0.0, 0.0]],
        "z_sin": [[0.0, 0.0, 0.0]],
        "n_field_periods": 3,
        "is_stellarator_symmetric": True,
    }
    design_hash = "design_hash_1"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(
            """
            INSERT INTO experiments (started_at, config_json, git_sha, constellaration_sha, notes)
            VALUES ('2026-01-01T00:00:00+00:00', '{}', 'sha', 'const_sha', NULL)
            """
        )
        conn.execute(
            """
            INSERT INTO candidates
            (experiment_id, problem, params_json, seed, status, design_hash, operator_family, model_route)
            VALUES (1, 'p3', ?, 123, 'pending', ?, 'seed', 'test')
            """,
            (json.dumps(boundary), design_hash),
        )
        conn.commit()
    finally:
        conn.close()
    return db_path, design_hash


def test_worker_creates_eval_directory_and_writes_eval(
    tmp_path: Path, monkeypatch
) -> None:
    db_path, design_hash = _setup_worker_db(tmp_path)
    run_dir = tmp_path / "run"

    def _fake_eval(
        *, problem: str, boundary: dict
    ) -> tuple[dict, p3_worker.EvalSummary]:
        assert problem == "p3"
        assert isinstance(boundary, dict)
        return (
            {"metrics": {"aspect_ratio": 8.0}, "violations": {}, "feasibility": 0.5},
            p3_worker.EvalSummary(
                feasibility=0.5,
                is_feasible=False,
                objective=1.0,
                aspect=8.0,
            ),
        )

    monkeypatch.setattr(p3_worker, "_evaluate_boundary", _fake_eval)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "p3_worker.py",
            "--problem",
            "p3",
            "--db",
            str(db_path),
            "--experiment-id",
            "1",
            "--run-dir",
            str(run_dir),
            "--worker-id",
            "1",
            "--limit",
            "1",
        ],
    )
    p3_worker.main()

    eval_path = run_dir / "eval" / f"{design_hash}.json"
    assert eval_path.exists()


def test_worker_handles_malformed_meta_as_failed_candidate(
    tmp_path: Path, monkeypatch
) -> None:
    db_path, design_hash = _setup_worker_db(tmp_path)
    run_dir = tmp_path / "run"
    candidates_dir = run_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    (candidates_dir / f"{design_hash}_meta.json").write_text("{invalid_json")

    called = {"evaluate": False}

    def _fake_eval(
        *, problem: str, boundary: dict
    ) -> tuple[dict, p3_worker.EvalSummary]:
        called["evaluate"] = True
        raise AssertionError("should not evaluate when meta parsing fails first")

    monkeypatch.setattr(p3_worker, "_evaluate_boundary", _fake_eval)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "p3_worker.py",
            "--problem",
            "p3",
            "--db",
            str(db_path),
            "--experiment-id",
            "1",
            "--run-dir",
            str(run_dir),
            "--worker-id",
            "1",
            "--limit",
            "1",
        ],
    )
    p3_worker.main()
    assert called["evaluate"] is False

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT status FROM candidates WHERE design_hash = ?", (design_hash,)
        ).fetchone()
        assert row is not None
        assert str(row["status"]).startswith("failed:")
    finally:
        conn.close()


def test_worker_handles_malformed_params_json_as_failed_candidate(
    tmp_path: Path, monkeypatch
) -> None:
    db_path, design_hash = _setup_worker_db(tmp_path)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "UPDATE candidates SET params_json = ? WHERE design_hash = ?",
            ("{invalid_json", design_hash),
        )
        conn.commit()
    finally:
        conn.close()

    run_dir = tmp_path / "run"
    called = {"evaluate": False}

    def _fake_eval(
        *, problem: str, boundary: dict
    ) -> tuple[dict, p3_worker.EvalSummary]:
        called["evaluate"] = True
        raise AssertionError("should not evaluate when params_json parsing fails")

    monkeypatch.setattr(p3_worker, "_evaluate_boundary", _fake_eval)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "p3_worker.py",
            "--problem",
            "p3",
            "--db",
            str(db_path),
            "--experiment-id",
            "1",
            "--run-dir",
            str(run_dir),
            "--worker-id",
            "1",
            "--limit",
            "1",
        ],
    )
    p3_worker.main()
    assert called["evaluate"] is False

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT status FROM candidates WHERE design_hash = ?",
            (design_hash,),
        ).fetchone()
        assert row is not None
        assert str(row["status"]).startswith("failed:")
    finally:
        conn.close()
