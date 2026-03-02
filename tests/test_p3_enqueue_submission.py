from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

from ai_scientist.memory.schema import init_db
from scripts import p3_enqueue_submission


def test_enqueue_submission_limit_applies_to_input_items(
    tmp_path: Path, monkeypatch
) -> None:
    db_path = tmp_path / "wm.sqlite"
    init_db(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
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

    run_dir = tmp_path / "run"
    boundary_a = {
        "r_cos": [[1.0, 0.0, 0.0]],
        "z_sin": [[0.0, 0.0, 0.0]],
        "n_field_periods": 3,
        "is_stellarator_symmetric": True,
    }
    boundary_b = {
        "r_cos": [[1.0, 0.1, 0.0]],
        "z_sin": [[0.0, 0.1, 0.0]],
        "n_field_periods": 3,
        "is_stellarator_symmetric": True,
    }
    submission_path = tmp_path / "submission.json"
    submission_path.write_text(
        json.dumps(
            [
                boundary_a,
                boundary_a,
                boundary_b,
            ]
        )
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "p3_enqueue_submission.py",
            "--db",
            str(db_path),
            "--experiment-id",
            "1",
            "--run-dir",
            str(run_dir),
            "--batch-id",
            "1",
            "--limit",
            "2",
            str(submission_path),
        ],
    )
    p3_enqueue_submission.main()

    conn2 = sqlite3.connect(str(db_path))
    conn2.row_factory = sqlite3.Row
    try:
        row = conn2.execute(
            "SELECT COUNT(*) AS n FROM candidates WHERE experiment_id = 1"
        ).fetchone()
        assert row is not None
        # Only first two input items are considered; they are duplicates.
        assert int(row["n"]) == 1
    finally:
        conn2.close()
