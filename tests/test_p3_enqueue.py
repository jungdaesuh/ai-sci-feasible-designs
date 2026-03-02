from __future__ import annotations

from pathlib import Path

from ai_scientist.p3_enqueue import connect_queue_db, enqueue_candidate


def test_enqueue_candidate_persists_problem_label(tmp_path: Path) -> None:
    db_path = tmp_path / "wm.sqlite"
    run_dir = tmp_path / "run"
    conn = connect_queue_db(db_path)
    try:
        conn.execute(
            """
            INSERT INTO experiments (started_at, config_json, git_sha, constellaration_sha, notes)
            VALUES ('2026-01-01T00:00:00+00:00', '{}', 'sha', 'const_sha', NULL)
            """
        )
        conn.commit()
        record = enqueue_candidate(
            conn,
            experiment_id=1,
            problem="p2",
            run_dir=run_dir,
            batch_id=1,
            boundary={
                "r_cos": [[1.0, 0.0, 0.0]],
                "z_sin": [[0.0, 0.0, 0.0]],
                "n_field_periods": 3,
                "is_stellarator_symmetric": True,
            },
            seed=123,
            move_family="seed",
            parents=[],
            knobs={},
            novelty_score=None,
            operator_family="seed",
            model_route="test",
        )
        assert record is not None
        row = conn.execute(
            "SELECT problem FROM candidates WHERE id = ?",
            (int(record.candidate_id),),
        ).fetchone()
        assert row is not None
        assert str(row["problem"]) == "p2"
    finally:
        conn.close()
