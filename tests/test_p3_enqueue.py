from __future__ import annotations

import json
from pathlib import Path

from ai_scientist.memory import hash_payload
from ai_scientist.p3_enqueue import candidate_seed, connect_queue_db, enqueue_candidate


def _boundary() -> dict:
    return {
        "r_cos": [[1.0, 0.01, 0.0], [0.0, 0.02, 0.0]],
        "z_sin": [[0.0, 0.0, 0.0], [0.0, 0.01, 0.0]],
        "n_field_periods": 3,
        "is_stellarator_symmetric": True,
    }


def test_candidate_seed_is_deterministic() -> None:
    seed = candidate_seed(seed_base=10, batch_id=7, index=3)
    assert seed == 7_000_013


def test_enqueue_candidate_persists_row_and_artifacts(tmp_path: Path) -> None:
    db_path = tmp_path / "world.sqlite"
    run_dir = tmp_path / "run"
    boundary = _boundary()
    parent_hash = hash_payload(boundary)

    conn = connect_queue_db(db_path)
    try:
        with conn:
            record = enqueue_candidate(
                conn,
                experiment_id=42,
                run_dir=run_dir,
                batch_id=1,
                boundary=boundary,
                seed=candidate_seed(seed_base=0, batch_id=1, index=0),
                move_family="blend",
                parents=[parent_hash],
                knobs={"t": 0.05},
                novelty_score=0.05,
                operator_family="blend",
                model_route="governor_static_recipe/mirror",
            )
        assert record is not None
        assert record.design_hash == hash_payload(boundary)
        assert record.novelty_score == 0.05

        row = conn.execute(
            """
            SELECT status, design_hash, lineage_parent_hashes_json, novelty_score,
                   operator_family, model_route
            FROM candidates
            WHERE experiment_id = ?
            """,
            (42,),
        ).fetchone()
        assert row is not None
        assert row["status"] == "pending"
        assert row["design_hash"] == record.design_hash
        assert json.loads(row["lineage_parent_hashes_json"]) == [parent_hash]
        assert row["novelty_score"] == 0.05
        assert row["operator_family"] == "blend"
        assert row["model_route"] == "governor_static_recipe/mirror"

        candidate_path = run_dir / "candidates" / f"{record.design_hash}.json"
        meta_path = run_dir / "candidates" / f"{record.design_hash}_meta.json"
        batch_log = run_dir / "batches" / "batch_001.jsonl"
        assert candidate_path.exists()
        assert meta_path.exists()
        assert batch_log.exists()

        batch_lines = batch_log.read_text().strip().splitlines()
        assert len(batch_lines) == 1
        batch_record = json.loads(batch_lines[0])
        assert batch_record["candidate_id"] == record.candidate_id
        assert batch_record["design_hash"] == record.design_hash
        assert batch_record["move_family"] == "blend"
        assert batch_record["model_route"] == "governor_static_recipe/mirror"

        artifact_rows = conn.execute(
            "SELECT kind FROM artifacts WHERE experiment_id = ? ORDER BY id",
            (42,),
        ).fetchall()
        assert [str(row["kind"]) for row in artifact_rows] == [
            "candidate_json",
            "candidate_meta",
        ]
    finally:
        conn.close()


def test_enqueue_candidate_skips_duplicate_design_hash(tmp_path: Path) -> None:
    db_path = tmp_path / "world.sqlite"
    run_dir = tmp_path / "run"
    boundary = _boundary()

    conn = connect_queue_db(db_path)
    try:
        with conn:
            first = enqueue_candidate(
                conn,
                experiment_id=7,
                run_dir=run_dir,
                batch_id=2,
                boundary=boundary,
                seed=candidate_seed(seed_base=11, batch_id=2, index=0),
                move_family="seed",
                parents=[],
                knobs={"index": 0},
                novelty_score=None,
                operator_family="seed",
                model_route="submission_seedlist",
            )
            second = enqueue_candidate(
                conn,
                experiment_id=7,
                run_dir=run_dir,
                batch_id=2,
                boundary=boundary,
                seed=candidate_seed(seed_base=11, batch_id=2, index=1),
                move_family="seed",
                parents=[],
                knobs={"index": 1},
                novelty_score=None,
                operator_family="seed",
                model_route="submission_seedlist",
            )
        assert first is not None
        assert second is None

        count_row = conn.execute(
            "SELECT COUNT(*) AS n FROM candidates WHERE experiment_id = ?",
            (7,),
        ).fetchone()
        assert count_row is not None
        assert int(count_row["n"]) == 1

        batch_log = run_dir / "batches" / "batch_002.jsonl"
        batch_lines = batch_log.read_text().strip().splitlines()
        assert len(batch_lines) == 1
    finally:
        conn.close()
