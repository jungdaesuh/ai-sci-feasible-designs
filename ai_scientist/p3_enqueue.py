"""Shared enqueue helpers for P3 candidate queue scripts."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from ai_scientist.memory import hash_payload
from ai_scientist.memory.schema import init_db


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect_queue_db(db_path: Path) -> sqlite3.Connection:
    init_db(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def candidate_seed(*, seed_base: int, batch_id: int, index: int) -> int:
    return int(seed_base) + int(batch_id) * 1_000_000 + int(index)


@dataclass(frozen=True)
class CandidateMeta:
    experiment_id: int
    batch_id: int
    seed: int
    move_family: str
    parents: list[str]
    knobs: dict[str, int | float]
    created_at: str


@dataclass(frozen=True)
class EnqueueRecord:
    candidate_id: int
    design_hash: str
    seed: int
    move_family: str
    model_route: str
    parents: list[str]
    knobs: dict[str, int | float]
    novelty_score: float | None
    created_at: str


def enqueue_candidate(
    conn: sqlite3.Connection,
    *,
    experiment_id: int,
    problem: str,
    run_dir: Path,
    batch_id: int,
    boundary: Mapping[str, object],
    seed: int,
    move_family: str,
    parents: list[str],
    knobs: Mapping[str, int | float],
    novelty_score: float | None,
    operator_family: str,
    model_route: str,
) -> EnqueueRecord | None:
    design_hash = hash_payload(boundary)
    row = conn.execute(
        "SELECT 1 FROM candidates WHERE experiment_id = ? AND design_hash = ? LIMIT 1",
        (int(experiment_id), design_hash),
    ).fetchone()
    if row is not None:
        return None

    candidates_dir = run_dir / "candidates"
    batches_dir = run_dir / "batches"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    batches_dir.mkdir(parents=True, exist_ok=True)
    batch_log = batches_dir / f"batch_{int(batch_id):03}.jsonl"

    candidate_path = candidates_dir / f"{design_hash}.json"
    meta_path = candidates_dir / f"{design_hash}_meta.json"

    created_at = _utc_now_iso()
    candidate_path.write_text(json.dumps(boundary, indent=2))
    meta = CandidateMeta(
        experiment_id=int(experiment_id),
        batch_id=int(batch_id),
        seed=int(seed),
        move_family=str(move_family),
        parents=[str(parent) for parent in parents],
        knobs={str(k): v for k, v in knobs.items()},
        created_at=created_at,
    )
    meta_path.write_text(json.dumps(asdict(meta), indent=2))

    cursor = conn.execute(
        """
        INSERT INTO candidates
        (experiment_id, problem, params_json, seed, status, design_hash, lineage_parent_hashes_json, novelty_score, operator_family, model_route)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            int(experiment_id),
            str(problem),
            json.dumps(boundary, separators=(",", ":")),
            int(seed),
            "pending",
            design_hash,
            json.dumps([str(parent) for parent in parents]),
            float(novelty_score) if novelty_score is not None else None,
            str(operator_family),
            str(model_route),
        ),
    )
    candidate_id = cursor.lastrowid
    assert candidate_id is not None

    conn.execute(
        "INSERT INTO artifacts (experiment_id, path, kind) VALUES (?, ?, ?)",
        (int(experiment_id), str(candidate_path), "candidate_json"),
    )
    conn.execute(
        "INSERT INTO artifacts (experiment_id, path, kind) VALUES (?, ?, ?)",
        (int(experiment_id), str(meta_path), "candidate_meta"),
    )

    record = EnqueueRecord(
        candidate_id=int(candidate_id),
        design_hash=design_hash,
        seed=int(seed),
        move_family=str(move_family),
        model_route=str(model_route),
        parents=[str(parent) for parent in parents],
        knobs={str(k): v for k, v in knobs.items()},
        novelty_score=float(novelty_score) if novelty_score is not None else None,
        created_at=created_at,
    )
    with batch_log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(record)) + "\n")
    return record
