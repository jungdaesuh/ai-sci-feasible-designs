#!/usr/bin/env python
# ruff: noqa: E402
"""Enqueue a P3 submission list (seed set) into the SQLite candidate queue.

Input format:
- JSON list of JSON-encoded strings (or objects), each representing a boundary.

This does not run physics. It inserts `pending` candidates and writes immutable
artifacts under:
  <RUN_DIR>/candidates/<design_hash>.json
  <RUN_DIR>/candidates/<design_hash>_meta.json
  <RUN_DIR>/batches/batch_<BATCH_ID>.jsonl
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ai_scientist.memory import hash_payload
from ai_scientist.memory.schema import init_db


@dataclass(frozen=True)
class CandidateMeta:
    experiment_id: int
    batch_id: int
    seed: int
    move_family: str
    parents: list[str]
    knobs: dict[str, int | float]
    created_at: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> object:
    payload = json.loads(path.read_text())
    if isinstance(payload, str):
        return json.loads(payload)
    return payload


def _extract_boundary(data: object) -> dict:
    if not isinstance(data, dict):
        raise TypeError("Boundary JSON must be an object.")
    if "r_cos" not in data or "z_sin" not in data:
        raise KeyError("Boundary JSON missing required keys: r_cos, z_sin.")
    boundary: dict = {
        "r_cos": data["r_cos"],
        "z_sin": data["z_sin"],
        "n_field_periods": int(data.get("n_field_periods", 3)),
        "is_stellarator_symmetric": bool(data.get("is_stellarator_symmetric", True)),
    }
    if data.get("r_sin") is not None:
        boundary["r_sin"] = data["r_sin"]
    if data.get("z_cos") is not None:
        boundary["z_cos"] = data["z_cos"]
    return boundary


def _connect(db_path: Path) -> sqlite3.Connection:
    init_db(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _already_present(
    conn: sqlite3.Connection, *, experiment_id: int, design_hash: str
) -> bool:
    row = conn.execute(
        "SELECT 1 FROM candidates WHERE experiment_id = ? AND design_hash = ? LIMIT 1",
        (experiment_id, design_hash),
    ).fetchone()
    return row is not None


def _insert_candidate(
    conn: sqlite3.Connection,
    *,
    experiment_id: int,
    boundary: dict,
    seed: int,
    design_hash: str,
) -> int:
    cursor = conn.execute(
        "INSERT INTO candidates (experiment_id, problem, params_json, seed, status, design_hash) VALUES (?, ?, ?, ?, ?, ?)",
        (
            experiment_id,
            "p3",
            json.dumps(boundary, separators=(",", ":")),
            int(seed),
            "pending",
            design_hash,
        ),
    )
    candidate_id = cursor.lastrowid
    assert candidate_id is not None
    return int(candidate_id)


def _insert_artifacts(
    conn: sqlite3.Connection,
    *,
    experiment_id: int,
    artifacts: Iterable[tuple[Path, str]],
) -> None:
    for path, kind in artifacts:
        conn.execute(
            "INSERT INTO artifacts (experiment_id, path, kind) VALUES (?, ?, ?)",
            (experiment_id, str(path), kind),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Enqueue a P3 seed submission list.")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("reports/p3_world_model.sqlite"),
        help="SQLite DB path for P3 runs.",
    )
    parser.add_argument("--experiment-id", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--batch-id", type=int, required=True)
    parser.add_argument(
        "--seed-base",
        type=int,
        default=0,
        help="Base seed for deterministic candidate seeds in this batch.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, enqueue at most this many items from the submission list.",
    )
    parser.add_argument("submission", type=Path, help="P3 submission JSON list.")
    args = parser.parse_args()

    candidates_dir = args.run_dir / "candidates"
    batches_dir = args.run_dir / "batches"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    batches_dir.mkdir(parents=True, exist_ok=True)
    batch_log = batches_dir / f"batch_{int(args.batch_id):03}.jsonl"

    raw = _load_json(args.submission)
    if not isinstance(raw, list):
        raise SystemExit("Expected a JSON list.")

    conn = _connect(args.db)
    inserted = 0
    skipped = 0
    limited = int(args.limit)
    try:
        with conn:
            for k, item in enumerate(raw):
                if limited > 0 and inserted >= limited:
                    break

                obj = json.loads(item) if isinstance(item, str) else item
                boundary = _extract_boundary(obj)
                design_hash = hash_payload(boundary)
                if _already_present(
                    conn, experiment_id=int(args.experiment_id), design_hash=design_hash
                ):
                    skipped += 1
                    continue

                seed = int(args.seed_base) + int(args.batch_id) * 1_000_000 + int(k)

                candidate_path = candidates_dir / f"{design_hash}.json"
                meta_path = candidates_dir / f"{design_hash}_meta.json"

                candidate_path.write_text(json.dumps(boundary, indent=2))
                meta = CandidateMeta(
                    experiment_id=int(args.experiment_id),
                    batch_id=int(args.batch_id),
                    seed=seed,
                    move_family="seed",
                    parents=[],
                    knobs={"index": int(k)},
                    created_at=_utc_now_iso(),
                )
                meta_path.write_text(json.dumps(asdict(meta), indent=2))

                candidate_id = _insert_candidate(
                    conn,
                    experiment_id=int(args.experiment_id),
                    boundary=boundary,
                    seed=seed,
                    design_hash=design_hash,
                )

                _insert_artifacts(
                    conn,
                    experiment_id=int(args.experiment_id),
                    artifacts=[
                        (candidate_path, "candidate_json"),
                        (meta_path, "candidate_meta"),
                    ],
                )

                record = {
                    "candidate_id": candidate_id,
                    "design_hash": design_hash,
                    "seed": seed,
                    "move_family": "seed",
                    "parents": [],
                    "knobs": {"index": int(k)},
                    "created_at": meta.created_at,
                }
                with batch_log.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")

                inserted += 1

    finally:
        conn.close()

    print(
        f"batch_id={int(args.batch_id)} family=seed inserted={inserted} skipped={skipped}"
    )


if __name__ == "__main__":
    main()
