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
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ai_scientist.p3_enqueue import candidate_seed, connect_queue_db, enqueue_candidate


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
    parser.add_argument(
        "--model-route",
        type=str,
        default="submission_seedlist",
        help="Route label recorded in candidate metadata for governance/reporting.",
    )
    parser.add_argument("submission", type=Path, help="P3 submission JSON list.")
    args = parser.parse_args()

    raw = _load_json(args.submission)
    if not isinstance(raw, list):
        raise SystemExit("Expected a JSON list.")

    conn = connect_queue_db(args.db)
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
                seed = candidate_seed(
                    seed_base=int(args.seed_base),
                    batch_id=int(args.batch_id),
                    index=int(k),
                )
                record = enqueue_candidate(
                    conn,
                    experiment_id=int(args.experiment_id),
                    run_dir=args.run_dir,
                    batch_id=int(args.batch_id),
                    boundary=boundary,
                    seed=seed,
                    move_family="seed",
                    parents=[],
                    knobs={"index": int(k)},
                    novelty_score=None,
                    operator_family="seed",
                    model_route=str(args.model_route),
                )
                if record is None:
                    skipped += 1
                    continue

                inserted += 1

    finally:
        conn.close()

    print(
        f"batch_id={int(args.batch_id)} family=seed inserted={inserted} skipped={skipped}"
    )


if __name__ == "__main__":
    main()
