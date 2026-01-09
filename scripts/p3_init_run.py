#!/usr/bin/env python
# ruff: noqa: E402
"""Initialize a persistent P3 run directory and WorldModel experiment row.

This script does not run physics. It creates:
- a timestamped run directory under artifacts/p3/<RUN_ID>/
- a SQLite DB (if missing) using the existing WorldModel schema
- an experiments row and a meta.json for reproducibility
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ai_scientist.memory.schema import init_db


@dataclass(frozen=True)
class RunMeta:
    run_id: str
    experiment_id: int
    created_at: str
    tag: str
    db_path: str
    run_dir: str
    workers: int
    git_sha: str
    constellaration_sha: str
    notes: str | None = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_sha(repo: Path) -> str:
    out = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    )
    return out.strip()


def _insert_experiment(
    *,
    db_path: Path,
    config_payload: dict,
    git_sha: str,
    constellaration_sha: str,
    notes: str | None,
) -> int:
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.execute(
            "INSERT INTO experiments (started_at, config_json, git_sha, constellaration_sha, notes) VALUES (?, ?, ?, ?, ?)",
            (
                _utc_now_iso(),
                json.dumps(config_payload, separators=(",", ":")),
                git_sha,
                constellaration_sha,
                notes,
            ),
        )
        experiment_id = cursor.lastrowid
        assert experiment_id is not None
        conn.commit()
        return int(experiment_id)
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize a P3 run directory.")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("reports/p3_world_model.sqlite"),
        help="SQLite DB path for P3 runs.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("artifacts/p3"),
        help="Directory under which a timestamped RUN_ID dir is created.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="default",
        help="Short tag included in RUN_ID (e.g. mirror_repair).",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default=None,
        help="Optional free-form notes stored in the experiments row.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Planned number of high-fidelity workers (informational).",
    )
    args = parser.parse_args()

    init_db(args.db)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_id = f"{stamp}_p3_{args.tag}"
    run_dir = args.run_root / run_id
    (run_dir / "candidates").mkdir(parents=True, exist_ok=False)
    (run_dir / "eval").mkdir(parents=True, exist_ok=False)
    (run_dir / "batches").mkdir(parents=True, exist_ok=False)
    (run_dir / "submissions").mkdir(parents=True, exist_ok=False)

    repo_root = Path(__file__).resolve().parents[1]
    git_sha = _git_sha(repo_root)
    try:
        constellaration_sha = _git_sha(repo_root / "constellaration")
    except subprocess.CalledProcessError:
        constellaration_sha = "unknown"

    config_payload = {
        "problem": "p3",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "tag": args.tag,
        "workers": int(args.workers),
        "created_at": _utc_now_iso(),
    }
    experiment_id = _insert_experiment(
        db_path=args.db,
        config_payload=config_payload,
        git_sha=git_sha,
        constellaration_sha=constellaration_sha,
        notes=args.notes,
    )

    meta = RunMeta(
        run_id=run_id,
        experiment_id=experiment_id,
        created_at=_utc_now_iso(),
        tag=args.tag,
        db_path=str(args.db),
        run_dir=str(run_dir),
        workers=int(args.workers),
        git_sha=git_sha,
        constellaration_sha=constellaration_sha,
        notes=args.notes,
    )
    (run_dir / "meta.json").write_text(json.dumps(asdict(meta), indent=2))

    print(f"experiment_id={experiment_id}")
    print(f"run_dir={run_dir}")
    print(f"db={args.db}")


if __name__ == "__main__":
    main()
