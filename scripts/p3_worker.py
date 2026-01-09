#!/usr/bin/env python
# ruff: noqa: E402
"""P3 high-fidelity worker: claim → evaluate → persist.

Each worker:
- claims one `pending` candidate from the SQLite queue
- runs official high-fidelity VMEC++/Boozer/QI metrics
- writes immutable eval artifacts under <RUN_DIR>/eval/
- records results into SQLite (`metrics`, `pareto_archive` when feasible)

This script is designed to be safe to restart; it only claims `pending` work.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ai_scientist.memory.schema import init_db

from constellaration import forward_model, problems
from constellaration.geometry import surface_rz_fourier


_P3_CONSTRAINT_NAMES = ["iota", "log10_qi", "mirror", "flux", "vacuum"]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _set_thread_env_defaults() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def _connect(db_path: Path) -> sqlite3.Connection:
    init_db(db_path)
    conn = sqlite3.connect(str(db_path), timeout=60.0)
    conn.row_factory = sqlite3.Row
    return conn


def _claim_next(
    conn: sqlite3.Connection, *, experiment_id: int, worker_id: int
) -> sqlite3.Row | None:
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute(
            "SELECT id, params_json, seed, design_hash FROM candidates WHERE experiment_id = ? AND status = 'pending' ORDER BY id LIMIT 1",
            (experiment_id,),
        ).fetchone()
        if row is None:
            conn.execute("COMMIT")
            return None
        new_status = f"running:{worker_id}:{now}"
        cur = conn.execute(
            "UPDATE candidates SET status = ? WHERE id = ? AND status = 'pending'",
            (new_status, int(row["id"])),
        )
        if cur.rowcount != 1:
            conn.execute("COMMIT")
            return None
        conn.execute("COMMIT")
        return row
    except Exception:
        conn.execute("ROLLBACK")
        raise


def _update_status(conn: sqlite3.Connection, *, candidate_id: int, status: str) -> None:
    conn.execute(
        "UPDATE candidates SET status = ? WHERE id = ?",
        (status, int(candidate_id)),
    )


def _insert_artifact(
    conn: sqlite3.Connection, *, experiment_id: int, path: Path, kind: str
) -> None:
    conn.execute(
        "INSERT INTO artifacts (experiment_id, path, kind) VALUES (?, ?, ?)",
        (experiment_id, str(path), kind),
    )


def _insert_metrics(
    conn: sqlite3.Connection,
    *,
    candidate_id: int,
    metrics_payload: dict,
    feasibility: float,
    objective: float | None,
    is_feasible: bool,
) -> int:
    cursor = conn.execute(
        "INSERT INTO metrics (candidate_id, raw_json, feasibility, objective, hv, is_feasible) VALUES (?, ?, ?, ?, ?, ?)",
        (
            int(candidate_id),
            json.dumps(metrics_payload, separators=(",", ":")),
            float(feasibility),
            float(objective) if objective is not None else None,
            None,
            1 if is_feasible else 0,
        ),
    )
    metrics_id = cursor.lastrowid
    assert metrics_id is not None
    return int(metrics_id)


def _insert_pareto_archive(
    conn: sqlite3.Connection,
    *,
    experiment_id: int,
    cycle: int,
    design_hash: str,
    fidelity: str,
    gradient: float,
    aspect: float,
    metrics_id: int,
    git_sha: str,
    constellaration_sha: str,
    settings_json: str,
    seed: int,
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO pareto_archive
        (experiment_id, cycle, design_hash, fidelity, gradient, aspect, metrics_id, git_sha, constellaration_sha, settings_json, seed)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            experiment_id,
            int(cycle),
            design_hash,
            fidelity,
            float(gradient),
            float(aspect),
            int(metrics_id),
            git_sha,
            constellaration_sha,
            settings_json,
            int(seed),
        ),
    )


def _load_meta(run_dir: Path, design_hash: str) -> dict | None:
    meta_path = run_dir / "candidates" / f"{design_hash}_meta.json"
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text())


@dataclass(frozen=True)
class EvalSummary:
    feasibility: float
    is_feasible: bool
    objective: float
    aspect: float


def _evaluate_p3(boundary: dict) -> tuple[dict, EvalSummary]:
    surface = surface_rz_fourier.SurfaceRZFourier.model_validate(boundary)
    settings = forward_model.ConstellarationSettings.default_high_fidelity()
    metrics, _equilibrium = forward_model.forward_model(surface, settings=settings)

    problem = problems.MHDStableQIStellarator()
    feasibility = problem.compute_feasibility(metrics)
    is_feasible = problem.is_feasible(metrics)
    violations_vec = problem._normalized_constraint_violations(metrics)
    violations = {
        name: float(val)
        for name, val in zip(_P3_CONSTRAINT_NAMES, violations_vec.tolist())
    }

    qi = metrics.qi
    log10_qi = math.log10(float(qi)) if qi is not None and qi > 0 else None

    metrics_payload = {
        "metrics": {
            "aspect_ratio": float(metrics.aspect_ratio),
            "lgradB": float(metrics.minimum_normalized_magnetic_gradient_scale_length),
            "iota_edge": float(metrics.edge_rotational_transform_over_n_field_periods),
            "qi": float(qi) if qi is not None else None,
            "log10_qi": float(log10_qi) if log10_qi is not None else None,
            "mirror": float(metrics.edge_magnetic_mirror_ratio),
            "flux_compression": float(
                metrics.flux_compression_in_regions_of_bad_curvature
            )
            if metrics.flux_compression_in_regions_of_bad_curvature is not None
            else None,
            "vacuum_well": float(metrics.vacuum_well),
            "max_elongation": float(metrics.max_elongation),
            "average_triangularity": float(metrics.average_triangularity),
        },
        "violations": violations,
        "feasibility": float(feasibility),
        "is_feasible": bool(is_feasible),
    }

    summary = EvalSummary(
        feasibility=float(feasibility),
        is_feasible=bool(is_feasible),
        objective=float(metrics.minimum_normalized_magnetic_gradient_scale_length),
        aspect=float(metrics.aspect_ratio),
    )
    return metrics_payload, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="P3 high-fidelity worker.")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("reports/p3_world_model.sqlite"),
        help="SQLite DB path for P3 runs.",
    )
    parser.add_argument("--experiment-id", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument(
        "--sleep-sec",
        type=float,
        default=2.0,
        help="Sleep interval when no pending candidates exist.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, stop after this many evaluations.",
    )
    args = parser.parse_args()

    _set_thread_env_defaults()

    conn = _connect(args.db)
    exp = conn.execute(
        "SELECT git_sha, constellaration_sha FROM experiments WHERE id = ?",
        (int(args.experiment_id),),
    ).fetchone()
    if exp is None:
        raise SystemExit(f"experiment_id {args.experiment_id} not found in DB")
    git_sha = str(exp["git_sha"])
    constellaration_sha = str(exp["constellaration_sha"])

    settings_json = json.dumps(
        {
            "vmec_fidelity": "high_fidelity",
            "boozer": True,
            "qi": True,
            "turbulence": True,
        },
        separators=(",", ":"),
    )

    done = 0
    try:
        while True:
            if args.limit > 0 and done >= args.limit:
                break

            row = _claim_next(
                conn,
                experiment_id=int(args.experiment_id),
                worker_id=int(args.worker_id),
            )
            if row is None:
                time.sleep(float(args.sleep_sec))
                continue

            candidate_id = int(row["id"])
            design_hash = str(row["design_hash"])
            boundary = json.loads(str(row["params_json"]))
            meta = _load_meta(args.run_dir, design_hash) or {}

            started_at = _utc_now_iso()
            t0 = time.monotonic()
            error: str | None = None
            eval_payload: dict
            summary: EvalSummary | None = None

            try:
                eval_payload, summary = _evaluate_p3(boundary)
            except Exception as exc:
                eval_payload = {}
                error = str(exc)

            wall_sec = time.monotonic() - t0
            finished_at = _utc_now_iso()

            record = {
                "design_hash": design_hash,
                "candidate_id": candidate_id,
                "experiment_id": int(args.experiment_id),
                "worker_id": int(args.worker_id),
                "started_at": started_at,
                "finished_at": finished_at,
                "wall_seconds": float(wall_sec),
                "meta": meta,
                "error": error,
                **eval_payload,
            }

            eval_path = args.run_dir / "eval" / f"{design_hash}.json"
            eval_path.write_text(json.dumps(record, indent=2))

            with conn:
                _insert_artifact(
                    conn,
                    experiment_id=int(args.experiment_id),
                    path=eval_path,
                    kind="eval_json",
                )

                if summary is None:
                    _insert_metrics(
                        conn,
                        candidate_id=candidate_id,
                        metrics_payload=record,
                        feasibility=float("inf"),
                        objective=None,
                        is_feasible=False,
                    )
                    _update_status(
                        conn,
                        candidate_id=candidate_id,
                        status=f"failed:{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}",
                    )
                else:
                    metrics_id = _insert_metrics(
                        conn,
                        candidate_id=candidate_id,
                        metrics_payload=record,
                        feasibility=summary.feasibility,
                        objective=summary.objective,
                        is_feasible=summary.is_feasible,
                    )
                    _update_status(
                        conn,
                        candidate_id=candidate_id,
                        status=f"done:{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}",
                    )

                    if summary.is_feasible:
                        cycle = int(meta.get("batch_id", 0))
                        seed = int(meta.get("seed", int(row["seed"])))
                        _insert_pareto_archive(
                            conn,
                            experiment_id=int(args.experiment_id),
                            cycle=cycle,
                            design_hash=design_hash,
                            fidelity="high_fidelity",
                            gradient=summary.objective,
                            aspect=summary.aspect,
                            metrics_id=metrics_id,
                            git_sha=git_sha,
                            constellaration_sha=constellaration_sha,
                            settings_json=settings_json,
                            seed=seed,
                        )

            done += 1

    finally:
        conn.close()


if __name__ == "__main__":
    main()
