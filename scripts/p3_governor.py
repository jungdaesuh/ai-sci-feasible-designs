#!/usr/bin/env python
# ruff: noqa: E402
"""P3 Governor: decide what to propose next based on DB progress.

This is the "intervention" layer:
- reads recent outcomes from the WorldModel SQLite DB
- identifies the highest-leverage near-feasible candidate and its worst constraint
- proposes the next batch by calling `scripts/p3_propose.py`

By default it prints the commands it would run. Use --execute to actually enqueue.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
from pymoo.indicators import hv

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ai_scientist.memory.schema import init_db


_DEFAULT_RECORD_HV = 135.15669906272515


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def _connect(db_path: Path) -> sqlite3.Connection:
    init_db(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _status_bucket(status: str) -> str:
    return status.split(":", 1)[0] if status else "unknown"


def _compute_hv(points: list[tuple[float, float]]) -> float:
    if not points:
        return 0.0
    X = np.array([(-lgradb, aspect) for lgradb, aspect in points], dtype=float)
    indicator = hv.Hypervolume(ref_point=np.array([1.0, 20.0], dtype=float))
    out = indicator(X)
    assert out is not None
    return float(out)


def _load_boundary(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise TypeError("Boundary JSON must be an object.")
    return payload


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _next_batch_id(run_dir: Path) -> int:
    candidates_dir = run_dir / "candidates"
    max_batch = 0
    for meta_path in candidates_dir.glob("*_meta.json"):
        meta = json.loads(meta_path.read_text())
        if not isinstance(meta, dict):
            continue
        batch_id = meta.get("batch_id")
        if isinstance(batch_id, int) and batch_id > max_batch:
            max_batch = batch_id
    return max_batch + 1


@dataclass(frozen=True)
class CandidateRow:
    candidate_id: int
    design_hash: str
    seed: int
    feasibility: float
    is_feasible: bool
    lgradb: float | None
    aspect: float | None
    violations: dict[str, float]
    metrics: dict
    meta: dict


def _fetch_candidates(
    conn: sqlite3.Connection, *, experiment_id: int, limit: int
) -> list[CandidateRow]:
    rows = conn.execute(
        """
        SELECT c.id AS candidate_id, c.design_hash AS design_hash, c.seed AS seed,
               m.feasibility AS feasibility, m.is_feasible AS is_feasible,
               m.objective AS objective, m.raw_json AS raw_json
        FROM metrics m
        JOIN candidates c ON c.id = m.candidate_id
        WHERE c.experiment_id = ?
        ORDER BY m.id DESC
        LIMIT ?
        """,
        (experiment_id, int(limit)),
    ).fetchall()

    out: list[CandidateRow] = []
    for row in rows:
        payload = json.loads(str(row["raw_json"]))
        metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
        violations = payload.get("violations", {}) if isinstance(payload, dict) else {}
        meta = payload.get("meta", {}) if isinstance(payload, dict) else {}

        aspect = None
        if isinstance(metrics, dict) and "aspect_ratio" in metrics:
            try:
                aspect = float(metrics["aspect_ratio"])
            except (TypeError, ValueError):
                aspect = None

        lgradb = None
        if row["objective"] is not None:
            lgradb = float(row["objective"])
        elif isinstance(metrics, dict) and "lgradB" in metrics:
            try:
                lgradb = float(metrics["lgradB"])
            except (TypeError, ValueError):
                lgradb = None

        vio_clean: dict[str, float] = {}
        if isinstance(violations, dict):
            for k, v in violations.items():
                try:
                    vio_clean[str(k)] = float(v)
                except (TypeError, ValueError):
                    continue

        out.append(
            CandidateRow(
                candidate_id=int(row["candidate_id"]),
                design_hash=str(row["design_hash"]),
                seed=int(row["seed"]),
                feasibility=float(row["feasibility"]),
                is_feasible=bool(int(row["is_feasible"])),
                lgradb=lgradb,
                aspect=aspect,
                violations=vio_clean,
                metrics=metrics if isinstance(metrics, dict) else {},
                meta=meta if isinstance(meta, dict) else {},
            )
        )
    return out


def _worst_constraint(violations: dict[str, float]) -> tuple[str | None, float]:
    if not violations:
        return None, 0.0
    worst_name = None
    worst_val = -float("inf")
    for name, val in violations.items():
        if val > worst_val:
            worst_val = val
            worst_name = name
    return worst_name, float(worst_val)


def _potential_area(lgradb: float, aspect: float) -> float:
    return max(0.0, 1.0 + float(lgradb)) * max(0.0, 20.0 - float(aspect))


def _focus_score(row: CandidateRow) -> float:
    if row.lgradb is None or row.aspect is None:
        return -float("inf")
    return _potential_area(row.lgradb, row.aspect) / max(row.feasibility, 1e-3)


def _choose_focus(
    candidates: list[CandidateRow], *, max_feasibility: float
) -> CandidateRow | None:
    pool = [
        c
        for c in candidates
        if (not c.is_feasible)
        and math.isfinite(c.feasibility)
        and c.feasibility > 1e-2
        and c.feasibility <= max_feasibility
        and c.lgradb is not None
        and c.aspect is not None
    ]
    if not pool:
        return None
    return max(pool, key=_focus_score)


def _choose_partner(
    candidates: list[CandidateRow],
    *,
    worst_constraint: str,
) -> CandidateRow | None:
    feasible = [
        c
        for c in candidates
        if c.is_feasible and c.lgradb is not None and c.aspect is not None
    ]
    pool = (
        feasible
        if feasible
        else [c for c in candidates if c.lgradb is not None and c.aspect is not None]
    )
    if not pool:
        return None

    key_name = {
        "mirror": "mirror",
        "log10_qi": "log10_qi",
        "flux": "flux_compression",
        "vacuum": "vacuum_well",
        "iota": "iota_edge",
    }.get(worst_constraint, "")

    if not key_name:
        return max(pool, key=lambda c: float(c.lgradb or -1e9))

    def metric_value(c: CandidateRow) -> float:
        if key_name not in c.metrics:
            return float("inf") if worst_constraint != "vacuum" else -float("inf")
        try:
            return float(c.metrics[key_name])
        except (TypeError, ValueError):
            return float("inf") if worst_constraint != "vacuum" else -float("inf")

    if worst_constraint == "vacuum":
        return max(pool, key=metric_value)
    return min(pool, key=metric_value)


@dataclass(frozen=True)
class ProposalCommand:
    argv: list[str]


def _cmd_str(cmd: ProposalCommand) -> str:
    return " ".join(cmd.argv)


def _build_blend_cmd(
    *,
    db: Path,
    experiment_id: int,
    run_dir: Path,
    batch_id: int,
    seed_base: int,
    parent_a: Path,
    parent_b: Path,
    t_min: float,
    t_max: float,
    t_step: float,
) -> ProposalCommand:
    return ProposalCommand(
        argv=[
            "python",
            "scripts/p3_propose.py",
            "--db",
            str(db),
            "--experiment-id",
            str(experiment_id),
            "--run-dir",
            str(run_dir),
            "--batch-id",
            str(batch_id),
            "--seed-base",
            str(seed_base),
            "--family",
            "blend",
            "--parent-a",
            str(parent_a),
            "--parent-b",
            str(parent_b),
            "--t-min",
            f"{t_min:.6f}",
            "--t-max",
            f"{t_max:.6f}",
            "--t-step",
            f"{t_step:.6f}",
        ]
    )


def _build_scale_cmd(
    *,
    db: Path,
    experiment_id: int,
    run_dir: Path,
    batch_id: int,
    seed_base: int,
    parent: Path,
    axisym_z: float | None = None,
    axisym_r: float | None = None,
    scale_abs_n: tuple[int, float] | None = None,
    scale_m_ge: tuple[int, float] | None = None,
) -> ProposalCommand:
    argv = [
        "python",
        "scripts/p3_propose.py",
        "--db",
        str(db),
        "--experiment-id",
        str(experiment_id),
        "--run-dir",
        str(run_dir),
        "--batch-id",
        str(batch_id),
        "--seed-base",
        str(seed_base),
        "--family",
        "scale_groups",
        "--parent",
        str(parent),
    ]
    if axisym_z is not None:
        argv += ["--axisym-z", f"{axisym_z:.6f}"]
    if axisym_r is not None:
        argv += ["--axisym-r", f"{axisym_r:.6f}"]
    if scale_abs_n is not None:
        abs_n, factor = scale_abs_n
        argv += ["--scale-abs-n", str(int(abs_n)), f"{float(factor):.6f}"]
    if scale_m_ge is not None:
        m_min, factor = scale_m_ge
        argv += ["--scale-m-ge", str(int(m_min)), f"{float(factor):.6f}"]
    return ProposalCommand(argv=argv)


def _ensure_parent_file(run_dir: Path, *, design_hash: str, boundary: dict) -> Path:
    path = run_dir / "candidates" / f"{design_hash}.json"
    if not path.exists():
        path.write_text(json.dumps(boundary, indent=2))
    return path


def _select_recipe(
    *,
    db: Path,
    experiment_id: int,
    run_dir: Path,
    batch_id: int,
    seed_base: int,
    focus: CandidateRow,
    partner: CandidateRow | None,
) -> tuple[list[ProposalCommand], dict]:
    # Ensure parent boundary JSONs exist (we can pull from DB if needed).
    conn = _connect(db)
    try:
        row = conn.execute(
            "SELECT params_json FROM candidates WHERE experiment_id = ? AND design_hash = ? LIMIT 1",
            (experiment_id, focus.design_hash),
        ).fetchone()
        if row is None:
            raise ValueError("Focus candidate missing from candidates table.")
        focus_boundary = json.loads(str(row["params_json"]))
    finally:
        conn.close()

    focus_path = _ensure_parent_file(
        run_dir, design_hash=focus.design_hash, boundary=focus_boundary
    )

    worst_name, worst_val = _worst_constraint(focus.violations)
    worst = str(worst_name) if worst_name is not None else ""

    cmds: list[ProposalCommand] = []

    # Scale sweep around the focus point (1D, ~4-6 candidates).
    if worst == "mirror":
        axisym_z_values = [
            0.95,
            0.96,
            0.965,
            0.97,
            0.975,
            0.98,
            0.985,
            0.99,
            0.995,
            1.0,
        ]
        for i, sz in enumerate(axisym_z_values):
            cmds.append(
                _build_scale_cmd(
                    db=db,
                    experiment_id=experiment_id,
                    run_dir=run_dir,
                    batch_id=batch_id,
                    seed_base=seed_base + 10 + i,
                    parent=focus_path,
                    axisym_z=sz,
                )
            )
        for j, factor in enumerate([0.85, 0.9, 0.95]):
            cmds.append(
                _build_scale_cmd(
                    db=db,
                    experiment_id=experiment_id,
                    run_dir=run_dir,
                    batch_id=batch_id,
                    seed_base=seed_base + 40 + j,
                    parent=focus_path,
                    scale_m_ge=(3, factor),
                )
            )
        combo_index = 0
        for sz in [0.96, 0.965, 0.97]:
            for factor in [0.9, 0.95]:
                cmds.append(
                    _build_scale_cmd(
                        db=db,
                        experiment_id=experiment_id,
                        run_dir=run_dir,
                        batch_id=batch_id,
                        seed_base=seed_base + 60 + combo_index,
                        parent=focus_path,
                        axisym_z=sz,
                        scale_m_ge=(3, factor),
                    )
                )
                combo_index += 1
    elif worst == "log10_qi":
        for i, factor in enumerate([0.85, 0.9, 0.95]):
            cmds.append(
                _build_scale_cmd(
                    db=db,
                    experiment_id=experiment_id,
                    run_dir=run_dir,
                    batch_id=batch_id,
                    seed_base=seed_base + 10 + i,
                    parent=focus_path,
                    scale_m_ge=(3, factor),
                )
            )
        for i, factor in enumerate([0.96, 0.98]):
            cmds.append(
                _build_scale_cmd(
                    db=db,
                    experiment_id=experiment_id,
                    run_dir=run_dir,
                    batch_id=batch_id,
                    seed_base=seed_base + 20 + i,
                    parent=focus_path,
                    scale_abs_n=(1, factor),
                )
            )
    elif worst == "flux":
        for i, factor in enumerate([0.7, 0.8, 0.9]):
            cmds.append(
                _build_scale_cmd(
                    db=db,
                    experiment_id=experiment_id,
                    run_dir=run_dir,
                    batch_id=batch_id,
                    seed_base=seed_base + 10 + i,
                    parent=focus_path,
                    scale_m_ge=(3, factor),
                )
            )
    elif worst == "vacuum":
        for i, factor in enumerate([0.9, 0.95, 0.98]):
            cmds.append(
                _build_scale_cmd(
                    db=db,
                    experiment_id=experiment_id,
                    run_dir=run_dir,
                    batch_id=batch_id,
                    seed_base=seed_base + 10 + i,
                    parent=focus_path,
                    scale_m_ge=(2, factor),
                )
            )
    elif worst == "iota":
        for i, factor in enumerate([1.02, 1.04, 1.06]):
            cmds.append(
                _build_scale_cmd(
                    db=db,
                    experiment_id=experiment_id,
                    run_dir=run_dir,
                    batch_id=batch_id,
                    seed_base=seed_base + 10 + i,
                    parent=focus_path,
                    scale_abs_n=(1, factor),
                )
            )

    # Blend sweep towards a partner that is better on the worst constraint.
    if partner is not None:
        conn2 = _connect(db)
        try:
            row2 = conn2.execute(
                "SELECT params_json FROM candidates WHERE experiment_id = ? AND design_hash = ? LIMIT 1",
                (experiment_id, partner.design_hash),
            ).fetchone()
            partner_boundary = (
                json.loads(str(row2["params_json"])) if row2 is not None else None
            )
        finally:
            conn2.close()

        if partner_boundary is not None:
            partner_path = _ensure_parent_file(
                run_dir, design_hash=partner.design_hash, boundary=partner_boundary
            )
            cmds.append(
                _build_blend_cmd(
                    db=db,
                    experiment_id=experiment_id,
                    run_dir=run_dir,
                    batch_id=batch_id,
                    seed_base=seed_base,
                    parent_a=focus_path,
                    parent_b=partner_path,
                    t_min=0.0,
                    t_max=0.1,
                    t_step=0.02,
                )
            )

    decision = {
        "batch_id": batch_id,
        "focus": {
            "design_hash": focus.design_hash,
            "candidate_id": focus.candidate_id,
            "feasibility": focus.feasibility,
            "aspect": focus.aspect,
            "lgradb": focus.lgradb,
            "worst_constraint": worst,
            "worst_violation": worst_val,
        },
        "partner": None
        if partner is None
        else {
            "design_hash": partner.design_hash,
            "candidate_id": partner.candidate_id,
            "aspect": partner.aspect,
            "lgradb": partner.lgradb,
        },
        "commands": [_cmd_str(c) for c in cmds],
        "created_at": _utc_now_iso(),
    }
    return cmds, decision


def _run_cmds(cmds: Iterable[ProposalCommand]) -> None:
    for cmd in cmds:
        subprocess.run(cmd.argv, check=True)


def _log_governor_artifact(
    conn: sqlite3.Connection, *, experiment_id: int, path: Path
) -> None:
    conn.execute(
        "INSERT INTO artifacts (experiment_id, path, kind) VALUES (?, ?, ?)",
        (experiment_id, str(path), "governor_decision"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="P3 proposal governor.")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("reports/p3_world_model.sqlite"),
        help="SQLite DB path for P3 runs.",
    )
    parser.add_argument("--experiment-id", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--queue-multiplier", type=int, default=2)
    parser.add_argument("--max-focus-feas", type=float, default=0.25)
    parser.add_argument("--recent-limit", type=int, default=500)
    parser.add_argument("--record-hv", type=float, default=_DEFAULT_RECORD_HV)
    parser.add_argument("--sleep-sec", type=float, default=15.0)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually enqueue proposals by invoking p3_propose.py.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Run continuously: keep queue filled and adjust batches over time.",
    )
    parser.add_argument(
        "--bootstrap-parent-a",
        type=Path,
        default=None,
        help="Optional bootstrap: parent A boundary JSON for an initial blend sweep.",
    )
    parser.add_argument(
        "--bootstrap-parent-b",
        type=Path,
        default=None,
        help="Optional bootstrap: parent B boundary JSON for an initial blend sweep.",
    )
    parser.add_argument("--bootstrap-t-min", type=float, default=0.85)
    parser.add_argument("--bootstrap-t-max", type=float, default=0.95)
    parser.add_argument("--bootstrap-t-step", type=float, default=0.005)
    args = parser.parse_args()

    target_queue = int(args.workers) * int(args.queue_multiplier)

    conn = _connect(args.db)
    try:
        while True:
            exp_id = int(args.experiment_id)
            pending = conn.execute(
                "SELECT COUNT(*) AS n FROM candidates WHERE experiment_id = ? AND status = 'pending'",
                (exp_id,),
            ).fetchone()
            pending_n = int(pending["n"]) if pending is not None else 0

            # Compute current feasible HV (best estimate from DB).
            feasible_rows = conn.execute(
                """
                SELECT m.objective AS lgradb, m.raw_json AS raw
                FROM metrics m
                JOIN candidates c ON m.candidate_id = c.id
                WHERE c.experiment_id = ? AND m.is_feasible = 1
                """,
                (exp_id,),
            ).fetchall()
            feasible_points: list[tuple[float, float]] = []
            for row in feasible_rows:
                payload = json.loads(str(row["raw"]))
                metrics = (
                    payload.get("metrics", {}) if isinstance(payload, dict) else {}
                )
                if not isinstance(metrics, dict):
                    continue
                if row["lgradb"] is None or "aspect_ratio" not in metrics:
                    continue
                feasible_points.append(
                    (float(row["lgradb"]), float(metrics["aspect_ratio"]))
                )
            hv_value = _compute_hv(feasible_points)

            print(
                f"[{_utc_now_iso()}] exp={exp_id} pending={pending_n}/{target_queue} hv={hv_value:.6f} record={float(args.record_hv):.6f}"
            )

            if pending_n >= target_queue:
                if not args.loop:
                    break
                time.sleep(float(args.sleep_sec))
                continue

            candidates = _fetch_candidates(
                conn, experiment_id=exp_id, limit=int(args.recent_limit)
            )
            if not candidates:
                if args.bootstrap_parent_a is None or args.bootstrap_parent_b is None:
                    print(
                        "No metrics yet and no bootstrap parents provided; nothing to propose."
                    )
                    if not args.loop:
                        break
                    time.sleep(float(args.sleep_sec))
                    continue

                batch_id = _next_batch_id(args.run_dir)
                seed_base = int(args.experiment_id) * 10_000_000 + int(batch_id) * 1_000
                cmd = _build_blend_cmd(
                    db=args.db,
                    experiment_id=exp_id,
                    run_dir=args.run_dir,
                    batch_id=batch_id,
                    seed_base=seed_base,
                    parent_a=args.bootstrap_parent_a,
                    parent_b=args.bootstrap_parent_b,
                    t_min=float(args.bootstrap_t_min),
                    t_max=float(args.bootstrap_t_max),
                    t_step=float(args.bootstrap_t_step),
                )
                decision = {
                    "batch_id": batch_id,
                    "bootstrap": True,
                    "commands": [_cmd_str(cmd)],
                    "created_at": _utc_now_iso(),
                }
                artifact_path = (
                    args.run_dir
                    / "governor"
                    / f"governor_bootstrap_batch_{batch_id:03}_{_utc_stamp()}.json"
                )
                _write_json(artifact_path, decision)
                with conn:
                    _log_governor_artifact(
                        conn, experiment_id=exp_id, path=artifact_path
                    )
                print(_cmd_str(cmd))
                if args.execute:
                    _run_cmds([cmd])
                if not args.loop:
                    break
                time.sleep(float(args.sleep_sec))
                continue

            focus = _choose_focus(
                candidates, max_feasibility=float(args.max_focus_feas)
            )
            if focus is None:
                print("No near-feasible focus candidate found in recent window.")
                if not args.loop:
                    break
                time.sleep(float(args.sleep_sec))
                continue

            worst_name, _worst_val = _worst_constraint(focus.violations)
            worst = str(worst_name) if worst_name is not None else ""
            partner = _choose_partner(candidates, worst_constraint=worst)

            batch_id = _next_batch_id(args.run_dir)
            seed_base = int(args.experiment_id) * 10_000_000 + int(batch_id) * 1_000
            cmds, decision = _select_recipe(
                db=args.db,
                experiment_id=exp_id,
                run_dir=args.run_dir,
                batch_id=batch_id,
                seed_base=seed_base,
                focus=focus,
                partner=partner,
            )

            decision["focus_meta"] = asdict(
                focus
            )  # traceability; includes violations/metrics/meta
            decision["partner_meta"] = None if partner is None else asdict(partner)
            decision["hv_at_decision"] = hv_value
            decision["record_hv"] = float(args.record_hv)

            artifact_path = (
                args.run_dir
                / "governor"
                / f"governor_batch_{batch_id:03}_{_utc_stamp()}.json"
            )
            _write_json(artifact_path, decision)
            with conn:
                _log_governor_artifact(conn, experiment_id=exp_id, path=artifact_path)

            for cmd in cmds:
                print(_cmd_str(cmd))
            if args.execute and cmds:
                _run_cmds(cmds)

            if not args.loop:
                break
            time.sleep(float(args.sleep_sec))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
