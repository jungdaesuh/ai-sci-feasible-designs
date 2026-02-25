#!/usr/bin/env python
# ruff: noqa: E402
"""Enqueue P3 candidates into the WorldModel SQLite queue (no physics).

This script generates candidates via a small set of structured move families and
inserts them into the `candidates` table with status `pending`.
Artifacts are written under:
  <RUN_DIR>/candidates/<design_hash>.json
  <RUN_DIR>/candidates/<design_hash>_meta.json
  <RUN_DIR>/batches/batch_<BATCH_ID>.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ai_scientist.memory import hash_payload
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


def _as_arrays(boundary: dict) -> tuple[np.ndarray, np.ndarray]:
    r_cos = np.asarray(boundary["r_cos"], dtype=float)
    z_sin = np.asarray(boundary["z_sin"], dtype=float)
    if r_cos.shape != z_sin.shape:
        raise ValueError(f"Shape mismatch: r_cos={r_cos.shape} z_sin={z_sin.shape}")
    return r_cos, z_sin


def _from_arrays(
    template: dict,
    *,
    r_cos: np.ndarray,
    z_sin: np.ndarray,
) -> dict:
    out = dict(template)
    out["r_cos"] = r_cos.tolist()
    out["z_sin"] = z_sin.tolist()
    return out


def _blend(a: dict, b: dict, t: float) -> dict:
    r_a, z_a = _as_arrays(a)
    r_b, z_b = _as_arrays(b)
    if r_a.shape != r_b.shape:
        raise ValueError(f"Shape mismatch: A={r_a.shape} B={r_b.shape}")
    r = (1.0 - t) * r_a + t * r_b
    z = (1.0 - t) * z_a + t * z_b
    return _from_arrays(a, r_cos=r, z_sin=z)


def _n_values(ntor: int) -> np.ndarray:
    return np.arange(-ntor, ntor + 1)


def _scale_groups(
    parent: dict,
    *,
    axisym_z: float | None,
    axisym_r: float | None,
    scale_abs_n: list[tuple[int, float]],
    scale_m_ge: list[tuple[int, float]],
) -> tuple[dict, dict[str, float]]:
    r, z = _as_arrays(parent)
    mpol, ncol = r.shape
    ntor = (ncol - 1) // 2
    nvals = _n_values(ntor)
    n0 = int(np.where(nvals == 0)[0][0])

    knobs: dict[str, float] = {}

    if axisym_z is not None:
        z = z.copy()
        z[1:, n0] *= float(axisym_z)
        knobs["axisym_z"] = float(axisym_z)

    if axisym_r is not None:
        r = r.copy()
        r[1:, n0] *= float(axisym_r)
        knobs["axisym_r"] = float(axisym_r)

    if scale_abs_n:
        r = r.copy()
        z = z.copy()
        for abs_n, factor in scale_abs_n:
            mask = np.abs(nvals) == int(abs_n)
            if not np.any(mask):
                continue
            r[:, mask] *= float(factor)
            z[:, mask] *= float(factor)
            knobs[f"abs_n_{int(abs_n)}"] = float(factor)

    if scale_m_ge:
        r = r.copy()
        z = z.copy()
        for m_min, factor in scale_m_ge:
            m_min_i = int(m_min)
            if m_min_i >= mpol:
                continue
            r[m_min_i:, :] *= float(factor)
            z[m_min_i:, :] *= float(factor)
            knobs[f"m_ge_{m_min_i}"] = float(factor)

    return _from_arrays(parent, r_cos=r, z_sin=z), knobs


def _iter_t_values(
    *,
    t_values: list[float] | None,
    t_min: float | None,
    t_max: float | None,
    t_step: float | None,
) -> list[float]:
    if t_values:
        return [float(t) for t in t_values]
    if t_min is None or t_max is None or t_step is None:
        raise ValueError("Provide either --t (repeatable) or --t-min/--t-max/--t-step.")
    out: list[float] = []
    t = float(t_min)
    t_max_f = float(t_max)
    step = float(t_step)
    if step <= 0:
        raise ValueError("--t-step must be > 0.")
    while t <= t_max_f + 1e-12:
        out.append(float(t))
        t += step
    return out


def _derive_novelty_score(
    *,
    knobs: Mapping[str, float],
    override: float | None,
) -> float:
    if override is not None:
        return float(override)
    if not knobs:
        return 0.0
    magnitudes: list[float] = []
    for key, value in knobs.items():
        v = float(value)
        if key == "t":
            magnitudes.append(abs(v))
        else:
            magnitudes.append(abs(v - 1.0))
    return float(max(magnitudes) if magnitudes else 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Enqueue P3 candidates.")
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
        "--family",
        choices=["blend", "scale_groups"],
        required=True,
        help="Move family used to generate candidates.",
    )
    parser.add_argument(
        "--model-route",
        type=str,
        default="governor_static_recipe",
        help="Route label recorded in candidate metadata for governance/reporting.",
    )
    parser.add_argument(
        "--novelty-score",
        type=float,
        default=None,
        help="Optional novelty score override for all candidates in this command.",
    )

    # blend args
    parser.add_argument("--parent-a", type=Path, default=None)
    parser.add_argument("--parent-b", type=Path, default=None)
    parser.add_argument("--t", type=float, action="append", default=None)
    parser.add_argument("--t-min", type=float, default=None)
    parser.add_argument("--t-max", type=float, default=None)
    parser.add_argument("--t-step", type=float, default=None)

    # scale_groups args
    parser.add_argument("--parent", type=Path, default=None)
    parser.add_argument("--axisym-z", type=float, default=None)
    parser.add_argument("--axisym-r", type=float, default=None)
    parser.add_argument(
        "--scale-abs-n",
        nargs=2,
        action="append",
        default=[],
        metavar=("ABS_N", "FACTOR"),
        help="Scale all |n|=ABS_N columns by FACTOR (applied to r_cos and z_sin).",
    )
    parser.add_argument(
        "--scale-m-ge",
        nargs=2,
        action="append",
        default=[],
        metavar=("M_MIN", "FACTOR"),
        help="Scale all rows m>=M_MIN by FACTOR (applied to r_cos and z_sin).",
    )

    args = parser.parse_args()

    conn = connect_queue_db(args.db)
    inserted = 0
    skipped = 0

    try:
        if args.family == "blend":
            if args.parent_a is None or args.parent_b is None:
                raise ValueError("--parent-a and --parent-b are required for blend.")
            a = _extract_boundary(_load_json(args.parent_a))
            b = _extract_boundary(_load_json(args.parent_b))
            t_vals = _iter_t_values(
                t_values=args.t,
                t_min=args.t_min,
                t_max=args.t_max,
                t_step=args.t_step,
            )
            proposals: list[tuple[dict, dict[str, float], list[str]]] = []
            for t in t_vals:
                boundary = _blend(a, b, t)
                knobs = {"t": float(t)}
                parents = [hash_payload(a), hash_payload(b)]
                proposals.append((boundary, knobs, parents))

        else:
            if args.parent is None:
                raise ValueError("--parent is required for scale_groups.")
            parent = _extract_boundary(_load_json(args.parent))
            scale_abs_n = [(int(n), float(f)) for n, f in args.scale_abs_n]
            scale_m_ge = [(int(m), float(f)) for m, f in args.scale_m_ge]
            boundary, knobs = _scale_groups(
                parent,
                axisym_z=args.axisym_z,
                axisym_r=args.axisym_r,
                scale_abs_n=scale_abs_n,
                scale_m_ge=scale_m_ge,
            )
            parents = [hash_payload(parent)]
            proposals = [(boundary, knobs, parents)]

        with conn:
            for k, (boundary, knobs, parents) in enumerate(proposals):
                seed = candidate_seed(
                    seed_base=int(args.seed_base),
                    batch_id=int(args.batch_id),
                    index=int(k),
                )
                novelty_score = _derive_novelty_score(
                    knobs={key: float(value) for key, value in knobs.items()},
                    override=args.novelty_score,
                )
                record = enqueue_candidate(
                    conn,
                    experiment_id=int(args.experiment_id),
                    run_dir=args.run_dir,
                    batch_id=int(args.batch_id),
                    boundary=boundary,
                    seed=seed,
                    move_family=args.family,
                    parents=parents,
                    knobs={key: float(value) for key, value in knobs.items()},
                    novelty_score=novelty_score,
                    operator_family=args.family,
                    model_route=str(args.model_route),
                )
                if record is None:
                    skipped += 1
                    continue

                inserted += 1

    finally:
        conn.close()

    print(
        f"batch_id={int(args.batch_id)} family={args.family} inserted={inserted} skipped={skipped}"
    )


if __name__ == "__main__":
    main()
