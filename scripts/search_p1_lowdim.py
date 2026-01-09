#!/usr/bin/env python
"""Low-dimensional P1 search using rotating-ellipse parameters (no helical noise)."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from constellaration.geometry import surface_rz_fourier
from constellaration.initial_guess import generate_rotating_ellipse

from ai_scientist.forward_model import forward_model_batch
from ai_scientist.tools.evaluation import _settings_for_stage


@dataclass(frozen=True)
class Range:
    low: float
    high: float

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.low, self.high))


def _build_boundary(
    *,
    aspect_ratio: float,
    elongation: float,
    rotational_transform: float,
    tri_scale: float,
    nfp: int,
    max_poloidal: int,
    max_toroidal: int,
) -> dict:
    surface = generate_rotating_ellipse(
        aspect_ratio=aspect_ratio,
        elongation=elongation,
        rotational_transform=rotational_transform,
        n_field_periods=nfp,
    )
    surface = surface_rz_fourier.set_max_mode_numbers(
        surface,
        max_poloidal_mode=max_poloidal,
        max_toroidal_mode=max_toroidal,
    )

    r_cos = np.asarray(surface.r_cos, dtype=float)
    z_sin = np.asarray(surface.z_sin, dtype=float)
    center = r_cos.shape[1] // 2
    minor = float(r_cos[1, center])

    if r_cos.shape[0] > 2:
        r_cos[2, center] = -tri_scale * minor

    if center > 0:
        r_cos[0, :center] = 0.0
        z_sin[0, : center + 1] = 0.0

    return {
        "r_cos": r_cos.tolist(),
        "z_sin": z_sin.tolist(),
        "n_field_periods": nfp,
        "n_periodicity": 1,
        "is_stellarator_symmetric": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search P1 feasibility via rotating-ellipse parameters."
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-trials", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--target-count", type=int, default=1)
    parser.add_argument("--nfp", type=int, default=3)
    parser.add_argument("--max-poloidal", type=int, default=3)
    parser.add_argument("--max-toroidal", type=int, default=3)
    parser.add_argument("--aspect-ratio-low", type=float, default=3.0)
    parser.add_argument("--aspect-ratio-high", type=float, default=3.6)
    parser.add_argument("--elongation-low", type=float, default=1.4)
    parser.add_argument("--elongation-high", type=float, default=2.2)
    parser.add_argument("--rot-transform-low", type=float, default=1.5)
    parser.add_argument("--rot-transform-high", type=float, default=2.2)
    parser.add_argument("--tri-scale-low", type=float, default=0.55)
    parser.add_argument("--tri-scale-high", type=float, default=0.8)
    parser.add_argument("--feasibility-target", type=float, default=1e-3)
    parser.add_argument(
        "--stage",
        type=str,
        default="screen",
        help="Evaluation stage: screen (low fidelity) or promote (high fidelity).",
    )
    parser.add_argument("--prerelax", action="store_true")
    parser.add_argument("--prerelax-steps", type=int, default=50)
    parser.add_argument("--prerelax-lr", type=float, default=1e-2)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/p1/low_feasible/lowdim_search"),
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    ranges = {
        "aspect_ratio": Range(args.aspect_ratio_low, args.aspect_ratio_high),
        "elongation": Range(args.elongation_low, args.elongation_high),
        "rotational_transform": Range(args.rot_transform_low, args.rot_transform_high),
        "tri_scale": Range(args.tri_scale_low, args.tri_scale_high),
    }

    settings = _settings_for_stage(args.stage, "p1", skip_qi=True)
    if args.prerelax:
        settings = settings.model_copy(
            update={
                "prerelax": True,
                "prerelax_steps": args.prerelax_steps,
                "prerelax_lr": args.prerelax_lr,
            }
        )

    found: list[dict] = []
    trials = 0
    best = None

    while trials < args.max_trials and len(found) < args.target_count:
        batch = []
        params = []
        for _ in range(min(args.batch_size, args.max_trials - trials)):
            sample = {
                "aspect_ratio": ranges["aspect_ratio"].sample(rng),
                "elongation": ranges["elongation"].sample(rng),
                "rotational_transform": ranges["rotational_transform"].sample(rng),
                "tri_scale": ranges["tri_scale"].sample(rng),
            }
            params.append(sample)
            batch.append(
                _build_boundary(
                    aspect_ratio=sample["aspect_ratio"],
                    elongation=sample["elongation"],
                    rotational_transform=sample["rotational_transform"],
                    tri_scale=sample["tri_scale"],
                    nfp=args.nfp,
                    max_poloidal=args.max_poloidal,
                    max_toroidal=args.max_toroidal,
                )
            )
        trials += len(batch)

        results = forward_model_batch(batch, settings, n_workers=1, pool_type="thread")
        batch_feasible = 0

        for sample, boundary, result in zip(params, batch, results):
            feas = float(result.feasibility)
            if not math.isfinite(feas):
                continue

            if best is None or feas < best["feasibility"]:
                best = {
                    "feasibility": feas,
                    "metrics": {
                        "aspect_ratio": float(result.metrics.aspect_ratio),
                        "average_triangularity": float(
                            result.metrics.average_triangularity
                        ),
                        "edge_rotational_transform_over_n_field_periods": float(
                            result.metrics.edge_rotational_transform_over_n_field_periods
                        ),
                        "max_elongation": float(result.metrics.max_elongation),
                    },
                    "params": dict(sample),
                }

            if feas <= args.feasibility_target:
                batch_feasible += 1
                record = dict(boundary)
                record["params"] = dict(sample)
                record["metrics"] = {
                    "aspect_ratio": float(result.metrics.aspect_ratio),
                    "average_triangularity": float(
                        result.metrics.average_triangularity
                    ),
                    "edge_rotational_transform_over_n_field_periods": float(
                        result.metrics.edge_rotational_transform_over_n_field_periods
                    ),
                    "max_elongation": float(result.metrics.max_elongation),
                    "feasibility": feas,
                    "feasibility_target": float(args.feasibility_target),
                }
                found.append(record)
                if len(found) >= args.target_count:
                    break

        best_feas = None if best is None else best["feasibility"]
        best_msg = "None" if best_feas is None else f"{best_feas:.4f}"
        print(
            f"trials={trials} feasible_in_batch={batch_feasible} "
            f"best_feas={best_msg} "
            f"target={args.feasibility_target:.1e}"
        )

    if best is not None:
        (output_dir / "best_infeasible.json").write_text(json.dumps(best, indent=2))

    if not found:
        print("No feasible designs found in the current search window.")
        return

    for idx, record in enumerate(found, start=1):
        path = output_dir / f"p1_real_feasible_{idx:03}.json"
        path.write_text(json.dumps(record, indent=2))
    print(f"Wrote {len(found)} feasible designs to {output_dir}.")


if __name__ == "__main__":
    main()
