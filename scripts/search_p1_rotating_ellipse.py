#!/usr/bin/env python
"""Search P1-feasible designs using rotating-ellipse parameter sampling."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from constellaration.geometry import surface_rz_fourier
from constellaration.initial_guess import generate_rotating_ellipse

from ai_scientist.forward_model import forward_model_batch
from ai_scientist.tools.evaluation import _settings_for_stage


@dataclass(frozen=True)
class ParamRange:
    low: float
    high: float

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.low, self.high))


def _make_boundary(
    *,
    aspect_ratio: float,
    elongation: float,
    rotational_transform: float,
    tri_scale: float,
    helical_scale: float,
    nfp: int,
    max_poloidal: int,
    max_toroidal: int,
    rng: np.random.Generator,
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

    for m in range(1, min(4, r_cos.shape[0])):
        if center + 1 < r_cos.shape[1]:
            amp_r = rng.normal(scale=helical_scale)
            amp_z = rng.normal(scale=helical_scale)
            r_cos[m, center + 1] = amp_r
            z_sin[m, center + 1] = amp_z
            r_cos[m, center - 1] = amp_r
            z_sin[m, center - 1] = amp_z

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
        description="Random search for P1-feasible rotating-ellipse designs."
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-trials", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--target-count", type=int, default=3)
    parser.add_argument("--nfp", type=int, default=3)
    parser.add_argument("--max-poloidal", type=int, default=4)
    parser.add_argument("--max-toroidal", type=int, default=4)
    parser.add_argument("--aspect-ratio-low", type=float, default=3.2)
    parser.add_argument("--aspect-ratio-high", type=float, default=3.8)
    parser.add_argument("--elongation-low", type=float, default=1.4)
    parser.add_argument("--elongation-high", type=float, default=1.9)
    parser.add_argument("--rot-transform-low", type=float, default=1.5)
    parser.add_argument("--rot-transform-high", type=float, default=1.9)
    parser.add_argument("--tri-scale-low", type=float, default=0.55)
    parser.add_argument("--tri-scale-high", type=float, default=0.7)
    parser.add_argument("--helical-scale-low", type=float, default=0.005)
    parser.add_argument("--helical-scale-high", type=float, default=0.02)
    parser.add_argument("--prerelax", action="store_true")
    parser.add_argument("--prerelax-steps", type=int, default=50)
    parser.add_argument("--prerelax-lr", type=float, default=1e-2)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/p1/low_feasible/real_search"),
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    aspect_range = ParamRange(args.aspect_ratio_low, args.aspect_ratio_high)
    elong_range = ParamRange(args.elongation_low, args.elongation_high)
    rot_range = ParamRange(args.rot_transform_low, args.rot_transform_high)
    tri_range = ParamRange(args.tri_scale_low, args.tri_scale_high)
    hel_range = ParamRange(args.helical_scale_low, args.helical_scale_high)

    settings = _settings_for_stage("screen", "p1", skip_qi=True)
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

    while trials < args.max_trials and len(found) < args.target_count:
        batch = []
        for _ in range(min(args.batch_size, args.max_trials - trials)):
            boundary = _make_boundary(
                aspect_ratio=aspect_range.sample(rng),
                elongation=elong_range.sample(rng),
                rotational_transform=rot_range.sample(rng),
                tri_scale=tri_range.sample(rng),
                helical_scale=hel_range.sample(rng),
                nfp=args.nfp,
                max_poloidal=args.max_poloidal,
                max_toroidal=args.max_toroidal,
                rng=rng,
            )
            batch.append(boundary)
        trials += len(batch)

        results = forward_model_batch(batch, settings, n_workers=1, pool_type="thread")
        for candidate, result in zip(batch, results):
            if result.feasibility <= 1e-2:
                record = dict(candidate)
                record["metrics"] = {
                    "aspect_ratio": float(result.metrics.aspect_ratio),
                    "average_triangularity": float(
                        result.metrics.average_triangularity
                    ),
                    "edge_rotational_transform_over_n_field_periods": float(
                        result.metrics.edge_rotational_transform_over_n_field_periods
                    ),
                    "max_elongation": float(result.metrics.max_elongation),
                    "feasibility": float(result.feasibility),
                }
                found.append(record)
                if len(found) >= args.target_count:
                    break

        print(
            f"trials={trials} found={len(found)} "
            f"last_batch_feasible={sum(r.feasibility <= 1e-2 for r in results)}"
        )

    if not found:
        print("No feasible designs found in the current search window.")
        return

    for idx, record in enumerate(found, start=1):
        path = output_dir / f"p1_real_feasible_{idx:03}.json"
        path.write_text(json.dumps(record, indent=2))
    print(f"Wrote {len(found)} feasible designs to {output_dir}.")


if __name__ == "__main__":
    main()
