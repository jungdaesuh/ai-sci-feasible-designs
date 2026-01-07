#!/usr/bin/env python
"""Evaluate a single rotating-ellipse boundary for P1 with configurable fidelity."""

from __future__ import annotations

import argparse

from constellaration.geometry import surface_rz_fourier
from constellaration.initial_guess import generate_rotating_ellipse

from ai_scientist.forward_model import forward_model_batch
from ai_scientist.tools.evaluation import _settings_for_stage


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate one P1 candidate boundary via constellaration."
    )
    parser.add_argument("--aspect-ratio", type=float, default=3.5)
    parser.add_argument("--elongation", type=float, default=1.4)
    parser.add_argument("--rot-transform", type=float, default=1.6)
    parser.add_argument("--tri-scale", type=float, default=0.8)
    parser.add_argument("--nfp", type=int, default=3)
    parser.add_argument("--max-poloidal", type=int, default=3)
    parser.add_argument("--max-toroidal", type=int, default=3)
    parser.add_argument("--stage", type=str, default="promote")
    args = parser.parse_args()

    surface = generate_rotating_ellipse(
        aspect_ratio=args.aspect_ratio,
        elongation=args.elongation,
        rotational_transform=args.rot_transform,
        n_field_periods=args.nfp,
    )
    surface = surface_rz_fourier.set_max_mode_numbers(
        surface,
        max_poloidal_mode=args.max_poloidal,
        max_toroidal_mode=args.max_toroidal,
    )

    r_cos = surface.r_cos.copy()
    z_sin = surface.z_sin.copy()
    center = r_cos.shape[1] // 2
    minor = float(r_cos[1, center])
    if r_cos.shape[0] > 2:
        r_cos[2, center] = -args.tri_scale * minor
    if center > 0:
        r_cos[0, :center] = 0.0
        z_sin[0, : center + 1] = 0.0

    boundary = {
        "r_cos": r_cos.tolist(),
        "z_sin": z_sin.tolist(),
        "n_field_periods": args.nfp,
        "n_periodicity": 1,
        "is_stellarator_symmetric": True,
    }

    settings = _settings_for_stage(args.stage, "p1", skip_qi=True)
    result = forward_model_batch([boundary], settings, n_workers=1, pool_type="thread")[
        0
    ]
    metrics = result.metrics
    print(
        "feas=",
        result.feasibility,
        "ar=",
        getattr(metrics, "aspect_ratio", None),
        "tri=",
        getattr(metrics, "average_triangularity", None),
        "iota=",
        getattr(metrics, "edge_rotational_transform_over_n_field_periods", None),
        "elong=",
        getattr(metrics, "max_elongation", None),
        "error=",
        result.error_message,
    )


if __name__ == "__main__":
    main()
