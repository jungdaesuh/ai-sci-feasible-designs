#!/usr/bin/env python
"""Continuation-based convergence funnel for P1 high-fidelity VMEC."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from constellaration.forward_model import ConstellarationSettings
from constellaration.geometry import surface_rz_fourier
from constellaration.initial_guess import generate_rotating_ellipse
from constellaration.mhd import vmec_settings as vmec_settings_module

from ai_scientist.forward_model import ForwardModelSettings, forward_model_batch


@dataclass(frozen=True)
class Params:
    aspect_ratio: float
    elongation: float
    rotational_transform: float
    tri_scale: float

    def to_array(self) -> np.ndarray:
        return np.array(
            [
                self.aspect_ratio,
                self.elongation,
                self.rotational_transform,
                self.tri_scale,
            ],
            dtype=float,
        )

    @staticmethod
    def from_array(values: np.ndarray) -> "Params":
        return Params(
            aspect_ratio=float(values[0]),
            elongation=float(values[1]),
            rotational_transform=float(values[2]),
            tri_scale=float(values[3]),
        )


def _build_boundary(
    params: Params,
    *,
    nfp: int,
    max_poloidal: int,
    max_toroidal: int,
) -> dict:
    surface = generate_rotating_ellipse(
        aspect_ratio=params.aspect_ratio,
        elongation=params.elongation,
        rotational_transform=params.rotational_transform,
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
        r_cos[2, center] = -params.tri_scale * minor
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


def _settings_for_fidelity(label: str) -> ForwardModelSettings:
    fidelity_map = {
        "very_low": "very_low_fidelity",
        "low": "low_fidelity",
        "from_boundary": "from_boundary_resolution",
        "high": "high_fidelity",
    }
    fidelity_key = fidelity_map[label]
    const_settings = ConstellarationSettings(
        vmec_preset_settings=vmec_settings_module.VmecPresetSettings(
            fidelity=fidelity_key,
        ),
        boozer_preset_settings=None,
        qi_settings=None,
        turbulent_settings=None,
    )
    return ForwardModelSettings(
        constellaration_settings=const_settings,
        problem="p1",
        stage=label,
        fidelity=label,
    )


def _load_params(path: Path) -> Params:
    data = json.loads(path.read_text())
    params = data.get("params")
    if params is None:
        raise ValueError(f"Missing params in {path}")
    return Params(
        aspect_ratio=float(params["aspect_ratio"]),
        elongation=float(params["elongation"]),
        rotational_transform=float(params["rotational_transform"]),
        tri_scale=float(params["tri_scale"]),
    )


def _evaluate(
    boundary: dict,
    settings: ForwardModelSettings,
    *,
    max_feasibility: float | None,
) -> tuple[dict, bool]:
    result = forward_model_batch([boundary], settings, n_workers=1, pool_type="thread")[
        0
    ]
    metrics = result.metrics
    feasible = (
        result.error_message is None
        and math.isfinite(result.feasibility)
        and math.isfinite(float(getattr(metrics, "aspect_ratio", float("inf"))))
        and math.isfinite(float(getattr(metrics, "max_elongation", float("inf"))))
    )
    if max_feasibility is not None:
        feasible = feasible and result.feasibility <= max_feasibility
    record = {
        "feasibility": float(result.feasibility),
        "error_message": result.error_message,
        "metrics": {
            "aspect_ratio": float(getattr(metrics, "aspect_ratio", float("inf"))),
            "average_triangularity": float(
                getattr(metrics, "average_triangularity", float("inf"))
            ),
            "edge_rotational_transform_over_n_field_periods": float(
                getattr(
                    metrics,
                    "edge_rotational_transform_over_n_field_periods",
                    float("inf"),
                )
            ),
            "max_elongation": float(getattr(metrics, "max_elongation", float("inf"))),
        },
    }
    return record, feasible


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Continuation-based convergence funnel for P1."
    )
    parser.add_argument("--seed-json", type=Path, default=None)
    parser.add_argument("--target-json", type=Path, default=None)
    parser.add_argument("--aspect-ratio", type=float, default=4.0)
    parser.add_argument("--elongation", type=float, default=3.0)
    parser.add_argument("--rot-transform", type=float, default=0.9)
    parser.add_argument("--tri-scale", type=float, default=0.1)
    parser.add_argument("--target-aspect-ratio", type=float, default=3.6)
    parser.add_argument("--target-elongation", type=float, default=1.5)
    parser.add_argument("--target-rot-transform", type=float, default=1.6)
    parser.add_argument("--target-tri-scale", type=float, default=0.6)
    parser.add_argument("--nfp", type=int, default=3)
    parser.add_argument("--max-poloidal", type=int, default=3)
    parser.add_argument("--max-toroidal", type=int, default=3)
    parser.add_argument("--stages", type=str, default="very_low,low,from_boundary,high")
    parser.add_argument("--max-feasibility", type=float, default=2.0)
    parser.add_argument("--step-initial", type=float, default=0.2)
    parser.add_argument("--step-min", type=float, default=0.02)
    parser.add_argument("--step-max", type=float, default=0.6)
    parser.add_argument("--step-up", type=float, default=1.25)
    parser.add_argument("--step-down", type=float, default=0.5)
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/p1/convergence_funnel"),
    )
    args = parser.parse_args()

    if args.seed_json is not None:
        current = _load_params(args.seed_json)
    else:
        current = Params(
            aspect_ratio=args.aspect_ratio,
            elongation=args.elongation,
            rotational_transform=args.rot_transform,
            tri_scale=args.tri_scale,
        )

    if args.target_json is not None:
        target = _load_params(args.target_json)
    else:
        target = Params(
            aspect_ratio=args.target_aspect_ratio,
            elongation=args.target_elongation,
            rotational_transform=args.target_rot_transform,
            tri_scale=args.target_tri_scale,
        )

    stage_labels = [item.strip() for item in args.stages.split(",") if item.strip()]
    stage_settings = {label: _settings_for_fidelity(label) for label in stage_labels}

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.jsonl"

    step = args.step_initial
    best = None

    for iteration in range(1, args.max_iter + 1):
        delta = target.to_array() - current.to_array()
        distance = float(np.linalg.norm(delta))
        if distance <= args.tolerance:
            break

        candidate = Params.from_array(current.to_array() + step * delta)
        boundary = _build_boundary(
            candidate,
            nfp=args.nfp,
            max_poloidal=args.max_poloidal,
            max_toroidal=args.max_toroidal,
        )

        stage_results = {}
        passed_all = True
        for label in stage_labels:
            record, ok = _evaluate(
                boundary,
                stage_settings[label],
                max_feasibility=args.max_feasibility,
            )
            stage_results[label] = record
            if not ok:
                passed_all = False
                break

        history = {
            "iteration": iteration,
            "step": step,
            "distance": distance,
            "params": {
                "aspect_ratio": candidate.aspect_ratio,
                "elongation": candidate.elongation,
                "rotational_transform": candidate.rotational_transform,
                "tri_scale": candidate.tri_scale,
            },
            "stage_results": stage_results,
            "accepted": passed_all,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(history) + "\n")

        if passed_all:
            current = candidate
            best = history
            step = min(args.step_max, step * args.step_up)
        else:
            step = max(args.step_min, step * args.step_down)

        if step <= args.step_min and not passed_all:
            break

    if best is not None:
        (output_dir / "best.json").write_text(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
