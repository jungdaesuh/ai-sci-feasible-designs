#!/usr/bin/env python
# ruff: noqa: E402
"""ALM-NGOpt style P1 optimization with multi-fidelity promotion.

KISS goal: get *any* candidate that:
1) converges under VMEC high-fidelity settings, and
2) satisfies P1 constraints within tolerance, and
3) has low max_elongation (high score).

This follows the ConStellaration baseline recipe:
- optimize with low-fidelity VMEC in the loop (fast / more robust)
- periodically promote/evaluate at higher fidelities
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import nevergrad as ng
import numpy as np
from constellaration.forward_model import ConstellarationSettings
from constellaration.geometry import surface_rz_fourier
from constellaration.initial_guess import generate_rotating_ellipse
from constellaration.mhd import vmec_settings as vmec_settings_module
from constellaration.optimization import augmented_lagrangian as al
from constellaration.utils import pytree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_scientist.forward_model import ForwardModelSettings, forward_model_batch


@dataclass(frozen=True)
class ConstraintTargets:
    aspect_ratio_max: float = 4.0
    triangularity_max: float = -0.5
    iota_min: float = 0.3


def _make_settings(*, vmec_fidelity: str) -> ForwardModelSettings:
    const_settings = ConstellarationSettings(
        vmec_preset_settings=vmec_settings_module.VmecPresetSettings(
            fidelity=vmec_fidelity,
        ),
        boozer_preset_settings=None,
        qi_settings=None,
        turbulent_settings=None,
    )
    return ForwardModelSettings(
        constellaration_settings=const_settings,
        problem="p1",
        stage=vmec_fidelity,
        fidelity=vmec_fidelity,
    )


def _normalize_score(max_elongation: float) -> float:
    # Matches constellaration.problems.GeometricalProblem._score scaling.
    if not math.isfinite(max_elongation):
        return 0.0
    normalized = (max_elongation - 1.0) / (10.0 - 1.0)
    normalized = float(np.clip(normalized, 0.0, 1.0))
    return 1.0 - normalized


def _normalized_constraint_violations(
    metrics: object,
    *,
    targets: ConstraintTargets,
    normalize_targets: ConstraintTargets,
) -> np.ndarray:
    ar = float(getattr(metrics, "aspect_ratio", float("inf")))
    tri = float(getattr(metrics, "average_triangularity", float("inf")))
    iota = float(
        getattr(
            metrics, "edge_rotational_transform_over_n_field_periods", float("-inf")
        )
    )
    tri_denom = max(abs(normalize_targets.triangularity_max), 1e-6)
    iota_denom = max(abs(normalize_targets.iota_min), 1e-6)
    return np.array(
        [
            (ar - targets.aspect_ratio_max) / abs(normalize_targets.aspect_ratio_max),
            (tri - targets.triangularity_max) / tri_denom,
            (targets.iota_min - iota) / iota_denom,
        ],
        dtype=float,
    )


def _feasibility_inf_from_metrics_map(
    metrics_map: dict,
    *,
    targets: ConstraintTargets,
) -> float:
    ar = float(metrics_map.get("aspect_ratio", float("inf")))
    tri = float(metrics_map.get("average_triangularity", float("inf")))
    iota = float(
        metrics_map.get("edge_rotational_transform_over_n_field_periods", float("-inf"))
    )
    tri_denom = max(abs(targets.triangularity_max), 1e-6)
    iota_denom = max(abs(targets.iota_min), 1e-6)
    violations = np.array(
        [
            (ar - targets.aspect_ratio_max) / abs(targets.aspect_ratio_max),
            (tri - targets.triangularity_max) / tri_denom,
            (targets.iota_min - iota) / iota_denom,
        ],
        dtype=float,
    )
    return float(np.max(np.maximum(violations, 0.0)))


def _is_p1_feasible(
    metrics: object,
    *,
    targets: ConstraintTargets,
    tol: float,
) -> bool:
    violations = _normalized_constraint_violations(
        metrics, targets=targets, normalize_targets=targets
    )
    return float(np.max(np.maximum(violations, 0.0))) <= tol


def _constraints_vector(
    metrics: object,
    *,
    targets: ConstraintTargets,
    normalize_targets: ConstraintTargets,
    feasibility_tolerance: float,
) -> np.ndarray:
    # Positive means violation. Normalized to match constellaration.problems.
    violations = _normalized_constraint_violations(
        metrics, targets=targets, normalize_targets=normalize_targets
    )
    return violations - float(feasibility_tolerance)


def _constraint_violation_inf(constraints: np.ndarray) -> float:
    return float(np.max(np.maximum(0.0, constraints)))


def _surface_to_boundary(surface: surface_rz_fourier.SurfaceRZFourier) -> dict:
    return {
        "r_cos": np.asarray(surface.r_cos, dtype=float).tolist(),
        "z_sin": np.asarray(surface.z_sin, dtype=float).tolist(),
        "n_field_periods": int(surface.n_field_periods),
        "n_periodicity": 1,
        "is_stellarator_symmetric": bool(surface.is_stellarator_symmetric),
    }


def _eval_boundary_once(boundary: dict, *, vmec_fidelity: str) -> dict:
    settings = _make_settings(vmec_fidelity=vmec_fidelity)
    result = forward_model_batch([boundary], settings, n_workers=1, pool_type="thread")[
        0
    ]
    metrics = result.metrics
    return {
        "error": result.error_message,
        "objective": float(result.objective),
        "feasibility": float(result.feasibility),
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


def _boundary_worker(
    queue: multiprocessing.Queue, boundary: dict, vmec_fidelity: str
) -> None:
    queue.put(_eval_boundary_once(boundary, vmec_fidelity=vmec_fidelity))


def _eval_boundary_with_timeout(
    boundary: dict,
    *,
    vmec_fidelity: str,
    timeout_sec: float,
) -> dict:
    ctx = multiprocessing.get_context("spawn")
    queue: multiprocessing.Queue = ctx.Queue()
    process = ctx.Process(
        target=_boundary_worker,
        args=(queue, boundary, vmec_fidelity),
    )
    process.start()
    process.join(timeout_sec)
    if process.is_alive():
        process.terminate()
        process.join()
        return {"error": f"timeout_after_{timeout_sec}s"}
    if queue.empty():
        return {"error": "no_result"}
    return queue.get()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ALM-NGOpt style optimization for P1 with multi-fidelity promotion."
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--nfp", type=int, default=3)
    parser.add_argument(
        "--init-boundary",
        type=Path,
        default=None,
        help="Optional JSON boundary file to use as the initial guess.",
    )
    parser.add_argument("--init-aspect-ratio", type=float, default=4.0)
    parser.add_argument("--init-elongation", type=float, default=3.0)
    parser.add_argument(
        "--init-rot-transform",
        type=float,
        default=None,
        help="Total rotational transform used by rotating-ellipse generator. "
        "Defaults to 0.3*nfp.",
    )
    parser.add_argument(
        "--init-tri-scale",
        type=float,
        default=0.0,
        help="If >0, set r_cos(2,0) = -tri_scale * minor_radius on the initial guess.",
    )
    parser.add_argument("--max-poloidal", type=int, default=4)
    parser.add_argument("--max-toroidal", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--outer-iters", type=int, default=10)
    parser.add_argument("--budget-initial", type=int, default=120)
    parser.add_argument("--budget-increment", type=int, default=80)
    parser.add_argument("--budget-max", type=int, default=1200)
    parser.add_argument("--bounds-initial", type=float, default=0.5)
    parser.add_argument("--bounds-reduction", type=float, default=0.9)
    parser.add_argument("--bounds-min", type=float, default=0.05)
    parser.add_argument("--penalty-initial", type=float, default=10.0)
    parser.add_argument("--penalty-increase", type=float, default=5.0)
    parser.add_argument("--penalty-max", type=float, default=1e8)
    parser.add_argument("--constraint-tol-factor", type=float, default=0.8)
    parser.add_argument("--triangularity-start", type=float, default=0.0)
    parser.add_argument("--triangularity-end", type=float, default=-0.5)
    parser.add_argument("--iota-start", type=float, default=0.25)
    parser.add_argument("--iota-end", type=float, default=0.3)
    parser.add_argument("--feas-tol", type=float, default=1e-2)
    parser.add_argument("--promote-every", type=int, default=1)
    parser.add_argument("--promote-budget", type=int, default=2)
    parser.add_argument("--promote-feasibility-max", type=float, default=0.05)
    parser.add_argument("--promote-objective-max", type=float, default=3.5)
    parser.add_argument("--promote-timeout-from-boundary", type=float, default=120.0)
    parser.add_argument("--promote-timeout-high", type=float, default=240.0)
    parser.add_argument(
        "--stop-score",
        type=float,
        default=None,
        help="If set, stop early once a promoted high-fidelity score reaches this value.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/p1/alm_ngopt"),
    )
    args = parser.parse_args()

    np.random.seed(args.seed)  # noqa: NPY002

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.jsonl"

    final_targets = ConstraintTargets()

    settings_low = _make_settings(vmec_fidelity="low_fidelity")

    # Register SurfaceRZFourier for pytree flattening.
    pytree.register_pydantic_data(
        surface_rz_fourier.SurfaceRZFourier,
        meta_fields=["n_field_periods", "is_stellarator_symmetric"],
    )

    # Initial guess: either a provided boundary JSON, or a rotating ellipse seed.
    if args.init_boundary is not None:
        raw = json.loads(args.init_boundary.read_text())
        if isinstance(raw, str):
            raw = json.loads(raw)
        if not isinstance(raw, dict) or "r_cos" not in raw or "z_sin" not in raw:
            raise SystemExit(f"Invalid boundary file: {args.init_boundary}")
        boundary = {
            "r_cos": raw["r_cos"],
            "z_sin": raw["z_sin"],
            "n_field_periods": raw.get("n_field_periods", args.nfp),
            "is_stellarator_symmetric": raw.get("is_stellarator_symmetric", True),
        }
        if raw.get("r_sin") is not None:
            boundary["r_sin"] = raw["r_sin"]
        if raw.get("z_cos") is not None:
            boundary["z_cos"] = raw["z_cos"]
        init_surface = surface_rz_fourier.SurfaceRZFourier.model_validate(boundary)
        init_surface = init_surface.model_copy(
            update={"n_field_periods": int(args.nfp), "is_stellarator_symmetric": True}
        )
    else:
        init_rot = args.init_rot_transform
        if init_rot is None:
            init_rot = 0.3 * args.nfp
        init_surface = generate_rotating_ellipse(
            aspect_ratio=args.init_aspect_ratio,
            elongation=args.init_elongation,
            rotational_transform=float(init_rot),
            n_field_periods=args.nfp,
        )
    init_surface = surface_rz_fourier.set_max_mode_numbers(
        init_surface,
        max_poloidal_mode=args.max_poloidal,
        max_toroidal_mode=args.max_toroidal,
    )
    if args.init_boundary is None and args.init_tri_scale > 0.0:
        r_cos = np.asarray(init_surface.r_cos, dtype=float)
        z_sin = np.asarray(init_surface.z_sin, dtype=float)
        center = r_cos.shape[1] // 2
        minor = float(r_cos[1, center])
        if r_cos.shape[0] > 2:
            r_cos[2, center] = -float(args.init_tri_scale) * minor
        if center > 0:
            r_cos[0, :center] = 0.0
            z_sin[0, : center + 1] = 0.0
        init_surface = init_surface.model_copy(update={"r_cos": r_cos, "z_sin": z_sin})
    mask = surface_rz_fourier.build_mask(
        init_surface,
        max_poloidal_mode=args.max_poloidal,
        max_toroidal_mode=args.max_toroidal,
    )
    initial_guess, unravel = pytree.mask_and_ravel(init_surface, mask)

    scale_grid = surface_rz_fourier.compute_infinity_norm_spectrum_scaling_fun(
        poloidal_modes=np.asarray(init_surface.poloidal_modes).flatten(),
        toroidal_modes=np.asarray(init_surface.toroidal_modes).flatten(),
        alpha=args.alpha,
    ).reshape(np.asarray(init_surface.poloidal_modes).shape)
    scale = np.concatenate(
        [
            np.asarray(scale_grid)[np.asarray(mask.r_cos)],
            np.asarray(scale_grid)[np.asarray(mask.z_sin)],
        ]
    )
    x0 = np.asarray(initial_guess, dtype=float) / scale

    def _eval_vector(
        x: np.ndarray,
        *,
        settings: ForwardModelSettings,
        targets: ConstraintTargets,
    ) -> tuple[float, np.ndarray, dict]:
        surface = unravel(jnp.asarray(x * scale))
        boundary = _surface_to_boundary(surface)
        result = forward_model_batch(
            [boundary], settings, n_workers=1, pool_type="thread"
        )[0]
        metrics = result.metrics
        if result.error_message is not None or not math.isfinite(result.objective):
            return (1e9, np.ones(3, dtype=float) * 1e9, {"error": result.error_message})

        objective = float(getattr(metrics, "max_elongation", float("inf")))
        constraints = _constraints_vector(
            metrics,
            targets=targets,
            normalize_targets=final_targets,
            feasibility_tolerance=args.feas_tol,
        )
        violations = _normalized_constraint_violations(
            metrics, targets=targets, normalize_targets=final_targets
        )
        feasibility_official = float(np.max(np.maximum(violations, 0.0)))
        record = {
            "objective": objective,
            "constraints": constraints.tolist(),
            "feasibility": feasibility_official,
            "constraint_violations": violations.tolist(),
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
                "max_elongation": objective,
            },
            "boundary": boundary,
        }
        return objective, constraints, record

    # Initialize state at x0 (low fidelity).
    init_targets = ConstraintTargets(
        aspect_ratio_max=final_targets.aspect_ratio_max,
        triangularity_max=float(args.triangularity_start),
        iota_min=float(args.iota_start),
    )
    obj0, cons0, rec0 = _eval_vector(x0, settings=settings_low, targets=init_targets)
    state = al.AugmentedLagrangianState(
        x=jnp.asarray(x0),
        multipliers=jnp.zeros_like(jnp.asarray(cons0)),
        penalty_parameters=jnp.ones_like(jnp.asarray(cons0)) * args.penalty_initial,
        objective=jnp.asarray(obj0),
        constraints=jnp.asarray(cons0),
        bounds=jnp.ones_like(jnp.asarray(x0)) * args.bounds_initial,
    )
    al_settings = al.AugmentedLagrangianSettings(
        constraint_violation_tolerance_reduction_factor=args.constraint_tol_factor,
        penalty_parameters_increase_factor=args.penalty_increase,
        bounds_reduction_factor=args.bounds_reduction,
        penalty_parameters_max=args.penalty_max,
        bounds_min=args.bounds_min,
    )

    best_low = {
        "objective": obj0,
        "feasibility": _constraint_violation_inf(cons0),
        "record": rec0,
        "x": x0.tolist(),
    }
    best_violation = {
        "objective": obj0,
        "feasibility": _constraint_violation_inf(cons0),
        "record": rec0,
        "x": x0.tolist(),
    }
    best_high = None

    budget = int(args.budget_initial)
    for outer in range(1, args.outer_iters + 1):
        denom = max(1, args.outer_iters - 1)
        frac = float((outer - 1) / denom)
        current_targets = ConstraintTargets(
            aspect_ratio_max=final_targets.aspect_ratio_max,
            triangularity_max=float(
                args.triangularity_start
                + frac * (args.triangularity_end - args.triangularity_start)
            ),
            iota_min=float(args.iota_start + frac * (args.iota_end - args.iota_start)),
        )
        # Trust-region bounds around current iterate.
        x_center = np.asarray(state.x, dtype=float)
        bounds = np.asarray(state.bounds, dtype=float)
        lower = x_center - bounds
        upper = x_center + bounds

        parametrization = ng.p.Array(init=x_center, lower=lower, upper=upper)
        optimizer = ng.optimizers.NGOpt(
            parametrization=parametrization,
            budget=budget,
            num_workers=1,
        )

        t0 = time.time()
        for inner in range(1, budget + 1):
            cand = optimizer.ask()
            x = np.asarray(cand.value, dtype=float)

            objective, constraints, record = _eval_vector(
                x, settings=settings_low, targets=current_targets
            )
            loss = float(
                al.augmented_lagrangian_function(
                    jnp.asarray(objective),
                    jnp.asarray(constraints),
                    state,
                )
            )
            optimizer.tell(cand, loss)

            viol = _constraint_violation_inf(constraints)
            if viol < best_violation["feasibility"]:
                best_violation = {
                    "objective": objective,
                    "feasibility": viol,
                    "record": record,
                    "x": x.tolist(),
                }

            is_feasible = viol <= 0.0
            if is_feasible and objective < best_low["objective"]:
                best_low = {
                    "objective": objective,
                    "feasibility": viol,
                    "record": record,
                    "x": x.tolist(),
                }

            log = {
                "outer": outer,
                "inner": inner,
                "budget": budget,
                "tri_target": current_targets.triangularity_max,
                "iota_target": current_targets.iota_min,
                "elapsed_sec": time.time() - t0,
                "loss": loss,
                "objective": objective,
                "constraint_violation_inf": viol,
                "constraints": constraints.tolist(),
                "error": record.get("error"),
            }
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(log) + "\n")

        rec = optimizer.provide_recommendation()
        x_rec = np.asarray(rec.value, dtype=float)
        obj_rec, cons_rec, record_rec = _eval_vector(
            x_rec, settings=settings_low, targets=current_targets
        )

        prev_violation = _constraint_violation_inf(np.asarray(state.constraints))
        new_violation = _constraint_violation_inf(cons_rec)
        new_bounds = np.asarray(state.bounds, dtype=float)
        if math.isfinite(new_violation) and new_violation < prev_violation:
            new_bounds = np.maximum(args.bounds_min, new_bounds * args.bounds_reduction)

        state = al.update_augmented_lagrangian_state(
            x=jnp.asarray(x_rec),
            objective=jnp.asarray(obj_rec),
            constraints=jnp.asarray(cons_rec),
            state=state,
            settings=al_settings,
        ).model_copy(update={"bounds": jnp.asarray(new_bounds)})

        budget = int(min(args.budget_max, budget + args.budget_increment))

        # Promotion: evaluate a handful of best low-fidelity points at higher fidelities.
        if args.promote_every > 0 and outer % args.promote_every == 0:
            candidates = [best_violation]
            if best_low["feasibility"] <= 0.0:
                candidates.append(best_low)
            if len(candidates) > args.promote_budget:
                candidates = candidates[: args.promote_budget]

            for cand in candidates:
                metrics_low = cand["record"].get("metrics", {})
                feasibility_final = _feasibility_inf_from_metrics_map(
                    metrics_low, targets=final_targets
                )

                objective_low = float(cand["record"].get("objective", float("inf")))
                if not math.isfinite(feasibility_final) or (
                    feasibility_final > args.promote_feasibility_max
                ):
                    continue
                if not math.isfinite(objective_low) or (
                    objective_low > args.promote_objective_max
                ):
                    continue
                x_cand = np.asarray(cand["x"], dtype=float)
                surface = unravel(jnp.asarray(x_cand * scale))
                boundary = _surface_to_boundary(surface)

                # Mid fidelity gate.
                mid = _eval_boundary_with_timeout(
                    boundary,
                    vmec_fidelity="from_boundary_resolution",
                    timeout_sec=args.promote_timeout_from_boundary,
                )
                if mid.get("error") is not None:
                    continue

                # High fidelity gate.
                hi = _eval_boundary_with_timeout(
                    boundary,
                    vmec_fidelity="high_fidelity",
                    timeout_sec=args.promote_timeout_high,
                )
                if hi.get("error") is not None:
                    continue

                metrics_hi = hi["metrics"]
                score = _normalize_score(float(metrics_hi["max_elongation"]))
                feasible_hi = _is_p1_feasible(
                    type("Metrics", (), metrics_hi),
                    targets=final_targets,
                    tol=args.feas_tol,
                )
                if not feasible_hi:
                    score = 0.0

                promoted = {
                    "score": score,
                    "feasible": feasible_hi,
                    "metrics": metrics_hi,
                    "high_fidelity_feasibility": float(
                        _feasibility_inf_from_metrics_map(
                            metrics_hi, targets=final_targets
                        )
                    ),
                    "boundary": boundary,
                    "x": cand["x"],
                }

                if best_high is None or promoted["score"] > best_high["score"]:
                    best_high = promoted
                    (output_dir / "best_high.json").write_text(
                        json.dumps(best_high, indent=2)
                    )
                    (output_dir / "best_high_boundary.json").write_text(
                        json.dumps(best_high["boundary"], indent=2)
                    )
                    if args.stop_score is not None and best_high["score"] >= float(
                        args.stop_score
                    ):
                        return

        (output_dir / "best_low.json").write_text(json.dumps(best_low, indent=2))
        best_low_boundary = best_low.get("record", {}).get("boundary")
        if best_low_boundary is not None:
            (output_dir / "best_low_boundary.json").write_text(
                json.dumps(best_low_boundary, indent=2)
            )
        (output_dir / "best_violation.json").write_text(
            json.dumps(best_violation, indent=2)
        )
        best_violation_boundary = best_violation.get("record", {}).get("boundary")
        if best_violation_boundary is not None:
            (output_dir / "best_violation_boundary.json").write_text(
                json.dumps(best_violation_boundary, indent=2)
            )


if __name__ == "__main__":
    main()
