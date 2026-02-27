#!/usr/bin/env python
# ruff: noqa: E402
"""ALM-NGOpt style P2 optimization with multi-fidelity promotion.

Goal: find a boundary that is high-fidelity feasible for P2 and maximizes
minimum_normalized_magnetic_gradient_scale_length (score = clip(L_gradB / 20)).

Strategy (KISS):
- run low-fidelity VMEC in the loop (but compute QI at low VMEC resolution)
- periodically promote to high-fidelity to measure the real score
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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_scientist.forward_model import ForwardModelSettings, forward_model_batch
from ai_scientist.restart_runtime import (
    append_restart_history,
    select_adaptive_restart_runtime,
)


@dataclass(frozen=True)
class ConstraintTargets:
    aspect_ratio_max: float = 10.0
    iota_min: float = 0.25
    log10_qi_max: float = -4.0
    edge_mirror_max: float = 0.2
    max_elongation_max: float = 5.0


def _make_settings(*, vmec_fidelity: str) -> ForwardModelSettings:
    const_settings = ConstellarationSettings().model_copy(
        update={
            "vmec_preset_settings": vmec_settings_module.VmecPresetSettings(
                fidelity=vmec_fidelity
            ),
            "turbulent_settings": None,
        }
    )
    return ForwardModelSettings(
        constellaration_settings=const_settings,
        problem="p2",
        stage="p2",  # stage-gated QI in ai_scientist backend
        fidelity=vmec_fidelity,
    )


def _normalize_score(lgradb: float) -> float:
    if not math.isfinite(lgradb):
        return 0.0
    return float(np.clip(lgradb / 20.0, 0.0, 1.0))


def _log10_or_large(qi_value: float | None) -> float:
    if qi_value is None or not math.isfinite(qi_value) or qi_value <= 0.0:
        return 10.0
    return float(math.log10(qi_value))


def _normalized_constraint_violations(
    metrics: object,
    *,
    targets: ConstraintTargets,
) -> np.ndarray:
    ar = float(getattr(metrics, "aspect_ratio", float("inf")))
    iota = float(
        getattr(
            metrics, "edge_rotational_transform_over_n_field_periods", float("-inf")
        )
    )
    mirror = float(getattr(metrics, "edge_magnetic_mirror_ratio", float("inf")))
    elong = float(getattr(metrics, "max_elongation", float("inf")))
    log10_qi = _log10_or_large(getattr(metrics, "qi", None))

    return np.array(
        [
            (ar - targets.aspect_ratio_max) / abs(targets.aspect_ratio_max),
            (targets.iota_min - iota) / abs(targets.iota_min),
            (log10_qi - targets.log10_qi_max) / abs(targets.log10_qi_max),
            (mirror - targets.edge_mirror_max) / abs(targets.edge_mirror_max),
            (elong - targets.max_elongation_max) / abs(targets.max_elongation_max),
        ],
        dtype=float,
    )


def _feasibility_official(
    metrics: object,
    *,
    targets: ConstraintTargets,
) -> float:
    violations = _normalized_constraint_violations(metrics, targets=targets)
    return float(np.max(np.maximum(violations, 0.0)))


def _constraints_for_alm(
    metrics: object,
    *,
    targets: ConstraintTargets,
    feasibility_tolerance: float,
) -> np.ndarray:
    # ALM expects c(x) <= 0 for feasibility.
    violations = _normalized_constraint_violations(metrics, targets=targets)
    return violations - float(feasibility_tolerance)


def _constraint_violation_inf(constraints: np.ndarray) -> float:
    return float(np.max(np.maximum(0.0, constraints)))


def _restart_lgradb(record: dict) -> float:
    value = record.get("lgradb", float("-inf"))
    result = float(value)
    return result if math.isfinite(result) else float("-inf")


def _telemetry_lgradb(record: dict) -> float | None:
    if "lgradb" not in record:
        return None
    value = float(record["lgradb"])
    return value if math.isfinite(value) else None


def _surface_to_boundary(surface: surface_rz_fourier.SurfaceRZFourier) -> dict:
    boundary = {
        "r_cos": np.asarray(surface.r_cos, dtype=float).tolist(),
        "z_sin": np.asarray(surface.z_sin, dtype=float).tolist(),
        "n_field_periods": int(surface.n_field_periods),
        "n_periodicity": 1,
        "is_stellarator_symmetric": bool(surface.is_stellarator_symmetric),
    }
    if getattr(surface, "r_sin", None) is not None:
        boundary["r_sin"] = np.asarray(surface.r_sin, dtype=float).tolist()
    if getattr(surface, "z_cos", None) is not None:
        boundary["z_cos"] = np.asarray(surface.z_cos, dtype=float).tolist()
    return boundary


def _eval_boundary_once(boundary: dict, *, vmec_fidelity: str) -> dict:
    settings = _make_settings(vmec_fidelity=vmec_fidelity)
    result = forward_model_batch([boundary], settings, n_workers=1, pool_type="thread")[
        0
    ]
    metrics = result.metrics
    return {
        "error": result.error_message,
        "equilibrium_converged": bool(getattr(result, "equilibrium_converged", True)),
        "objective": float(result.objective),
        "feasibility": _feasibility_official(metrics, targets=ConstraintTargets()),
        "metrics": {
            "aspect_ratio": float(getattr(metrics, "aspect_ratio", float("inf"))),
            "edge_rotational_transform_over_n_field_periods": float(
                getattr(
                    metrics,
                    "edge_rotational_transform_over_n_field_periods",
                    float("-inf"),
                )
            ),
            "qi": getattr(metrics, "qi", None),
            "edge_magnetic_mirror_ratio": float(
                getattr(metrics, "edge_magnetic_mirror_ratio", float("inf"))
            ),
            "max_elongation": float(getattr(metrics, "max_elongation", float("inf"))),
            "minimum_normalized_magnetic_gradient_scale_length": float(
                getattr(
                    metrics,
                    "minimum_normalized_magnetic_gradient_scale_length",
                    float("-inf"),
                )
            ),
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
        description="Run ALM-NGOpt style optimization for P2 with multi-fidelity promotion."
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--nfp", type=int, default=3)
    parser.add_argument("--init-boundary", type=Path, default=None)
    parser.add_argument("--init-aspect-ratio", type=float, default=10.0)
    parser.add_argument("--init-elongation", type=float, default=3.0)
    parser.add_argument(
        "--init-rot-transform",
        type=float,
        default=None,
        help="Total rotational transform used by rotating-ellipse generator. Defaults to 0.25*nfp.",
    )
    parser.add_argument("--max-poloidal", type=int, default=None)
    parser.add_argument("--max-toroidal", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--outer-iters", type=int, default=12)
    parser.add_argument("--budget-initial", type=int, default=120)
    parser.add_argument("--budget-increment", type=int, default=80)
    parser.add_argument("--budget-max", type=int, default=1500)
    parser.add_argument("--bounds-initial", type=float, default=0.3)
    parser.add_argument("--bounds-reduction", type=float, default=0.9)
    parser.add_argument("--bounds-min", type=float, default=0.03)
    parser.add_argument("--penalty-initial", type=float, default=10.0)
    parser.add_argument("--penalty-increase", type=float, default=5.0)
    parser.add_argument("--penalty-max", type=float, default=1e8)
    parser.add_argument("--constraint-tol-factor", type=float, default=0.8)
    parser.add_argument(
        "--adaptive-restart",
        action="store_true",
        help="Enable adaptive restart seed selector across ALM outer iterations.",
    )
    parser.add_argument("--restart-feasibility-weight", type=float, default=0.45)
    parser.add_argument("--restart-objective-weight", type=float, default=0.45)
    parser.add_argument("--restart-diversity-weight", type=float, default=0.10)
    parser.add_argument("--restart-saturation-penalty", type=float, default=0.15)
    parser.add_argument(
        "--restart-novelty-min-distance",
        type=float,
        default=0.0,
        help=(
            "Minimum L2 distance from current state required for restart-seed "
            "novelty gating; falls back to ungated selection if no candidate passes."
        ),
    )
    parser.add_argument(
        "--restart-novelty-feasibility-max",
        type=float,
        default=float("inf"),
        help=(
            "Optional maximum feasibility violation allowed by restart novelty gate "
            "(infinity disables feasibility filtering)."
        ),
    )
    parser.add_argument(
        "--restart-novelty-near-duplicate-distance",
        type=float,
        default=0.08,
        help=(
            "Near-duplicate distance band for two-stage novelty gate LLM adjudication. "
            "Must be >= --restart-novelty-min-distance."
        ),
    )
    parser.add_argument(
        "--restart-novelty-judge-mode",
        choices=["disabled", "heuristic"],
        default="heuristic",
        help=(
            "Second-stage near-duplicate adjudication mode. "
            "Use 'heuristic' as deterministic fallback before provider-backed LLM judge."
        ),
    )
    parser.add_argument("--feas-tol", type=float, default=1e-2)
    parser.add_argument("--promote-every", type=int, default=2)
    parser.add_argument("--promote-budget", type=int, default=2)
    parser.add_argument("--promote-feasibility-max", type=float, default=0.02)
    parser.add_argument("--promote-timeout-high", type=float, default=600.0)
    parser.add_argument(
        "--stop-score",
        type=float,
        default=None,
        help="If set, stop early once a promoted high-fidelity score reaches this value.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/p2/alm_ngopt"),
    )
    args = parser.parse_args()

    np.random.seed(args.seed)  # noqa: NPY002

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.jsonl"

    final_targets = ConstraintTargets()

    settings_low = _make_settings(vmec_fidelity="low_fidelity")

    pytree.register_pydantic_data(
        surface_rz_fourier.SurfaceRZFourier,
        meta_fields=["n_field_periods", "is_stellarator_symmetric"],
    )

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
            init_rot = 0.25 * args.nfp
        init_surface = generate_rotating_ellipse(
            aspect_ratio=args.init_aspect_ratio,
            elongation=args.init_elongation,
            rotational_transform=float(init_rot),
            n_field_periods=args.nfp,
        )

    if args.max_poloidal is None:
        args.max_poloidal = int(init_surface.max_poloidal_mode)
    if args.max_toroidal is None:
        args.max_toroidal = int(init_surface.max_toroidal_mode)

    init_surface = surface_rz_fourier.set_max_mode_numbers(
        init_surface,
        max_poloidal_mode=int(args.max_poloidal),
        max_toroidal_mode=int(args.max_toroidal),
    )
    mask = surface_rz_fourier.build_mask(
        init_surface,
        max_poloidal_mode=int(args.max_poloidal),
        max_toroidal_mode=int(args.max_toroidal),
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
    ) -> tuple[float, np.ndarray, dict]:
        surface = unravel(jnp.asarray(x * scale))
        boundary = _surface_to_boundary(surface)
        result = forward_model_batch(
            [boundary], settings, n_workers=1, pool_type="thread"
        )[0]
        metrics = result.metrics

        if not bool(getattr(result, "equilibrium_converged", True)):
            return (
                -1e9,
                np.ones(5, dtype=float) * 1e9,
                {"error": result.error_message},
            )

        lgradb = float(
            getattr(
                metrics,
                "minimum_normalized_magnetic_gradient_scale_length",
                float("-inf"),
            )
        )
        if not math.isfinite(lgradb):
            return (
                -1e9,
                np.ones(5, dtype=float) * 1e9,
                {"error": result.error_message},
            )

        objective = -lgradb  # minimize negative => maximize L_gradB
        constraints = _constraints_for_alm(
            metrics, targets=final_targets, feasibility_tolerance=args.feas_tol
        )
        violations = _normalized_constraint_violations(metrics, targets=final_targets)
        feasibility = float(np.max(np.maximum(violations, 0.0)))

        record = {
            "lgradb": lgradb,
            "objective": objective,
            "constraints": constraints.tolist(),
            "constraint_violations": violations.tolist(),
            "feasibility": feasibility,
            "metrics": {
                "aspect_ratio": float(getattr(metrics, "aspect_ratio", float("inf"))),
                "edge_rotational_transform_over_n_field_periods": float(
                    getattr(
                        metrics,
                        "edge_rotational_transform_over_n_field_periods",
                        float("-inf"),
                    )
                ),
                "qi": getattr(metrics, "qi", None),
                "edge_magnetic_mirror_ratio": float(
                    getattr(metrics, "edge_magnetic_mirror_ratio", float("inf"))
                ),
                "max_elongation": float(
                    getattr(metrics, "max_elongation", float("inf"))
                ),
                "minimum_normalized_magnetic_gradient_scale_length": lgradb,
            },
            "boundary": boundary,
            "error": result.error_message,
        }
        return objective, constraints, record

    obj0, cons0, rec0 = _eval_vector(x0, settings=settings_low)
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
        "lgradb": _restart_lgradb(rec0),
        "objective": obj0,
        "feasibility": _constraint_violation_inf(cons0),
        "record": rec0,
        "x": x0.tolist(),
    }
    best_violation = {
        "lgradb": _restart_lgradb(rec0),
        "objective": obj0,
        "feasibility": _constraint_violation_inf(cons0),
        "record": rec0,
        "x": x0.tolist(),
    }
    best_high = None
    restart_selection_counts: dict[str, int] = {}
    restart_history_path = output_dir / "restart_history.jsonl"

    budget = int(args.budget_initial)
    for outer in range(1, args.outer_iters + 1):
        state_x = np.asarray(state.x, dtype=float)
        x_center = state_x
        restart_decision = None
        if args.adaptive_restart:
            state_feasibility = _constraint_violation_inf(np.asarray(state.constraints))
            best_high_x = (
                None if best_high is None else np.asarray(best_high["x"], dtype=float)
            )
            best_high_metrics = (
                {} if best_high is None else best_high.get("metrics", {})
            )
            best_high_objective = (
                None
                if best_high is None
                else float(
                    best_high_metrics.get(
                        "minimum_normalized_magnetic_gradient_scale_length",
                        float("-inf"),
                    )
                )
            )
            best_high_feasibility = (
                None
                if best_high is None
                else float(best_high.get("high_fidelity_feasibility", float("inf")))
            )
            (
                x_center,
                selected_seed_label,
                selected_identity,
                restart_decision,
                restart_selection_counts,
            ) = select_adaptive_restart_runtime(
                problem="p2",
                state_x=state_x,
                state_objective=-float(state.objective),
                state_feasibility=state_feasibility,
                best_violation_x=np.asarray(best_violation["x"], dtype=float),
                best_violation_objective=float(best_violation["lgradb"]),
                best_violation_feasibility=float(best_violation["feasibility"]),
                best_low_x=np.asarray(best_low["x"], dtype=float),
                best_low_objective=float(best_low["lgradb"]),
                best_low_feasibility=float(best_low["feasibility"]),
                best_high_x=best_high_x,
                best_high_objective=best_high_objective,
                best_high_feasibility=best_high_feasibility,
                selection_counts=restart_selection_counts,
                feasibility_weight=float(args.restart_feasibility_weight),
                objective_weight=float(args.restart_objective_weight),
                diversity_weight=float(args.restart_diversity_weight),
                saturation_penalty=float(args.restart_saturation_penalty),
                novelty_min_distance=float(args.restart_novelty_min_distance),
                novelty_feasibility_max=float(args.restart_novelty_feasibility_max),
                novelty_near_duplicate_distance=float(
                    args.restart_novelty_near_duplicate_distance
                ),
                novelty_judge_mode=str(args.restart_novelty_judge_mode),
            )
            append_restart_history(
                restart_history_path,
                outer=outer,
                selected_seed=selected_seed_label,
                selected_seed_identity=selected_identity,
                counts=restart_selection_counts,
                decision=restart_decision,
            )
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

            objective, constraints, record = _eval_vector(x, settings=settings_low)
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
                    "lgradb": _restart_lgradb(record),
                    "objective": objective,
                    "feasibility": viol,
                    "record": record,
                    "x": x.tolist(),
                }

            is_feasible = viol <= 0.0
            record_lgradb = _restart_lgradb(record)
            if is_feasible and record_lgradb > best_low["lgradb"]:
                best_low = {
                    "lgradb": record_lgradb,
                    "objective": objective,
                    "feasibility": viol,
                    "record": record,
                    "x": x.tolist(),
                }

            log = {
                "outer": outer,
                "inner": inner,
                "budget": budget,
                "elapsed_sec": time.time() - t0,
                "loss": loss,
                "lgradb": _telemetry_lgradb(record),
                "constraint_violation_inf": viol,
                "feasibility_official": float(record.get("feasibility", float("inf"))),
                "error": record.get("error"),
                "restart_seed": None
                if restart_decision is None
                else restart_decision.get("selected_label"),
            }
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(log) + "\n")

        rec = optimizer.provide_recommendation()
        x_rec = np.asarray(rec.value, dtype=float)
        obj_rec, cons_rec, _record_rec = _eval_vector(x_rec, settings=settings_low)

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

        if args.promote_every > 0 and outer % args.promote_every == 0:
            candidates = [best_violation]
            if best_low["feasibility"] <= 0.0:
                candidates.append(best_low)
            if len(candidates) > args.promote_budget:
                candidates = candidates[: args.promote_budget]

            for cand in candidates:
                feasibility_low = float(cand["record"].get("feasibility", float("inf")))
                if not math.isfinite(feasibility_low) or (
                    feasibility_low > args.promote_feasibility_max
                ):
                    continue

                x_cand = np.asarray(cand["x"], dtype=float)
                surface = unravel(jnp.asarray(x_cand * scale))
                boundary = _surface_to_boundary(surface)

                hi = _eval_boundary_with_timeout(
                    boundary,
                    vmec_fidelity="high_fidelity",
                    timeout_sec=args.promote_timeout_high,
                )
                if hi.get("error") is not None or not bool(
                    hi.get("equilibrium_converged", True)
                ):
                    continue

                metrics_hi = hi["metrics"]
                lgradb_hi = float(
                    metrics_hi.get(
                        "minimum_normalized_magnetic_gradient_scale_length",
                        float("-inf"),
                    )
                )
                score = _normalize_score(lgradb_hi)
                feasible_hi = (
                    float(hi.get("feasibility", float("inf"))) <= args.feas_tol
                )
                if not feasible_hi:
                    score = 0.0

                promoted = {
                    "score": score,
                    "feasible": feasible_hi,
                    "metrics": metrics_hi,
                    "high_fidelity_feasibility": float(
                        hi.get("feasibility", float("inf"))
                    ),
                    "boundary": boundary,
                    "x": cand["x"],
                }

                if best_high is None or promoted["score"] > best_high["score"]:
                    best_high = promoted
                    (output_dir / "best_high.json").write_text(
                        json.dumps(best_high, indent=2)
                    )
                    if args.stop_score is not None and best_high["score"] >= float(
                        args.stop_score
                    ):
                        return

        (output_dir / "best_low.json").write_text(json.dumps(best_low, indent=2))
        (output_dir / "best_violation.json").write_text(
            json.dumps(best_violation, indent=2)
        )


if __name__ == "__main__":
    main()
