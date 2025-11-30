"""Bridge between ai_scientist ASO loop and constellaration ALM infrastructure.

This module provides a steppable interface to the ALM optimization loop,
allowing supervision injection between outer iterations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple
import multiprocessing
from concurrent import futures
import time

import jax.numpy as jnp
import numpy as np
import nevergrad
from nevergrad.parametrization import parameter as param

# =============================================================================
# CONSTELLARATION API CONTRACT
# =============================================================================
# This module depends on specific constellaration function signatures.
# If these imports fail or type-check errors occur, constellaration has changed.
# Last verified: 2025-11-30
# =============================================================================

from constellaration.optimization.augmented_lagrangian import (
    AugmentedLagrangianState,
    AugmentedLagrangianSettings,
    augmented_lagrangian_function,
    update_augmented_lagrangian_state,
)
from constellaration.optimization.settings import (
    AugmentedLagrangianMethodSettings,
    NevergradSettings,
    OptimizationSettings,
)
from constellaration.geometry import surface_rz_fourier as rz_fourier
from constellaration.utils import pytree
import constellaration.forward_model as forward_model
import constellaration.problems as problems
from constellaration.optimization.augmented_lagrangian_runner import objective_constraints


@dataclass
class ALMStepResult:
    """Result of a single ALM outer iteration."""
    state: AugmentedLagrangianState
    n_evals: int
    objective: float
    max_violation: float
    metrics: Optional[forward_model.ConstellarationMetrics]


@dataclass
class ALMContext:
    """Context for ALM optimization (created once, reused across steps)."""
    scale: jnp.ndarray
    unravel_fn: Callable[[jnp.ndarray], rz_fourier.SurfaceRZFourier]
    problem: problems.SingleObjectiveProblem | problems.MHDStableQIStellarator
    forward_settings: forward_model.ConstellarationSettings
    alm_settings: AugmentedLagrangianMethodSettings
    aspect_ratio_upper_bound: Optional[float]


def _safe_objective_constraints(
    x: jnp.ndarray,
    scale: jnp.ndarray,
    problem: problems.SingleObjectiveProblem | problems.MHDStableQIStellarator,
    unravel_fn: Callable[[jnp.ndarray], rz_fourier.SurfaceRZFourier],
    settings: forward_model.ConstellarationSettings,
    aspect_ratio_upper_bound: float | None,
    constraint_shape: tuple[int, ...],
) -> tuple[
    tuple[jnp.ndarray, jnp.ndarray], forward_model.ConstellarationMetrics | None
]:
    """Wrapper for objective_constraints that handles exceptions gracefully."""
    try:
        return objective_constraints(
            x, scale, problem, unravel_fn, settings, aspect_ratio_upper_bound
        )
    except Exception as e:
        # Return high objective/constraints to discourage this point
        # We use a string error indicator in the first element of the tuple if needed,
        # but here we follow the pattern of returning a result that won't crash the loop.
        # However, the guide suggests: return (("ERROR", str(e), x.tolist()[:5]), None)
        # But objective_constraints return signature is strict (tuple[jnp, jnp], metrics).
        # constellaration's objective_constraints handles its own exceptions and returns 
        # NAN_TO_HIGH_VALUE. We will trust it mostly, but wrap strictly for unhandled ones.
        NAN_VALUE = 1e6
        objective = jnp.array(NAN_VALUE)
        print(f"Evaluation failed: {e}")
        # Use the passed shape to create the fallback constraints
        return ((objective, jnp.ones(constraint_shape) * NAN_VALUE), None)


def create_alm_context(
    boundary: rz_fourier.SurfaceRZFourier,
    problem: problems.SingleObjectiveProblem | problems.MHDStableQIStellarator,
    settings: OptimizationSettings,
    aspect_ratio_upper_bound: Optional[float] = None,
) -> Tuple[ALMContext, AugmentedLagrangianState]:
    """
    Initialize ALM context and starting state from a boundary.

    Returns:
        Tuple of (context for reuse, initial ALM state)
    """
    # Apply mode limits
    boundary = rz_fourier.set_max_mode_numbers(
        surface=boundary,
        max_poloidal_mode=settings.max_poloidal_mode,
        max_toroidal_mode=settings.max_toroidal_mode,
    )

    # Build mask for optimization
    mask = rz_fourier.build_mask(
        boundary,
        max_poloidal_mode=settings.max_poloidal_mode,
        max_toroidal_mode=settings.max_toroidal_mode,
    )

    # Flatten to optimization vector
    initial_guess, unravel_fn = pytree.mask_and_ravel(pytree=boundary, mask=mask)

    # Compute scaling
    scale = rz_fourier.compute_infinity_norm_spectrum_scaling_fun(
        poloidal_modes=boundary.poloidal_modes.flatten(),
        toroidal_modes=boundary.toroidal_modes.flatten(),
        alpha=settings.infinity_norm_spectrum_scaling,
    ).reshape(boundary.poloidal_modes.shape)
    scale = jnp.array(np.concatenate([scale[mask.r_cos], scale[mask.z_sin]]))

    x0 = jnp.array(initial_guess) / scale

    # Evaluate initial point
    (objective, constraints), _ = objective_constraints(
        x0,
        scale,
        problem,
        unravel_fn,
        settings.forward_model_settings,
        aspect_ratio_upper_bound,
    )

    # Create initial state
    assert isinstance(
        settings.optimizer_settings, AugmentedLagrangianMethodSettings
    )
    alm_settings = settings.optimizer_settings
    
    state = AugmentedLagrangianState(
        x=jnp.copy(x0),
        multipliers=jnp.zeros_like(constraints),
        penalty_parameters=alm_settings.penalty_parameters_initial
        * jnp.ones_like(constraints),
        objective=objective,
        constraints=constraints,
        bounds=jnp.ones_like(x0) * alm_settings.bounds_initial,
    )

    context = ALMContext(
        scale=scale,
        unravel_fn=unravel_fn,
        problem=problem,
        forward_settings=settings.forward_model_settings,
        alm_settings=alm_settings,
        aspect_ratio_upper_bound=aspect_ratio_upper_bound,
    )

    return context, state


def step_alm(
    context: ALMContext,
    state: AugmentedLagrangianState,
    budget: int,
    *,
    penalty_override: Optional[jnp.ndarray] = None,
    bounds_override: Optional[jnp.ndarray] = None,
    num_workers: int = 4,
) -> ALMStepResult:
    """
    Execute ONE outer ALM iteration.

    This is the steppable interface that allows supervision between iterations.

    Args:
        context: ALM context (created once via create_alm_context)
        state: Current ALM state
        budget: Number of function evaluations for this step
        penalty_override: Optional direct override of penalty parameters
        bounds_override: Optional direct override of trust region bounds
        num_workers: Parallel workers for evaluation

    Returns:
        ALMStepResult with updated state and metrics
    """
    # Apply overrides to state before optimization
    if penalty_override is not None:
        state = state.model_copy(update={"penalty_parameters": penalty_override})
    if bounds_override is not None:
        state = state.model_copy(update={"bounds": bounds_override})

    # Setup trust region parametrization
    parametrization = nevergrad.p.Array(
        init=np.array(state.x),
        lower=np.array(state.x - state.bounds),
        upper=np.array(state.x + state.bounds),
    )
    random_state = np.random.get_state()  # noqa: NPY002
    parametrization.random_state.set_state(random_state)

    oracle = nevergrad.optimizers.NGOpt(
        parametrization=parametrization,
        budget=budget,
        num_workers=num_workers,
    )
    oracle.suggest(np.array(state.x))

    n_evals = 0
    last_metrics = None
    
    # Extract constraint shape for safe fallback
    constraint_shape = state.constraints.shape

    mp_context = multiprocessing.get_context("forkserver")

    with futures.ProcessPoolExecutor(
        max_workers=num_workers, mp_context=mp_context
    ) as executor:
        running_evaluations: list[Tuple[futures.Future, param.Parameter]] = []
        rest_budget = budget
        
        # Using the same batch_mode logic as the original runner (defaulting to False here as not in settings)
        batch_mode = False
        
        # Time limit check omitted as per step-based design, but can be added if needed.
        
        while rest_budget or running_evaluations:
            # Submit new evaluations
            while len(running_evaluations) < min(num_workers, rest_budget):
                candidate = oracle.ask()
                
                future = executor.submit(
                    _safe_objective_constraints,
                    jnp.array(candidate.value),
                    context.scale,
                    context.problem,
                    context.unravel_fn,
                    context.forward_settings,
                    context.aspect_ratio_upper_bound,
                    constraint_shape,
                )
                running_evaluations.append((future, candidate))
                rest_budget -= 1

            # Wait for completions
            return_when = (
                futures.ALL_COMPLETED if batch_mode else futures.FIRST_COMPLETED
            )
            completed, _ = futures.wait(
                [f for f, _ in running_evaluations],
                return_when=return_when,
            )

            for future, candidate in running_evaluations:
                if future in completed:
                    n_evals += 1
                    (obj, cons), metrics = future.result()
                    last_metrics = metrics

                    oracle.tell(
                        candidate,
                        augmented_lagrangian_function(obj, cons, state).item(),
                    )

            # Remove completed from the running list
            running_evaluations = [
                (fut, cand) for fut, cand in running_evaluations if fut not in completed
            ]

    # Get final recommendation
    recommendation = oracle.provide_recommendation()
    x = recommendation.value

    # Evaluate final point
    (objective, constraints), final_metrics = objective_constraints(
        x,
        context.scale,
        context.problem,
        context.unravel_fn,
        context.forward_settings,
        context.aspect_ratio_upper_bound,
    )

    # Update ALM state
    new_state = update_augmented_lagrangian_state(
        x=jnp.copy(x),
        objective=objective,
        constraints=constraints,
        state=state,
        settings=context.alm_settings.augmented_lagrangian_settings,
    )

    max_violation = float(jnp.max(jnp.maximum(0.0, constraints)))

    return ALMStepResult(
        state=new_state,
        n_evals=n_evals,
        objective=float(objective),
        max_violation=max_violation,
        metrics=final_metrics,
    )


def state_to_boundary_params(
    context: ALMContext,
    state: AugmentedLagrangianState,
) -> Dict[str, Any]:
    """Convert ALM state back to boundary parameter dict."""
    boundary = context.unravel_fn(state.x * context.scale)
    return {
        "r_cos": np.asarray(boundary.r_cos).tolist(),
        "z_sin": np.asarray(boundary.z_sin).tolist(),
        "n_field_periods": boundary.n_field_periods,
        "is_stellarator_symmetric": boundary.is_stellarator_symmetric,
    }