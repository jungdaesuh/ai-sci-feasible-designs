"""Differentiable Optimization Module (Phase 3.1).

This module implements gradient-based optimization on input parameters
using a differentiable surrogate model.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence, Tuple

import jax.numpy as jnp
import numpy as np
import torch
from constellaration.geometry import surface_rz_fourier as surface_module

from ai_scientist import config as ai_config
from ai_scientist import tools
from ai_scientist.optim import geometry
from ai_scientist.optim.surrogate_v2 import NeuralOperatorSurrogate
from ai_scientist.utils import pytree

# Register SurfaceRZFourier as a Pytree to ensure JAX compatibility
# (In case the installed constellaration library hasn't registered it yet)
# We treat scalar metadata as meta_fields so they aren't traversed by mask_and_ravel
try:
    pytree.register_pydantic_data(
        surface_module.SurfaceRZFourier,
        meta_fields=["n_field_periods", "is_stellarator_symmetric"],
    )
except ValueError:
    # Already registered
    pass

MAX_ELONGATION = 5.0


def _is_maximization_problem(problem: str) -> bool:
    """Return True if the problem's PHYSICS objective should be maximized.

    Based on forward_model.compute_objective():
        P1: max_elongation -> minimize
        P2: gradient_scale_length -> MAXIMIZE
        P3: aspect_ratio -> minimize

    Note: CycleExecutor may train surrogates on HV for P3, but the physics
    objective direction should match the benchmark definition.
    """
    problem_lower = (problem or "").lower()
    return problem_lower.startswith("p2")


def _compute_index_mapping(
    boundary_template: Any, max_poloidal: int, max_toroidal: int, device: str
) -> Tuple[torch.Tensor, int]:
    """Compute the index mapping from compact (optimized) vector to dense (surrogate) vector.

    Returns:
        dense_indices: Tensor of shape (compact_size,) containing indices in the dense vector
                       where each compact element should be placed.
                       Sorted such that dense_indices[i] corresponds to compact_vector[i].
        dense_size: Size of the dense vector.
    """

    dummy_params = {
        "r_cos": np.zeros((max_poloidal + 1, 2 * max_toroidal + 1)),
        "z_sin": np.zeros((max_poloidal + 1, 2 * max_toroidal + 1)),
        "n_field_periods": boundary_template.n_field_periods,
        "is_stellarator_symmetric": True,
    }

    boundary = tools.make_boundary_from_params(dummy_params)

    # Set max modes (redundant if dummy_params matched, but safe)
    boundary = surface_module.set_max_mode_numbers(
        boundary,
        max_poloidal_mode=max_poloidal,
        max_toroidal_mode=max_toroidal,
    )

    mask = surface_module.build_mask(
        boundary,
        max_poloidal_mode=max_poloidal,
        max_toroidal_mode=max_toroidal,
    )

    flat_jax, unravel_fn = pytree.mask_and_ravel(boundary, mask)
    compact_size = flat_jax.size

    # IDs 1..N
    ids = jnp.arange(1, compact_size + 1, dtype=float)
    boundary_ids = unravel_fn(ids)

    params_ids = {
        "r_cos": np.asarray(boundary_ids.r_cos).tolist(),
        "z_sin": np.asarray(boundary_ids.z_sin).tolist(),
        "n_field_periods": boundary_ids.n_field_periods,
        "is_stellarator_symmetric": boundary_ids.is_stellarator_symmetric,
    }

    dense_vector, _ = tools.structured_flatten(params_ids)
    dense_size = dense_vector.size

    # Collect pairs (compact_id, dense_idx)
    # Use explicit integer comparison for robustness (IDs are integers 1..N stored as floats)
    pairs = []
    for i, val in enumerate(dense_vector):
        int_val = int(round(val))
        if int_val >= 1:  # Valid IDs are 1, 2, 3, ...
            compact_id = int_val - 1
            pairs.append((compact_id, i))

    # Sort by compact_id so that dense_indices[k] corresponds to compact_vector[k]
    pairs.sort(key=lambda x: x[0])

    # Extract dense indices in order
    sorted_dense_indices = [p[1] for p in pairs]

    return torch.tensor(
        sorted_dense_indices, dtype=torch.long, device=device
    ), dense_size


def gradient_descent_on_inputs(
    candidates: Sequence[Mapping[str, Any]],
    surrogate: NeuralOperatorSurrogate,
    cfg: ai_config.ExperimentConfig,
    *,
    steps: int = 100,
    lr: float = 1e-2,
    device: str = "cpu",
) -> list[Mapping[str, Any]]:
    """Optimize candidate parameters using gradient descent on the surrogate.

    Args:
        candidates: List of candidate dictionaries.
        surrogate: The differentiable surrogate model.
        cfg: Experiment configuration.
        steps: Number of optimization steps.
        lr: Learning rate.
        device: Device to run optimization on.

    Returns:
        List of optimized candidate dictionaries.
    """
    optimized_candidates = []

    # Determine masking parameters from template
    template = cfg.boundary_template
    max_poloidal = max(1, template.n_poloidal_modes - 1)
    max_toroidal = max(1, (template.n_toroidal_modes - 1) // 2)

    # Precompute mapping
    dense_indices, dense_size = _compute_index_mapping(
        template, max_poloidal, max_toroidal, device
    )

    weights = cfg.constraint_weights

    for candidate in candidates:
        params = candidate["params"]

        # 1. Convert to SurfaceRZFourier
        boundary = tools.make_boundary_from_params(params)

        # 2. Build mask for optimization (optimize only active modes)
        boundary = surface_module.set_max_mode_numbers(
            boundary,
            max_poloidal_mode=max_poloidal,
            max_toroidal_mode=max_toroidal,
        )

        mask = surface_module.build_mask(
            boundary,
            max_poloidal_mode=max_poloidal,
            max_toroidal_mode=max_toroidal,
        )

        # 3. Flatten to JAX array -> Numpy -> Torch
        flat_jax, unravel_fn = pytree.mask_and_ravel(boundary, mask)
        flat_np = np.array(flat_jax)
        x_torch = torch.tensor(
            flat_np, dtype=torch.float32, device=device, requires_grad=True
        )

        # 4. Optimization Loop
        optimizer = torch.optim.Adam([x_torch], lr=lr)

        for _ in range(steps):
            optimizer.zero_grad()

            # Map Compact -> Dense
            x_dense = torch.zeros(dense_size, device=device, dtype=x_torch.dtype)
            x_dense[dense_indices] = x_torch

            # Append n_field_periods
            nfp_val = float(boundary.n_field_periods)
            nfp_tensor = torch.tensor([nfp_val], device=device, dtype=x_torch.dtype)
            x_input = torch.cat([x_dense, nfp_tensor], dim=0)

            # Predict
            (
                pred_obj,
                std_obj,
                pred_mhd,
                std_mhd,
                pred_qi,
                std_qi,
            ) = surrogate.predict_torch(x_input.unsqueeze(0))

            # Compute Elongation directly (Exact, so std=0)
            mpol = surrogate._schema.mpol
            ntor = surrogate._schema.ntor
            grid_h = mpol + 1
            grid_w = 2 * ntor + 1
            half_size = grid_h * grid_w

            x_spectral = x_input[:-1].unsqueeze(0)  # (1, dense_size)
            r_cos = x_spectral[:, :half_size].view(1, grid_h, grid_w)
            z_sin = x_spectral[:, half_size:].view(1, grid_h, grid_w)

            pred_elo = geometry.elongation(r_cos, z_sin, x_input[-1])
            std_elo = torch.zeros_like(pred_elo)

            # Loss Formulation (Risk-Averse: Penalize Uncertainty)
            # Direction depends on problem:
            # - P1: MINIMIZE elongation -> UCB (mean + beta*std)
            # - P2/P3: MAXIMIZE objective -> LCB, then negate for minimization
            beta = 0.1
            if _is_maximization_problem(cfg.problem):
                # Maximize: use LCB (pessimistic for maximization), then negate
                loss_obj = -(pred_obj.squeeze() - beta * std_obj.squeeze())
            else:
                # Minimize: use UCB (pessimistic for minimization)
                loss_obj = pred_obj.squeeze() + beta * std_obj.squeeze()

            # Constraints (Penalty Method) with Pessimistic Bounds
            # MHD: Good if > 0. Pessimistic: mean - beta*std
            viol_mhd = torch.relu(-(pred_mhd.squeeze() - beta * std_mhd.squeeze()))

            # QI constraint: log10(qi) <= -3.5 (P3 default threshold)
            # Surrogate predicts raw QI, so we convert to log10 for comparison
            qi_raw = pred_qi.squeeze()
            qi_positive = qi_raw.abs() + QI_EPS
            log10_qi = torch.log10(qi_positive)
            s_qi = std_qi.squeeze()
            s_log10_qi = s_qi / (qi_positive * 2.302585)  # Uncertainty propagation
            viol_qi = torch.relu((log10_qi + beta * s_log10_qi) - LOG10_QI_THRESHOLD_P3)

            # Elongation: Good if < MAX_ELONGATION
            viol_elo = torch.relu(
                (pred_elo.squeeze() + beta * std_elo.squeeze()) - MAX_ELONGATION
            )

            loss_penalty = (
                weights.mhd * viol_mhd
                + weights.qi * viol_qi
                + weights.elongation * viol_elo
            )

            loss = loss_obj + loss_penalty

            loss.backward()
            optimizer.step()

        # 5. Reconstruction
        x_final_np = x_torch.detach().cpu().numpy()
        boundary_new = unravel_fn(jnp.array(x_final_np))

        # 6. Convert back to dict params
        params_new = {
            "r_cos": np.asarray(boundary_new.r_cos).tolist(),
            "z_sin": np.asarray(boundary_new.z_sin).tolist(),
            "n_field_periods": boundary_new.n_field_periods,
            "is_stellarator_symmetric": boundary_new.is_stellarator_symmetric,
        }
        if boundary_new.r_sin is not None:
            params_new["r_sin"] = np.asarray(boundary_new.r_sin).tolist()
        if boundary_new.z_cos is not None:
            params_new["z_cos"] = np.asarray(boundary_new.z_cos).tolist()

        optimized_candidates.append(
            {
                **candidate,
                "params": params_new,
                "design_hash": tools.design_hash(params_new),
                "source": "gradient_descent",
            }
        )

    return optimized_candidates


# Problem-dependent QI thresholds
# The physics constraints are defined in LOG space: log10(qi) <= threshold
# We use LOG10 thresholds here and compare log10(qi) to threshold.
# This is more numerically stable than comparing raw QI values, especially
# since QI spans many orders of magnitude (1e-6 to 1e-1).
LOG10_QI_THRESHOLD_P2 = -4.0  # Constraint: log10(qi) <= -4.0
LOG10_QI_THRESHOLD_P3 = -3.5  # Constraint: log10(qi) <= -3.5
QI_EPS = 1e-12  # Small epsilon for numerical stability in log computation


def _get_log10_qi_threshold(problem: str) -> float:
    """Get the log10(qi) threshold for the given problem."""
    if problem.lower().startswith("p2"):
        return LOG10_QI_THRESHOLD_P2
    return LOG10_QI_THRESHOLD_P3


def optimize_alm_inner_loop(
    x_initial: np.ndarray,
    scale: np.ndarray,
    surrogate: NeuralOperatorSurrogate,
    alm_state: Mapping[str, Any],
    *,
    n_field_periods_val: int,
    problem: str = "p3",
    steps: int = 10,
    lr: float = 1e-2,
    device: str = "cpu",
) -> np.ndarray:
    """Optimize the ALM inner loop using gradient descent on the surrogate."""

    if surrogate._schema is None:
        raise RuntimeError(
            "Surrogate schema is missing, cannot infer mapping for ALM optimization."
        )

    mpol = surrogate._schema.mpol
    ntor = surrogate._schema.ntor

    class DummyTemplate:
        def __init__(self, n_field_periods):
            self.n_field_periods = n_field_periods

    dense_indices, dense_size = _compute_index_mapping(
        DummyTemplate(n_field_periods_val),
        max_poloidal=mpol,
        max_toroidal=ntor,
        device=device,
    )

    # Convert inputs to Torch
    x_torch = torch.tensor(
        x_initial, dtype=torch.float32, device=device, requires_grad=True
    )
    scale_torch = torch.tensor(scale, dtype=torch.float32, device=device)

    multipliers = torch.tensor(
        np.asarray(alm_state["multipliers"]), dtype=torch.float32, device=device
    )
    penalty_params = torch.tensor(
        np.asarray(alm_state["penalty_parameters"]), dtype=torch.float32, device=device
    )

    optimizer = torch.optim.Adam([x_torch], lr=lr)

    log10_qi_threshold = _get_log10_qi_threshold(problem)
    beta = 0.1  # Uncertainty penalty factor

    for _ in range(steps):
        optimizer.zero_grad()

        # Unscale
        x_unscaled = x_torch * scale_torch

        # Map to Dense
        x_dense = torch.zeros(dense_size, device=device, dtype=x_torch.dtype)
        x_dense[dense_indices] = x_unscaled

        # Append n_field_periods
        nfp_tensor = torch.tensor(
            [float(n_field_periods_val)], device=device, dtype=x_torch.dtype
        )
        x_input = torch.cat([x_dense, nfp_tensor], dim=0)

        # Predict
        (
            pred_obj,
            std_obj,
            pred_mhd,
            std_mhd,
            pred_qi,
            std_qi,
        ) = surrogate.predict_torch(x_input.unsqueeze(0))

        # Compute Elongation directly
        grid_h = mpol + 1
        grid_w = 2 * ntor + 1
        half_size = grid_h * grid_w

        x_spectral = x_input[:-1].unsqueeze(0)
        r_cos = x_spectral[:, :half_size].view(1, grid_h, grid_w)
        z_sin = x_spectral[:, half_size:].view(1, grid_h, grid_w)

        pred_elo = geometry.elongation(r_cos, z_sin, x_input[-1])
        std_elo = torch.zeros_like(pred_elo)

        obj = pred_obj.squeeze() + beta * std_obj.squeeze()
        mhd = pred_mhd.squeeze()
        qi_raw = pred_qi.squeeze()  # Raw QI value from surrogate
        elo = pred_elo.squeeze()

        # Std
        s_mhd = std_mhd.squeeze()
        s_qi = std_qi.squeeze()
        s_elo = std_elo.squeeze()

        # Objective term (pessimistic for both directions)
        # Use _is_maximization_problem to handle P2 AND P3 correctly
        if _is_maximization_problem(problem):
            # P2/P3: MAXIMIZE objective -> minimize -objective
            # LCB: mean - beta*std (pessimistic for maximization)
            obj_term = -(pred_obj.squeeze() - beta * std_obj.squeeze())
        else:
            # P1: MINIMIZE objective
            # Pessimistic: mean + beta*std
            obj_term = obj

        # QI constraint: log10(qi) <= threshold
        # The surrogate predicts RAW qi, so we convert to log10 for comparison
        # This is dimensionally correct since the physics constraint is in log space
        qi_positive = qi_raw.abs() + QI_EPS
        log10_qi = torch.log10(qi_positive)

        # Uncertainty propagation: for log transformation, d(log10(x))/dx = 1/(x*ln(10))
        # So std(log10(qi)) ≈ std(qi) / (qi * ln(10))
        s_log10_qi = s_qi / (qi_positive * 2.302585)  # ln(10) ≈ 2.302585

        # Constraints (Pessimistic)
        # MHD: Good if > 0. Pessimistic: mean - beta*std (lower confidence bound)
        c1 = torch.relu(-(mhd - beta * s_mhd))
        # QI: Good if log10(qi) < threshold. Pessimistic: mean + beta*std (upper confidence bound)
        c2 = torch.relu((log10_qi + beta * s_log10_qi) - log10_qi_threshold)
        # Elongation: Good if < MAX_ELONGATION. Pessimistic: mean + beta*std
        c3 = torch.relu((elo + beta * s_elo) - MAX_ELONGATION)

        constraints = torch.stack([c1, c2, c3])

        augmented_term = (
            0.5
            * penalty_params
            * (
                torch.relu(multipliers / penalty_params + constraints) ** 2
                - (multipliers / penalty_params) ** 2
            )
        )

        loss = obj_term + torch.sum(augmented_term)

        loss.backward()
        optimizer.step()

    return x_torch.detach().cpu().numpy()
