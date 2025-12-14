"""Differentiable Optimization Module (Phase 3.1).

This module implements gradient-based optimization on input parameters
using a differentiable surrogate model.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence, Tuple

import numpy as np
import torch
from constellaration.geometry import surface_rz_fourier as surface_module

from ai_scientist import config as ai_config
from ai_scientist import tools
from ai_scientist.constraints import (
    EARLY_STOPPING_MIN_IMPROVEMENT,
    EARLY_STOPPING_PATIENCE,
    LAMBDA_FOURIER_DECAY,
    LAMBDA_R00_REGULARIZATION,
    MAX_ELONGATION,
    QI_EPS,
    get_log10_qi_threshold,
)
from ai_scientist.optim import geometry
from ai_scientist.optim.surrogate_v2 import NeuralOperatorSurrogate
from ai_scientist.objective_types import TargetKind
from ai_scientist.utils import pytree

_logger = logging.getLogger(__name__)

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


def _is_maximization_problem(problem: str, target: TargetKind) -> bool:
    """Return True if the target should be maximized.

    Args:
        problem: Problem identifier (p1, p2, p3).
        target: The surrogate training target (TargetKind.OBJECTIVE or TargetKind.HV).

    Direction logic:
        - HV (hypervolume) is ALWAYS maximized regardless of problem.
        - Physics objectives vary by problem:
            P1: max_elongation -> minimize
            P2: gradient_scale_length -> MAXIMIZE
            P3: aspect_ratio -> minimize
    """
    # HV is always maximized (Pareto hypervolume metric)
    if target == TargetKind.HV:
        return True

    # Physics objective direction
    problem_lower = (problem or "").lower()
    return problem_lower.startswith("p2")


def _build_local_mask(
    max_poloidal: int,
    max_toroidal: int,
    *,
    include_r00: bool = False,
) -> np.ndarray:
    """Build stellarator-symmetric mask without depending on constellaration.

    Args:
        max_poloidal: Maximum poloidal mode number.
        max_toroidal: Maximum toroidal mode number.
        include_r00: If True, include R₀₀ (m=0, n=0) in optimization.
                     Default False for backward compatibility.

    Returns:
        Boolean mask of shape (mpol+1, 2*ntor+1) where True indicates
        an active coefficient that should be optimized.
    """
    grid_h = max_poloidal + 1

    poloidal = np.arange(grid_h)[:, None]
    toroidal = np.arange(-max_toroidal, max_toroidal + 1)[None, :]

    # Stellarator symmetry: m=0 only has n >= 1 active, m>0 has all n
    # Optionally include R₀₀ (m=0, n=0) for major radius optimization
    if include_r00:
        return (poloidal > 0) | ((poloidal == 0) & (toroidal >= 0))
    else:
        return (poloidal > 0) | ((poloidal == 0) & (toroidal >= 1))


def _extract_masked_params(
    params: Mapping[str, Any],
    max_poloidal: int,
    max_toroidal: int,
    *,
    include_r00: bool = False,
) -> Tuple[np.ndarray, dict]:
    """Extract active Fourier coefficients from params as a flat vector.

    Args:
        params: Boundary parameters dict with r_cos, z_sin.
        max_poloidal: Maximum poloidal mode number.
        max_toroidal: Maximum toroidal mode number.
        include_r00: If True, include R₀₀ (m=0, n=0) in optimization.

    Returns:
        flat_vector: 1D array of active coefficients [r_cos_masked, z_sin_masked]
        metadata: dict with n_field_periods, is_stellarator_symmetric, and shapes
    """
    mask = _build_local_mask(max_poloidal, max_toroidal, include_r00=include_r00)
    grid_h = max_poloidal + 1
    grid_w = 2 * max_toroidal + 1

    r_cos = np.asarray(params.get("r_cos", []), dtype=float)
    z_sin = np.asarray(params.get("z_sin", []), dtype=float)

    # Resize to expected shape (pad with zeros or truncate)
    r_cos_resized = np.zeros((grid_h, grid_w), dtype=float)
    z_sin_resized = np.zeros((grid_h, grid_w), dtype=float)

    if r_cos.size > 0:
        src_h, src_w = r_cos.shape
        src_ntor = (src_w - 1) // 2
        # P0 FIX: Warn if high-frequency toroidal modes will be truncated
        if src_ntor > max_toroidal:
            _logger.warning(
                "Truncating r_cos ntor from %d to %d - high-frequency modes will be lost",
                src_ntor,
                max_toroidal,
            )
        copy_h = min(src_h, grid_h)
        # Center the source around the middle column (n=0)
        src_center = src_w // 2
        dst_center = grid_w // 2
        src_start = max(0, src_center - dst_center)
        dst_start = max(0, dst_center - src_center)
        copy_w_actual = min(src_w - src_start, grid_w - dst_start)
        r_cos_resized[:copy_h, dst_start : dst_start + copy_w_actual] = r_cos[
            :copy_h, src_start : src_start + copy_w_actual
        ]

    if z_sin.size > 0:
        src_h, src_w = z_sin.shape
        src_ntor = (src_w - 1) // 2
        # P0 FIX: Warn if high-frequency toroidal modes will be truncated
        if src_ntor > max_toroidal:
            _logger.warning(
                "Truncating z_sin ntor from %d to %d - high-frequency modes will be lost",
                src_ntor,
                max_toroidal,
            )
        copy_h = min(src_h, grid_h)
        src_center = src_w // 2
        dst_center = grid_w // 2
        src_start = max(0, src_center - dst_center)
        dst_start = max(0, dst_center - src_center)
        copy_w_actual = min(src_w - src_start, grid_w - dst_start)
        z_sin_resized[:copy_h, dst_start : dst_start + copy_w_actual] = z_sin[
            :copy_h, src_start : src_start + copy_w_actual
        ]

    # Extract masked values in row-major order
    r_cos_masked = r_cos_resized[mask]
    z_sin_masked = z_sin_resized[mask]

    flat_vector = np.concatenate([r_cos_masked, z_sin_masked])

    metadata = {
        "n_field_periods": params.get("n_field_periods", 1),
        "is_stellarator_symmetric": params.get("is_stellarator_symmetric", True),
        "r_cos_full": r_cos_resized,
        "z_sin_full": z_sin_resized,
        "mask": mask,
        "n_masked": int(mask.sum()),
    }

    return flat_vector, metadata


def _reconstruct_params(flat_vector: np.ndarray, metadata: dict) -> dict[str, Any]:
    """Reconstruct params dict from flat vector and metadata."""
    mask = metadata["mask"]
    n_masked = metadata["n_masked"]

    # Split vector into r_cos and z_sin portions
    r_cos_masked = flat_vector[:n_masked]
    z_sin_masked = flat_vector[n_masked:]

    # Reconstruct full matrices
    r_cos_full = metadata["r_cos_full"].copy()
    z_sin_full = metadata["z_sin_full"].copy()

    r_cos_full[mask] = r_cos_masked
    z_sin_full[mask] = z_sin_masked

    return {
        "r_cos": r_cos_full.tolist(),
        "z_sin": z_sin_full.tolist(),
        "n_field_periods": metadata["n_field_periods"],
        "is_stellarator_symmetric": metadata["is_stellarator_symmetric"],
    }


def _compute_index_mapping(
    boundary_template: Any,
    max_poloidal: int,
    max_toroidal: int,
    device: str,
    *,
    dense_mpol: int | None = None,
    dense_ntor: int | None = None,
    include_r00: bool = False,
) -> Tuple[torch.Tensor, int]:
    """Compute the index mapping from compact (masked) to dense (surrogate) vector.

    This function computes the mapping algebraically by understanding both orderings:

    1. **Compact (masked)**: Elements extracted by `mask_and_ravel` from a SurfaceRZFourier
       pytree. The mask follows stellarator symmetry rules:
       - For m=0: only n >= 1 are active (or n >= 0 if include_r00=True)
       - For m>0: all n in [-ntor, ntor] are active
       Order: r_cos masked elements, then z_sin masked elements (pytree leaf order).

    2. **Dense (structured_flatten)**: Layout is [r_cos..., z_sin...] where each matrix
       is flattened in row-major order: m=0..mpol, n=-ntor..ntor.

    Args:
        boundary_template: Template object (used for type hints only).
        max_poloidal: Maximum poloidal mode number.
        max_toroidal: Maximum toroidal mode number.
        device: Torch device string.
        dense_mpol: Target dense mpol (defaults to max_poloidal).
        dense_ntor: Target dense ntor (defaults to max_toroidal).
        include_r00: If True, include R₀₀ (m=0, n=0) in optimization.

    Returns:
        dense_indices: Tensor of shape (compact_size,) containing indices in the dense
                       vector where each compact element should be placed.
        dense_size: Size of the dense vector.
    """
    source_grid_h = max_poloidal + 1

    target_mpol = max_poloidal if dense_mpol is None else int(dense_mpol)
    target_ntor = max_toroidal if dense_ntor is None else int(dense_ntor)
    if target_mpol < max_poloidal or target_ntor < max_toroidal:
        raise ValueError(
            "Dense schema must be >= compact schema. "
            f"Got dense_mpol={target_mpol}, dense_ntor={target_ntor} "
            f"for compact max_poloidal={max_poloidal}, max_toroidal={max_toroidal}."
        )

    dense_grid_h = target_mpol + 1
    dense_grid_w = 2 * target_ntor + 1
    half_dense = dense_grid_h * dense_grid_w
    dense_size = 2 * half_dense  # r_cos + z_sin

    # Build the mask with stellarator symmetry rules:
    # - For m=0: n >= 1 (or n >= 0 if include_r00=True)
    # - For m>0: all n are kept
    poloidal = np.arange(source_grid_h)[:, None]  # (mpol+1, 1)
    toroidal = np.arange(-max_toroidal, max_toroidal + 1)[None, :]  # (1, 2*ntor+1)

    if include_r00:
        # Include R₀₀ (m=0, n=0) for major radius optimization
        mask_array = (poloidal > 0) | ((poloidal == 0) & (toroidal >= 0))
    else:
        # Default: exclude R₀₀ for backward compatibility
        mask_array = (poloidal > 0) | ((poloidal == 0) & (toroidal >= 1))

    # Get (m, n_idx) pairs where mask is True, in row-major order
    # np.argwhere returns in row-major order: iterates m first, then n_idx within each m
    masked_indices = np.argwhere(mask_array)  # Shape (N, 2) with [m, n_idx]

    compact_to_dense = []

    # r_cos portion (first half of dense vector)
    # structured_flatten ordering: for m in 0..mpol, for n in -ntor..ntor:
    #   index = m * dense_grid_w + (n + dense_ntor)
    for m, n_idx in masked_indices:
        n = int(n_idx) - max_toroidal
        dense_idx = int(m) * dense_grid_w + (n + target_ntor)
        compact_to_dense.append(int(dense_idx))

    # z_sin portion (second half of dense vector, offset by half_dense)
    # Same mask applies to z_sin
    for m, n_idx in masked_indices:
        n = int(n_idx) - max_toroidal
        dense_idx = half_dense + int(m) * dense_grid_w + (n + target_ntor)
        compact_to_dense.append(int(dense_idx))

    return torch.tensor(compact_to_dense, dtype=torch.long, device=device), dense_size


def gradient_descent_on_inputs(
    candidates: Sequence[Mapping[str, Any]],
    surrogate: NeuralOperatorSurrogate,
    cfg: ai_config.ExperimentConfig,
    *,
    steps: int = 100,
    lr: float = 1e-2,
    device: str = "cpu",
    target: TargetKind,
) -> list[Mapping[str, Any]]:
    """Optimize candidate parameters using gradient descent on the surrogate.

    Args:
        candidates: List of candidate dictionaries.
        surrogate: The differentiable surrogate model.
        cfg: Experiment configuration.
        steps: Number of optimization steps.
        lr: Learning rate.
        device: Device to run optimization on.
        target: Surrogate training target (TargetKind.OBJECTIVE or TargetKind.HV).
                Use get_training_target() to obtain this from problem type.

    Returns:
        List of optimized candidate dictionaries.
    """
    optimized_candidates = []

    # Determine masking parameters from template
    template = cfg.boundary_template
    max_poloidal = max(1, template.n_poloidal_modes - 1)
    max_toroidal = max(1, (template.n_toroidal_modes - 1) // 2)

    if surrogate._schema is None:
        raise RuntimeError(
            "Surrogate schema is missing; cannot build dense input vector."
        )
    schema = surrogate._schema
    # Tests sometimes inject a MagicMock schema with only (mpol, ntor). Normalize to the
    # real FlattenSchema so structured_flatten has deterministic rounding metadata.
    if not isinstance(schema, tools.FlattenSchema):
        schema = tools.FlattenSchema(mpol=int(schema.mpol), ntor=int(schema.ntor))

    # Precompute mapping (compact mask -> dense surrogate schema)
    # H1 Fix: Check if major radius optimization is enabled
    include_r00 = getattr(cfg, "optimize_major_radius", False)
    dense_indices, dense_size = _compute_index_mapping(
        template,
        max_poloidal,
        max_toroidal,
        device,
        dense_mpol=schema.mpol,
        dense_ntor=schema.ntor,
        include_r00=include_r00,
    )

    weights = cfg.constraint_weights
    problem = getattr(cfg, "problem", "p3")
    ntor = schema.ntor  # For R₀₀ index calculation

    for candidate in candidates:
        params = candidate["params"]

        # 1. Extract masked params using local helper (mock-safe)
        flat_np, metadata = _extract_masked_params(
            params, max_poloidal, max_toroidal, include_r00=include_r00
        )
        n_field_periods = metadata["n_field_periods"]

        x_torch = torch.tensor(
            flat_np, dtype=torch.float32, device=device, requires_grad=True
        )

        # 4. Optimization Loop with early stopping and LR scheduling (Issue #3 fix)
        optimizer = torch.optim.Adam([x_torch], lr=lr)
        # Cosine annealing scheduler: lr decreases smoothly from initial to ~0 over steps
        # This improves convergence by allowing aggressive early exploration then fine-tuning
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=steps, eta_min=lr * 0.01
        )
        best_loss = float("inf")
        best_x_torch = x_torch.detach().clone()  # Capture initial state as best
        patience_counter = 0

        for step in range(steps):
            optimizer.zero_grad()

            # Map Compact -> Dense
            base_params = {
                "r_cos": np.asarray(metadata["r_cos_full"]).tolist(),
                "z_sin": np.asarray(metadata["z_sin_full"]).tolist(),
                "n_field_periods": int(n_field_periods),
                "is_stellarator_symmetric": bool(
                    metadata.get("is_stellarator_symmetric", True)
                ),
            }
            base_vec, _ = tools.structured_flatten(base_params, schema=schema)
            x_dense = torch.tensor(base_vec, device=device, dtype=x_torch.dtype)
            x_dense[dense_indices] = x_torch

            # Append n_field_periods
            nfp_val = float(n_field_periods)
            nfp_tensor = torch.tensor([nfp_val], device=device, dtype=x_torch.dtype)
            x_input = torch.cat([x_dense, nfp_tensor], dim=0)

            # Predict
            # Issue #1 FIX: predict_torch now returns 12 values (6 metrics × mean/std)
            (
                pred_obj,
                std_obj,
                pred_mhd,
                std_mhd,
                pred_qi,
                std_qi,
                _pred_iota,  # Not used in optimization loop
                _std_iota,  # Not used in optimization loop
                _pred_mirror,  # P3 constraint - not used in GD loop
                _std_mirror,
                _pred_flux,  # P3 constraint - not used in GD loop
                _std_flux,
            ) = surrogate.predict_torch(x_input.unsqueeze(0))

            # Compute Elongation directly (Exact, so std=0)
            mpol = schema.mpol
            ntor = schema.ntor
            grid_h = mpol + 1
            grid_w = 2 * ntor + 1
            half_size = grid_h * grid_w

            x_spectral = x_input[:-1].unsqueeze(0)  # (1, dense_size)
            r_cos = x_spectral[:, :half_size].view(1, grid_h, grid_w)
            z_sin = x_spectral[:, half_size:].view(1, grid_h, grid_w)

            # B5 FIX: Use elongation_isoperimetric for physics-correct calculation
            pred_elo = geometry.elongation_isoperimetric(r_cos, z_sin, x_input[-1])
            std_elo = torch.zeros_like(pred_elo)

            # Loss Formulation (Risk-Averse: Penalize Uncertainty)
            # Direction depends on problem and target:
            # - P1: MINIMIZE elongation -> UCB (mean + beta*std)
            # - P2: MAXIMIZE gradient -> LCB, then negate
            # - P3 with HV target: MAXIMIZE HV -> LCB, then negate
            beta = 0.1
            # Use explicit target (required parameter - no fallback inference)
            if _is_maximization_problem(problem, target):
                # Maximize: use LCB (pessimistic for maximization), then negate
                loss_obj = -(pred_obj.squeeze() - beta * std_obj.squeeze())
            else:
                # Minimize: use UCB (pessimistic for minimization)
                loss_obj = pred_obj.squeeze() + beta * std_obj.squeeze()

            # Constraints (Penalty Method) with Pessimistic Bounds
            # MHD: Good if > 0. Pessimistic: mean - beta*std
            viol_mhd = torch.relu(-(pred_mhd.squeeze() - beta * std_mhd.squeeze()))

            # QI constraint: log10(qi) <= threshold (problem-dependent)
            # Surrogate predicts raw QI, so we convert to log10 for comparison
            qi_raw = pred_qi.squeeze()
            qi_positive = qi_raw.abs() + QI_EPS
            log10_qi = torch.log10(qi_positive)
            s_qi = std_qi.squeeze()

            # Uncertainty propagation with fourth-order bias correction
            # Taylor expansion of E[log(X)] for X ~ N(μ, σ²):
            #   E[log(X)] ≈ log(μ) - σ²/(2μ²) - 3σ⁴/(4μ⁴) + O(σ⁶)
            # Converting to log₁₀ by dividing by ln(10):
            #   E[log₁₀(X)] ≈ log₁₀(μ) - σ²/(2μ²ln10) - 3σ⁴/(4μ⁴ln10)
            ln10 = 2.302585
            s_log10_qi = s_qi / (qi_positive * ln10)
            # Second-order + fourth-order bias correction
            cv_squared = (s_qi / qi_positive) ** 2  # (σ/μ)²
            bias_correction = cv_squared / (2 * ln10) + 3 * cv_squared**2 / (4 * ln10)
            log10_qi_corrected = log10_qi - bias_correction

            log10_qi_threshold = get_log10_qi_threshold(problem)
            viol_qi = torch.relu(
                (log10_qi_corrected + beta * s_log10_qi) - log10_qi_threshold
            )

            # Elongation: Good if < MAX_ELONGATION
            viol_elo = torch.relu(
                (pred_elo.squeeze() + beta * std_elo.squeeze()) - MAX_ELONGATION
            )

            loss_penalty = (
                weights.mhd * viol_mhd
                + weights.qi * viol_qi
                + weights.elongation * viol_elo
            )

            # Fourier decay regularization (L2 penalty weighted by mode order)
            m_idx = torch.arange(grid_h, device=device, dtype=x_torch.dtype)
            n_idx_arr = torch.arange(
                -ntor, ntor + 1, device=device, dtype=x_torch.dtype
            )
            mode_weight = torch.sqrt(m_idx[:, None] ** 2 + n_idx_arr[None, :] ** 2)
            fourier_penalty = LAMBDA_FOURIER_DECAY * torch.sum(
                mode_weight * (r_cos.squeeze() ** 2 + z_sin.squeeze() ** 2)
            )

            # H1 Fix: R₀₀ regularization (if included in optimization)
            # Strong regularization toward dataset mean (~1.0) prevents scale drift
            r00_penalty = torch.tensor(0.0, device=device, dtype=x_torch.dtype)
            if include_r00:
                # R₀₀ is at position (0, ntor) in the r_cos coefficient matrix
                # In the dense vector, index = 0 * dense_grid_w + ntor = ntor
                r00_value = x_dense[ntor]
                r00_target = 1.0
                r00_penalty = LAMBDA_R00_REGULARIZATION * (r00_value - r00_target) ** 2

            loss = loss_obj + loss_penalty + fourier_penalty + r00_penalty

            loss.backward()
            optimizer.step()
            scheduler.step()  # Issue #3 fix: update LR each step

            # Early stopping check: track best parameters seen
            current_loss = loss.item()
            if current_loss < best_loss - EARLY_STOPPING_MIN_IMPROVEMENT:
                best_loss = current_loss
                best_x_torch = x_torch.detach().clone()  # Capture best parameters
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    break  # Converged

        # 5. Reconstruction using best parameters (not just last step)
        x_final_np = best_x_torch.cpu().numpy()
        params_new = _reconstruct_params(x_final_np, metadata)

        optimized_candidates.append(
            {
                **candidate,
                "params": params_new,
                "design_hash": tools.design_hash(params_new),
                "source": "gradient_descent",
            }
        )

    return optimized_candidates


# NOTE: Physics constants (QI_EPS, LOG10_QI_THRESHOLDS, MAX_ELONGATION, etc.)
# are imported from ai_scientist.constraints (SSOT for optimization constants).
# Use get_log10_qi_threshold(problem) to get problem-specific QI thresholds.


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
    target: TargetKind,
    include_r00: bool = False,
) -> np.ndarray:
    """Optimize the ALM inner loop using gradient descent on the surrogate.

    Args:
        x_initial: Initial parameter vector (scaled).
        scale: Scaling factors for parameters.
        surrogate: The differentiable surrogate model.
        alm_state: ALM state with multipliers and penalty parameters.
        n_field_periods_val: Number of field periods.
        problem: Problem identifier (p1, p2, p3).
        steps: Number of optimization steps.
        lr: Learning rate.
        device: Device to run optimization on.
        target: Surrogate training target (TargetKind.OBJECTIVE or TargetKind.HV).
        include_r00: If True, include R₀₀ in optimization with regularization.
    """

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
        include_r00=include_r00,
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
    # Cosine annealing scheduler for better convergence (Issue #3 fix)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=lr * 0.01
    )

    log10_qi_threshold = get_log10_qi_threshold(problem)
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
        # Issue #1 FIX: predict_torch now returns 12 values (6 metrics × mean/std)
        (
            pred_obj,
            std_obj,
            pred_mhd,
            std_mhd,
            pred_qi,
            std_qi,
            _pred_iota,  # Not used in ALM loop
            _std_iota,  # Not used in ALM loop
            _pred_mirror,  # P3 constraint - not used in ALM loop
            _std_mirror,
            _pred_flux,  # P3 constraint - not used in ALM loop
            _std_flux,
        ) = surrogate.predict_torch(x_input.unsqueeze(0))

        # Compute Elongation directly
        grid_h = mpol + 1
        grid_w = 2 * ntor + 1
        half_size = grid_h * grid_w

        x_spectral = x_input[:-1].unsqueeze(0)
        r_cos = x_spectral[:, :half_size].view(1, grid_h, grid_w)
        z_sin = x_spectral[:, half_size:].view(1, grid_h, grid_w)

        # B5 FIX: Use elongation_isoperimetric for physics-correct calculation
        pred_elo = geometry.elongation_isoperimetric(r_cos, z_sin, x_input[-1])
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
        # Use explicit target (required parameter - no fallback inference)
        if _is_maximization_problem(problem, target):
            # P2/P3: MAXIMIZE target -> minimize -objective
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

        # Uncertainty propagation with fourth-order bias correction
        # Taylor expansion: E[log₁₀(X)] ≈ log₁₀(μ) - σ²/(2μ²ln10) - 3σ⁴/(4μ⁴ln10)
        ln10 = 2.302585
        s_log10_qi = s_qi / (qi_positive * ln10)
        # Second-order + fourth-order bias correction
        cv_squared = (s_qi / qi_positive) ** 2  # (σ/μ)²
        bias_correction = cv_squared / (2 * ln10) + 3 * cv_squared**2 / (4 * ln10)
        log10_qi_corrected = log10_qi - bias_correction

        # Constraints (Pessimistic)
        # MHD: Good if > 0. Pessimistic: mean - beta*std (lower confidence bound)
        c1 = torch.relu(-(mhd - beta * s_mhd))
        # QI: Good if log10(qi) < threshold. Pessimistic: mean + beta*std (upper confidence bound)
        c2 = torch.relu((log10_qi_corrected + beta * s_log10_qi) - log10_qi_threshold)
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

        # Fourier decay regularization (L2 penalty weighted by mode order)
        m_idx = torch.arange(grid_h, device=device, dtype=x_torch.dtype)
        n_idx_arr = torch.arange(-ntor, ntor + 1, device=device, dtype=x_torch.dtype)
        mode_weight = torch.sqrt(m_idx[:, None] ** 2 + n_idx_arr[None, :] ** 2)
        fourier_penalty = LAMBDA_FOURIER_DECAY * torch.sum(
            mode_weight * (r_cos.squeeze() ** 2 + z_sin.squeeze() ** 2)
        )

        # H1 Fix: R₀₀ regularization (if included in optimization)
        r00_penalty = torch.tensor(0.0, device=device, dtype=x_torch.dtype)
        if include_r00:
            r00_value = x_dense[ntor]
            r00_target = 1.0
            r00_penalty = LAMBDA_R00_REGULARIZATION * (r00_value - r00_target) ** 2

        loss = obj_term + torch.sum(augmented_term) + fourier_penalty + r00_penalty

        loss.backward()
        optimizer.step()
        scheduler.step()  # Issue #3 fix: update LR each step

    return x_torch.detach().cpu().numpy()
