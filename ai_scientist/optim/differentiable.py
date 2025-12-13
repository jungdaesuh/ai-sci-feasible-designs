"""Differentiable Optimization Module (Phase 3.1).

This module implements gradient-based optimization on input parameters
using a differentiable surrogate model.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence, Tuple

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

# Fourier decay regularization: penalize high-frequency coefficients
LAMBDA_FOURIER_DECAY = 0.01

# Early stopping parameters
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_IMPROVEMENT = 1e-4


def _is_maximization_problem(problem: str, target: str = "objective") -> bool:
    """Return True if the target should be maximized.

    Args:
        problem: Problem identifier (p1, p2, p3).
        target: The surrogate training target ("objective" or "hv").

    Direction logic:
        - HV (hypervolume) is ALWAYS maximized regardless of problem.
        - Physics objectives vary by problem:
            P1: max_elongation -> minimize
            P2: gradient_scale_length -> MAXIMIZE
            P3: aspect_ratio -> minimize
    """
    # HV is always maximized (Pareto hypervolume metric)
    if target.lower() == "hv":
        return True

    # Physics objective direction
    problem_lower = (problem or "").lower()
    return problem_lower.startswith("p2")


def _build_local_mask(max_poloidal: int, max_toroidal: int) -> np.ndarray:
    """Build stellarator-symmetric mask without depending on constellaration.

    Returns a boolean mask of shape (mpol+1, 2*ntor+1) where True indicates
    an active coefficient that should be optimized.
    """
    grid_h = max_poloidal + 1

    poloidal = np.arange(grid_h)[:, None]
    toroidal = np.arange(-max_toroidal, max_toroidal + 1)[None, :]

    # Stellarator symmetry: m=0 only has n >= 1 active, m>0 has all n
    return (poloidal > 0) | ((poloidal == 0) & (toroidal >= 1))


def _extract_masked_params(
    params: Mapping[str, Any], max_poloidal: int, max_toroidal: int
) -> Tuple[np.ndarray, dict]:
    """Extract active Fourier coefficients from params as a flat vector.

    Returns:
        flat_vector: 1D array of active coefficients [r_cos_masked, z_sin_masked]
        metadata: dict with n_field_periods, is_stellarator_symmetric, and shapes
    """
    mask = _build_local_mask(max_poloidal, max_toroidal)
    grid_h = max_poloidal + 1
    grid_w = 2 * max_toroidal + 1

    r_cos = np.asarray(params.get("r_cos", []), dtype=float)
    z_sin = np.asarray(params.get("z_sin", []), dtype=float)

    # Resize to expected shape (pad with zeros or truncate)
    r_cos_resized = np.zeros((grid_h, grid_w), dtype=float)
    z_sin_resized = np.zeros((grid_h, grid_w), dtype=float)

    if r_cos.size > 0:
        src_h, src_w = r_cos.shape
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
) -> Tuple[torch.Tensor, int]:
    """Compute the index mapping from compact (masked) to dense (surrogate) vector.

    This function computes the mapping algebraically by understanding both orderings:

    1. **Compact (masked)**: Elements extracted by `mask_and_ravel` from a SurfaceRZFourier
       pytree. The mask follows stellarator symmetry rules:
       - For m=0: only n >= 1 are active
       - For m>0: all n in [-ntor, ntor] are active
       Order: r_cos masked elements, then z_sin masked elements (pytree leaf order).

    2. **Dense (structured_flatten)**: Layout is [r_cos..., z_sin...] where each matrix
       is flattened in row-major order: m=0..mpol, n=-ntor..ntor.

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

    # Build the mask the same way constellaration does (stellarator symmetric):
    # For m=0: only n >= 1 are active (R_cos(0,0) is major radius, kept; but mask excludes n<1)
    # Wait - looking at build_mask more carefully:
    # (poloidal > 0) | ((poloidal == 0) & (toroidal >= 1))
    # So for m=0: n >= 1 is kept
    # For m>0: all n are kept
    poloidal = np.arange(source_grid_h)[:, None]  # (mpol+1, 1)
    toroidal = np.arange(-max_toroidal, max_toroidal + 1)[None, :]  # (1, 2*ntor+1)

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
    dense_indices, dense_size = _compute_index_mapping(
        template,
        max_poloidal,
        max_toroidal,
        device,
        dense_mpol=schema.mpol,
        dense_ntor=schema.ntor,
    )

    weights = cfg.constraint_weights
    problem = getattr(cfg, "problem", "p3")

    for candidate in candidates:
        params = candidate["params"]

        # 1. Extract masked params using local helper (mock-safe)
        flat_np, metadata = _extract_masked_params(params, max_poloidal, max_toroidal)
        n_field_periods = metadata["n_field_periods"]

        x_torch = torch.tensor(
            flat_np, dtype=torch.float32, device=device, requires_grad=True
        )

        # 4. Optimization Loop with early stopping
        optimizer = torch.optim.Adam([x_torch], lr=lr)
        best_loss = float("inf")
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
            # B3 FIX: predict_torch now returns 8 values including iota
            (
                pred_obj,
                std_obj,
                pred_mhd,
                std_mhd,
                pred_qi,
                std_qi,
                _pred_iota,  # Not used in optimization loop
                _std_iota,  # Not used in optimization loop
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
            # Determine target for P3 (uses HV in CycleExecutor)
            target = "hv" if problem.lower().startswith("p3") else "objective"
            if _is_maximization_problem(problem, target):
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

            # Fourier decay regularization (L2 penalty weighted by mode order)
            m_idx = torch.arange(grid_h, device=device, dtype=x_torch.dtype)
            n_idx = torch.arange(-ntor, ntor + 1, device=device, dtype=x_torch.dtype)
            mode_weight = torch.sqrt(m_idx[:, None] ** 2 + n_idx[None, :] ** 2)
            fourier_penalty = LAMBDA_FOURIER_DECAY * torch.sum(
                mode_weight * (r_cos.squeeze() ** 2 + z_sin.squeeze() ** 2)
            )

            loss = loss_obj + loss_penalty + fourier_penalty

            loss.backward()
            optimizer.step()

            # Early stopping check
            current_loss = loss.item()
            if current_loss < best_loss - EARLY_STOPPING_MIN_IMPROVEMENT:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    break  # Converged

        # 5. Reconstruction using local helper (mock-safe)
        x_final_np = x_torch.detach().cpu().numpy()
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
        # B3 FIX: predict_torch now returns 8 values including iota
        (
            pred_obj,
            std_obj,
            pred_mhd,
            std_mhd,
            pred_qi,
            std_qi,
            _pred_iota,  # Not used in ALM loop
            _std_iota,  # Not used in ALM loop
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
        # Determine target for P3 (uses HV in CycleExecutor)
        target = "hv" if problem.lower().startswith("p3") else "objective"
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

        # Fourier decay regularization (L2 penalty weighted by mode order)
        m_idx = torch.arange(grid_h, device=device, dtype=x_torch.dtype)
        n_idx = torch.arange(-ntor, ntor + 1, device=device, dtype=x_torch.dtype)
        mode_weight = torch.sqrt(m_idx[:, None] ** 2 + n_idx[None, :] ** 2)
        fourier_penalty = LAMBDA_FOURIER_DECAY * torch.sum(
            mode_weight * (r_cos.squeeze() ** 2 + z_sin.squeeze() ** 2)
        )

        loss = obj_term + torch.sum(augmented_term) + fourier_penalty

        loss.backward()
        optimizer.step()

    return x_torch.detach().cpu().numpy()
