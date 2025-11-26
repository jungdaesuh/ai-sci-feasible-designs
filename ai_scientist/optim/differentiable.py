"""Differentiable Optimization Module (Phase 3.1).

This module implements gradient-based optimization on input parameters
using a differentiable surrogate model.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence, Tuple

import numpy as np
import torch
import jax.numpy as jnp

from ai_scientist import config as ai_config
from ai_scientist import tools
from ai_scientist.optim.surrogate_v2 import NeuralOperatorSurrogate
from constellaration.geometry import surface_rz_fourier as surface_module
from constellaration.utils import pytree


# Ensure SurfaceRZFourier is registered as a JAX PyTree
pytree.register_pydantic_data(
    surface_module.SurfaceRZFourier,
    meta_fields=["n_field_periods", "is_stellarator_symmetric"],
)


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
    
    weights = cfg.constraint_weights

    for candidate in candidates:
        params = candidate["params"]
        
        # 1. Convert to SurfaceRZFourier
        boundary = tools.make_boundary_from_params(params)
        
        # 2. Build mask for optimization (optimize only active modes)
        # Ensure boundary has correct max modes set
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
        x_torch = torch.tensor(flat_np, dtype=torch.float32, device=device, requires_grad=True)
        
        # 4. Optimization Loop
        optimizer = torch.optim.Adam([x_torch], lr=lr)
        
        for _ in range(steps):
            optimizer.zero_grad()
            
            # Predict
            # Note: surrogate.predict_torch expects a batch, so unsqueeze
            pred_obj, pred_mhd, pred_qi, pred_elo = surrogate.predict_torch(x_torch.unsqueeze(0))
            
            # Loss Formulation
            # Objective: minimize_objective flag check needed?
            # Usually P1/P3 minimize objective (or maximize negative).
            # Assuming minimize for now based on runner.py logic.
            
            loss_obj = pred_obj.squeeze()
            
            # Constraints (Penalty Method)
            # Violations are max(0, value - threshold) or similar
            # MHD: violation if < 0? Runner says: max(0, -mhd)
            viol_mhd = torch.relu(-pred_mhd.squeeze())
            # QI: violation if > 0? Runner says: max(0, qi)
            viol_qi = torch.relu(pred_qi.squeeze())
            # Elongation: violation if > threshold? Runner says: max(0, elo)
            viol_elo = torch.relu(pred_elo.squeeze())
            
            loss_penalty = (
                weights.mhd * viol_mhd + 
                weights.qi * viol_qi + 
                weights.elongation * viol_elo
            )
            
            loss = loss_obj + loss_penalty
            
            loss.backward()
            optimizer.step()
            
        # 5. Reconstruction
        x_final_np = x_torch.detach().cpu().numpy()
        boundary_new = unravel_fn(jnp.array(x_final_np))
        
        # 6. Convert back to dict params
        # Note: SurfaceRZFourier to dict
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
            
        optimized_candidates.append({
            **candidate,
            "params": params_new,
            "design_hash": tools.design_hash(params_new),
            "source": "gradient_descent",
        })
        
    return optimized_candidates


def optimize_alm_inner_loop(
    x_initial: np.ndarray,
    scale: np.ndarray,
    surrogate: NeuralOperatorSurrogate,
    alm_state: Mapping[str, Any],
    *,
    steps: int = 10,
    lr: float = 1e-2,
    device: str = "cpu",
) -> np.ndarray:
    """Optimize the ALM inner loop using gradient descent on the surrogate.

    Args:
        x_initial: Initial scaled parameters (Numpy).
        scale: Scaling factors (Numpy).
        surrogate: Differentiable surrogate model.
        alm_state: Dictionary containing 'multipliers', 'penalty_parameters', 'constraints'.
        steps: Number of optimization steps.
        lr: Learning rate.
        device: Device for optimization.

    Returns:
        Optimized scaled parameters (Numpy).
    """
    # Convert inputs to Torch
    x_torch = torch.tensor(x_initial, dtype=torch.float32, device=device, requires_grad=True)
    scale_torch = torch.tensor(scale, dtype=torch.float32, device=device)
    
    multipliers = torch.tensor(np.asarray(alm_state["multipliers"]), dtype=torch.float32, device=device)
    penalty_params = torch.tensor(np.asarray(alm_state["penalty_parameters"]), dtype=torch.float32, device=device)
    
    optimizer = torch.optim.Adam([x_torch], lr=lr)
    
    # Feasibility cutoff for QI (from runner.py)
    # FEASIBILITY_CUTOFF = 1e-2
    # Ideally passed in, but hardcoding for now to match runner.
    FEASIBILITY_CUTOFF = 1e-2 
    
    for _ in range(steps):
        optimizer.zero_grad()
        
        # Unscale for prediction
        x_unscaled = x_torch * scale_torch
        
        # Predict
        # unsqueeze for batch dim
        pred_obj, pred_mhd, pred_qi, pred_elo = surrogate.predict_torch(x_unscaled.unsqueeze(0))
        
        obj = pred_obj.squeeze()
        mhd = pred_mhd.squeeze()
        qi = pred_qi.squeeze()
        elo = pred_elo.squeeze()
        
        # Construct Constraint Vector (matching runner.py logic)
        # runner.py:
        # max(0, -mhd)
        # max(0, qi - FEASIBILITY_CUTOFF)
        # max(0, elongation - 5.0)
        
        c1 = torch.relu(-mhd)
        c2 = torch.relu(qi - FEASIBILITY_CUTOFF)
        c3 = torch.relu(elo - 5.0)
        
        constraints = torch.stack([c1, c2, c3])
        
        # ALM Loss Formula (PHR)
        # value = objective + sum(0.5 * mu * (max(0, lambda/mu + c)^2 - (lambda/mu)^2))
        
        augmented_term = 0.5 * penalty_params * (
            torch.relu(multipliers / penalty_params + constraints)**2 - 
            (multipliers / penalty_params)**2
        )
        
        loss = obj + torch.sum(augmented_term)
        
        loss.backward()
        optimizer.step()
        
    return x_torch.detach().cpu().numpy()
