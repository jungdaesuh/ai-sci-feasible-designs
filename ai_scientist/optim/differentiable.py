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
