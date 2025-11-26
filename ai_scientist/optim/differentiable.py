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


def _compute_index_mapping(
    boundary_template: Any,
    max_poloidal: int,
    max_toroidal: int,
    device: str
) -> Tuple[torch.Tensor, int]:
    """Compute the index mapping from compact (optimized) vector to dense (surrogate) vector."""
    
    # Create a dummy boundary with unique identifiers as coefficients
    # We use a linear range of values to track indices
    
    # 1. Create dummy boundary structure
    # We need to create a SurfaceRZFourier with the correct shape
    # We can use generate_rotating_ellipse or similar as a base, then fill with IDs
    # Or just manually construct.
    
    # Let's use the template to define the structure.
    # We need to ensure the shapes match what pytree.mask_and_ravel expects.
    
    # Create dummy params matching the template size
    dummy_params = {
        "r_cos": np.zeros((boundary_template.n_poloidal_modes, 2 * boundary_template.n_toroidal_modes + 1)),
        "z_sin": np.zeros((boundary_template.n_poloidal_modes, 2 * boundary_template.n_toroidal_modes + 1)),
        "n_field_periods": boundary_template.n_field_periods,
        "is_stellarator_symmetric": True
    }
    
    boundary = tools.make_boundary_from_params(dummy_params)
    
    # Set max modes as done in optimization
    boundary = surface_module.set_max_mode_numbers(
        boundary,
        max_poloidal_mode=max_poloidal,
        max_toroidal_mode=max_toroidal,
    )
    
    # 2. Identify indices in the compact vector
    mask = surface_module.build_mask(
        boundary,
        max_poloidal_mode=max_poloidal,
        max_toroidal_mode=max_toroidal,
    )
    
    # We want to map: Compact Index i -> Dense Index j
    # We can assign a unique ID to each coefficient in the 'dense' space (structured_flatten space)
    # Then apply mask_and_ravel to see where those IDs end up in 'compact' space? 
    # No, reverse is easier:
    # Assign unique IDs in 'compact' space. Unravel to boundary. Flatten with structured_flatten.
    # Then we see where each compact index went.
    
    flat_jax, unravel_fn = pytree.mask_and_ravel(boundary, mask)
    compact_size = flat_jax.size
    
    # Create IDs: 1, 2, ..., size
    ids = jnp.arange(1, compact_size + 1, dtype=float)
    
    # Unravel IDs to boundary
    boundary_ids = unravel_fn(ids)
    
    # Convert boundary_ids to params dict for structured_flatten
    params_ids = {
        "r_cos": np.asarray(boundary_ids.r_cos).tolist(),
        "z_sin": np.asarray(boundary_ids.z_sin).tolist(),
        "n_field_periods": boundary_ids.n_field_periods,
        "is_stellarator_symmetric": boundary_ids.is_stellarator_symmetric,
    }
    
    # Flatten with structured_flatten
    dense_vector, _ = tools.structured_flatten(params_ids)
    dense_size = dense_vector.size
    
    # Find where IDs ended up
    # dense_vector contains values from 1..compact_size at specific positions. 0 elsewhere.
    
    dense_indices = []
    compact_indices = []
    
    for i, val in enumerate(dense_vector):
        if val > 0.5: # Non-zero ID
            compact_idx = int(round(val)) - 1
            dense_indices.append(i)
            compact_indices.append(compact_idx)
            
    return torch.tensor(dense_indices, dtype=torch.long, device=device), dense_size


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
    dense_indices, dense_size = _compute_index_mapping(template, max_poloidal, max_toroidal, device)
    
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
            
            # Map Compact -> Dense
            # Initialize dense vector with zeros (differentiable)
            x_dense = torch.zeros(dense_size, device=device, dtype=x_torch.dtype)
            # Scatter values (we can use index assignment because we computed dense_indices)
            # But x_dense[dense_indices] = x_torch[compact_indices]
            # Note: _compute_index_mapping returned dense_indices corresponding to 0, 1, 2... of compact vector sorted by position?
            # Actually, we iterated dense vector. 
            # We need to map x_torch values to x_dense positions.
            # We need to know which x_torch index goes to which x_dense index.
            # The previous logic gave us: dense_indices[k] corresponds to compact value ID k (if we sort by compact ID).
            # Let's refine _compute_index_mapping logic implicitly or check return.
            # Returned: dense_indices (list of positions in dense vec), compact_indices (list of values/indices in compact vec).
            # We need to assign: x_dense[dense_indices[k]] = x_torch[compact_indices[k]]
            # To vectorize: x_dense.scatter_(0, dense_indices_tensor, x_torch_reordered)
            
            # It's simpler if we sort the mapping pairs by compact index.
            # Re-implement mapping usage here efficiently:
            # We assume x_torch has size N. We map all N elements to N positions in Dense.
            # The precomputed 'dense_indices' should be sorted such that:
            # x_dense[dense_indices[i]] = x_torch[i]
            # This implies dense_indices[i] is the position in dense vector for the i-th element of x_torch.
            
            # Let's re-verify mapping logic in _compute_index_mapping or fix it here.
            # If we return `dense_indices` such that `dense_indices[i]` is the dense-location for `x_torch[i]`:
            # Then `x_dense.scatter_(0, dense_indices, x_torch)` works? No, scatter expects source to match index size.
            # Yes: x_dense[dense_indices] = x_torch
            # Provided `dense_indices` has same length as `x_torch` and is ordered by x_torch index.
            
            # In _compute_index_mapping:
            # We iterated dense_vector. Found ID `compact_idx`.
            # So we have pairs (dense_pos, compact_idx).
            # We should sort these pairs by `compact_idx`.
            # Then `dense_indices` will be ordered by compact index.
            
            # Let's trust that we update the helper to return sorted indices.
            # Updating usage here:
            
            x_dense = torch.zeros(dense_size, device=device, dtype=x_torch.dtype)
            
            # We need 'dense_indices' sorted by 'compact_indices'
            # Let's do this sorting inside the loop or helper.
            # Assuming helper returns sorted dense_indices.
            
            x_dense[dense_indices] = x_torch # Differentiable scatter
            
            # Predict
            pred_obj, pred_mhd, pred_qi, pred_elo = surrogate.predict_torch(x_dense.unsqueeze(0))
            
            # Loss Formulation
            loss_obj = pred_obj.squeeze()
            
            # Constraints (Penalty Method)
            viol_mhd = torch.relu(-pred_mhd.squeeze())
            viol_qi = torch.relu(pred_qi.squeeze())
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
    """Optimize the ALM inner loop using gradient descent on the surrogate."""
    
    # We need to reconstruct mapping here too.
    # But we don't have easy access to template/mask config here unless passed.
    # However, NeuralOperatorSurrogate has `_schema` which implies the DENSE structure.
    # But we need the MASK structure.
    # We assume x_initial matches the surrogate's schema implied masking?
    # No, x_initial comes from ALM runner which used `boundary_template` from cfg.
    # We should probably pass the mapping or recompute it.
    # For now, we will infer it from the surrogate schema (assuming standard masking).
    
    # NOTE: This is a limitation. We assume the standard mask logic applies.
    mpol = surrogate._schema.mpol
    ntor = surrogate._schema.ntor
    
    # Create a temporary template config wrapper
    # We only need n_poloidal, n_toroidal etc.
    class MockTemplate:
        n_poloidal_modes = mpol + 1
        n_toroidal_modes = ntor * 2 + 1 # Rough approx? No, ntor is max mode.
        # schema.ntor is max toroidal mode.
        # template.n_toroidal_modes usually means number of modes or max?
        # tools._derive_schema_from_params: ntor = (shape[1]-1)//2.
        # So shape[1] = 2*ntor + 1.
        n_field_periods = 1 # We don't know this! But it doesn't affect flattening size usually.
    
    # Wait, max_poloidal in optimization is template.n_poloidal_modes - 1.
    # So if schema.mpol = 6, then n_poloidal_modes = 7.
    
    # We need exact mask logic.
    # Since we cannot easily pass template here without changing signature (which breaks runner calls),
    # we will rely on the fact that 'x_initial' size + surrogate schema *might* let us infer,
    # OR we update runner.py to pass mapping? 
    # No, runner.py calls this.
    
    # Let's regenerate mapping using schema info.
    
    dense_indices, dense_size = _compute_index_mapping(
        type("MockTemplate", (), {
            "n_poloidal_modes": mpol + 1,
            "n_toroidal_modes": ntor * 2 + 1, # This argument is unused if we pass max modes directly
            "n_field_periods": 1 # Dummy
        }),
        max_poloidal=mpol,
        max_toroidal=ntor,
        device=device
    )
    
    # Fix ordering of indices (Sort by compact index)
    # Our _compute_index_mapping returns unsorted. We need to sort it.
    # Refactoring _compute_index_mapping to return sorted indices:
    # (Done in logic below)
    
    # We need to ensure the helper returns sorted indices. 
    # Let's patch the helper implementation in the file writing.

    # Convert inputs to Torch
    x_torch = torch.tensor(x_initial, dtype=torch.float32, device=device, requires_grad=True)
    scale_torch = torch.tensor(scale, dtype=torch.float32, device=device)
    
    multipliers = torch.tensor(np.asarray(alm_state["multipliers"]), dtype=torch.float32, device=device)
    penalty_params = torch.tensor(np.asarray(alm_state["penalty_parameters"]), dtype=torch.float32, device=device)
    
    optimizer = torch.optim.Adam([x_torch], lr=lr)
    
    FEASIBILITY_CUTOFF = 1e-2 
    
    for _ in range(steps):
        optimizer.zero_grad()
        
        # Unscale
        x_unscaled = x_torch * scale_torch
        
        # Map to Dense
        x_dense = torch.zeros(dense_size, device=device, dtype=x_torch.dtype)
        x_dense[dense_indices] = x_unscaled
        
        # Predict
        pred_obj, pred_mhd, pred_qi, pred_elo = surrogate.predict_torch(x_dense.unsqueeze(0))
        
        obj = pred_obj.squeeze()
        mhd = pred_mhd.squeeze()
        qi = pred_qi.squeeze()
        elo = pred_elo.squeeze()
        
        c1 = torch.relu(-mhd)
        c2 = torch.relu(qi - FEASIBILITY_CUTOFF)
        c3 = torch.relu(elo - 5.0)
        
        constraints = torch.stack([c1, c2, c3])
        
        augmented_term = 0.5 * penalty_params * (
            torch.relu(multipliers / penalty_params + constraints)**2 - 
            (multipliers / penalty_params)**2
        )
        
        loss = obj + torch.sum(augmented_term)
        
        loss.backward()
        optimizer.step()
        
    return x_torch.detach().cpu().numpy()


# Patch the helper to return sorted indices
def _compute_index_mapping(
    boundary_template: Any,
    max_poloidal: int,
    max_toroidal: int,
    device: str
) -> Tuple[torch.Tensor, int]:
    """Compute the index mapping from compact (optimized) vector to dense (surrogate) vector."""
    
    dummy_params = {
        "r_cos": np.zeros((max_poloidal + 1, 2 * max_toroidal + 1)),
        "z_sin": np.zeros((max_poloidal + 1, 2 * max_toroidal + 1)),
        "n_field_periods": boundary_template.n_field_periods,
        "is_stellarator_symmetric": True
    }
    
    boundary = tools.make_boundary_from_params(dummy_params)
    
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
    pairs = []
    for i, val in enumerate(dense_vector):
        if val > 0.5:
            compact_id = int(round(val)) - 1
            pairs.append((compact_id, i))
            
    # Sort by compact_id
    pairs.sort(key=lambda x: x[0])
    
    # Extract dense indices in order
    sorted_dense_indices = [p[1] for p in pairs]
    
    return torch.tensor(sorted_dense_indices, dtype=torch.long, device=device), dense_size

