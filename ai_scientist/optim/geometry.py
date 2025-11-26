"""Geometry utilities for converting between Fourier and Real-space representations.

This module implements Phase 1.1 of the V2 upgrade: Hybrid Representation.
It provides utilities to convert the compact Fourier coefficient representation
(SurfaceRZFourier) into 3D point clouds or meshes, which are more suitable
for Geometric Deep Learning (GNNs) and equivariant networks.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import torch

# Attempt to import SurfaceRZFourier for type hinting, but keep it optional
# to avoid hard dependency on constellaration if just using raw arrays.
try:
    from constellaration.geometry.surface_rz_fourier import SurfaceRZFourier
except ImportError:
    SurfaceRZFourier = Any


def fourier_to_real_space(
    r_cos: np.ndarray | torch.Tensor,
    z_sin: np.ndarray | torch.Tensor,
    n_theta: int = 32,
    n_zeta: int = 32,
    n_field_periods: int = 1,
) -> Tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
    """
    Convert Fourier coefficients (R_cos, Z_sin) to real-space coordinates (R, Z, Phi).
    
    The surface is defined as:
    R(theta, zeta) = sum_{m,n} r_cos[m,n] * cos(m*theta - n*Nfp*zeta)
    Z(theta, zeta) = sum_{m,n} z_sin[m,n] * sin(m*theta - n*Nfp*zeta)
    
    Args:
        r_cos: Fourier coefficients for R [mpol+1, 2*ntor+1] (numpy or torch)
        z_sin: Fourier coefficients for Z [mpol+1, 2*ntor+1] (numpy or torch)
        n_theta: Number of poloidal grid points per section.
        n_zeta: Number of toroidal grid points per field period.
        n_field_periods: Number of field periods (Nfp).
        
    Returns:
        Tuple of (R, Z, Phi) grids in real space.
        Shapes will be (n_theta, n_zeta * n_field_periods).
    """
    is_torch = isinstance(r_cos, torch.Tensor)
    
    if is_torch:
        mpol_plus_1, two_ntor_plus_1 = r_cos.shape
        xp = torch
        cos = torch.cos
        sin = torch.sin
        pi = torch.pi
    else:
        mpol_plus_1, two_ntor_plus_1 = r_cos.shape
        xp = np
        cos = np.cos
        sin = np.sin
        pi = np.pi

    mpol = mpol_plus_1 - 1
    ntor = (two_ntor_plus_1 - 1) // 2
    
    # Create grid
    if is_torch:
        # torch.linspace includes endpoint by default. Simulate endpoint=False by adding one step and slicing.
        theta = torch.linspace(0, 2 * pi, n_theta + 1)[:-1]
        zeta_one_period = torch.linspace(0, 2 * pi / n_field_periods, n_zeta + 1)[:-1]
    else:
        theta = np.linspace(0, 2 * pi, n_theta, endpoint=False)
        zeta_one_period = np.linspace(0, 2 * pi / n_field_periods, n_zeta, endpoint=False)
    
    # Repeat for all field periods? Or just return one period? 
    # Usually GNNs might operate on one period or the whole torus.
    # Let's generate the full torus for completeness in visualization/point clouds.
    zeta_list = []
    for i in range(n_field_periods):
        zeta_list.append(zeta_one_period + i * (2 * pi / n_field_periods))
    
    if is_torch:
        zeta = torch.cat(zeta_list)
    else:
        zeta = np.concatenate(zeta_list)
        
    # Meshgrid (theta, zeta)
    # Shapes: T (n_theta, 1), Z (1, n_zeta_total)
    theta_grid = theta[:, None] if not is_torch else theta.unsqueeze(1)
    zeta_grid = zeta[None, :] if not is_torch else zeta.unsqueeze(0)
    
    # Evaluate series
    # R = sum r_cos * cos(angle)
    # angle = m*theta - n*Nfp*zeta
    
    R = xp.zeros_like(theta_grid * zeta_grid)
    Z = xp.zeros_like(R)
    
    # Loop over modes (vectorization is possible but clearer loop first)
    # Optimization: precompute angle grid?
    # angle[m, n] = m*theta - n*Nfp*zeta
    
    for m in range(mpol + 1):
        for n_idx in range(2 * ntor + 1):
            n = n_idx - ntor # Maps index 0..2*ntor to -ntor..ntor
            
            # Check efficient coefficient magnitude to skip small ones?
            rc = r_cos[m, n_idx]
            zs = z_sin[m, n_idx]
            
            # Using the VMEC convention: angle = m*theta - n*Nfp*zeta
            # (Note: sign of n might vary by convention, check VMEC++)
            angle = m * theta_grid - n * n_field_periods * zeta_grid
            
            R = R + rc * cos(angle)
            Z = Z + zs * sin(angle)
            
    Phi = zeta_grid.expand_as(R) if is_torch else np.broadcast_to(zeta_grid, R.shape)
    return R, Z, Phi


def batch_fourier_to_real_space(
    r_cos: torch.Tensor,
    z_sin: torch.Tensor,
    n_field_periods: int | torch.Tensor,
    n_theta: int = 32,
    n_zeta: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batched version of fourier_to_real_space for PyTorch.
    
    Args:
        r_cos: (Batch, mpol+1, 2*ntor+1)
        z_sin: (Batch, mpol+1, 2*ntor+1)
        n_field_periods: Nfp. Can be int (fixed) or Tensor (Batch,).
                         If Tensor, n_zeta is TOTAL points over [0, 2pi].
                         If int, n_zeta is points PER PERIOD (legacy).
        
    Returns:
        R, Z, Phi with shape (Batch, n_theta, n_zeta_total)
    """
    batch_size, mpol_plus_1, two_ntor_plus_1 = r_cos.shape
    mpol = mpol_plus_1 - 1
    ntor = (two_ntor_plus_1 - 1) // 2
    
    # Create grid (shared across batch)
    device = r_cos.device
    pi = torch.pi
    
    is_variable_nfp = isinstance(n_field_periods, torch.Tensor)
    
    if is_variable_nfp:
        # n_zeta is total points over [0, 2pi]
        # We generate a fixed grid for the whole torus
        zeta = torch.linspace(0, 2 * pi, n_zeta + 1, device=device)[:-1] # (n_zeta,)
        
        # Ensure n_field_periods is (Batch, 1, 1) for broadcasting
        nfp_tensor = n_field_periods.view(batch_size, 1, 1)
    else:
        # Legacy mode: n_zeta per period
        zeta_one = torch.linspace(0, 2 * pi / n_field_periods, n_zeta + 1, device=device)[:-1]
        zeta_list = [zeta_one + i * (2 * pi / n_field_periods) for i in range(n_field_periods)]
        zeta = torch.cat(zeta_list) # (n_zeta * nfp,)
        nfp_tensor = float(n_field_periods) # Scalar broadcast
    
    theta = torch.linspace(0, 2 * pi, n_theta + 1, device=device)[:-1]
    
    # (1, n_theta, 1)
    theta_grid = theta.view(1, n_theta, 1)
    # (1, 1, n_zeta_total)
    zeta_grid = zeta.view(1, 1, -1)
    
    R = torch.zeros(batch_size, n_theta, zeta.size(0), device=device)
    Z = torch.zeros_like(R)
    
    for m in range(mpol + 1):
        # m*theta term
        m_theta = m * theta_grid # (1, T, 1)
        
        for n_idx in range(2 * ntor + 1):
            n = n_idx - ntor
            # n*Nfp*zeta term
            # If nfp is tensor: (B, 1, 1) * (1, 1, Z) -> (B, 1, Z)
            n_zeta_term = n * nfp_tensor * zeta_grid 
            
            angle = m_theta - n_zeta_term # (B, T, Z)
            
            cos_angle = torch.cos(angle)
            sin_angle = torch.sin(angle)
            
            # Coefficients: (B, 1, 1)
            rc = r_cos[:, m, n_idx].view(batch_size, 1, 1)
            zs = z_sin[:, m, n_idx].view(batch_size, 1, 1)
            
            R = R + rc * cos_angle
            Z = Z + zs * sin_angle
            
    Phi = zeta_grid.expand(batch_size, n_theta, zeta.size(0))
    
    return R, Z, Phi


def to_cartesian(
    R: np.ndarray | torch.Tensor, 
    Z: np.ndarray | torch.Tensor, 
    Phi: np.ndarray | torch.Tensor
) -> Tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
    """Convert Cylindrical (R, Z, Phi) to Cartesian (X, Y, Z)."""
    is_torch = isinstance(R, torch.Tensor)
    if is_torch:
        cos = torch.cos
        sin = torch.sin
    else:
        cos = np.cos
        sin = np.sin
        
    X = R * cos(Phi)
    Y = R * sin(Phi)
    # Z is Z
    return X, Y, Z


def surface_to_point_cloud(
    r_cos: np.ndarray | torch.Tensor,
    z_sin: np.ndarray | torch.Tensor,
    n_field_periods: int,
    n_theta: int = 32,
    n_zeta: int = 32,
    as_tensor: bool = True
) -> torch.Tensor | np.ndarray:
    """
    High-level utility to convert coefficients directly to a (N, 3) point cloud.
    
    Args:
        r_cos, z_sin: Fourier coefficients.
        n_field_periods: Nfp.
        n_theta, n_zeta: Grid resolution per period.
        as_tensor: If True, returns torch.Tensor, else numpy array.
        
    Returns:
        Point cloud of shape (N_points, 3) where N = n_theta * n_zeta * Nfp.
        Columns are [x, y, z].
    """
    R, Z, Phi = fourier_to_real_space(r_cos, z_sin, n_theta, n_zeta, n_field_periods)
    X, Y, Z_cart = to_cartesian(R, Z, Phi)
    
    is_torch = isinstance(X, torch.Tensor)
    
    if is_torch:
        # Flatten and stack
        points = torch.stack([X.flatten(), Y.flatten(), Z_cart.flatten()], dim=1)
        return points
    else:
        points = np.stack([X.flatten(), Y.flatten(), Z_cart.flatten()], axis=1)
        if as_tensor:
            return torch.from_numpy(points).float()
        return points
