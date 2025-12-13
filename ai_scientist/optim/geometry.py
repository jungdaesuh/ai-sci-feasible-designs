# ruff: noqa: F722, F821
# pyright: reportUndefinedVariable=false
"""Geometry utilities for stellarator optimization.

This module implements Phase 1.1 of the V2 upgrade: Hybrid Representation.
It provides utilities to convert the compact Fourier coefficient representation
(SurfaceRZFourier) into 3D point clouds or meshes, which are more suitable
for Geometric Deep Learning (GNNs) and equivariant networks.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import torch
from jaxtyping import Float

# Attempt to import SurfaceRZFourier for type hinting, but keep it optional
# to avoid hard dependency on constellaration if just using raw arrays.
try:
    from constellaration.geometry.surface_rz_fourier import SurfaceRZFourier
except ImportError:
    SurfaceRZFourier = Any


def fourier_to_real_space(
    r_cos: Float[np.ndarray, "mpol_plus_1 two_ntor_plus_1"]
    | Float[torch.Tensor, "mpol_plus_1 two_ntor_plus_1"],
    z_sin: Float[np.ndarray, "mpol_plus_1 two_ntor_plus_1"]
    | Float[torch.Tensor, "mpol_plus_1 two_ntor_plus_1"],
    n_theta: int = 32,
    n_zeta: int = 32,
    n_field_periods: int = 1,
) -> Tuple[
    Float[np.ndarray, "n_theta n_zeta_total"]
    | Float[torch.Tensor, "n_theta n_zeta_total"],
    Float[np.ndarray, "n_theta n_zeta_total"]
    | Float[torch.Tensor, "n_theta n_zeta_total"],
    Float[np.ndarray, "n_theta n_zeta_total"]
    | Float[torch.Tensor, "n_theta n_zeta_total"],
]:
    """
    Convert Fourier coefficients (R_cos, Z_sin) to real-space coordinates (R, Z, Phi).

    The surface is defined as:
    R(theta, zeta) = sum_{m,n} r_cos[m,n] * cos(m*theta - n*Nfp*zeta)
    Z(theta, zeta) = sum_{m,n} z_sin[m,n] * sin(m*theta - n*Nfp*zeta)

    Memory Optimization:
        Uses trigonometric separability to avoid building a full (M, N, T, Z) angle tensor.
        cos(mθ - nNζ) = cos(mθ)cos(nNζ) + sin(mθ)sin(nNζ)
        sin(mθ - nNζ) = sin(mθ)cos(nNζ) - cos(mθ)sin(nNζ)
        This reduces peak memory from O(M*N*T*Z) to O(M*Z) + O(M*T) + O(N*Z).

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
        pi = torch.pi
    else:
        mpol_plus_1, two_ntor_plus_1 = r_cos.shape
        pi = np.pi

    mpol = mpol_plus_1 - 1
    ntor = (two_ntor_plus_1 - 1) // 2

    # Create grid
    if is_torch:
        theta = torch.linspace(0, 2 * pi, n_theta + 1)[:-1]
        zeta_one_period = torch.linspace(0, 2 * pi / n_field_periods, n_zeta + 1)[:-1]
    else:
        theta = np.linspace(0, 2 * pi, n_theta, endpoint=False)
        zeta_one_period = np.linspace(
            0, 2 * pi / n_field_periods, n_zeta, endpoint=False
        )

    # Generate full torus zeta grid
    zeta_list = []
    for i in range(n_field_periods):
        zeta_list.append(zeta_one_period + i * (2 * pi / n_field_periods))

    if is_torch:
        zeta = torch.cat(zeta_list)
    else:
        zeta = np.concatenate(zeta_list)

    # Precompute mode indices
    if is_torch:
        m_idx = torch.arange(mpol + 1, dtype=r_cos.dtype, device=r_cos.device)
        n_idx = torch.arange(2 * ntor + 1, dtype=r_cos.dtype, device=r_cos.device)
    else:
        m_idx = np.arange(mpol + 1, dtype=r_cos.dtype)
        n_idx = np.arange(2 * ntor + 1, dtype=r_cos.dtype)

    # n values: maps 0..2*ntor to -ntor..ntor
    n_vals = n_idx - ntor

    # =========================================================================
    # Memory-optimized Fourier summation using trig separability
    # =========================================================================
    # Instead of building full (M, N, T, Z) angle tensor, we use:
    #   cos(mθ - nNζ) = cos(mθ)cos(nNζ) + sin(mθ)sin(nNζ)
    #   sin(mθ - nNζ) = sin(mθ)cos(nNζ) - cos(mθ)sin(nNζ)
    #
    # This allows two-stage contraction:
    #   Stage 1: Contract over n to get (M, Z) intermediates
    #   Stage 2: Contract over m to get (T, Z) result
    # =========================================================================

    if is_torch:
        # Precompute separable trig terms
        # cos(mθ), sin(mθ): shape (M, T)
        m_theta = m_idx[:, None] * theta[None, :]  # (M, T)
        cos_m_theta = torch.cos(m_theta)  # (M, T)
        sin_m_theta = torch.sin(m_theta)  # (M, T)

        # cos(nNζ), sin(nNζ): shape (N, Z)
        n_nfp_zeta = (n_vals * n_field_periods)[:, None] * zeta[None, :]  # (N, Z)
        cos_n_zeta = torch.cos(n_nfp_zeta)  # (N, Z)
        sin_n_zeta = torch.sin(n_nfp_zeta)  # (N, Z)

        # Stage 1: Contract over n -> (M, Z)
        # A_rc[m,z] = sum_n r_cos[m,n] * cos(nNζ[z])
        # B_rc[m,z] = sum_n r_cos[m,n] * sin(nNζ[z])
        A_rc = torch.einsum("mn,nz->mz", r_cos, cos_n_zeta)  # (M, Z)
        B_rc = torch.einsum("mn,nz->mz", r_cos, sin_n_zeta)  # (M, Z)

        # Same for z_sin
        A_zs = torch.einsum("mn,nz->mz", z_sin, cos_n_zeta)  # (M, Z)
        B_zs = torch.einsum("mn,nz->mz", z_sin, sin_n_zeta)  # (M, Z)

        # Stage 2: Contract over m -> (T, Z)
        # R[t,z] = sum_m cos(mθ[t]) * A_rc[m,z] + sin(mθ[t]) * B_rc[m,z]
        R = torch.einsum("mt,mz->tz", cos_m_theta, A_rc) + torch.einsum(
            "mt,mz->tz", sin_m_theta, B_rc
        )

        # Z[t,z] = sum_m sin(mθ[t]) * A_zs[m,z] - cos(mθ[t]) * B_zs[m,z]
        Z = torch.einsum("mt,mz->tz", sin_m_theta, A_zs) - torch.einsum(
            "mt,mz->tz", cos_m_theta, B_zs
        )

        Phi = zeta[None, :].expand(n_theta, -1)
    else:
        # NumPy version
        m_theta = m_idx[:, None] * theta[None, :]  # (M, T)
        cos_m_theta = np.cos(m_theta)
        sin_m_theta = np.sin(m_theta)

        n_nfp_zeta = (n_vals * n_field_periods)[:, None] * zeta[None, :]  # (N, Z)
        cos_n_zeta = np.cos(n_nfp_zeta)
        sin_n_zeta = np.sin(n_nfp_zeta)

        # Stage 1: Contract over n -> (M, Z)
        A_rc = np.einsum("mn,nz->mz", r_cos, cos_n_zeta)
        B_rc = np.einsum("mn,nz->mz", r_cos, sin_n_zeta)
        A_zs = np.einsum("mn,nz->mz", z_sin, cos_n_zeta)
        B_zs = np.einsum("mn,nz->mz", z_sin, sin_n_zeta)

        # Stage 2: Contract over m -> (T, Z)
        R = np.einsum("mt,mz->tz", cos_m_theta, A_rc) + np.einsum(
            "mt,mz->tz", sin_m_theta, B_rc
        )
        Z = np.einsum("mt,mz->tz", sin_m_theta, A_zs) - np.einsum(
            "mt,mz->tz", cos_m_theta, B_zs
        )

        Phi = np.broadcast_to(zeta[None, :], R.shape)

    return R, Z, Phi


def batch_fourier_to_real_space(
    r_cos: Float[torch.Tensor, "batch mpol_plus_1 two_ntor_plus_1"],
    z_sin: Float[torch.Tensor, "batch mpol_plus_1 two_ntor_plus_1"],
    n_field_periods: int | Float[torch.Tensor, "batch"],
    n_theta: int = 32,
    n_zeta: int = 32,
) -> Tuple[
    Float[torch.Tensor, "batch n_theta n_zeta_total"],
    Float[torch.Tensor, "batch n_theta n_zeta_total"],
    Float[torch.Tensor, "batch n_theta n_zeta_total"],
]:
    """
    Batched version of fourier_to_real_space for PyTorch.

    Memory Optimization:
        Uses trigonometric separability to avoid building a full (B, M, N, T, Z) angle tensor.
        cos(mθ - nNζ) = cos(mθ)cos(nNζ) + sin(mθ)sin(nNζ)
        sin(mθ - nNζ) = sin(mθ)cos(nNζ) - cos(mθ)sin(nNζ)
        This reduces peak memory from O(B*M*N*T*Z) to O(B*M*Z) + O(M*T) + O(B*N*Z).

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
        zeta = torch.linspace(0, 2 * pi, n_zeta + 1, device=device)[:-1]  # (Z,)
        nfp_tensor = n_field_periods.view(batch_size, 1)  # (B, 1)
    else:
        # Legacy mode: n_zeta per period
        zeta_one = torch.linspace(
            0, 2 * pi / n_field_periods, n_zeta + 1, device=device
        )[:-1]
        zeta_list = [
            zeta_one + i * (2 * pi / n_field_periods) for i in range(n_field_periods)
        ]
        zeta = torch.cat(zeta_list)  # (Z,)
        nfp_scalar = float(n_field_periods)

    theta = torch.linspace(0, 2 * pi, n_theta + 1, device=device)[:-1]  # (T,)

    # Precompute mode indices
    m_idx = torch.arange(mpol + 1, dtype=r_cos.dtype, device=device)  # (M,)
    n_idx_arr = torch.arange(2 * ntor + 1, dtype=r_cos.dtype, device=device)  # (N,)
    n_vals = n_idx_arr - ntor  # Maps 0..2*ntor to -ntor..ntor

    # =========================================================================
    # Memory-optimized Fourier summation using trig separability
    # =========================================================================
    # cos(mθ - nNζ) = cos(mθ)cos(nNζ) + sin(mθ)sin(nNζ)
    # sin(mθ - nNζ) = sin(mθ)cos(nNζ) - cos(mθ)sin(nNζ)
    #
    # For fixed Nfp: both m*theta and n*Nfp*zeta terms are shared across batch
    # For variable Nfp: m*theta is still shared, but n*Nfp*zeta depends on batch
    # =========================================================================

    # Precompute cos(mθ), sin(mθ) - shape (M, T), shared across batch
    m_theta = m_idx[:, None] * theta[None, :]  # (M, T)
    cos_m_theta = torch.cos(m_theta)  # (M, T)
    sin_m_theta = torch.sin(m_theta)  # (M, T)

    if is_variable_nfp:
        # Variable Nfp: n*Nfp*zeta depends on batch
        # n_vals: (N,), nfp_tensor: (B, 1), zeta: (Z,)
        # n_nfp_zeta: (B, N, Z)
        n_nfp_zeta = (
            n_vals[None, :, None] * nfp_tensor[:, :, None] * zeta[None, None, :]
        )
        cos_n_zeta = torch.cos(n_nfp_zeta)  # (B, N, Z)
        sin_n_zeta = torch.sin(n_nfp_zeta)  # (B, N, Z)

        # Stage 1: Contract over n -> (B, M, Z)
        # A_rc[b,m,z] = sum_n r_cos[b,m,n] * cos(n*Nfp[b]*zeta[z])
        A_rc = torch.einsum("bmn,bnz->bmz", r_cos, cos_n_zeta)  # (B, M, Z)
        B_rc = torch.einsum("bmn,bnz->bmz", r_cos, sin_n_zeta)  # (B, M, Z)
        A_zs = torch.einsum("bmn,bnz->bmz", z_sin, cos_n_zeta)  # (B, M, Z)
        B_zs = torch.einsum("bmn,bnz->bmz", z_sin, sin_n_zeta)  # (B, M, Z)

        # Stage 2: Contract over m -> (B, T, Z)
        # R[b,t,z] = sum_m cos(mθ[t]) * A_rc[b,m,z] + sin(mθ[t]) * B_rc[b,m,z]
        R = torch.einsum("mt,bmz->btz", cos_m_theta, A_rc) + torch.einsum(
            "mt,bmz->btz", sin_m_theta, B_rc
        )
        Z = torch.einsum("mt,bmz->btz", sin_m_theta, A_zs) - torch.einsum(
            "mt,bmz->btz", cos_m_theta, B_zs
        )
    else:
        # Fixed Nfp: fully separable, shared trig terms
        # cos(nNζ), sin(nNζ): shape (N, Z)
        n_nfp_zeta = (n_vals * nfp_scalar)[:, None] * zeta[None, :]  # (N, Z)
        cos_n_zeta = torch.cos(n_nfp_zeta)  # (N, Z)
        sin_n_zeta = torch.sin(n_nfp_zeta)  # (N, Z)

        # Stage 1: Contract over n -> (B, M, Z)
        A_rc = torch.einsum("bmn,nz->bmz", r_cos, cos_n_zeta)  # (B, M, Z)
        B_rc = torch.einsum("bmn,nz->bmz", r_cos, sin_n_zeta)  # (B, M, Z)
        A_zs = torch.einsum("bmn,nz->bmz", z_sin, cos_n_zeta)  # (B, M, Z)
        B_zs = torch.einsum("bmn,nz->bmz", z_sin, sin_n_zeta)  # (B, M, Z)

        # Stage 2: Contract over m -> (B, T, Z)
        R = torch.einsum("mt,bmz->btz", cos_m_theta, A_rc) + torch.einsum(
            "mt,bmz->btz", sin_m_theta, B_rc
        )
        Z = torch.einsum("mt,bmz->btz", sin_m_theta, A_zs) - torch.einsum(
            "mt,bmz->btz", cos_m_theta, B_zs
        )

    Phi = zeta[None, None, :].expand(batch_size, n_theta, -1)

    return R, Z, Phi


def to_cartesian(
    R: Float[np.ndarray, "..."] | Float[torch.Tensor, "..."],
    Z: Float[np.ndarray, "..."] | Float[torch.Tensor, "..."],
    Phi: Float[np.ndarray, "..."] | Float[torch.Tensor, "..."],
) -> Tuple[
    Float[np.ndarray, "..."] | Float[torch.Tensor, "..."],
    Float[np.ndarray, "..."] | Float[torch.Tensor, "..."],
    Float[np.ndarray, "..."] | Float[torch.Tensor, "..."],
]:
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
    r_cos: Float[np.ndarray, "mpol_plus_1 two_ntor_plus_1"]
    | Float[torch.Tensor, "mpol_plus_1 two_ntor_plus_1"],
    z_sin: Float[np.ndarray, "mpol_plus_1 two_ntor_plus_1"]
    | Float[torch.Tensor, "mpol_plus_1 two_ntor_plus_1"],
    n_field_periods: int,
    n_theta: int = 32,
    n_zeta: int = 32,
    as_tensor: bool = True,
) -> Float[torch.Tensor, "n_points 3"] | Float[np.ndarray, "n_points 3"]:
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


# -----------------------------------------------------------------------------
# Phase 1.1: Differentiable Geometry Metrics (The "Engineer")
# -----------------------------------------------------------------------------


def _compute_derivatives(
    r_cos: Float[torch.Tensor, "batch mpol_plus_1 two_ntor_plus_1"],
    z_sin: Float[torch.Tensor, "batch mpol_plus_1 two_ntor_plus_1"],
    n_field_periods: int | Float[torch.Tensor, "batch"],
    n_theta: int = 64,
    n_zeta: int = 64,
) -> dict[str, Float[torch.Tensor, "batch n_theta n_zeta"]]:
    """
    Compute surface derivatives R_theta, R_zeta, Z_theta, Z_zeta, etc.

    Memory Optimization:
        Uses trigonometric separability to avoid building a full (B, M, N, T, Z) angle tensor.
        This reduces peak memory from O(B*M*N*T*Z) to O(B*M*Z) + O(M*T) + O(B*N*Z).

        Key identities:
            cos(mθ - nNζ) = cos(mθ)cos(nNζ) + sin(mθ)sin(nNζ)
            sin(mθ - nNζ) = sin(mθ)cos(nNζ) - cos(mθ)sin(nNζ)

        Derivatives introduce scaling factors (m for θ-derivatives, nNfp for ζ-derivatives)
        which are absorbed into the two-stage contraction.

    If n_field_periods is a Tensor (batched), we evaluate over the FULL torus (0 to 2pi)
    to preserve the correct winding and dependence on N_fp.
    If n_field_periods is an int (scalar), we evaluate over ONE field period (0 to 2pi/Nfp)
    and rely on symmetry.

    Args:
        r_cos, z_sin: Fourier coefficients (Batch, mpol+1, 2*ntor+1).
        n_field_periods: Number of field periods (scalar or batch).
        n_theta, n_zeta: Grid resolution.

    Returns:
        Dict containing R, Z, R_t, R_z, Z_t, Z_z, R_tt, R_tz, R_zz, Z_tt, Z_tz, Z_zz.
        All shapes: (Batch, n_theta, n_zeta).
    """
    batch_size, mpol_plus_1, two_ntor_plus_1 = r_cos.shape
    mpol = mpol_plus_1 - 1
    ntor = (two_ntor_plus_1 - 1) // 2
    device = r_cos.device

    # Handle Nfp
    is_tensor_nfp = isinstance(n_field_periods, torch.Tensor)
    if is_tensor_nfp:
        nfp = n_field_periods.view(batch_size, 1)  # (B, 1)
        # Batched N_fp: Evaluate over FULL TORUS [0, 2pi]
        zeta = torch.linspace(0, 2 * torch.pi, n_zeta + 1, device=device)[:-1]  # (Z,)
    else:
        nfp_scalar = float(n_field_periods)
        nfp = nfp_scalar
        # Fixed N_fp: Evaluate over ONE FIELD PERIOD [0, 2pi/nfp]
        zeta = torch.linspace(0, 2 * torch.pi / nfp_scalar, n_zeta + 1, device=device)[
            :-1
        ]  # (Z,)

    theta = torch.linspace(0, 2 * torch.pi, n_theta + 1, device=device)[:-1]  # (T,)

    # Precompute mode indices
    m_idx = torch.arange(mpol + 1, dtype=r_cos.dtype, device=device)  # (M,)
    n_idx_arr = torch.arange(2 * ntor + 1, dtype=r_cos.dtype, device=device)  # (N,)
    n_vals = n_idx_arr - ntor  # Maps 0..2*ntor to -ntor..ntor

    # =========================================================================
    # Memory-optimized derivative computation using trig separability
    # =========================================================================
    # Stage 1: Precompute θ-dependent terms (shared across batch)
    # cos(mθ), sin(mθ): shape (M, T)
    m_theta_angles = m_idx[:, None] * theta[None, :]  # (M, T)
    cos_m = torch.cos(m_theta_angles)  # (M, T)
    sin_m = torch.sin(m_theta_angles)  # (M, T)

    # m-weighted versions for derivatives
    m_cos_m = m_idx[:, None] * cos_m  # (M, T)
    m_sin_m = m_idx[:, None] * sin_m  # (M, T)
    m2_cos_m = (m_idx[:, None] ** 2) * cos_m  # (M, T)
    m2_sin_m = (m_idx[:, None] ** 2) * sin_m  # (M, T)

    # Stage 2: Precompute ζ-dependent terms
    if is_tensor_nfp:
        # Variable Nfp: n*Nfp depends on batch
        # n_vals: (N,), nfp: (B, 1), zeta: (Z,)
        # n_nfp: (B, N) = n_vals[None, :] * nfp
        n_nfp = n_vals[None, :] * nfp  # (B, N)
        n_nfp_zeta = n_nfp[:, :, None] * zeta[None, None, :]  # (B, N, Z)

        cos_nz = torch.cos(n_nfp_zeta)  # (B, N, Z)
        sin_nz = torch.sin(n_nfp_zeta)  # (B, N, Z)

        # n*nfp-weighted versions for ζ-derivatives
        # Note: factor_z in original code is -n*nfp, so we use n_nfp directly
        n_nfp_expanded = n_nfp[:, :, None]  # (B, N, 1)
        n_nfp_cos_nz = n_nfp_expanded * cos_nz  # (B, N, Z)
        n_nfp_sin_nz = n_nfp_expanded * sin_nz  # (B, N, Z)
        n_nfp2_cos_nz = (n_nfp_expanded**2) * cos_nz  # (B, N, Z)
        n_nfp2_sin_nz = (n_nfp_expanded**2) * sin_nz  # (B, N, Z)

        # Stage 3: Contract over n for r_cos coefficients -> (B, M, Z)
        # Basic: A_rc = sum_n r_cos[m,n] * cos(nNζ), B_rc = sum_n r_cos[m,n] * sin(nNζ)
        A_rc = torch.einsum("bmn,bnz->bmz", r_cos, cos_nz)
        B_rc = torch.einsum("bmn,bnz->bmz", r_cos, sin_nz)
        # n-weighted: for ζ-derivatives
        nA_rc = torch.einsum("bmn,bnz->bmz", r_cos, n_nfp_cos_nz)
        nB_rc = torch.einsum("bmn,bnz->bmz", r_cos, n_nfp_sin_nz)
        # n²-weighted: for ζζ-derivatives
        n2A_rc = torch.einsum("bmn,bnz->bmz", r_cos, n_nfp2_cos_nz)
        n2B_rc = torch.einsum("bmn,bnz->bmz", r_cos, n_nfp2_sin_nz)

        # Same for z_sin coefficients
        A_zs = torch.einsum("bmn,bnz->bmz", z_sin, cos_nz)
        B_zs = torch.einsum("bmn,bnz->bmz", z_sin, sin_nz)
        nA_zs = torch.einsum("bmn,bnz->bmz", z_sin, n_nfp_cos_nz)
        nB_zs = torch.einsum("bmn,bnz->bmz", z_sin, n_nfp_sin_nz)
        n2A_zs = torch.einsum("bmn,bnz->bmz", z_sin, n_nfp2_cos_nz)
        n2B_zs = torch.einsum("bmn,bnz->bmz", z_sin, n_nfp2_sin_nz)

    else:
        # Fixed Nfp: n*Nfp shared across batch
        n_nfp = n_vals * nfp_scalar  # (N,)
        n_nfp_zeta = n_nfp[:, None] * zeta[None, :]  # (N, Z)

        cos_nz = torch.cos(n_nfp_zeta)  # (N, Z)
        sin_nz = torch.sin(n_nfp_zeta)  # (N, Z)

        # n*nfp-weighted versions
        n_nfp_expanded = n_nfp[:, None]  # (N, 1)
        n_nfp_cos_nz = n_nfp_expanded * cos_nz  # (N, Z)
        n_nfp_sin_nz = n_nfp_expanded * sin_nz  # (N, Z)
        n_nfp2_cos_nz = (n_nfp_expanded**2) * cos_nz  # (N, Z)
        n_nfp2_sin_nz = (n_nfp_expanded**2) * sin_nz  # (N, Z)

        # Contract over n -> (B, M, Z)
        A_rc = torch.einsum("bmn,nz->bmz", r_cos, cos_nz)
        B_rc = torch.einsum("bmn,nz->bmz", r_cos, sin_nz)
        nA_rc = torch.einsum("bmn,nz->bmz", r_cos, n_nfp_cos_nz)
        nB_rc = torch.einsum("bmn,nz->bmz", r_cos, n_nfp_sin_nz)
        n2A_rc = torch.einsum("bmn,nz->bmz", r_cos, n_nfp2_cos_nz)
        n2B_rc = torch.einsum("bmn,nz->bmz", r_cos, n_nfp2_sin_nz)

        A_zs = torch.einsum("bmn,nz->bmz", z_sin, cos_nz)
        B_zs = torch.einsum("bmn,nz->bmz", z_sin, sin_nz)
        nA_zs = torch.einsum("bmn,nz->bmz", z_sin, n_nfp_cos_nz)
        nB_zs = torch.einsum("bmn,nz->bmz", z_sin, n_nfp_sin_nz)
        n2A_zs = torch.einsum("bmn,nz->bmz", z_sin, n_nfp2_cos_nz)
        n2B_zs = torch.einsum("bmn,nz->bmz", z_sin, n_nfp2_sin_nz)

    # =========================================================================
    # Stage 4: Contract over m to get final (B, T, Z) outputs
    # =========================================================================
    # Using trig identities:
    #   cos(mθ - nNζ) = cos(mθ)cos(nNζ) + sin(mθ)sin(nNζ)
    #   sin(mθ - nNζ) = sin(mθ)cos(nNζ) - cos(mθ)sin(nNζ)

    # R = sum r_cos * cos(α) = cos(mθ)*A_rc + sin(mθ)*B_rc
    R = torch.einsum("mt,bmz->btz", cos_m, A_rc) + torch.einsum(
        "mt,bmz->btz", sin_m, B_rc
    )

    # Z = sum z_sin * sin(α) = sin(mθ)*A_zs - cos(mθ)*B_zs
    Z = torch.einsum("mt,bmz->btz", sin_m, A_zs) - torch.einsum(
        "mt,bmz->btz", cos_m, B_zs
    )

    # R_t = sum r_cos * (-m) * sin(α) = (-m*sin(mθ))*A_rc + (m*cos(mθ))*B_rc
    R_t = torch.einsum("mt,bmz->btz", -m_sin_m, A_rc) + torch.einsum(
        "mt,bmz->btz", m_cos_m, B_rc
    )

    # R_z = sum r_cos * (nNfp) * sin(α) = sin(mθ)*nA_rc - cos(mθ)*nB_rc
    R_z = torch.einsum("mt,bmz->btz", sin_m, nA_rc) - torch.einsum(
        "mt,bmz->btz", cos_m, nB_rc
    )

    # Z_t = sum z_sin * m * cos(α) = (m*cos(mθ))*A_zs + (m*sin(mθ))*B_zs
    Z_t = torch.einsum("mt,bmz->btz", m_cos_m, A_zs) + torch.einsum(
        "mt,bmz->btz", m_sin_m, B_zs
    )

    # Z_z = sum z_sin * (-nNfp) * cos(α) = -cos(mθ)*nA_zs - sin(mθ)*nB_zs
    Z_z = -torch.einsum("mt,bmz->btz", cos_m, nA_zs) - torch.einsum(
        "mt,bmz->btz", sin_m, nB_zs
    )

    # R_tt = sum r_cos * (-m²) * cos(α) = (-m²*cos(mθ))*A_rc + (-m²*sin(mθ))*B_rc
    R_tt = torch.einsum("mt,bmz->btz", -m2_cos_m, A_rc) + torch.einsum(
        "mt,bmz->btz", -m2_sin_m, B_rc
    )

    # R_zz = sum r_cos * (-(nNfp)²) * cos(α) = -cos(mθ)*n2A_rc - sin(mθ)*n2B_rc
    R_zz = -torch.einsum("mt,bmz->btz", cos_m, n2A_rc) - torch.einsum(
        "mt,bmz->btz", sin_m, n2B_rc
    )

    # R_tz = sum r_cos * m*(nNfp) * cos(α) = (m*cos(mθ))*nA_rc + (m*sin(mθ))*nB_rc
    R_tz = torch.einsum("mt,bmz->btz", m_cos_m, nA_rc) + torch.einsum(
        "mt,bmz->btz", m_sin_m, nB_rc
    )

    # Z_tt = sum z_sin * (-m²) * sin(α) = (-m²*sin(mθ))*A_zs + (m²*cos(mθ))*B_zs
    Z_tt = torch.einsum("mt,bmz->btz", -m2_sin_m, A_zs) + torch.einsum(
        "mt,bmz->btz", m2_cos_m, B_zs
    )

    # Z_zz = sum z_sin * (-(nNfp)²) * sin(α) = -sin(mθ)*n2A_zs + cos(mθ)*n2B_zs
    Z_zz = -torch.einsum("mt,bmz->btz", sin_m, n2A_zs) + torch.einsum(
        "mt,bmz->btz", cos_m, n2B_zs
    )

    # Z_tz = sum z_sin * m*(nNfp) * sin(α) = (m*sin(mθ))*nA_zs - (m*cos(mθ))*nB_zs
    Z_tz = torch.einsum("mt,bmz->btz", m_sin_m, nA_zs) - torch.einsum(
        "mt,bmz->btz", m_cos_m, nB_zs
    )

    # Return nfp in original format for compatibility
    nfp_return = n_field_periods.view(batch_size, 1, 1) if is_tensor_nfp else nfp_scalar

    return {
        "R": R,
        "Z": Z,
        "R_t": R_t,
        "R_z": R_z,
        "Z_t": Z_t,
        "Z_z": Z_z,
        "R_tt": R_tt,
        "R_zz": R_zz,
        "R_tz": R_tz,
        "Z_tt": Z_tt,
        "Z_zz": Z_zz,
        "Z_tz": Z_tz,
        "nfp": nfp_return,
    }


def elongation(
    r_cos: Float[torch.Tensor, "batch mpol_plus_1 two_ntor_plus_1"],
    z_sin: Float[torch.Tensor, "batch mpol_plus_1 two_ntor_plus_1"],
    n_field_periods: int | Float[torch.Tensor, "batch"],
    n_theta: int = 64,
    n_zeta: int = 64,
) -> Float[torch.Tensor, "batch"]:
    """
    Compute the maximum elongation of poloidal cross-sections.

    Method:
        1. Compute R, Z grid.
        2. Center the cross-sections at each zeta.
        3. Compute covariance matrix of (R, Z) points.
        4. Elongation = sqrt(lambda_max / lambda_min).

    Returns:
        Tensor of shape (Batch,) with max elongation.
    """
    d = _compute_derivatives(r_cos, z_sin, n_field_periods, n_theta, n_zeta)
    R, Z = d["R"], d["Z"]  # (B, T, Z)

    # Center per zeta slice
    R_mean = torch.mean(R, dim=1, keepdim=True)
    Z_mean = torch.mean(Z, dim=1, keepdim=True)

    Rc = R - R_mean
    Zc = Z - Z_mean

    # Covariance (B, Z)
    var_R = torch.mean(Rc**2, dim=1)
    var_Z = torch.mean(Zc**2, dim=1)
    cov_RZ = torch.mean(Rc * Zc, dim=1)

    # Eigenvalues of 2x2 covariance matrix: [[var_R, cov_RZ], [cov_RZ, var_Z]]
    # Eigenvalues: lambda = (tr ± sqrt(tr² - 4*det)) / 2
    tr = var_R + var_Z
    det = var_R * var_Z - cov_RZ**2

    # Numerical stability: prevent sqrt of negative values from floating-point error
    # (det can exceed tr²/4 slightly due to numerical noise)
    discriminant = torch.clamp(tr**2 - 4 * det, min=0.0)
    sqrt_discriminant = torch.sqrt(discriminant)

    # Compute l1 (larger eigenvalue) directly
    l1 = (tr + sqrt_discriminant) / 2

    # NUMERICAL STABILITY FIX:
    # Instead of l2 = (tr - sqrt_discriminant) / 2, which suffers from catastrophic
    # cancellation when tr ≈ sqrt_discriminant (near-circular cross-sections),
    # we use: l2 = det / l1 (since l1 * l2 = det)
    # This is stable because l1 is always well-conditioned.
    l1_safe = torch.clamp(l1, min=1e-10)
    l2 = det / l1_safe

    # Detect degenerate cross-sections (point-like, near-zero variance in both axes)
    # These should return elongation = 1.0 (isotropic/circular assumption)
    MIN_TRACE = 1e-6
    is_degenerate = tr < MIN_TRACE

    # For elongation, we need sqrt(l1/l2) = sqrt(l1² / det)
    # This avoids issues when l2 is very small
    l2_safe = torch.clamp(l2, min=1e-10)

    # Elongation per slice
    elo_slice = torch.sqrt(l1_safe / l2_safe)

    # Override degenerate cases with 1.0 (circular cross-section)
    elo_slice = torch.where(is_degenerate, torch.ones_like(elo_slice), elo_slice)

    # Max over zeta
    return torch.max(elo_slice, dim=1)[0]


def elongation_isoperimetric(
    r_cos: Float[torch.Tensor, "batch mpol_plus_1 two_ntor_plus_1"],
    z_sin: Float[torch.Tensor, "batch mpol_plus_1 two_ntor_plus_1"],
    n_field_periods: int | Float[torch.Tensor, "batch"],
    n_theta: int = 64,
    n_zeta: int = 64,
) -> Float[torch.Tensor, "batch"]:
    """
    Compute elongation via isoperimetric quotient (physics-corrected).

    This provides a differentiable approximation that better matches the
    benchmark's ellipse-fitting approach than the covariance method.

    Physics Correction (B5):
        The original elongation() uses covariance eigenvalues which measure
        statistical spread, not geometric extent. For non-elliptic shapes
        (common in stellarators), this can underestimate elongation by ~25%.

        The benchmark fits an ellipse matching both perimeter AND area.
        This function uses the isoperimetric quotient Q = 4πA/P² as a proxy:
            - For circle: Q = 1, elongation = 1
            - For ellipse with κ = a/b: Q < 1, elongation > 1

        The relationship Q ↔ κ is monotonic, enabling a differentiable mapping.

    Returns:
        Tensor of shape (Batch,) with max elongation over zeta slices.
    """
    d = _compute_derivatives(r_cos, z_sin, n_field_periods, n_theta, n_zeta)
    R, Z = d["R"], d["Z"]
    R_t, Z_t = d["R_t"], d["Z_t"]

    n_theta_pts = R.shape[1]
    dtheta = 2 * torch.pi / n_theta_pts

    # Perimeter per zeta slice: P = ∫ ds = ∫ sqrt(R_θ² + Z_θ²) dθ
    ds = torch.sqrt(R_t**2 + Z_t**2 + 1e-8)  # (B, T, Z)
    perimeter = torch.sum(ds, dim=1) * dtheta  # (B, Z)

    # Area via Green's theorem: A = 0.5 * |∮ (R dZ - Z dR)|
    integrand = R * Z_t - Z * R_t
    area = 0.5 * torch.abs(torch.sum(integrand, dim=1) * dtheta)  # (B, Z)

    # Isoperimetric quotient: Q = 4πA/P²
    # Q = 1 for circle, Q < 1 for non-circular shapes
    Q = 4 * torch.pi * area / (perimeter**2 + 1e-8)

    # Clamp Q to valid range [epsilon, 1] for numerical stability
    # Q > 1 can occur due to numerical errors; Q << 1 indicates extreme shapes
    Q = torch.clamp(Q, min=0.05, max=1.0)

    # Map Q to elongation using ellipse relationship
    # For an ellipse with semi-axes a, b (a ≥ b), elongation κ = a/b:
    #   Q = π² / (4κ E(m)²)  where m = 1 - 1/κ², E = complete elliptic integral
    #
    # The relationship Q ↔ κ is monotonic. We derive an analytical approximation
    # by noting that for moderate elongations:
    #   E(m)² ≈ (π/2)² × (1/2 + 1/(2κ²))
    #
    # This gives: Q ≈ 2κ / (κ² + 1)
    # Inverting: κ = (1 + √(1 - Q²)) / Q
    #
    # This base formula has ~16% max error for κ ∈ [1, 6].
    # We apply an empirically-calibrated correction factor:
    #   κ_corrected = κ_base × (1 + 0.333 × (1 - Q))
    #
    # This reduces max error to ~3.6% across the range κ ∈ [1, 6].

    # Compute discriminant safely (avoid sqrt of negative due to numerical noise)
    discriminant = torch.clamp(1.0 - Q**2, min=1e-8)

    # Base formula: κ = (1 + √(1-Q²)) / Q
    kappa_base = (1.0 + torch.sqrt(discriminant)) / Q

    # Empirical correction (calibrated against exact ellipse perimeter integrals)
    correction = 1.0 + 0.333 * (1.0 - Q)
    elongation_per_slice = kappa_base * correction

    # Max over zeta slices
    return torch.max(elongation_per_slice, dim=1)[0]


def aspect_ratio(
    r_cos: Float[torch.Tensor, "batch mpol_plus_1 two_ntor_plus_1"],
    z_sin: Float[torch.Tensor, "batch mpol_plus_1 two_ntor_plus_1"],
    n_field_periods: int | Float[torch.Tensor, "batch"],
    n_theta: int = 64,
    n_zeta: int = 64,
) -> Float[torch.Tensor, "batch"]:
    """
    Compute Aspect Ratio = R_major / r_minor_eff.

    R_major: Mean of R along the magnetic axis (geometric center at theta=0),
             averaged over all toroidal angles zeta. This correctly accounts
             for all toroidal mode contributions (n != 0).

    r_minor_eff: Average effective radius sqrt(Area/pi) where Area is computed
                 using Green's theorem: Area = 0.5 * int (R * Z_theta - Z * R_theta) dtheta

    Note: Using R(m=0, n=0) coefficient directly would be incorrect for stellarators
    with significant n != 0 modes, as it ignores the toroidal oscillations.
    """
    d = _compute_derivatives(r_cos, z_sin, n_field_periods, n_theta, n_zeta)
    R, Z = d["R"], d["Z"]
    R_t, Z_t = d["R_t"], d["Z_t"]  # d/dtheta

    # R_major: Average R at the geometric center of each cross-section
    # The geometric center at each zeta is approximately R_mean over theta
    # A more precise definition: mean of R along theta=0 (outboard midplane)
    # For a centered stellarator, this equals the average R over the surface
    # We use the mean over all (theta, zeta) as a robust approximation
    R_center_per_zeta = torch.mean(R, dim=1)  # Mean over theta -> (B, n_zeta)
    R_major = torch.mean(R_center_per_zeta, dim=1)  # Mean over zeta -> (B,)

    # Area = 0.5 * integral (R dZ - Z dR)
    #      = 0.5 * integral (R Z_t - Z R_t) dtheta
    # Discrete integral: mean(...) * 2pi
    integrand = R * Z_t - Z * R_t
    cross_section_area = 0.5 * torch.mean(integrand, dim=1) * 2 * torch.pi

    # Ensure positive area and compute effective minor radius
    r_minor = torch.sqrt(torch.abs(cross_section_area) / torch.pi)
    mean_r_minor = torch.mean(r_minor, dim=1)

    # Clamp minor radius to avoid division by zero
    mean_r_minor = torch.clamp(mean_r_minor, min=1e-6)

    return R_major / mean_r_minor


def aspect_ratio_arc_length(
    r_cos: Float[torch.Tensor, "batch mpol_plus_1 two_ntor_plus_1"],
    z_sin: Float[torch.Tensor, "batch mpol_plus_1 two_ntor_plus_1"],
    n_field_periods: int | Float[torch.Tensor, "batch"],
    n_theta: int = 64,
    n_zeta: int = 64,
) -> Float[torch.Tensor, "batch"]:
    """
    Compute Aspect Ratio with arc-length weighted centroid (physics-corrected).

    This fixes the parametric bias in the original aspect_ratio() function.

    Physics Correction (B4):
        The original implementation uses mean(R(θ)) which assumes uniform θ
        corresponds to uniform arc-length. For non-circular cross-sections,
        this is incorrect and introduces ~5% bias.

        Correct definition:
            R_centroid = ∫ R ds / ∫ ds
        where ds = sqrt(R_θ² + Z_θ²) dθ is the arc-length element.

    Returns:
        Tensor of shape (Batch,) with aspect ratio = R_major / r_minor.
    """
    d = _compute_derivatives(r_cos, z_sin, n_field_periods, n_theta, n_zeta)
    R, Z = d["R"], d["Z"]
    R_t, Z_t = d["R_t"], d["Z_t"]  # d/dtheta

    # Arc-length element: ds = sqrt(R_θ² + Z_θ²)
    # This weights regions with high curvature appropriately
    ds = torch.sqrt(R_t**2 + Z_t**2 + 1e-8)  # (B, T, Z)

    # Arc-length weighted centroid per zeta slice
    # R_centroid = ∫ R ds / ∫ ds
    R_weighted = R * ds
    R_centroid_per_zeta = torch.sum(R_weighted, dim=1) / torch.sum(ds, dim=1)  # (B, Z)
    R_major = torch.mean(R_centroid_per_zeta, dim=1)  # (B,)

    # Minor radius via Green's theorem (unchanged - mathematically correct)
    # Area = 0.5 * ∮ (R dZ - Z dR) = 0.5 * ∫ (R Z_θ - Z R_θ) dθ
    integrand = R * Z_t - Z * R_t
    cross_section_area = 0.5 * torch.mean(integrand, dim=1) * 2 * torch.pi

    # Effective minor radius: r = sqrt(Area / π)
    r_minor = torch.sqrt(torch.abs(cross_section_area) / torch.pi)
    mean_r_minor = torch.mean(r_minor, dim=1)

    # Clamp minor radius to avoid division by zero
    mean_r_minor = torch.clamp(mean_r_minor, min=1e-6)

    return R_major / mean_r_minor


def mean_curvature(
    r_cos: Float[torch.Tensor, "batch mpol_plus_1 two_ntor_plus_1"],
    z_sin: Float[torch.Tensor, "batch mpol_plus_1 two_ntor_plus_1"],
    n_field_periods: int | Float[torch.Tensor, "batch"],
    n_theta: int = 64,
    n_zeta: int = 64,
) -> Float[torch.Tensor, "batch"]:
    """
    Compute the Mean Curvature (H) of the surface.

    Returns:
        Scalar metric per batch: Mean of |H| over the surface.
    """
    d = _compute_derivatives(r_cos, z_sin, n_field_periods, n_theta, n_zeta)

    R = d["R"]
    R_t, R_z = d["R_t"], d["R_z"]
    Z_t, Z_z = d["Z_t"], d["Z_z"]
    R_tt, R_zz, R_tz = d["R_tt"], d["R_zz"], d["R_tz"]
    Z_tt, Z_zz, Z_tz = d["Z_tt"], d["Z_zz"], d["Z_tz"]

    # First Fundamental Form
    E = R_t**2 + Z_t**2
    F = R_t * R_z + Z_t * Z_z
    G = R_z**2 + R**2 + Z_z**2

    EG_F2 = E * G - F**2

    # Normal vector n (unnormalized)
    n_R = -R * Z_t
    n_phi = Z_t * R_z - R_t * Z_z
    n_Z = R * R_t

    norm_n = torch.sqrt(n_R**2 + n_phi**2 + n_Z**2 + 1e-8)

    # Second Fundamental Form coefficients L, M, N
    # L = r_tt . n / |n|
    L_dot = R_tt * n_R + Z_tt * n_Z

    # M = r_tz . n
    M_dot = R_tz * n_R + R_t * n_phi + Z_tz * n_Z

    # N = r_zz . n
    N_dot = (R_zz - R) * n_R + (2 * R_z) * n_phi + Z_zz * n_Z

    L = L_dot / norm_n
    M = M_dot / norm_n
    N = N_dot / norm_n

    # Mean Curvature H
    numerator = E * N + G * L - 2 * F * M
    denominator = 2 * EG_F2

    H = numerator / (denominator + 1e-8)

    # Return mean absolute curvature
    return torch.mean(torch.abs(H), dim=(1, 2))


def surface_area(
    r_cos: Float[torch.Tensor, "batch mpol_plus_1 two_ntor_plus_1"],
    z_sin: Float[torch.Tensor, "batch mpol_plus_1 two_ntor_plus_1"],
    n_field_periods: int | Float[torch.Tensor, "batch"],
    n_theta: int = 64,
    n_zeta: int = 64,
) -> Float[torch.Tensor, "batch"]:
    """
    Compute total surface area.
    """
    d = _compute_derivatives(r_cos, z_sin, n_field_periods, n_theta, n_zeta)

    R = d["R"]
    R_t, R_z = d["R_t"], d["R_z"]
    Z_t, Z_z = d["Z_t"], d["Z_z"]

    n_R = -R * Z_t
    n_phi = Z_t * R_z - R_t * Z_z
    n_Z = R * R_t

    norm_n = torch.sqrt(n_R**2 + n_phi**2 + n_Z**2 + 1e-8)

    nfp = d["nfp"]
    dt = (2 * torch.pi) / n_theta

    # Handle integration range
    if isinstance(nfp, torch.Tensor):
        # We integrated over [0, 2pi] (Full Torus)
        dz = (2 * torch.pi) / n_zeta
        area_total = torch.sum(norm_n * dt * dz, dim=(1, 2))
        return area_total
    else:
        # We integrated over [0, 2pi/nfp] (One Period)
        dz = (2 * torch.pi / nfp) / n_zeta
        area_per_period = torch.sum(norm_n * dt * dz, dim=(1, 2))
        return area_per_period * nfp


def check_self_intersection(
    r_cos: Float[torch.Tensor, "batch mpol_plus_1 two_ntor_plus_1"],
    z_sin: Float[torch.Tensor, "batch mpol_plus_1 two_ntor_plus_1"],
    n_field_periods: int | Float[torch.Tensor, "batch"],
    n_theta: int = 64,
    n_zeta: int = 64,
) -> Float[torch.Tensor, "batch"]:
    """Check if surface self-intersects.

    Detection methods:
    1. Jacobian sign flip: The surface Jacobian J = |∂r/∂θ × ∂r/∂ζ| should be
       positive everywhere. A sign change indicates the surface folds back.
    2. Cross-section overlap: At each ζ, the poloidal curve (R(θ), Z(θ))
       should not cross itself (convexity proxy via signed area).

    Args:
        r_cos, z_sin: Fourier coefficients (Batch, mpol+1, 2*ntor+1).
        n_field_periods: Number of field periods.
        n_theta, n_zeta: Grid resolution.

    Returns:
        Tensor of shape (Batch,) with values:
            0.0 = No self-intersection detected
            1.0 = Jacobian sign flip detected
            2.0 = Cross-section overlap detected
        Values > 0 indicate self-intersection.
    """
    d = _compute_derivatives(r_cos, z_sin, n_field_periods, n_theta, n_zeta)

    R = d["R"]  # (B, T, Z)
    Z = d["Z"]  # (B, T, Z)
    R_t, R_z = d["R_t"], d["R_z"]
    Z_t, Z_z = d["Z_t"], d["Z_z"]

    batch_size = R.shape[0]
    device = R.device

    result = torch.zeros(batch_size, device=device)

    # =========================================================================
    # Method 1: Jacobian sign flip detection
    # =========================================================================
    # The Jacobian of the surface parameterization is |∂r/∂θ × ∂r/∂ζ|
    # For stellarator surfaces, the normal vector n = ∂r/∂θ × ∂r/∂ζ
    # should point consistently outward (or inward). A sign flip indicates folding.

    # Normal vector components (cross product in cylindrical coords)
    _n_R = -R * Z_t  # noqa: F841 - computed for documentation clarity
    n_phi = Z_t * R_z - R_t * Z_z
    _n_Z = R * R_t  # noqa: F841 - computed for documentation clarity

    # Use the phi-component of normal as a proxy for Jacobian sign
    # For a proper stellarator surface, n_phi should maintain consistent sign
    # across the entire surface (all positive or all negative).

    n_phi_min = torch.min(n_phi.view(batch_size, -1), dim=1)[0]  # (B,)
    n_phi_max = torch.max(n_phi.view(batch_size, -1), dim=1)[0]  # (B,)

    # Sign flip detected if min < 0 and max > 0 (with some tolerance)
    tolerance = 0.01
    has_sign_flip = (n_phi_min < -tolerance) & (n_phi_max > tolerance)
    result = torch.where(has_sign_flip, torch.ones_like(result), result)

    # =========================================================================
    # Method 2: Cross-section overlap (convexity proxy)
    # =========================================================================
    # For each zeta slice, the poloidal curve (R(θ), Z(θ)) should not cross itself.
    # A simple proxy: check if the curve's signed area is consistent.
    # Self-intersecting curves (figure-8) have regions of opposite signed area.

    # Discrete signed area via shoelace formula per zeta slice
    # Area = 0.5 * sum(R[i] * Z[i+1] - R[i+1] * Z[i])
    # For a non-self-intersecting closed curve, this should be single-signed.

    # Roll to get next point in theta direction
    R_next = torch.roll(R, shifts=-1, dims=1)  # (B, T, Z)
    Z_next = torch.roll(Z, shifts=-1, dims=1)

    # Incremental signed area contribution per segment
    dA = R * Z_next - R_next * Z  # (B, T, Z)

    # Sum over theta gives total signed area per zeta slice
    signed_area = torch.sum(dA, dim=1)  # (B, Z)

    # Check for sign changes in signed area across zeta
    # This catches cases where the cross-section "flips" orientation
    area_min = torch.min(signed_area, dim=1)[0]  # (B,)
    area_max = torch.max(signed_area, dim=1)[0]  # (B,)

    # Normalize by mean area to make threshold scale-invariant
    area_mean = torch.mean(torch.abs(signed_area), dim=1)  # (B,)
    area_variation = (area_max - area_min) / (area_mean + 1e-8)

    # Large variation (> 1.0) suggests cross-section issues
    has_overlap = area_variation > 1.0
    result = torch.where(
        has_overlap & (result == 0), 2.0 * torch.ones_like(result), result
    )

    return result
