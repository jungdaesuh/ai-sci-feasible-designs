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
        zeta_one_period = np.linspace(
            0, 2 * pi / n_field_periods, n_zeta, endpoint=False
        )

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
            n = n_idx - ntor  # Maps index 0..2*ntor to -ntor..ntor

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
        zeta = torch.linspace(0, 2 * pi, n_zeta + 1, device=device)[:-1]  # (n_zeta,)

        # Ensure n_field_periods is (Batch, 1, 1) for broadcasting
        nfp_tensor = n_field_periods.view(batch_size, 1, 1)
    else:
        # Legacy mode: n_zeta per period
        zeta_one = torch.linspace(
            0, 2 * pi / n_field_periods, n_zeta + 1, device=device
        )[:-1]
        zeta_list = [
            zeta_one + i * (2 * pi / n_field_periods) for i in range(n_field_periods)
        ]
        zeta = torch.cat(zeta_list)  # (n_zeta * nfp,)
        nfp_tensor = float(n_field_periods)  # Scalar broadcast

    theta = torch.linspace(0, 2 * pi, n_theta + 1, device=device)[:-1]

    # (1, n_theta, 1)
    theta_grid = theta.view(1, n_theta, 1)
    # (1, 1, n_zeta_total)
    zeta_grid = zeta.view(1, 1, -1)

    R = torch.zeros(batch_size, n_theta, zeta.size(0), device=device)
    Z = torch.zeros_like(R)

    for m in range(mpol + 1):
        # m*theta term
        m_theta = m * theta_grid  # (1, T, 1)

        for n_idx in range(2 * ntor + 1):
            n = n_idx - ntor
            # n*Nfp*zeta term
            # If nfp is tensor: (B, 1, 1) * (1, 1, Z) -> (B, 1, Z)
            n_zeta_term = n * nfp_tensor * zeta_grid

            angle = m_theta - n_zeta_term  # (B, T, Z)

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
    if isinstance(n_field_periods, torch.Tensor):
        nfp = n_field_periods.view(batch_size, 1, 1)
        # Batched N_fp: Evaluate over FULL TORUS [0, 2pi] to capture winding explicitly.
        # zeta_grid corresponds to physical toroidal angle zeta.
        zeta = torch.linspace(0, 2 * torch.pi, n_zeta + 1, device=device)[:-1]
        zeta_grid = zeta.view(1, 1, -1)  # (1, 1, Z)
    else:
        nfp = float(n_field_periods)
        # Fixed N_fp: Evaluate over ONE FIELD PERIOD [0, 2pi/nfp].
        zeta = torch.linspace(0, 2 * torch.pi / nfp, n_zeta + 1, device=device)[:-1]
        zeta_grid = zeta.view(1, 1, -1)  # (1, 1, Z)

    theta = torch.linspace(0, 2 * torch.pi, n_theta + 1, device=device)[:-1]
    theta_grid = theta.view(1, n_theta, 1)  # (1, T, 1)

    # Initialize accumulators
    dims = (batch_size, n_theta, n_zeta)
    R = torch.zeros(dims, device=device)
    Z = torch.zeros(dims, device=device)
    R_t = torch.zeros(dims, device=device)
    R_z = torch.zeros(dims, device=device)
    Z_t = torch.zeros(dims, device=device)
    Z_z = torch.zeros(dims, device=device)
    R_tt = torch.zeros(dims, device=device)
    R_tz = torch.zeros(dims, device=device)
    R_zz = torch.zeros(dims, device=device)
    Z_tt = torch.zeros(dims, device=device)
    Z_tz = torch.zeros(dims, device=device)
    Z_zz = torch.zeros(dims, device=device)

    for m in range(mpol + 1):
        m_val = float(m)
        m_theta = m_val * theta_grid

        for n_idx in range(2 * ntor + 1):
            n = n_idx - ntor
            n_val = float(n)

            # angle = m*theta - n*nfp*zeta
            # If nfp is tensor (B,1,1), nfp*zeta_grid is (B,1,Z) scaled by nfp correctly.
            # Since zeta_grid is [0, 2pi], nfp*zeta_grid wraps nfp times.
            angle = m_theta - n_val * nfp * zeta_grid

            c = torch.cos(angle)
            s = torch.sin(angle)

            # Coefficients (B, 1, 1)
            rc = r_cos[:, m, n_idx].view(batch_size, 1, 1)
            zs = z_sin[:, m, n_idx].view(batch_size, 1, 1)

            # Values
            R = R + rc * c
            Z = Z + zs * s

            factor_z = -n_val * nfp

            # 1st Derivatives
            # R = sum(rc * cos(u)), u = m*t - n*N*z
            # dR/dt = sum(-rc * m * sin(u))
            # dR/dz = sum(-rc * (-n*N) * sin(u)) = sum(rc * n*N * sin(u))

            R_t = R_t - rc * m_val * s
            R_z = R_z - rc * factor_z * s

            Z_t = Z_t + zs * m_val * c
            Z_z = Z_z + zs * factor_z * c

            # 2nd Derivatives
            # d2/dt2: -m^2 cos
            # d2/dz2: -(nN)^2 cos
            # d2/dtz: -m(nN) cos

            R_tt = R_tt - rc * m_val**2 * c
            R_zz = R_zz - rc * factor_z**2 * c
            R_tz = R_tz - rc * m_val * factor_z * c

            Z_tt = Z_tt - zs * m_val**2 * s
            Z_zz = Z_zz - zs * factor_z**2 * s
            Z_tz = Z_tz - zs * m_val * factor_z * s

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
        "nfp": nfp,
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

    # Eigenvalues of 2x2 covariance matrix
    tr = var_R + var_Z
    det = var_R * var_Z - cov_RZ**2

    # Numerical stability: prevent sqrt of negative values from floating-point error
    # (det can exceed tr²/4 slightly due to numerical noise)
    gap = torch.clamp(tr**2 - 4 * det, min=0.0)
    sqrt_gap = torch.sqrt(gap)

    l1 = (tr + sqrt_gap) / 2
    l2 = (tr - sqrt_gap) / 2

    # Detect degenerate cross-sections (point-like, near-zero variance in both axes)
    # These should return elongation = 1.0 (isotropic/circular assumption)
    MIN_TRACE = 1e-6
    is_degenerate = tr < MIN_TRACE

    # Safe ratio: clamp both eigenvalues to prevent 0/0 or inf scenarios
    l1_safe = torch.clamp(l1, min=1e-8)
    l2_safe = torch.clamp(l2, min=1e-8)

    # Elongation per slice
    elo_slice = torch.sqrt(l1_safe / l2_safe)

    # Override degenerate cases with 1.0 (circular cross-section)
    elo_slice = torch.where(is_degenerate, torch.ones_like(elo_slice), elo_slice)

    # Max over zeta
    return torch.max(elo_slice, dim=1)[0]


def aspect_ratio(
    r_cos: Float[torch.Tensor, "batch mpol_plus_1 two_ntor_plus_1"],
    z_sin: Float[torch.Tensor, "batch mpol_plus_1 two_ntor_plus_1"],
    n_field_periods: int | Float[torch.Tensor, "batch"],
    n_theta: int = 64,
    n_zeta: int = 64,
) -> Float[torch.Tensor, "batch"]:
    """
    Compute Aspect Ratio = R_major / r_minor_eff.

    R_major: R(0,0) component (approximate).
    r_minor_eff: Average effective radius sqrt(Area/pi).

    Corrected to use Green's theorem symmetric form for Area:
    Area = 0.5 * int (R * Z_theta - Z * R_theta) dtheta
    """
    ntor = (r_cos.shape[2] - 1) // 2
    # Major radius approx (m=0, n=0)
    R00 = r_cos[:, 0, ntor]

    d = _compute_derivatives(r_cos, z_sin, n_field_periods, n_theta, n_zeta)
    R, Z = d["R"], d["Z"]
    R_t, Z_t = d["R_t"], d["Z_t"]  # d/dtheta

    # Area = 0.5 * integral (R dZ - Z dR)
    #      = 0.5 * integral (R Z_t - Z R_t) dtheta
    # Discrete integral: mean(...) * 2pi

    integrand = R * Z_t - Z * R_t
    cross_section_area = 0.5 * torch.mean(integrand, dim=1) * 2 * torch.pi

    # Ensure positive area
    r_minor = torch.sqrt(torch.abs(cross_section_area) / torch.pi)
    mean_r_minor = torch.mean(r_minor, dim=1)

    return R00 / mean_r_minor


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
