"""Geometric pre-relaxation module (StellarForge Phase 2).

This module implements the "Geometric Pre-Relaxation" step inspired by LEGO-xtal (Ridwan et al., 2025).
It performs fast, gradient-based optimization on the Fourier boundary coefficients to minimize
geometric energy (curvature, self-intersection proxies) before submitting to expensive VMEC++ evaluations.
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import torch
from ai_scientist.optim import geometry

_LOGGER = logging.getLogger(__name__)


def geometric_energy(
    r_cos: torch.Tensor,
    z_sin: torch.Tensor,
    nfp: int | torch.Tensor,
    target_ar: float = 8.0,
) -> torch.Tensor:
    """
    Compute geometric energy for pre-relaxation.

    Energy Components:
    1. Mean Curvature (smoothness): Penalizes high curvature.
    2. Aspect Ratio: Penalizes deviation from target (e.g. 8.0).
    3. Elongation: Penalizes cross-section elongation > 10.0 (pinching).

    Args:
        r_cos, z_sin: Batched Fourier coefficients (B, m, n).
        nfp: Number of field periods.
        target_ar: Target aspect ratio.

    Returns:
        Scalar loss per batch element (B,).
    """
    # 1. Mean Curvature Penalty (Smoothness)
    # Typical values ~2-5. We want to minimize sharp edges.
    H_mean = geometry.mean_curvature(r_cos, z_sin, nfp)

    # 2. Aspect Ratio Target
    # We want to maintain the design intent (e.g. A=8).
    ar = geometry.aspect_ratio(r_cos, z_sin, nfp)
    ar_loss = (ar - target_ar) ** 2

    # 3. Elongation Constraint (Soft)
    # Avoid pinching (elongation > 10 is usually invalid).
    elo = geometry.elongation(r_cos, z_sin, nfp)
    elo_penalty = torch.relu(elo - 10.0) ** 2

    # Total Energy
    # Weights tuning: Curvature is primary (~1.0), AR maintenance (~0.5), Elongation is hard constraint (~10.0)
    loss = 1.0 * H_mean + 0.5 * ar_loss + 10.0 * elo_penalty
    return loss


def prerelax_boundary(
    boundary_params: Dict[str, Any],
    steps: int = 50,
    lr: float = 1e-2,
    target_ar: float = 8.0,
    nfp: int = 3,
    device: str = "cpu",
) -> Tuple[Dict[str, Any], float]:
    """
    Optimize boundary parameters to minimize geometric energy.

    Args:
        boundary_params: Dictionary containing "r_cos", "z_sin".
        steps: Number of optimization steps.
        lr: Learning rate.
        target_ar: Target aspect ratio to maintain.
        nfp: Number of field periods.
        device: "cpu", "cuda", or "mps".

    Returns:
        Tuple of (optimized_params, final_energy).
    """
    # Extract
    r_cos_np = np.array(boundary_params["r_cos"], dtype=np.float32)
    z_sin_np = np.array(boundary_params["z_sin"], dtype=np.float32)

    # To Torch
    r_cos = torch.tensor(r_cos_np, device=device).unsqueeze(0)  # (1, m, n)
    z_sin = torch.tensor(z_sin_np, device=device).unsqueeze(0)

    r_cos.requires_grad_(True)
    z_sin.requires_grad_(True)

    optimizer = torch.optim.Adam([r_cos, z_sin], lr=lr)

    final_loss = 0.0

    for _ in range(steps):
        optimizer.zero_grad()

        loss = geometric_energy(r_cos, z_sin, nfp, target_ar=target_ar)

        # Mean over batch (size 1)
        loss_scalar = loss.mean()
        loss_scalar.backward()
        optimizer.step()

        final_loss = loss_scalar.item()

    # Export back
    new_params = boundary_params.copy()
    new_params["r_cos"] = r_cos.detach().cpu().squeeze(0).numpy().tolist()
    new_params["z_sin"] = z_sin.detach().cpu().squeeze(0).numpy().tolist()

    return new_params, final_loss
