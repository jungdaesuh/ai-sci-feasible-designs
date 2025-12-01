import numpy as np
import torch

from ai_scientist.optim import geometry


def test_geometry_metrics_run_and_differentiate():
    """
    Test that elongation, curvature, aspect_ratio run and have gradients.
    """
    batch_size = 2
    mpol = 3
    ntor = 3

    # Create random coefficients
    r_cos = torch.randn(
        batch_size, mpol + 1, 2 * ntor + 1, requires_grad=True, dtype=torch.float32
    )
    z_sin = torch.randn(
        batch_size, mpol + 1, 2 * ntor + 1, requires_grad=True, dtype=torch.float32
    )

    # Make them somewhat realistic to avoid NaNs (Major radius > minor radius)
    with torch.no_grad():
        r_cos[:, 0, ntor] = 5.0  # Major radius
        r_cos[:, 1, ntor] = 1.0  # Minor radius
        z_sin[:, 1, ntor] = 1.0

    nfp = 3

    # 1. Elongation
    elo = geometry.elongation(r_cos, z_sin, n_field_periods=nfp)
    assert elo.shape == (batch_size,)
    assert torch.all(elo > 0)

    loss_elo = torch.sum(elo)
    loss_elo.backward()

    assert r_cos.grad is not None
    assert z_sin.grad is not None

    # Reset grads
    r_cos.grad.zero_()
    z_sin.grad.zero_()

    # 2. Aspect Ratio
    ar = geometry.aspect_ratio(r_cos, z_sin, n_field_periods=nfp)
    assert ar.shape == (batch_size,)

    loss_ar = torch.sum(ar)
    loss_ar.backward()
    assert r_cos.grad is not None

    r_cos.grad.zero_()
    z_sin.grad.zero_()

    # 3. Mean Curvature
    # We expect curvature to be non-negative (mean abs curvature)
    curv = geometry.mean_curvature(r_cos, z_sin, n_field_periods=nfp)
    assert curv.shape == (batch_size,)
    assert torch.all(curv >= 0)

    loss_curv = torch.sum(curv)
    loss_curv.backward()
    assert r_cos.grad is not None

    r_cos.grad.zero_()

    # 4. Surface Area
    area = geometry.surface_area(r_cos, z_sin, n_field_periods=nfp)
    assert area.shape == (batch_size,)
    assert torch.all(area > 0)

    loss_area = torch.sum(area)
    loss_area.backward()
    assert r_cos.grad is not None


def test_geometry_metrics_batched_nfp():
    """Test with variable nfp per batch item."""
    batch_size = 3
    mpol = 2
    ntor = 2
    r_cos = torch.randn(batch_size, mpol + 1, 2 * ntor + 1)
    z_sin = torch.randn(batch_size, mpol + 1, 2 * ntor + 1)
    nfp = torch.tensor([1.0, 3.0, 5.0])

    elo = geometry.elongation(r_cos, z_sin, n_field_periods=nfp)
    assert elo.shape == (batch_size,)

    # Add robustness test: aspect ratio with tilted cross section
    # Circle: R = R0 + a cos t, Z = a sin t. Area = pi a^2.
    # Tilted: R = R0 + a cos(t+d), Z = a sin(t+d). Same area.
    # Stretched and Tilted:
    # x = a cos t, y = b sin t. Rotated by alpha.
    # R = R0 + (a cos t cos a - b sin t sin a)
    # Z = (a cos t sin a + b sin t cos a)
    # Area should be pi * a * b.

    R0 = 5.0
    a = 1.0
    alpha = np.pi / 4  # 45 deg

    # Construct manually
    # cos(t) term:
    # R: a cos a. Z: a sin a.
    # sin(t) term:
    # R: -b sin a. Z: b cos a.

    rc = torch.zeros(1, mpol + 1, 2 * ntor + 1)
    zs = torch.zeros(1, mpol + 1, 2 * ntor + 1)
    nt = ntor

    rc[0, 0, nt] = R0  # m=0, n=0

    # m=1, n=0
    rc[0, 1, nt] = a * np.cos(alpha)
    zs[0, 1, nt] = a * np.sin(alpha)

    # We need -b sin t. But fourier basis is usually +sin t.
    # So coeff is -b sin a.
    # Wait, z_sin corresponds to sin terms?
    # R = sum rc cos(mt). Z = sum zs sin(mt).
    # We need R to have sin(t) term? Not supported by R_cos basis usually?
    # R_cos basis assumes stellarator symmetry (R is even in theta, Z is odd in theta).
    # R(theta, zeta) = sum R_mn cos(m theta - n N zeta).
    # Z(theta, zeta) = sum Z_mn sin(m theta - n N zeta).
    # If stellarator symmetry holds, we cannot have tilted ellipses in the R-Z plane at phi=0.
    # Tilted ellipses imply broken stellarator symmetry (R has sin terms, Z has cos terms).
    # The current code `SurfaceRZFourier` (and my code) generally assumes `r_cos` and `z_sin`.
    # Wait, my `fourier_to_real_space` ONLY uses `r_cos` and `z_sin`.
    # So it strictly enforces stellarator symmetry at phi=0 (for m terms).
    # So we cannot represent a tilted ellipse at phi=0 with just r_cos, z_sin.
    # So the "Area bias" issue for tilted plasmas might only manifest if we had `r_sin`, `z_cos`.
    # BUT, `angle = m*theta - n*zeta`.
    # If n != 0, we have `cos(theta - zeta) = cos t cos z + sin t sin z`.
    # So at zeta != 0, we DO have sin(theta) terms.
    # So cross-sections can be tilted at zeta != 0.
    # The aspect ratio averages area over zeta.
    # So correct area formula is needed.

    # Let's try a mode that induces tilt at some zeta.
    # m=1, n=1.
    # R ~ cos(t - z). Z ~ sin(t - z).
    # at z=pi/4: R ~ cos(t-pi/4), Z ~ sin(t-pi/4).
    # This is just rotated phase, still a circle.
    # Ellipse: R ~ 2 cos(t-z), Z ~ sin(t-z).
    # z=0: R=2cos t, Z=sin t. (Standard ellipse).
    # z=pi/2: R=2sin t, Z=-cos t. (Rotated 90 deg).
    # Area is always pi * 2 * 1 = 2pi.

    rc[0, 1, nt + 1] = 2.0  # m=1, n=1
    zs[0, 1, nt + 1] = 1.0  # m=1, n=1
    rc[0, 0, nt] = 5.0  # R0

    # Calc area
    ar_val = geometry.aspect_ratio(rc, zs, n_field_periods=1)
    assert ar_val > 0


def test_nfp_dependence():
    """
    Regression test for [P1]: Ensure that different nfp values produce
    different geometries when nfp is passed as a batch tensor.

    Previous bug: normalizing zeta grid by nfp canceled out nfp in the phase argument,
    making the surface independent of nfp.
    """
    mpol = 2
    ntor = 2
    nt = ntor

    # Construct a surface where N matters.
    # Base torus (m=1, n=0) + Rotating perturbation (m=2, n=1).
    # R ~ R0 + r cos(t) + delta cos(2t - Nz)
    # Z ~      r sin(t) + delta sin(2t - Nz)
    # This creates a spiraling ridge. Higher N = more spirals = higher area/curvature.

    rc = torch.zeros(2, mpol + 1, 2 * ntor + 1)
    zs = torch.zeros(2, mpol + 1, 2 * ntor + 1)

    # Identical coefficients for both batch items
    rc[:, 0, nt] = 5.0  # R0
    rc[:, 1, nt] = 1.0  # m=1, n=0 (Base circle)
    zs[:, 1, nt] = 1.0

    rc[:, 2, nt + 1] = 0.3  # m=2, n=1 (Perturbation)
    zs[:, 2, nt + 1] = 0.3

    # Different Nfp
    nfp = torch.tensor([1.0, 5.0])

    # Calculate Mean Curvature
    curv = geometry.mean_curvature(rc, zs, n_field_periods=nfp)

    # Check that the two curvatures are DIFFERENT.
    assert not torch.isclose(curv[0], curv[1], rtol=1e-3).item(), (
        f"Curvatures should differ for different Nfp. Got {curv[0]} and {curv[1]}"
    )

    # Also check Surface Area
    area = geometry.surface_area(rc, zs, n_field_periods=nfp)
    assert not torch.isclose(area[0], area[1], rtol=1e-3).item(), (
        f"Areas should differ for different Nfp. Got {area[0]} and {area[1]}"
    )
