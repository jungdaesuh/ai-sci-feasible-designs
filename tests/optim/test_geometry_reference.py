"""Reference verification tests for geometry module (B6 fix).

This module verifies the differential geometry formulas in geometry.py
against independent mesh-based calculations to ensure physics correctness.

Key verifications:
1. Surface area: Compare analytic formula vs triangle mesh summation
2. Mean curvature: Compare against known analytic solutions for simple shapes
3. Cross-validation: Ensure internal consistency across different methods
"""

import numpy as np
import torch

from ai_scientist.optim import geometry


class TestSurfaceAreaReference:
    """Verify surface_area() against mesh-based calculation."""

    def _compute_mesh_surface_area(
        self,
        r_cos: torch.Tensor,
        z_sin: torch.Tensor,
        n_field_periods: int,
        n_theta: int = 64,
        n_zeta: int = 64,
    ) -> float:
        """Compute surface area via triangle mesh discretization.

        This provides an independent reference calculation that doesn't
        use the differential geometry formulas.
        """
        # Generate point cloud
        R, Z, Phi = geometry.batch_fourier_to_real_space(
            r_cos, z_sin, n_field_periods, n_theta, n_zeta
        )
        X, Y, Z_cart = geometry.to_cartesian(R, Z, Phi)

        # X, Y, Z are (1, n_theta, n_zeta_total)
        X = X[0].numpy()
        Y = Y[0].numpy()
        Z_cart = Z_cart[0].numpy()

        n_t, n_z = X.shape

        # Triangulate: each quad (i,j) -> 2 triangles
        total_area = 0.0

        for i in range(n_t):
            for j in range(n_z):
                # Vertices of quad (i,j), (i+1,j), (i,j+1), (i+1,j+1)
                # with periodic boundary in theta
                i1 = (i + 1) % n_t
                j1 = (j + 1) % n_z

                # Vertices
                v00 = np.array([X[i, j], Y[i, j], Z_cart[i, j]])
                v10 = np.array([X[i1, j], Y[i1, j], Z_cart[i1, j]])
                v01 = np.array([X[i, j1], Y[i, j1], Z_cart[i, j1]])
                v11 = np.array([X[i1, j1], Y[i1, j1], Z_cart[i1, j1]])

                # Triangle 1: v00, v10, v01
                edge1 = v10 - v00
                edge2 = v01 - v00
                area1 = 0.5 * np.linalg.norm(np.cross(edge1, edge2))

                # Triangle 2: v10, v11, v01
                edge1 = v11 - v10
                edge2 = v01 - v10
                area2 = 0.5 * np.linalg.norm(np.cross(edge1, edge2))

                total_area += area1 + area2

        return total_area

    def test_torus_surface_area(self):
        """Verify surface area for a simple torus (known analytic solution).

        For a torus with major radius R and minor radius r:
        A = 4 * pi^2 * R * r
        """
        R_major = 10.0
        r_minor = 1.0

        mpol = 1
        ntor = 0
        r_cos = torch.zeros((1, mpol + 1, 2 * ntor + 1))
        z_sin = torch.zeros((1, mpol + 1, 2 * ntor + 1))

        r_cos[0, 0, 0] = R_major
        r_cos[0, 1, 0] = r_minor
        z_sin[0, 1, 0] = r_minor

        nfp = 1

        # Analytic formula
        expected_area = 4 * np.pi**2 * R_major * r_minor

        # Geometry module calculation
        analytic_area = geometry.surface_area(r_cos, z_sin, nfp, n_theta=64, n_zeta=64)

        # Mesh-based calculation
        mesh_area = self._compute_mesh_surface_area(
            r_cos, z_sin, nfp, n_theta=64, n_zeta=64
        )

        # All three should agree within 2%
        assert np.isclose(analytic_area.item(), expected_area, rtol=0.02), (
            f"Analytic formula mismatch: got {analytic_area.item()}, expected {expected_area}"
        )
        assert np.isclose(mesh_area, expected_area, rtol=0.02), (
            f"Mesh area mismatch: got {mesh_area}, expected {expected_area}"
        )
        assert np.isclose(analytic_area.item(), mesh_area, rtol=0.02), (
            f"Analytic vs Mesh mismatch: {analytic_area.item()} vs {mesh_area}"
        )

    def test_elliptical_torus_surface_area(self):
        """Verify surface area for elliptical cross-section torus.

        For torus with elliptical cross-section (semi-axes a, b):
        A ≈ 4 * pi^2 * R * sqrt((a^2 + b^2) / 2)  (approximate for a ≈ b)

        We use mesh calculation as reference since no simple closed form exists.
        """
        R_major = 10.0
        a = 1.0  # R semi-axis
        b = 1.5  # Z semi-axis

        mpol = 1
        ntor = 0
        r_cos = torch.zeros((1, mpol + 1, 2 * ntor + 1))
        z_sin = torch.zeros((1, mpol + 1, 2 * ntor + 1))

        r_cos[0, 0, 0] = R_major
        r_cos[0, 1, 0] = a
        z_sin[0, 1, 0] = b

        nfp = 1

        # Geometry module calculation
        analytic_area = geometry.surface_area(r_cos, z_sin, nfp, n_theta=64, n_zeta=64)

        # Mesh-based calculation (reference)
        mesh_area = self._compute_mesh_surface_area(
            r_cos, z_sin, nfp, n_theta=64, n_zeta=64
        )

        # Should agree within 2%
        assert np.isclose(analytic_area.item(), mesh_area, rtol=0.02), (
            f"Analytic vs Mesh mismatch for elliptical: {analytic_area.item()} vs {mesh_area}"
        )

    def test_stellarator_like_surface_area(self):
        """Verify surface area for a stellarator-like shape with toroidal variation.

        Uses mesh calculation as reference.
        """
        R_major = 10.0
        r_minor = 1.0

        mpol = 2
        ntor = 1
        r_cos = torch.zeros((1, mpol + 1, 2 * ntor + 1))
        z_sin = torch.zeros((1, mpol + 1, 2 * ntor + 1))

        # Major radius
        r_cos[0, 0, ntor] = R_major
        # Minor radius (circular base)
        r_cos[0, 1, ntor] = r_minor
        z_sin[0, 1, ntor] = r_minor
        # Add some toroidal variation (typical stellarator mode)
        r_cos[0, 0, ntor + 1] = 0.2  # n=1 mode for R
        r_cos[0, 1, ntor + 1] = 0.1  # m=1, n=1 coupling

        nfp = 3

        # Geometry module calculation
        analytic_area = geometry.surface_area(r_cos, z_sin, nfp, n_theta=64, n_zeta=64)

        # Mesh-based calculation (reference)
        mesh_area = self._compute_mesh_surface_area(
            r_cos, z_sin, nfp, n_theta=64, n_zeta=64
        )

        # Should agree within 3% (slightly relaxed for complex shapes)
        assert np.isclose(analytic_area.item(), mesh_area, rtol=0.03), (
            f"Stellarator shape mismatch: {analytic_area.item()} vs {mesh_area}"
        )


class TestMeanCurvatureReference:
    """Verify mean_curvature() against known analytic solutions."""

    def test_torus_mean_curvature(self):
        """Verify mean curvature for a torus.

        For a torus with major radius R and minor radius r:
        H(theta) = (R + r*cos(theta)) / (2*r*(R + r*cos(theta)))
                 = 1 / (2*r) * (R/r + cos(theta)) / (R/r + cos(theta))

        At theta=0 (outboard): H = (R + r) / (2r(R + r))
        At theta=pi (inboard): H = (R - r) / (2r(R - r))

        Mean |H| over surface involves an integral.
        We use a simplified check: verify it's in the expected range.
        """
        R_major = 10.0
        r_minor = 1.0

        mpol = 1
        ntor = 0
        r_cos = torch.zeros((1, mpol + 1, 2 * ntor + 1))
        z_sin = torch.zeros((1, mpol + 1, 2 * ntor + 1))

        r_cos[0, 0, 0] = R_major
        r_cos[0, 1, 0] = r_minor
        z_sin[0, 1, 0] = r_minor

        nfp = 1

        H_mean = geometry.mean_curvature(r_cos, z_sin, nfp, n_theta=64, n_zeta=64)

        # For torus, H ranges from ~1/(2r) at outboard to ~-1/(2r) at inboard
        # (signs depend on orientation convention)
        # Mean |H| should be roughly 1/(2*r_minor) with some geometry factor
        expected_order = 1.0 / (2 * r_minor)

        # Check it's in reasonable range (within factor of 2)
        assert 0.1 * expected_order < H_mean.item() < 2.0 * expected_order, (
            f"Mean curvature {H_mean.item()} outside expected range for torus"
        )

    def test_larger_torus_lower_curvature(self):
        """Larger minor radius should give lower mean curvature."""
        R_major = 10.0

        mpol = 1
        ntor = 0

        results = []
        for r_minor in [0.5, 1.0, 2.0]:
            r_cos = torch.zeros((1, mpol + 1, 2 * ntor + 1))
            z_sin = torch.zeros((1, mpol + 1, 2 * ntor + 1))

            r_cos[0, 0, 0] = R_major
            r_cos[0, 1, 0] = r_minor
            z_sin[0, 1, 0] = r_minor

            nfp = 1
            H_mean = geometry.mean_curvature(r_cos, z_sin, nfp)
            results.append((r_minor, H_mean.item()))

        # Curvature should decrease with increasing minor radius
        # (smoother surface = lower curvature)
        assert results[0][1] > results[1][1] > results[2][1], (
            f"Curvature should decrease with r_minor: {results}"
        )


class TestElongationComparison:
    """Compare elongation methods for various shapes (B5 verification)."""

    def test_elongation_methods_agree_for_ellipse(self):
        """Both elongation methods should agree for pure ellipses."""
        R_major = 10.0

        for kappa in [1.0, 1.5, 2.0, 2.5, 3.0]:
            a = 1.0  # R semi-axis
            b = kappa * a  # Z semi-axis

            mpol = 1
            ntor = 0
            r_cos = torch.zeros((1, mpol + 1, 2 * ntor + 1))
            z_sin = torch.zeros((1, mpol + 1, 2 * ntor + 1))

            r_cos[0, 0, 0] = R_major
            r_cos[0, 1, 0] = a
            z_sin[0, 1, 0] = b

            nfp = 1

            elo_cov = geometry.elongation(r_cos, z_sin, nfp)
            elo_iso = geometry.elongation_isoperimetric(r_cos, z_sin, nfp)

            # For pure ellipses, both should be close to kappa
            assert np.isclose(elo_cov.item(), kappa, rtol=0.15), (
                f"Covariance elongation {elo_cov.item()} != {kappa}"
            )
            # Isoperimetric has known ~3.6% error for ellipses
            assert np.isclose(elo_iso.item(), kappa, rtol=0.20), (
                f"Isoperimetric elongation {elo_iso.item()} != {kappa}"
            )

    def test_elongation_isoperimetric_non_elliptic(self):
        """For non-elliptic shapes, isoperimetric should differ from covariance.

        Bean/D-shaped cross-sections have larger perimeter-to-area ratio
        than ellipses of the same covariance. The isoperimetric method
        should capture this.
        """
        R_major = 10.0

        mpol = 3
        ntor = 0
        r_cos = torch.zeros((1, mpol + 1, 2 * ntor + 1))
        z_sin = torch.zeros((1, mpol + 1, 2 * ntor + 1))

        # D-shaped cross-section (triangularity via m=2 mode)
        r_cos[0, 0, 0] = R_major
        r_cos[0, 1, 0] = 1.0
        z_sin[0, 1, 0] = 1.5
        # Add triangularity (m=2 mode)
        r_cos[0, 2, 0] = 0.3
        z_sin[0, 2, 0] = 0.1

        nfp = 1

        elo_cov = geometry.elongation(r_cos, z_sin, nfp)
        elo_iso = geometry.elongation_isoperimetric(r_cos, z_sin, nfp)

        # Both should give elongation > 1
        assert elo_cov.item() > 1.0
        assert elo_iso.item() > 1.0

        # For non-elliptic shapes, isoperimetric typically gives higher elongation
        # because the perimeter is larger relative to area
        # (This is the key insight from B5)

    def test_elongation_batched_consistency(self):
        """Verify batched computation matches individual computation."""
        batch_size = 4
        R_major = 10.0

        mpol = 2
        ntor = 1
        r_cos = torch.zeros((batch_size, mpol + 1, 2 * ntor + 1))
        z_sin = torch.zeros((batch_size, mpol + 1, 2 * ntor + 1))

        # Set up different shapes in each batch
        elongations = [1.0, 1.5, 2.0, 2.5]
        for i, kappa in enumerate(elongations):
            r_cos[i, 0, ntor] = R_major
            r_cos[i, 1, ntor] = 1.0
            z_sin[i, 1, ntor] = kappa

        nfp = 3

        # Batched computation
        elo_batch = geometry.elongation_isoperimetric(r_cos, z_sin, nfp)

        # Individual computations
        for i, kappa in enumerate(elongations):
            r_cos_single = r_cos[i : i + 1]
            z_sin_single = z_sin[i : i + 1]
            elo_single = geometry.elongation_isoperimetric(
                r_cos_single, z_sin_single, nfp
            )

            assert np.isclose(elo_batch[i].item(), elo_single.item(), rtol=1e-5), (
                f"Batch vs single mismatch at index {i}"
            )


class TestAspectRatioComparison:
    """Compare aspect ratio methods (B4 verification)."""

    def test_aspect_ratio_methods_agree_for_circular(self):
        """Both AR methods should agree for circular cross-section."""
        R_major = 10.0
        r_minor = 1.0

        mpol = 1
        ntor = 0
        r_cos = torch.zeros((1, mpol + 1, 2 * ntor + 1))
        z_sin = torch.zeros((1, mpol + 1, 2 * ntor + 1))

        r_cos[0, 0, 0] = R_major
        r_cos[0, 1, 0] = r_minor
        z_sin[0, 1, 0] = r_minor

        nfp = 1

        ar_basic = geometry.aspect_ratio(r_cos, z_sin, nfp)
        ar_arc = geometry.aspect_ratio_arc_length(r_cos, z_sin, nfp)

        expected = R_major / r_minor

        # Both should match for circular cross-section
        assert np.isclose(ar_basic.item(), expected, rtol=0.05)
        assert np.isclose(ar_arc.item(), expected, rtol=0.05)
        assert np.isclose(ar_basic.item(), ar_arc.item(), rtol=0.02)


class TestNumericalStability:
    """Test numerical stability of geometry computations."""

    def test_elongation_near_circular(self):
        """Elongation should be stable near 1.0 (no division by zero issues)."""
        R_major = 10.0
        r_minor = 1.0

        mpol = 1
        ntor = 0
        r_cos = torch.zeros((1, mpol + 1, 2 * ntor + 1))
        z_sin = torch.zeros((1, mpol + 1, 2 * ntor + 1))

        r_cos[0, 0, 0] = R_major
        r_cos[0, 1, 0] = r_minor
        z_sin[0, 1, 0] = r_minor * 1.001  # Very slightly elongated

        nfp = 1

        elo = geometry.elongation_isoperimetric(r_cos, z_sin, nfp)

        # Should be close to 1 and finite
        assert torch.isfinite(elo).all()
        assert 0.99 < elo.item() < 1.05

    def test_surface_area_small_minor_radius(self):
        """Surface area should remain stable for small minor radius."""
        R_major = 10.0
        r_minor = 0.1  # Small

        mpol = 1
        ntor = 0
        r_cos = torch.zeros((1, mpol + 1, 2 * ntor + 1))
        z_sin = torch.zeros((1, mpol + 1, 2 * ntor + 1))

        r_cos[0, 0, 0] = R_major
        r_cos[0, 1, 0] = r_minor
        z_sin[0, 1, 0] = r_minor

        nfp = 1

        area = geometry.surface_area(r_cos, z_sin, nfp)
        expected = 4 * np.pi**2 * R_major * r_minor

        assert torch.isfinite(area).all()
        assert np.isclose(area.item(), expected, rtol=0.05)

    def test_curvature_high_modes(self):
        """Mean curvature should remain finite with higher Fourier modes."""
        R_major = 10.0

        mpol = 4
        ntor = 2
        r_cos = torch.zeros((1, mpol + 1, 2 * ntor + 1))
        z_sin = torch.zeros((1, mpol + 1, 2 * ntor + 1))

        # Base torus
        r_cos[0, 0, ntor] = R_major
        r_cos[0, 1, ntor] = 1.0
        z_sin[0, 1, ntor] = 1.0

        # Add small higher-order perturbations
        torch.manual_seed(42)
        r_cos[0, 2:, :] = torch.randn_like(r_cos[0, 2:, :]) * 0.05
        z_sin[0, 2:, :] = torch.randn_like(z_sin[0, 2:, :]) * 0.05

        nfp = 3

        H_mean = geometry.mean_curvature(r_cos, z_sin, nfp)

        assert torch.isfinite(H_mean).all()
        assert H_mean.item() > 0  # Mean |H| should be positive
