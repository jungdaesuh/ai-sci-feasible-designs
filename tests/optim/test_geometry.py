import numpy as np
import torch
import hypothesis
from hypothesis import strategies as st

from ai_scientist.optim import geometry


def test_fourier_to_real_space_numpy():
    """Test basic conversion with numpy arrays."""
    # Simple torus: R = 10 + cos(theta), Z = sin(theta)
    # m=0, n=0 -> R=10 (major radius)
    # m=1, n=0 -> r_cos=1, z_sin=1 (minor radius 1)

    mpol = 1
    ntor = 0
    r_cos = np.zeros((mpol + 1, 2 * ntor + 1))
    z_sin = np.zeros((mpol + 1, 2 * ntor + 1))

    # Major radius 10 at m=0, n=0
    r_cos[0, 0] = 10.0
    # Minor radius 1 at m=1, n=0
    r_cos[1, 0] = 1.0
    z_sin[1, 0] = 1.0

    R, Z, Phi = geometry.fourier_to_real_space(
        r_cos, z_sin, n_theta=4, n_zeta=4, n_field_periods=1
    )

    assert R.shape == (4, 4)
    assert Z.shape == (4, 4)
    assert Phi.shape == (4, 4)

    # Check Major Radius approx
    assert np.allclose(np.mean(R), 10.0)
    # Check Z is centered
    assert np.allclose(np.mean(Z), 0.0)


def test_fourier_to_real_space_torch():
    """Test basic conversion with torch tensors."""
    mpol = 1
    ntor = 0
    r_cos = torch.zeros((mpol + 1, 2 * ntor + 1))
    z_sin = torch.zeros((mpol + 1, 2 * ntor + 1))

    r_cos[0, 0] = 5.0
    r_cos[1, 0] = 0.5
    z_sin[1, 0] = 0.5

    R, Z, Phi = geometry.fourier_to_real_space(
        r_cos, z_sin, n_theta=8, n_zeta=8, n_field_periods=1
    )

    assert isinstance(R, torch.Tensor)
    assert R.shape == (8, 8)
    assert torch.allclose(torch.mean(R), torch.tensor(5.0))


def test_surface_to_point_cloud():
    """Test generation of (N, 3) point cloud."""
    mpol = 1
    ntor = 1  # Test with some toroidal modes
    r_cos = np.zeros((mpol + 1, 2 * ntor + 1))
    z_sin = np.zeros((mpol + 1, 2 * ntor + 1))

    r_cos[0, 1] = 10.0  # Center index for n=0

    nfp = 3
    points = geometry.surface_to_point_cloud(
        r_cos, z_sin, n_field_periods=nfp, n_theta=10, n_zeta=10
    )

    # Expected size: theta(10) * zeta(10) * nfp(3) = 300 points
    assert points.shape == (300, 3)
    assert isinstance(points, torch.Tensor)  # Default is tensor

    # Check approximate bounds
    # R=10, so x^2 + y^2 should be approx 100
    radii = torch.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
    assert torch.allclose(radii, torch.tensor(10.0), atol=1e-5)


def test_coordinate_transform():
    """Verify cylindrical to cartesian transform."""
    R = np.array([[1.0]])
    Z = np.array([[2.0]])
    Phi = np.array([[0.0]])  # At phi=0, X=R, Y=0

    X, Y, Z_out = geometry.to_cartesian(R, Z, Phi)
    assert X[0, 0] == 1.0
    assert Y[0, 0] == 0.0
    assert Z_out[0, 0] == 2.0

    Phi = np.array([[np.pi / 2]])  # At phi=90, X=0, Y=R
    X, Y, Z_out = geometry.to_cartesian(R, Z, Phi)
    assert np.isclose(X[0, 0], 0.0, atol=1e-7)
    assert np.isclose(Y[0, 0], 1.0, atol=1e-7)


class TestGeometryInvariants:
    """Property-based tests for geometry mathematical invariants using actual implementation."""

    @hypothesis.given(
        R_major=st.floats(0.5, 10.0),
        R_minor=st.floats(0.1, 2.0),
    )
    def test_aspect_ratio_positive(self, R_major, R_minor):
        """Aspect ratio should always be positive when computed via geometry module."""
        hypothesis.assume(R_minor < R_major)

        # Construct a simple torus with these parameters
        # m=0, n=0 -> R_major
        # m=1, n=0 -> R_minor (for both R and Z to make a circular cross-section)
        mpol = 1
        ntor = 0
        r_cos = torch.zeros((mpol + 1, 2 * ntor + 1))
        z_sin = torch.zeros((mpol + 1, 2 * ntor + 1))

        r_cos[0, 0] = R_major
        r_cos[1, 0] = R_minor
        z_sin[1, 0] = R_minor

        # Add batch dimension
        r_cos = r_cos.unsqueeze(0)
        z_sin = z_sin.unsqueeze(0)
        nfp = torch.tensor([1.0])

        # Compute aspect ratio using the actual function
        ar = geometry.aspect_ratio(r_cos, z_sin, nfp)

        assert ar.item() > 0
        # For a circular torus, AR should be approx R_major / R_minor
        assert np.isclose(ar.item(), R_major / R_minor, rtol=0.1)

    @hypothesis.given(
        coeffs=st.lists(st.floats(-1.0, 1.0, allow_nan=False), min_size=4, max_size=16),
    )
    @hypothesis.settings(max_examples=50)
    def test_fourier_to_real_smooth(self, coeffs):
        """Real-space representation should be smooth (no NaN/inf) using actual conversion."""
        # Construct a valid shape (rows, 1) which corresponds to ntor=0
        rows = len(coeffs)
        cols = 1

        r_cos = np.array(coeffs).reshape(rows, cols)
        z_sin = np.array(coeffs).reshape(rows, cols)

        # Use the actual function
        R, Z, Phi = geometry.fourier_to_real_space(
            r_cos, z_sin, n_theta=32, n_zeta=32, n_field_periods=1
        )

        assert np.all(np.isfinite(R))
        assert np.all(np.isfinite(Z))
        assert np.all(np.isfinite(Phi))
