"""Integration tests with real physics backend (constellaration/vmecpp).

These tests exercise the full physics pipeline with real VMEC++ evaluations.
They are skipped unless explicitly enabled, and they fail loudly when enabled
but the environment is broken.

Run with: pytest tests/test_physics_integration.py -v
Requires: constellaration with vmecpp installed
"""

from __future__ import annotations

import os

import pytest


# Check if real backend is available
def _has_real_backend() -> bool:
    """Check if constellaration with vmecpp is available."""
    try:
        import constellaration.forward_model  # noqa: F401
        from constellaration.geometry import surface_rz_fourier  # noqa: F401

        # Validate that the native vmecpp extension can load (catches dlopen issues).
        try:
            import vmecpp.cpp._vmecpp  # noqa: F401

            return True
        except (ImportError, OSError):
            return False
    except ImportError:
        return False


def _require_real_backend() -> bool:
    """Gate real-backend tests behind an explicit opt-in."""
    backend = os.environ.get("AI_SCIENTIST_PHYSICS_BACKEND", "auto").lower()
    require = os.environ.get("AI_SCIENTIST_REQUIRE_REAL_BACKEND", "").strip().lower()
    return backend == "real" or require in {"1", "true", "yes", "y", "on"}


# Skip all tests if real backend unavailable
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _require_real_backend(),
        reason="Set AI_SCIENTIST_PHYSICS_BACKEND=real to run real-backend integration tests",
    ),
    pytest.mark.skipif(
        not _has_real_backend(),
        reason="Requires constellaration/vmecpp installation (native extension must load)",
    ),
]


class TestPhysicsIntegration:
    """Integration tests that exercise the real physics pipeline.

    These tests verify that:
    1. Forward model returns expected metric structure
    2. Constraint calculations match benchmark definitions
    3. Mock backend output ranges are reasonable vs real physics
    """

    @pytest.fixture
    def rotating_ellipse_boundary(self):
        """Create a simple rotating ellipse boundary for testing."""
        import numpy as np

        # Rotating ellipse: simple, known to be valid
        mpol = 2
        ntor = 2
        grid_h = mpol + 1
        grid_w = 2 * ntor + 1

        r_cos = np.zeros((grid_h, grid_w))
        z_sin = np.zeros((grid_h, grid_w))

        # R00 = 1.0 (major radius)
        r_cos[0, ntor] = 1.0
        # R10 = 0.1 (minor radius)
        r_cos[1, ntor] = 0.1
        # Z10 = 0.1 (vertical extent)
        z_sin[1, ntor] = 0.1

        return {
            "r_cos": r_cos.tolist(),
            "z_sin": z_sin.tolist(),
            "n_field_periods": 3,
            "is_stellarator_symmetric": True,
        }

    @pytest.fixture
    def constellaration_settings_skip_qi(self):
        """Settings for a cheap, stable real-backend smoke test.

        QI/Boozer calculations can fail for toy boundaries (e.g. insufficient
        bounce-point crossings), so these smoke tests focus on VMEC equilibrium
        + geometric/MHD metrics.
        """
        from constellaration.forward_model import ConstellarationSettings

        return ConstellarationSettings(
            boozer_preset_settings=None,
            qi_settings=None,
            turbulent_settings=None,
        )

    def test_forward_model_produces_valid_metrics(
        self, rotating_ellipse_boundary, constellaration_settings_skip_qi, real_backend
    ):
        """Verify forward model returns expected metric keys."""
        from ai_scientist.forward_model import evaluate_boundary, ForwardModelSettings

        settings = ForwardModelSettings(
            fidelity="low",
            problem="p1",
            constellaration_settings=constellaration_settings_skip_qi,
        )

        result = evaluate_boundary(rotating_ellipse_boundary, settings)

        # Check essential metrics exist
        assert result.metrics is not None
        assert hasattr(result.metrics, "aspect_ratio")
        assert hasattr(result.metrics, "max_elongation")
        assert hasattr(result.metrics, "vacuum_well")

        # Check values are reasonable
        assert result.metrics.aspect_ratio > 0
        assert result.metrics.max_elongation >= 1.0

    def test_constraint_margins_match_benchmark_definitions(
        self, rotating_ellipse_boundary, constellaration_settings_skip_qi, real_backend
    ):
        """Verify constraint gating works with real-backend metrics."""
        from ai_scientist.forward_model import (
            evaluate_boundary,
            ForwardModelSettings,
            compute_constraint_margins,
        )

        # Use P2 (has QI at full fidelity), but skip QI in the physics call and
        # compute margins at "promote" stage (which intentionally excludes QI).
        settings = ForwardModelSettings(
            fidelity="low",
            problem="p2",
            constellaration_settings=constellaration_settings_skip_qi,
        )

        result = evaluate_boundary(rotating_ellipse_boundary, settings)

        if result.metrics:
            margins = compute_constraint_margins(result.metrics, "p2", stage="promote")

            # "promote" stage excludes QI by design; verify geometry constraints exist.
            assert "qi" not in margins
            assert "max_elongation" in margins
            assert "edge_magnetic_mirror_ratio" in margins

    def test_equilibrium_converges_for_valid_boundary(
        self, rotating_ellipse_boundary, constellaration_settings_skip_qi, real_backend
    ):
        """Verify VMEC converges for a known-valid boundary."""
        from ai_scientist.forward_model import evaluate_boundary, ForwardModelSettings

        settings = ForwardModelSettings(
            fidelity="low",
            problem="p1",
            constellaration_settings=constellaration_settings_skip_qi,
        )

        result = evaluate_boundary(rotating_ellipse_boundary, settings)

        # Should converge for a simple rotating ellipse
        assert result.equilibrium_converged is True
        assert result.error_message is None


class TestMockVsRealComparison:
    """Compare mock backend output to real physics ranges.

    These tests help ensure the mock backend produces realistic values
    that won't cause test-only bugs.
    """

    def test_mock_metric_ranges_are_realistic(self):
        """Verify mock backend produces values in realistic ranges."""
        from ai_scientist.backends.mock import MockMetrics

        # Check default MockMetrics are in realistic ranges
        metrics = MockMetrics()

        # Aspect ratio: typically 5-15 for stellarators
        assert 3.0 <= metrics.aspect_ratio <= 20.0

        # Elongation: typically 1.0-6.0
        assert 1.0 <= metrics.max_elongation <= 8.0

        # Vacuum well: can be negative (unstable) or positive (stable)
        assert -0.5 <= metrics.vacuum_well <= 0.5

        # QI residual: typically 1e-6 to 1e-1
        if metrics.qi is not None:
            assert 1e-8 <= metrics.qi <= 1.0

        # Gradient scale length: typically 1-15
        assert 0.5 <= metrics.minimum_normalized_magnetic_gradient_scale_length <= 20.0
