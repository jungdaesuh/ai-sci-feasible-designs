"""Tests for ai_scientist.problems."""

# ruff: noqa: E402
import sys
from unittest.mock import MagicMock, patch
import pytest
import numpy as np


@pytest.fixture(autouse=True)
def mock_dependencies():
    """Mock dependencies for all tests in this module."""
    mock_vmecpp = MagicMock()
    mock_constellaration = MagicMock()

    # Setup mock modules dictionary
    mock_modules = {
        "vmecpp": mock_vmecpp,
        "vmecpp.cpp": MagicMock(),
        "vmecpp.cpp._vmecpp": MagicMock(),
        "constellaration": mock_constellaration,
        "constellaration.forward_model": MagicMock(),
        "constellaration.boozer": MagicMock(),
        "constellaration.mhd": MagicMock(),
        "constellaration.geometry": MagicMock(),
        "constellaration.geometry.surface_rz_fourier": MagicMock(),
        "constellaration.optimization": MagicMock(),
        "constellaration.optimization.augmented_lagrangian": MagicMock(),
        "constellaration.optimization.settings": MagicMock(),
        "constellaration.utils": MagicMock(),
        "constellaration.utils.pytree": MagicMock(),
        "constellaration.problems": MagicMock(),
        "constellaration.initial_guess": MagicMock(),
    }

    # Apply mocks using patch.dict
    with patch.dict(sys.modules, mock_modules):
        # Force re-import of ai_scientist.problems to pick up mocks
        import importlib

        if "ai_scientist.problems" in sys.modules:
            importlib.reload(sys.modules["ai_scientist.problems"])
        else:
            importlib.import_module("ai_scientist.problems")

        yield

        # Cleanup handled by patch.dict for mocks, but we should remove the reloaded module
        # so subsequent tests don't use this mocked version
        if "ai_scientist.problems" in sys.modules:
            del sys.modules["ai_scientist.problems"]


class TestProblemABC:
    def test_template_methods(self):
        """Test is_feasible, compute_feasibility, max_violation template methods."""
        from ai_scientist.problems import Problem

        class MockProblem(Problem):
            @property
            def name(self):
                return "mock"

            @property
            def constraint_names(self):
                return ["c1", "c2"]

            def _normalized_constraint_violations(self, metrics):
                return np.array([metrics["v1"], metrics["v2"]])

            def get_objective(self, metrics):
                return 0.0

        p = MockProblem()

        # Case 1: Feasible
        metrics_feasible = {"v1": -0.1, "v2": 0.0}
        assert p.is_feasible(metrics_feasible)
        assert p.compute_feasibility(metrics_feasible) == 0.0
        assert p.max_violation(metrics_feasible) == 0.0

        # Case 2: Infeasible
        metrics_infeasible = {"v1": 0.1, "v2": -0.5}
        assert not p.is_feasible(metrics_infeasible)
        assert p.compute_feasibility(metrics_infeasible) == 0.1
        assert p.max_violation(metrics_infeasible) == 0.1

        # Case 3: Mixed
        metrics_mixed = {"v1": 0.2, "v2": 0.3}
        assert not p.is_feasible(metrics_mixed)
        assert p.compute_feasibility(metrics_mixed) == 0.3
        assert p.max_violation(metrics_mixed) == 0.3


class TestP1Problem:
    def test_constraints(self):
        from ai_scientist.problems import P1Problem

        p = P1Problem()
        assert p.name == "p1"
        assert len(p.constraint_names) == 3

        # Feasible case
        # aspect_ratio <= 4.0
        # average_triangularity <= -0.5
        # edge_rotational_transform_over_n_field_periods >= 0.3
        metrics = {
            "aspect_ratio": 3.9,
            "average_triangularity": -0.6,
            "edge_rotational_transform_over_n_field_periods": 0.35,
            "max_elongation": 2.0,
        }
        assert p.is_feasible(metrics)
        assert p.get_objective(metrics) == 2.0

        # Infeasible case
        metrics_bad = {
            "aspect_ratio": 4.1,  # Violation
            "average_triangularity": -0.6,
            "edge_rotational_transform_over_n_field_periods": 0.35,
        }
        assert not p.is_feasible(metrics_bad)
        violations = p._normalized_constraint_violations(metrics_bad)
        assert violations[0] > 0  # AR violation


class TestP2Problem:
    def test_constraints(self):
        from ai_scientist.problems import P2Problem

        p = P2Problem()
        assert p.name == "p2"

        # Feasible case
        # aspect_ratio <= 10.0
        # edge_rotational_transform_over_n_field_periods >= 0.25
        # log10(qi) <= -4.0 => qi <= 1e-4
        # edge_magnetic_mirror_ratio <= 0.2
        # max_elongation <= 5.0
        metrics = {
            "aspect_ratio": 9.0,
            "edge_rotational_transform_over_n_field_periods": 0.3,
            "qi": 1e-5,
            "edge_magnetic_mirror_ratio": 0.1,
            "max_elongation": 4.0,
            "minimum_normalized_magnetic_gradient_scale_length": 1.5,
        }
        assert p.is_feasible(metrics)
        assert p.get_objective(metrics) == -1.5

        # Infeasible case
        metrics_bad = metrics.copy()
        metrics_bad["qi"] = 1e-3  # log10(-3) > -4
        assert not p.is_feasible(metrics_bad)


class TestP3Problem:
    def test_constraints(self):
        from ai_scientist.problems import P3Problem

        p = P3Problem()
        assert p.name == "p3"

        # Feasible case
        # edge_rotational_transform_over_n_field_periods >= 0.25
        # log10(qi) <= -3.5 => qi <= 10^-3.5 ~= 3.16e-4
        # edge_magnetic_mirror_ratio <= 0.25
        # flux_compression_in_regions_of_bad_curvature <= 0.9
        # vacuum_well >= 0.0
        metrics = {
            "aspect_ratio": 8.0,  # P3 objective: minimize aspect_ratio
            "edge_rotational_transform_over_n_field_periods": 0.3,
            "qi": 1e-4,
            "edge_magnetic_mirror_ratio": 0.2,
            "flux_compression_in_regions_of_bad_curvature": 0.8,
            "vacuum_well": 0.01,
            "minimum_normalized_magnetic_gradient_scale_length": 2.0,
        }
        assert p.is_feasible(metrics)
        assert p.get_objective(metrics) == pytest.approx(
            -0.25
        )  # Scalarized: -L_∇B / AR = -2.0 / 8.0

        # Infeasible case (vacuum well)
        metrics_bad = metrics.copy()
        metrics_bad["vacuum_well"] = -0.01
        assert not p.is_feasible(metrics_bad)


def test_get_problem():
    from ai_scientist.problems import get_problem, P1Problem, P2Problem, P3Problem

    assert isinstance(get_problem("p1"), P1Problem)
    assert isinstance(get_problem("P1"), P1Problem)
    assert isinstance(get_problem("p2"), P2Problem)
    assert isinstance(get_problem("p3"), P3Problem)
    assert isinstance(get_problem("p3_variant"), P3Problem)  # Default/fallback behavior
