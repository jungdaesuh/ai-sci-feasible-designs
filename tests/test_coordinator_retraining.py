# ruff: noqa: E402
"""Tests for periodic retraining in coordinator."""

import dataclasses
import sys
from unittest.mock import MagicMock, patch

import pytest

# Check if real constellaration is available BEFORE any mocking
_CONSTELLARATION_AVAILABLE = False
try:
    import constellaration.forward_model as _check_fm

    _CONSTELLARATION_AVAILABLE = True
    del _check_fm
except ImportError:
    pass


@pytest.fixture(autouse=True, scope="module")
def mock_constellaration_modules():
    """Mock constellaration modules for testing when not available.

    Uses patch.dict context manager to ensure cleanup after tests.
    This prevents pollution of sys.modules for subsequent test files.
    """
    if _CONSTELLARATION_AVAILABLE:
        # Real modules available, no mocking needed
        yield
        return

    # Create mock modules
    mock_constellaration = MagicMock()
    mock_alm = MagicMock()
    mock_alm.AugmentedLagrangianState = MagicMock

    mock_modules = {
        "vmecpp": MagicMock(),
        "vmecpp.cpp": MagicMock(),
        "vmecpp.cpp._vmecpp": MagicMock(),
        "constellaration": mock_constellaration,
        "constellaration.forward_model": MagicMock(),
        "constellaration.boozer": MagicMock(),
        "constellaration.mhd": MagicMock(),
        "constellaration.geometry": MagicMock(),
        "constellaration.geometry.surface_rz_fourier": MagicMock(),
        "constellaration.optimization": MagicMock(),
        "constellaration.optimization.augmented_lagrangian": mock_alm,
        "constellaration.optimization.settings": MagicMock(),
        "constellaration.utils": MagicMock(),
        "constellaration.utils.pytree": MagicMock(),
        "constellaration.problems": MagicMock(),
        "constellaration.initial_guess": MagicMock(),
    }

    with patch.dict(sys.modules, mock_modules):
        # Force re-import of coordinator to pick up mocks
        if "ai_scientist.coordinator" in sys.modules:
            del sys.modules["ai_scientist.coordinator"]
        yield
        # Cleanup: remove coordinator so it gets re-imported fresh next time
        if "ai_scientist.coordinator" in sys.modules:
            del sys.modules["ai_scientist.coordinator"]


from ai_scientist.config import RetrainingConfig, load_experiment_config
from ai_scientist.coordinator import Coordinator


class TestRetrainingConfig:
    """Tests for RetrainingConfig dataclass and loading."""

    def test_retraining_config_defaults(self):
        """Verify RetrainingConfig has expected default values."""
        config = RetrainingConfig()

        assert config.enabled is True
        assert config.cycle_cadence == 5
        assert config.min_elites == 32
        assert config.hv_stagnation_threshold == 0.005
        assert config.hv_stagnation_lookback == 3

    def test_retraining_config_custom_values(self):
        """Verify RetrainingConfig accepts custom values."""
        config = RetrainingConfig(
            enabled=False,
            cycle_cadence=10,
            min_elites=64,
            hv_stagnation_threshold=0.01,
            hv_stagnation_lookback=5,
        )

        assert config.enabled is False
        assert config.cycle_cadence == 10
        assert config.min_elites == 64
        assert config.hv_stagnation_threshold == 0.01
        assert config.hv_stagnation_lookback == 5

    def test_experiment_config_has_retraining(self):
        """Verify ExperimentConfig includes retraining field."""
        cfg = load_experiment_config("configs/experiment.example.yaml")
        assert hasattr(cfg, "retraining")
        assert cfg.retraining is not None


class TestShouldRetrain:
    """Tests for _should_retrain() trigger logic."""

    @pytest.fixture
    def mock_cfg(self):
        """Load real config."""
        return load_experiment_config("configs/experiment.example.yaml")

    @pytest.fixture
    def coordinator(self, mock_cfg):
        """Create Coordinator with mocked dependencies."""
        with (
            patch("ai_scientist.coordinator.OptimizationWorker"),
            patch("ai_scientist.coordinator.ExplorationWorker"),
            patch("ai_scientist.coordinator.GeometerWorker"),
            patch("ai_scientist.coordinator.PreRelaxWorker"),
            patch("ai_scientist.coordinator.RLRefinementWorker"),
        ):
            wm = MagicMock()
            planner = MagicMock()
            coord = Coordinator(cfg=mock_cfg, world_model=wm, planner=planner)
            coord.constraint_names = ["c1", "c2", "c3"]
            return coord

    def test_should_retrain_disabled(self, coordinator, mock_cfg):
        """Verify returns False when retraining is disabled."""
        # Create config with retraining disabled
        disabled_retraining = RetrainingConfig(enabled=False)
        coordinator.cfg = dataclasses.replace(mock_cfg, retraining=disabled_retraining)

        should, reason = coordinator._should_retrain(cycle=5, experiment_id=1)

        assert should is False
        assert reason == "disabled"

    def test_should_retrain_cycle_cadence(self, coordinator, mock_cfg):
        """Verify triggers at cycle_cadence intervals."""
        # Enable with cycle_cadence=5
        retraining = RetrainingConfig(enabled=True, cycle_cadence=5)
        coordinator.cfg = dataclasses.replace(mock_cfg, retraining=retraining)

        # Mock world_model to return None for HV delta (no stagnation)
        coordinator.world_model.average_recent_hv_delta.return_value = None

        # Cycles 5, 10, 15 should trigger
        should, reason = coordinator._should_retrain(cycle=5, experiment_id=1)
        assert should is True
        assert "cycle_cadence" in reason

        should, reason = coordinator._should_retrain(cycle=10, experiment_id=1)
        assert should is True

        # Cycles 1, 2, 3, 4 should not trigger
        should, reason = coordinator._should_retrain(cycle=1, experiment_id=1)
        assert should is False

        should, reason = coordinator._should_retrain(cycle=4, experiment_id=1)
        assert should is False

    def test_should_retrain_hv_stagnation(self, coordinator, mock_cfg):
        """Verify triggers on HV stagnation."""
        retraining = RetrainingConfig(
            enabled=True,
            cycle_cadence=100,  # High so it doesn't trigger
            hv_stagnation_threshold=0.005,
        )
        coordinator.cfg = dataclasses.replace(mock_cfg, retraining=retraining)

        # Mock HV stagnation (delta below threshold)
        coordinator.world_model.average_recent_hv_delta.return_value = 0.001

        should, reason = coordinator._should_retrain(cycle=3, experiment_id=1)

        assert should is True
        assert "hv_stagnation" in reason

    def test_should_retrain_no_trigger(self, coordinator, mock_cfg):
        """Verify returns False when no trigger conditions met."""
        retraining = RetrainingConfig(
            enabled=True,
            cycle_cadence=100,  # High so it doesn't trigger
            hv_stagnation_threshold=0.001,  # Low threshold
        )
        coordinator.cfg = dataclasses.replace(mock_cfg, retraining=retraining)

        # HV delta above threshold (no stagnation)
        coordinator.world_model.average_recent_hv_delta.return_value = 0.05

        should, reason = coordinator._should_retrain(cycle=3, experiment_id=1)

        assert should is False
        assert reason == "no_trigger"
