"""Tests for ExperimentConfig factory methods."""

import sys
from unittest.mock import MagicMock

# Mock heavy dependencies to avoid ImportError due to broken environment (libtorch)
sys.modules["ai_scientist.experiment_runner"] = MagicMock()
sys.modules["ai_scientist.forward_model"] = MagicMock()

from ai_scientist.config import ExperimentConfig


def test_p3_high_fidelity_factory():
    """Verify p3_high_fidelity factory method returns correct config."""
    config = ExperimentConfig.p3_high_fidelity()
    
    assert config.problem == "p3"
    assert config.cycles == 10
    assert config.aso.enabled is True
    assert config.aso.supervision_mode == "event_triggered"
    assert config.surrogate.backend == "neural_operator"
    
    # Check budgets
    assert config.budgets.screen_evals_per_cycle == 50
    assert config.budgets.promote_top_k == 5
    assert config.budgets.max_high_fidelity_evals_per_cycle == 3


def test_p3_quick_validation_factory():
    """Verify p3_quick_validation factory method returns correct config."""
    config = ExperimentConfig.p3_quick_validation()
    
    assert config.problem == "p3"
    assert config.cycles == 2
    assert config.aso.enabled is False
    
    # Check budgets
    assert config.budgets.screen_evals_per_cycle == 5
    assert config.budgets.promote_top_k == 2
    assert config.budgets.max_high_fidelity_evals_per_cycle == 1


def test_p3_aso_enabled_factory():
    """Verify p3_aso_enabled factory method returns correct config."""
    config = ExperimentConfig.p3_aso_enabled()
    
    assert config.problem == "p3"
    assert config.cycles == 5
    
    # Check ASO config
    assert config.aso.enabled is True
    assert config.aso.supervision_mode == "event_triggered"
    assert config.aso.max_stagnation_steps == 5
