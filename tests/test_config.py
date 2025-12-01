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


def test_surrogate_preservation():
    """Verify that factory methods preserve surrogate settings from defaults."""
    from unittest.mock import patch
    from ai_scientist.config import SurrogateConfig

    # Create a mock default config with non-default surrogate settings
    mock_surrogate = SurrogateConfig(
        backend="random_forest",
        n_ensembles=5,  # Non-default
        hidden_dim=128  # Non-default
    )
    
    with patch("ai_scientist.config.load_experiment_config") as mock_load:
        # Setup the mock to return a config with our custom surrogate
        mock_defaults = MagicMock()
        mock_defaults.surrogate = mock_surrogate
        # Mock other required fields to avoid errors
        mock_defaults.random_seed = 0
        mock_defaults.budgets = MagicMock()
        mock_defaults.adaptive_budgets = MagicMock()
        mock_defaults.fidelity_ladder = MagicMock()
        mock_defaults.boundary_template = MagicMock()
        mock_defaults.stage_gates = MagicMock()
        mock_defaults.governance = MagicMock()
        mock_defaults.proposal_mix = MagicMock()
        mock_defaults.constraint_weights = MagicMock()
        mock_defaults.generative = MagicMock()
        
        mock_load.return_value = mock_defaults
        
        # Test p3_high_fidelity - should override backend but keep n_ensembles
        config_high = ExperimentConfig.p3_high_fidelity()
        assert config_high.surrogate.backend == "neural_operator"
        assert config_high.surrogate.n_ensembles == 5
        
        # Test p3_quick_validation - should keep everything
        config_quick = ExperimentConfig.p3_quick_validation()
        assert config_quick.surrogate.backend == "random_forest"
        assert config_quick.surrogate.n_ensembles == 5
        
        # Test p3_aso_enabled - should keep everything
        config_aso = ExperimentConfig.p3_aso_enabled()
        assert config_aso.surrogate.backend == "random_forest"
        assert config_aso.surrogate.n_ensembles == 5


def test_aso_config_factories():
    """Verify ASOConfig factory methods."""
    from ai_scientist.config import ASOConfig

    # Default event triggered
    config = ASOConfig.default_event_triggered()
    assert config.enabled is True
    assert config.supervision_mode == "event_triggered"
    assert config.max_stagnation_steps == 5
    assert config.use_heuristic_fallback is True

    # Default periodic
    config = ASOConfig.default_periodic(interval=10)
    assert config.enabled is True
    assert config.supervision_mode == "periodic"
    assert config.supervision_interval == 10

    # Disabled
    config = ASOConfig.disabled()
    assert config.enabled is False


def test_alm_config_factories():
    """Verify ALMConfig factory methods."""
    from ai_scientist.config import ALMConfig

    # Default
    config = ALMConfig.default()
    assert config.penalty_parameters_increase_factor == 2.0  # Default value

    # Aggressive penalties
    config = ALMConfig.aggressive_penalties()
    assert config.penalty_parameters_increase_factor == 4.0
    assert config.penalty_parameters_initial == 10.0

    # Conservative
    config = ALMConfig.conservative()
    assert config.penalty_parameters_increase_factor == 1.5
    assert config.bounds_reduction_factor == 0.98
    assert config.maxit == 50
