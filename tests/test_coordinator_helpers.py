from unittest.mock import MagicMock

from ai_scientist import config as ai_config
from ai_scientist.coordinator import Coordinator


def test_coordinator_helpers_imports():
    """Verify helper methods import dependencies correctly."""

    # Load real config to ensure all fields are present
    cfg = ai_config.load_experiment_config("configs/experiment.example.yaml")

    # We can't easily instantiate Coordinator without real WorldModel/Planner
    # But we can instantiate it if we mock them
    wm = MagicMock()
    planner = MagicMock()

    # Mock generative model if needed by workers
    gen_model = MagicMock()

    coord = Coordinator(cfg, wm, planner, generative_model=gen_model)

    # Test _get_problem
    problem = coord._get_problem(cfg)
    assert problem is not None
    print(f"Problem created: {type(problem)}")

    # Test _build_optimization_settings
    settings = coord._build_optimization_settings(cfg)
    assert settings is not None
    print(f"Settings created: {type(settings)}")
    assert settings.forward_model_settings is not None
