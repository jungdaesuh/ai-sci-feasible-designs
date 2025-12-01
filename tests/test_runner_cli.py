import sys
from unittest.mock import MagicMock, patch
import pytest

@pytest.fixture
def runner_module():
    """
    Fixture that mocks heavy dependencies and imports ai_scientist.runner.
    Restores state after test to avoid polluting global sys.modules.
    """
    # Create dummy classes for types used in annotations to satisfy jaxtyping/isinstance checks
    class MockTensor:
        pass
    
    class MockArray:
        pass

    mock_torch = MagicMock()
    mock_torch.Tensor = MockTensor
    
    mock_jax = MagicMock()
    mock_jax.Array = MockArray
    
    mock_jax_numpy = MagicMock()
    mock_jax_numpy.ndarray = MockArray
    mock_jax.numpy = mock_jax_numpy

    mock_modules = {
        "torch": mock_torch,
        "torch.nn": MagicMock(),
        "torch.nn.functional": MagicMock(),
        "torch.optim": MagicMock(),
        "torch.distributions": MagicMock(),
        "torch.utils": MagicMock(),
        "torch.utils.data": MagicMock(),
        "vmecpp": MagicMock(),
        "jax": mock_jax,
        "jaxlib": MagicMock(),
        "jax.numpy": mock_jax_numpy,
        "jax.tree_util": MagicMock(),
        "ai_scientist.coordinator": MagicMock(),
        "ai_scientist.forward_model": MagicMock(),
        "ai_scientist.optim.surrogate_v2": MagicMock(),
    }

    with patch.dict(sys.modules, mock_modules):
        # Ensure we get a fresh import of runner using the mocks
        # We must remove it from sys.modules if it exists to force re-import with mocks
        if "ai_scientist.runner" in sys.modules:
            del sys.modules["ai_scientist.runner"]
        
        import ai_scientist.runner
        yield ai_scientist.runner
        
        # Cleanup: remove the mocked runner so subsequent tests don't use it
        if "ai_scientist.runner" in sys.modules:
            del sys.modules["ai_scientist.runner"]


def test_parser_help_mentions_screen_stage(runner_module) -> None:
    help_text = runner_module._build_argument_parser().format_help()
    assert "S1" in help_text
    assert "screen" in help_text.lower()
    assert "promote" in help_text.lower()


def test_parse_args_errors_when_screen_and_promote_both_set(runner_module) -> None:
    with pytest.raises(SystemExit) as excinfo:
        runner_module.parse_args(["--screen", "--promote"])
    assert excinfo.value.code == 2


def test_main_exits_when_screen_flag_conflicts_with_promote_preset(runner_module, monkeypatch) -> None:
    monkeypatch.setattr(
        sys, "argv", ["runner", "--screen", "--run-preset", "promote_only"]
    )
    with pytest.raises(SystemExit) as excinfo:
        runner_module.main()
    assert excinfo.value.code == 2


def test_parse_args_captures_preset(runner_module) -> None:
    """Verify that --preset argument is correctly parsed."""
    args = runner_module.parse_args(["--preset", "p3-quick"])
    assert args.preset == "p3-quick"


def test_parse_args_preset_defaults_to_none(runner_module) -> None:
    """Verify that preset is None by default."""
    args = runner_module.parse_args([])
    assert args.preset is None
