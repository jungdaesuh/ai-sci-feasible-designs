import sys
from unittest.mock import MagicMock

# Mock heavy dependencies to avoid ImportError due to broken environment (libtorch)
sys.modules["torch"] = MagicMock()
sys.modules["torch.nn"] = MagicMock()
sys.modules["torch.nn.functional"] = MagicMock()
sys.modules["torch.optim"] = MagicMock()
sys.modules["torch.distributions"] = MagicMock()
sys.modules["torch.utils"] = MagicMock()
sys.modules["torch.utils.data"] = MagicMock()
sys.modules["vmecpp"] = MagicMock()
sys.modules["ai_scientist.coordinator"] = MagicMock()
sys.modules["ai_scientist.forward_model"] = MagicMock()
sys.modules["ai_scientist.optim.surrogate_v2"] = MagicMock()

import pytest
from ai_scientist import runner

def test_parser_help_mentions_screen_stage() -> None:
    help_text = runner._build_argument_parser().format_help()
    assert "S1" in help_text
    assert "screen" in help_text.lower()
    assert "promote" in help_text.lower()


def test_parse_args_errors_when_screen_and_promote_both_set() -> None:
    with pytest.raises(SystemExit) as excinfo:
        runner.parse_args(["--screen", "--promote"])
    assert excinfo.value.code == 2


def test_main_exits_when_screen_flag_conflicts_with_promote_preset(monkeypatch) -> None:
    monkeypatch.setattr(
        sys, "argv", ["runner", "--screen", "--run-preset", "promote_only"]
    )
    with pytest.raises(SystemExit) as excinfo:
        runner.main()
    assert excinfo.value.code == 2


def test_parse_args_captures_preset() -> None:
    """Verify that --preset argument is correctly parsed."""
    args = runner.parse_args(["--preset", "p3-quick"])
    assert args.preset == "p3-quick"


def test_parse_args_preset_defaults_to_none() -> None:
    """Verify that preset is None by default."""
    args = runner.parse_args([])
    assert args.preset is None
