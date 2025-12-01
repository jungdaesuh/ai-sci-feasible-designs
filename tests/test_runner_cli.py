import sys
from unittest.mock import MagicMock, patch

import pytest




def test_parser_help_mentions_screen_stage(runner_module) -> None:
    help_text = runner_module.build_argument_parser().format_help()
    assert "S1" in help_text
    assert "screen" in help_text.lower()
    assert "promote" in help_text.lower()


def test_parse_args_errors_when_screen_and_promote_both_set(runner_module) -> None:
    with pytest.raises(SystemExit) as excinfo:
        runner_module.parse_args(["--screen", "--promote"])
    assert excinfo.value.code == 2


def test_main_exits_when_screen_flag_conflicts_with_promote_preset(
    runner_module, monkeypatch
) -> None:
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
