import sys

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
