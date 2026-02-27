import sys
from dataclasses import replace


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


def test_parse_args_no_rl_sets_flag(runner_module) -> None:
    args = runner_module.parse_args(["--no-rl"])
    assert args.disable_rl is True


def test_main_enables_aso_by_default_for_agent_planner(
    runner_module, monkeypatch, tmp_path
) -> None:
    cli = runner_module.RunnerCLIConfig(
        config_path=tmp_path / "cfg.yaml",
        problem=None,
        cycles=None,
        memory_db=None,
        eval_budget=None,
        workers=None,
        pool_type=None,
        screen_only=False,
        promote_only=False,
        slow=False,
        verbose=False,
        log_cache_stats=False,
        run_preset=None,
        planner="agent",
        aso=False,
        preset=None,
    )

    base = replace(
        runner_module.ai_config.ExperimentConfig.p3_quick_validation(),
        reporting_dir=tmp_path / "reports",
        memory_db=tmp_path / "world.db",
    )

    captured = {}

    def _fake_run_experiment(experiment, runtime=None):
        captured["experiment"] = experiment
        captured["runtime"] = runtime

    runner_module.main(
        parse_args_fn=lambda: cli,
        apply_run_preset_fn=lambda runtime: runtime,
        validate_runtime_flags_fn=lambda runtime: None,
        load_experiment_config_fn=lambda _p: base,
        run_experiment_fn=_fake_run_experiment,
    )

    assert captured["runtime"].planner == "agent"
    assert captured["experiment"].planner == "agent"
    assert captured["experiment"].aso.enabled is True
