from dataclasses import replace
from unittest.mock import patch

from ai_scientist import config as ai_config, memory, runner


def _stub_evaluator():
    def _evaluate(boundary_params, *, stage, use_cache=True):
        return {
            "metrics": {
                "minimum_normalized_magnetic_gradient_scale_length": 1.1,
                "aspect_ratio": 2.2,
            },
            "feasibility": 0.0,
            "objective": 2.2,
            "minimize_objective": True,
            "stage": stage,
        }

    return _evaluate


def test_run_presets_emit_expected_stage_history(tmp_path):
    base_config = ai_config.load_experiment_config(
        ai_config.DEFAULT_EXPERIMENT_CONFIG_PATH
    )
    presets = runner._load_run_presets()
    assert presets, "Run presets should be defined in configs/run_presets.yaml"
    for preset_name in sorted(presets):
        runtime = runner.RunnerCLIConfig(
            config_path=base_config.source_config,
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
            run_preset=preset_name,
            planner="deterministic",
        )
        activated = runner._apply_run_preset(runtime)
        expected_stage = "s2" if activated.promote_only else "s1"
        cfg = replace(
            base_config,
            reporting_dir=tmp_path / f"reports_{preset_name}",
            memory_db=tmp_path / f"memory_{preset_name}.db",
        )
        with memory.WorldModel(cfg.memory_db) as wm:
            experiment_id = wm.start_experiment({"problem": cfg.problem}, "stubsha")
            with (
                patch(
                    "ai_scientist.runner._problem_evaluator",
                    return_value=_stub_evaluator(),
                ),
                patch(
                    "ai_scientist.runner._problem_tool_name", return_value="evaluate_p3"
                ),
            ):
                budget_controller = runner.BudgetController(cfg)
                runner._run_cycle(
                    cfg,
                    cycle_index=0,
                    world_model=wm,
                    experiment_id=experiment_id,
                    governance_stage=expected_stage,
                    git_sha="deadbeef",
                    constellaration_sha="deadbeef",
                    surrogate_model=runner.SurrogateBundle(),
                    runtime=activated,
                    budget_controller=budget_controller,
                )
            history = wm.stage_history(experiment_id)
            assert history, "Stage history should record at least one cycle"
            assert history[-1].stage == expected_stage
