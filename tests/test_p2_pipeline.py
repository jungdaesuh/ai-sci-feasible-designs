from unittest.mock import MagicMock, patch
from ai_scientist.experiment_runner import run_experiment
from ai_scientist.config import ExperimentConfig
from ai_scientist.experiment_setup import RunnerCLIConfig
from ai_scientist.problems import get_problem
from pathlib import Path


def test_p2_problem_definition():
    """Verify P2 problem definition matches requirements."""
    problem = get_problem("p2")
    assert problem.name == "p2"
    assert "aspect_ratio" in problem.constraint_names
    assert "qi" in problem.constraint_names
    assert "max_elongation" in problem.constraint_names

    # Verify constraint bounds (indirectly via private attributes or behavior)
    assert problem._aspect_ratio_upper_bound == 10.0
    assert problem._log10_qi_upper_bound == -4.0


@patch("ai_scientist.experiment_runner.CycleExecutor")
@patch("ai_scientist.experiment_runner.memory.WorldModel")
@patch("ai_scientist.experiment_runner.rag.ensure_index")
def test_p2_pipeline_execution(
    mock_ensure_index, mock_world_model, mock_cycle_executor
):
    """Test that running with --problem p2 initializes the correct components."""

    # Setup mocks
    mock_ensure_index.return_value = MagicMock(chunks_indexed=10, index_path="dummy")
    mock_wm_instance = MagicMock()
    mock_world_model.return_value.__enter__.return_value = mock_wm_instance
    mock_wm_instance.start_experiment.return_value = 123

    # Config for P2
    # Use real dataclasses where asdict is called
    from ai_scientist.config import (
        BoundaryTemplateConfig,
        BudgetConfig,
        AdaptiveBudgetConfig,
        ProposalMixConfig,
        FidelityLadder,
        StageGateConfig,
        GovernanceConfig,
        ASOConfig,
    )

    cfg = MagicMock(spec=ExperimentConfig)
    cfg.problem = "p2"
    cfg.cycles = 1
    cfg.random_seed = 42

    # Real dataclasses for fields that get serialized via asdict
    # We need to provide required fields for these dataclasses
    cfg.budgets = BudgetConfig(
        screen_evals_per_cycle=10,
        promote_top_k=1,
        max_high_fidelity_evals_per_cycle=1,
        wall_clock_minutes=1.0,
        n_workers=1,
        pool_type="thread",
    )
    cfg.adaptive_budgets = AdaptiveBudgetConfig(
        enabled=False,
        hv_slope_reference=0.0,
        feasibility_target=0.0,
        cache_hit_target=0.0,
        screen_bounds=MagicMock(),
        promote_top_k_bounds=MagicMock(),
        high_fidelity_bounds=MagicMock(),
    )
    cfg.proposal_mix = ProposalMixConfig(constraint_ratio=0.5, exploration_ratio=0.5)
    cfg.fidelity_ladder = FidelityLadder(screen="s1", promote="s2")
    cfg.boundary_template = BoundaryTemplateConfig(
        n_poloidal_modes=1,
        n_toroidal_modes=1,
        n_field_periods=1,
        base_major_radius=1.0,
        base_minor_radius=0.1,
        perturbation_scale=0.0,
    )
    cfg.stage_gates = StageGateConfig(
        s1_to_s2_feasibility_margin=0.0,
        s1_to_s2_objective_improvement=0.0,
        s1_to_s2_lookback_cycles=1,
        s2_to_s3_hv_delta=0.0,
        s2_to_s3_lookback_cycles=1,
    )
    cfg.governance = GovernanceConfig(min_feasible_for_promotion=1, hv_lookback=1)
    cfg.aso = ASOConfig(enabled=False)

    cfg.reporting = {}
    cfg.planner = "deterministic"
    cfg.run_overrides = {}
    cfg.source_config = "dummy_path"  # Needed for serialization
    cfg.optimizer_backend = "gradient_descent"

    # Add surrogate and generative mocks
    cfg.surrogate = MagicMock()
    cfg.surrogate.backend = "mock_backend"
    cfg.generative = MagicMock()
    cfg.generative.enabled = False
    runtime = RunnerCLIConfig(
        config_path=Path("dummy"),
        problem="p2",
        cycles=1,
        memory_db=None,
        eval_budget=None,
        workers=None,
        pool_type=None,
        screen_only=True,
        promote_only=False,
        slow=False,
        verbose=False,
        log_cache_stats=False,
        run_preset=None,
        planner="deterministic",
        resume_from=None,
        aso=False,
        preset=None,
    )

    # Configure mock CycleExecutor to return a serializable CycleResult
    from ai_scientist.cycle_executor import CycleResult

    mock_result = CycleResult(
        cycle_index=0,
        candidates_evaluated=10,
        candidates_promoted=1,
        best_objective=0.5,
        hypervolume=0.8,
        feasibility_rate=0.9,
        report_path=None,
        best_eval={"objective": 0.5, "feasibility": 0.0, "design_hash": "abc"},
        p3_summary=None,
    )
    mock_cycle_executor.return_value.run_cycle.return_value = mock_result

    # Run experiment
    run_experiment(cfg, runtime=runtime)

    # Verify CycleExecutor was initialized (it handles the problem logic internally via config)
    # The key check is that the runner didn't crash and passed the problem override to the experiment config
    # Since Coordinator is no longer used, we verify via CycleExecutor
    mock_cycle_executor.assert_called_once()
    _, kwargs = mock_cycle_executor.call_args
    passed_cfg = kwargs.get("config")
    assert passed_cfg.problem == "p2"

    # Verify start_experiment was called
    mock_wm_instance.start_experiment.assert_called_once()
