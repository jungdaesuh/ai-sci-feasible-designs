import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from ai_scientist.cycle_executor import CycleExecutor
from ai_scientist import config as ai_config
from ai_scientist import memory


@pytest.fixture
def mock_config():
    # Create a minimal valid config
    mix = ai_config.ProposalMixConfig(
        constraint_ratio=0.5,
        exploration_ratio=0.5,
        exploitation_ratio=0.0,
        jitter_scale=0.01,
        surrogate_pool_multiplier=2.0,
        sampler_type="standard",
    )
    budgets = ai_config.BudgetConfig(
        screen_evals_per_cycle=10,
        promote_top_k=5,
        max_high_fidelity_evals_per_cycle=5,
        wall_clock_minutes=1.0,
        n_workers=1,
        pool_type="process",
    )
    br = ai_config.BudgetRangeConfig(min=1, max=100)
    adaptive = ai_config.AdaptiveBudgetConfig(
        enabled=False,
        hv_slope_reference=0.1,
        feasibility_target=0.1,
        cache_hit_target=0.5,
        screen_bounds=br,
        promote_top_k_bounds=br,
        high_fidelity_bounds=br,
    )
    ladder = ai_config.FidelityLadder(screen="low", promote="high")
    boundary = ai_config.BoundaryTemplateConfig(
        n_poloidal_modes=1,
        n_toroidal_modes=1,
        n_field_periods=1,
        base_major_radius=1.0,
        base_minor_radius=0.1,
        perturbation_scale=0.01,
        seed_path=None,
    )
    gates = ai_config.StageGateConfig(
        s1_to_s2_feasibility_margin=0.0,
        s1_to_s2_objective_improvement=0.0,
        s1_to_s2_lookback_cycles=1,
        s2_to_s3_hv_delta=0.0,
        s2_to_s3_lookback_cycles=1,
    )
    gov = ai_config.GovernanceConfig(min_feasible_for_promotion=1, hv_lookback=1)
    weights = ai_config.ConstraintWeightsConfig(mhd=1.0, qi=1.0, elongation=1.0)
    gen = ai_config.GenerativeConfig(enabled=False)
    sur = ai_config.SurrogateConfig(
        backend="random_forest",
        n_ensembles=1,
        learning_rate=0.001,
        epochs=100,
        hidden_dim=64,
        use_offline_dataset=False,
    )
    alm = ai_config.ALMConfig(
        penalty_parameters_increase_factor=2.0,
        constraint_violation_tolerance_reduction_factor=0.5,
        bounds_reduction_factor=0.95,
        penalty_parameters_max=1e8,
        bounds_min=0.05,
        maxit=25,
        penalty_parameters_initial=1.0,
        bounds_initial=2.0,
        oracle_budget_initial=100,
        oracle_budget_increment=26,
        oracle_budget_max=200,
        oracle_num_workers=4,
    )
    aso = ai_config.ASOConfig(
        enabled=False,
        supervision_mode="event_triggered",
        supervision_interval=5,
        feasibility_threshold=1e-3,
        stagnation_objective_threshold=1e-5,
        stagnation_violation_threshold=0.05,
        max_stagnation_steps=5,
        violation_increase_threshold=0.05,
        violation_decrease_threshold=0.05,
        steps_per_supervision=1,
        max_constraint_weight=1000.0,
        max_penalty_boost=4.0,
        llm_timeout_seconds=10.0,
        llm_max_retries=2,
        use_heuristic_fallback=True,
    )

    return ai_config.ExperimentConfig(
        problem="p3",
        cycles=5,
        random_seed=42,
        budgets=budgets,
        adaptive_budgets=adaptive,
        proposal_mix=mix,
        fidelity_ladder=ladder,
        boundary_template=boundary,
        stage_gates=gates,
        governance=gov,
        source_config=Path("test.yaml"),
        reporting_dir=Path("reports"),
        memory_db=Path("memory.db"),
        constraint_weights=weights,
        generative=gen,
        surrogate=sur,
        alm=alm,
        aso=aso,
        optimizer_backend="nevergrad",
        initialization_strategy="template",
        run_overrides={},
        reporting={},
        planner="deterministic",
    )


@patch("ai_scientist.cycle_executor.Coordinator")
def test_coordinator_rebuilt_with_overrides(MockCoordinator, mock_config):
    # Setup
    mock_world_model = MagicMock(spec=memory.WorldModel)
    mock_budget_controller = MagicMock()
    mock_budget_controller.snapshot.return_value = MagicMock(
        screen_evals_per_cycle=10, promote_top_k=5, max_high_fidelity_evals_per_cycle=5
    )
    mock_fidelity_controller = MagicMock()

    executor = CycleExecutor(
        config=mock_config,
        world_model=mock_world_model,
        planner=MagicMock(),
        budget_controller=mock_budget_controller,
        fidelity_controller=mock_fidelity_controller,
    )

    # Override budgets
    overrides = {"budgets": {"screen_evals_per_cycle": 999}}

    # Execute
    # We need to mock internal calls to avoid crash
    with (
        patch(
            "ai_scientist.cycle_executor._propose_p3_candidates_for_cycle"
        ) as mock_propose,
        patch(
            "ai_scientist.cycle_executor._surrogate_rank_screen_candidates"
        ) as mock_rank,
        patch(
            "ai_scientist.cycle_executor.tools.summarize_p3_candidates"
        ) as mock_summary,
    ):
        mock_propose.return_value = ([], 0, 0, 0)  # Return empty candidates
        mock_rank.return_value = []

        # Fix: Mock summary object needs integer properties for comparisons
        mock_summary_obj = MagicMock()
        mock_summary_obj.feasible_count = 0
        mock_summary_obj.hv_score = 0.0
        mock_summary.return_value = mock_summary_obj

        executor.run_cycle(
            cycle_index=1,
            experiment_id=1,
            governance_stage="test",
            git_sha="sha",
            constellaration_sha="sha",
            surrogate_model=MagicMock(),
            config_overrides=overrides,
        )

    # Verification
    # Coordinator should have been instantiated with the active config containing the override
    assert MockCoordinator.called
    call_args = MockCoordinator.call_args
    passed_config = call_args[0][0]  # First arg is config

    assert passed_config.budgets.screen_evals_per_cycle == 999
    # Verify it was NOT the initial coordinator reused (or if it was reused, it was updated - but our fix forces rebuild)
    # Since we mocked the class, if it was reused, the class wouldn't be called.
    # So MockCoordinator.called proves it was rebuilt.


@patch("ai_scientist.cycle_executor.Coordinator")
def test_generative_model_fit_called_once(MockCoordinator, mock_config):
    # Setup
    mock_world_model = MagicMock(spec=memory.WorldModel)
    # Return some history to trigger fit
    mock_world_model.surrogate_training_data.return_value = [({"p": 1}, 0.5)]

    mock_budget_controller = MagicMock()
    mock_budget_controller.snapshot.return_value = MagicMock(
        screen_evals_per_cycle=10, promote_top_k=5, max_high_fidelity_evals_per_cycle=5
    )
    mock_fidelity_controller = MagicMock()
    mock_gen_model = MagicMock()

    executor = CycleExecutor(
        config=mock_config,
        world_model=mock_world_model,
        planner=MagicMock(),
        budget_controller=mock_budget_controller,
        fidelity_controller=mock_fidelity_controller,
    )

    # Execute
    with (
        patch(
            "ai_scientist.cycle_executor._propose_p3_candidates_for_cycle"
        ) as mock_propose,
        patch(
            "ai_scientist.cycle_executor._surrogate_rank_screen_candidates"
        ) as mock_rank,
        patch("ai_scientist.cycle_executor.tools.get_cache_stats"),
    ):
        mock_propose.return_value = ([], 0, 0, 0)
        mock_rank.return_value = []

        executor.run_cycle(
            cycle_index=1,
            experiment_id=1,
            governance_stage="test",
            git_sha="sha",
            constellaration_sha="sha",
            surrogate_model=MagicMock(),
            generative_model=mock_gen_model,
        )

    # Verification
    # fit should be called exactly once
    assert mock_gen_model.fit.call_count == 1
