import unittest
from unittest.mock import MagicMock, patch

from ai_scientist.cycle_executor import CycleExecutor
from ai_scientist.problems import P2Problem, get_problem


class TestP2Pipeline(unittest.TestCase):
    def test_p2_problem_definition(self):
        """Verify P2 problem constraints and objective."""
        p2 = get_problem("p2")
        self.assertIsInstance(p2, P2Problem)
        self.assertEqual(p2.name, "p2")

        # Test constraints
        # 1. aspect_ratio <= 10.0
        # 2. iota >= 0.25
        # 3. log10(qi) <= -4.0
        # 4. mirror <= 0.2
        # 5. elongation <= 5.0

        # Case 1: Feasible
        metrics_feasible = {
            "aspect_ratio": 9.0,
            "edge_rotational_transform_over_n_field_periods": 0.3,
            "qi": 1e-5,  # log10 = -5
            "edge_magnetic_mirror_ratio": 0.1,
            "max_elongation": 4.0,
            "minimum_normalized_magnetic_gradient_scale_length": 2.0,
        }
        self.assertTrue(p2.is_feasible(metrics_feasible))
        self.assertEqual(p2.compute_feasibility(metrics_feasible), 0.0)

        # Case 2: Infeasible (Aspect Ratio)
        metrics_infeasible = metrics_feasible.copy()
        metrics_infeasible["aspect_ratio"] = 11.0
        self.assertFalse(p2.is_feasible(metrics_infeasible))
        self.assertGreater(p2.compute_feasibility(metrics_infeasible), 0.0)

        # Test Objective (Maximize gradient scale length -> minimize negative)
        obj = p2.get_objective(metrics_feasible)
        self.assertEqual(obj, -2.0)

    @patch("ai_scientist.cycle_executor.tools")
    def test_p2_cycle_execution_mock(self, mock_tools):
        """
        Test that CycleExecutor calls the correct P2 evaluator.
        We mock the heavy components to verify orchestration logic.
        """
        # Use real config to avoid dataclasses.replace issues
        from ai_scientist.config import load_experiment_config
        from ai_scientist.budget_manager import BudgetSnapshot

        real_config = load_experiment_config()
        # Override for P2 test
        from dataclasses import replace

        real_config = replace(
            real_config, problem="p2", optimizer_backend="gradient_descent"
        )

        # Mock components
        mock_world_model = MagicMock()
        mock_planner = MagicMock()
        mock_coordinator = MagicMock()
        mock_budget_controller = MagicMock()
        mock_fidelity_controller = MagicMock()

        # Mock budget_controller.snapshot() to return a proper BudgetSnapshot
        mock_budget_snapshot = BudgetSnapshot(
            screen_evals_per_cycle=10,
            promote_top_k=2,
            max_high_fidelity_evals_per_cycle=5,
            remaining_budget=100,
        )
        mock_budget_controller.snapshot.return_value = mock_budget_snapshot
        mock_budget_controller.consume = MagicMock()
        mock_budget_controller.adjust_for_cycle = MagicMock()
        mock_budget_controller.capture_cache_hit_rate = MagicMock(return_value=0.5)
        mock_budget_controller.to_dict = MagicMock(return_value={})

        # Mock world_model methods
        mock_world_model.surrogate_training_data = MagicMock(return_value=[])
        mock_world_model.previous_best_hv = MagicMock(return_value=0.0)
        mock_world_model.record_cycle = MagicMock()
        mock_world_model.record_cycle_summary = MagicMock()
        mock_world_model.record_cycle_hv = MagicMock()
        mock_world_model.record_deterministic_snapshot = MagicMock()
        mock_world_model.log_candidate = MagicMock(return_value=(1, 1))
        mock_world_model.log_artifact = MagicMock()
        mock_world_model.record_stage_history = MagicMock()
        mock_world_model.statements_for_cycle = MagicMock(return_value=[])
        mock_world_model.stage_history = MagicMock(return_value=[])
        mock_world_model.property_graph_summary = MagicMock(return_value={})
        mock_world_model.transaction = MagicMock(
            return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock())
        )
        mock_world_model.upsert_pareto = MagicMock()
        mock_world_model.record_pareto_archive = MagicMock()

        # Setup CycleExecutor
        executor = CycleExecutor(
            config=real_config,
            world_model=mock_world_model,
            planner=mock_planner,
            coordinator=mock_coordinator,
            budget_controller=mock_budget_controller,
            fidelity_controller=mock_fidelity_controller,
        )

        # Mock Coordinator to return candidates
        mock_coordinator.produce_candidates.return_value = [
            {
                "params": {
                    "r_cos": [[1.0, 0.0, 0.0]],
                    "z_sin": [[0.0, 0.5, 0.0]],
                    "n_field_periods": 1,
                    "is_stellarator_symmetric": True,
                },
                "design_hash": "test_hash_123",
                "seed": 42,
            }
        ]

        # Mock Surrogate Ranking
        with patch(
            "ai_scientist.cycle_executor._surrogate_rank_screen_candidates"
        ) as mock_rank:
            mock_rank.return_value = [
                {
                    "params": {
                        "r_cos": [[1.0, 0.0, 0.0]],
                        "z_sin": [[0.0, 0.5, 0.0]],
                        "n_field_periods": 1,
                        "is_stellarator_symmetric": True,
                    },
                    "design_hash": "test_hash_123",
                    "seed": 42,
                }
            ]

            # Mock Fidelity Controller to return results
            mock_fidelity_controller.evaluate_stage.return_value = [
                {
                    "params": {
                        "r_cos": [[1.0, 0.0, 0.0]],
                        "z_sin": [[0.0, 0.5, 0.0]],
                        "n_field_periods": 1,
                        "is_stellarator_symmetric": True,
                    },
                    "design_hash": "test_hash_123",
                    "seed": 42,
                    "evaluation": {
                        "aspect_ratio": 9.0,
                        "edge_rotational_transform_over_n_field_periods": 0.3,
                        "qi": 1e-5,
                        "edge_magnetic_mirror_ratio": 0.1,
                        "max_elongation": 4.0,
                        "minimum_normalized_magnetic_gradient_scale_length": 2.0,
                        "objective": -2.0,
                        "feasibility": 0.0,
                        "stage": "p2",
                        "metrics": {
                            "aspect_ratio": 9.0,
                            "edge_rotational_transform_over_n_field_periods": 0.3,
                            "qi": 1e-5,
                            "edge_magnetic_mirror_ratio": 0.1,
                            "max_elongation": 4.0,
                            "minimum_normalized_magnetic_gradient_scale_length": 2.0,
                        },
                    },
                }
            ]
            mock_fidelity_controller.get_promotion_candidates = MagicMock(
                return_value=[]
            )

            # Mock tools.summarize_p3_candidates (it's named p3 but used generically in runner)
            # We need to ensure it doesn't crash with P2 data
            from ai_scientist.tools.hypervolume import P3Summary, ParetoEntry

            mock_summary = P3Summary(
                feasible_count=1,
                hv_score=0.5,
                reference_point=(1.0, 20.0),
                archive_size=1,
                pareto_entries=[
                    ParetoEntry(
                        design_hash="test_hash_123",
                        seed=42,
                        gradient=2.0,
                        aspect_ratio=9.0,
                        objective=-2.0,
                        feasibility=0.0,
                        stage="p2",
                    )
                ],
            )
            mock_tools.summarize_p3_candidates.return_value = mock_summary
            mock_tools.get_cache_stats = MagicMock(return_value={})
            mock_tools.design_hash = MagicMock(return_value="test_hash_123")
            mock_tools.compute_constraint_margins = MagicMock(return_value={})

            # Run Cycle
            result = executor.run_cycle(
                cycle_index=0,
                experiment_id=1,
                governance_stage="s1",
                git_sha="abc",
                constellaration_sha="def",
                surrogate_model=MagicMock(),
                verbose=False,
            )

            # Verify P2 evaluator was requested
            # The runner uses _problem_tool_name("p2") -> "evaluate_p2"
            # fidelity_controller.evaluate_stage should receive tool_name="evaluate_p2"
            mock_fidelity_controller.evaluate_stage.assert_called()
            call_args = mock_fidelity_controller.evaluate_stage.call_args
            self.assertEqual(call_args.kwargs["tool_name"], "evaluate_p2")

            # Verify result
            self.assertEqual(result.candidates_evaluated, 1)
