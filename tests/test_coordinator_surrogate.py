import unittest
from unittest.mock import MagicMock, patch
import sys


class TestCoordinatorSurrogate(unittest.TestCase):
    def setUp(self):
        # Create mocks
        self.mock_modules = {
            "simsopt": MagicMock(),
            "simsoptpp": MagicMock(),
            "simsopt.geo": MagicMock(),
            "simsopt.field": MagicMock(),
            "pymoo": MagicMock(),
            "pymoo.indicators": MagicMock(),
            "pymoo.indicators.hv": MagicMock(),
            "jax": MagicMock(),
            "jax.numpy": MagicMock(),
            "jax.tree_util": MagicMock(),
            "constellaration": MagicMock(),
            "constellaration.geometry": MagicMock(),
            "constellaration.geometry.surface_rz_fourier": MagicMock(),
            "constellaration.optimization": MagicMock(),
            "constellaration.optimization.augmented_lagrangian": MagicMock(),
            "constellaration.optimization.settings": MagicMock(),
            "constellaration.forward_model": MagicMock(),
            "constellaration.problems": MagicMock(),
            "constellaration.utils": MagicMock(),
            "constellaration.initial_guess": MagicMock(),
            "jaxtyping": MagicMock(),
            "nevergrad": MagicMock(),
            "torch": MagicMock(),
            "torch.nn": MagicMock(),
            "torch.nn.functional": MagicMock(),
            "torch.optim": MagicMock(),
            "torch.utils": MagicMock(),
            "torch.utils.data": MagicMock(),
        }

        # Start patcher
        self.modules_patcher = patch.dict(sys.modules, self.mock_modules)
        self.modules_patcher.start()

        # Import modules under test (now using mocks)
        # We need to reload them to ensure they pick up the mocks
        import ai_scientist.coordinator
        import ai_scientist.optim.surrogate
        import importlib

        importlib.reload(ai_scientist.optim.surrogate)
        importlib.reload(ai_scientist.coordinator)

        self.Coordinator = ai_scientist.coordinator.Coordinator
        self.SurrogatePrediction = ai_scientist.optim.surrogate.SurrogatePrediction

        # Mock configuration
        self.cfg = MagicMock()
        self.cfg.proposal_mix.surrogate_pool_multiplier = 5.0
        self.cfg.proposal_mix.exploration_ratio = 0.1
        self.cfg.boundary_template = MagicMock()

        self.mock_world_model = MagicMock()
        self.mock_planner = MagicMock()
        self.mock_surrogate = MagicMock()
        self.mock_surrogate._trained = True

        self.coordinator = self.Coordinator(
            cfg=self.cfg,
            world_model=self.mock_world_model,
            planner=self.mock_planner,
            surrogate=self.mock_surrogate,
        )

        # Mock workers to avoid real computation
        self.coordinator.explore_worker = MagicMock()
        self.coordinator.geo_worker = MagicMock()
        self.coordinator.opt_worker = MagicMock()

    def tearDown(self):
        self.modules_patcher.stop()
        # Reload real modules if they were loaded before
        # This is a bit heavy-handed but ensures safety
        if "ai_scientist.coordinator" in sys.modules:
            import ai_scientist.coordinator
            import importlib

            importlib.reload(ai_scientist.coordinator)

    def test_surrogate_rank_seeds(self):
        # Setup seeds
        seeds = [{"id": 1, "params": {}}, {"id": 2, "params": {}}]

        # Setup surrogate predictions
        # Setup surrogate predictions
        pred1 = self.SurrogatePrediction(
            expected_value=0.8,
            prob_feasible=0.9,
            predicted_objective=0.5,
            minimize_objective=True,
            metadata=seeds[0],
        )
        pred2 = self.SurrogatePrediction(
            expected_value=0.2,
            prob_feasible=0.4,
            predicted_objective=0.9,
            minimize_objective=True,
            metadata=seeds[1],
        )

        # Mock rank_candidates to return sorted predictions (best first)
        self.mock_surrogate.rank_candidates.return_value = [pred1, pred2]

        # Call method
        ranked = self.coordinator._surrogate_rank_seeds(seeds, cycle=1)

        # Verify
        self.assertEqual(len(ranked), 2)
        self.assertEqual(ranked[0]["id"], 1)
        self.assertEqual(ranked[1]["id"], 2)
        self.assertEqual(ranked[0]["surrogate_score"], 0.8)
        self.assertEqual(ranked[1]["surrogate_score"], 0.2)
        self.assertEqual(ranked[0]["surrogate_rank"], 0)

        # Verify surrogate call
        self.mock_surrogate.rank_candidates.assert_called_once()

    def test_surrogate_rank_seeds_untrained(self):
        self.mock_surrogate._trained = False
        seeds = [{"id": 1}, {"id": 2}]

        ranked = self.coordinator._surrogate_rank_seeds(seeds, cycle=1)

        # Should return seeds (shuffled, but length same)
        self.assertEqual(len(ranked), 2)
        self.mock_surrogate.rank_candidates.assert_not_called()

    @patch("ai_scientist.coordinator.TrajectoryState")
    @patch("ai_scientist.coordinator.create_alm_context")
    @patch("ai_scientist.coordinator.step_alm")
    def test_produce_candidates_aso_uses_surrogate(
        self, mock_step, mock_create_alm, mock_traj_cls
    ):
        # Setup
        n_needed = 1
        pool_size = int(
            max(10, n_needed * self.cfg.proposal_mix.surrogate_pool_multiplier)
        )  # Should be 10

        # Mock prepare_seeds to return pool_size seeds
        mock_seeds = [
            {"id": i, "params": {"r_cos": [1.0], "z_sin": [0.0]}}
            for i in range(pool_size)
        ]
        self.coordinator._prepare_seeds = MagicMock(return_value=mock_seeds)

        # Mock surrogate ranking to reverse them (so id=4 is best)
        # We just mock _surrogate_rank_seeds directly to simplify
        ranked_seeds = sorted(mock_seeds, key=lambda x: x["id"], reverse=True)
        ranked_seeds[0]["surrogate_score"] = 0.99
        self.coordinator._surrogate_rank_seeds = MagicMock(return_value=ranked_seeds)

        # Mock ALM loop internals to exit immediately
        mock_traj_instance = MagicMock()
        mock_traj_instance.budget_used = 100  # Stop immediately
        mock_traj_instance.status = "active"
        mock_traj_instance.seed = ranked_seeds[0]
        mock_traj_instance.model_copy.return_value = mock_traj_instance
        mock_traj_cls.return_value = mock_traj_instance

        mock_create_alm.return_value = (MagicMock(), MagicMock())

        # Call
        self.coordinator.produce_candidates_aso(
            cycle=1,
            experiment_id=1,
            eval_budget=100,
            template=self.cfg.boundary_template,
        )

        # Verify
        # 1. Check pool generation size
        self.coordinator._prepare_seeds.assert_called_with(None, 1, pool_size)

        # 2. Check ranking called
        self.coordinator._surrogate_rank_seeds.assert_called_with(mock_seeds, 1)

        # 3. Check best seed selected (id=4)
        # The trajectory should be initialized with the first ranked seed
        # We can check the call to TrajectoryState constructor or the print output
        # Here we check that the trajectory was created with the best seed
        # Note: We mocked TrajectoryState, so we check the call args
        call_args = mock_traj_cls.call_args
        self.assertEqual(call_args.kwargs["seed"]["id"], 9)


if __name__ == "__main__":
    unittest.main()
