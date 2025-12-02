import unittest
from unittest.mock import MagicMock, patch
import sys

# Mock simsopt/simsoptpp before importing anything that uses them
sys.modules["simsopt"] = MagicMock()
sys.modules["simsoptpp"] = MagicMock()
sys.modules["simsopt.geo"] = MagicMock()
sys.modules["simsopt.field"] = MagicMock()
sys.modules["pymoo"] = MagicMock()
sys.modules["pymoo.indicators"] = MagicMock()
sys.modules["pymoo.indicators.hv"] = MagicMock()

# Create a package-like mock for jax
jax = MagicMock()
jax.numpy = MagicMock()
jax.tree_util = MagicMock()
sys.modules["jax"] = jax
sys.modules["jax.numpy"] = jax.numpy
sys.modules["jax.tree_util"] = jax.tree_util

sys.modules["constellaration"] = MagicMock()
sys.modules["constellaration.geometry"] = MagicMock()
sys.modules["constellaration.geometry.surface_rz_fourier"] = MagicMock()
sys.modules["constellaration.optimization"] = MagicMock()
sys.modules["constellaration.optimization.augmented_lagrangian"] = MagicMock()
sys.modules["constellaration.optimization.settings"] = MagicMock()
sys.modules["constellaration.forward_model"] = MagicMock()
sys.modules["constellaration.problems"] = MagicMock()
sys.modules["constellaration.utils"] = MagicMock()
sys.modules["constellaration.initial_guess"] = MagicMock()
sys.modules["jaxtyping"] = MagicMock()
sys.modules["nevergrad"] = MagicMock()
# Create a package-like mock for torch
torch = MagicMock()
torch.nn = MagicMock()
torch.nn.functional = MagicMock()
torch.optim = MagicMock()
torch.utils = MagicMock()
torch.utils.data = MagicMock()
sys.modules["torch"] = torch
sys.modules["torch.nn"] = torch.nn
sys.modules["torch.nn.functional"] = torch.nn.functional
sys.modules["torch.optim"] = torch.optim
sys.modules["torch.utils"] = torch.utils
sys.modules["torch.utils.data"] = torch.utils.data

from ai_scientist.coordinator import Coordinator  # noqa: E402
from ai_scientist.optim.surrogate import SurrogatePrediction  # noqa: E402


class TestCoordinatorSurrogate(unittest.TestCase):
    def setUp(self):
        # Mock configuration
        self.cfg = MagicMock()
        self.cfg.proposal_mix.surrogate_pool_multiplier = 5.0
        self.cfg.proposal_mix.exploration_ratio = 0.1
        self.cfg.boundary_template = MagicMock()

        self.mock_world_model = MagicMock()
        self.mock_planner = MagicMock()
        self.mock_surrogate = MagicMock()
        self.mock_surrogate._trained = True

        self.coordinator = Coordinator(
            cfg=self.cfg,
            world_model=self.mock_world_model,
            planner=self.mock_planner,
            surrogate=self.mock_surrogate,
        )

        # Mock workers to avoid real computation
        self.coordinator.explore_worker = MagicMock()
        self.coordinator.geo_worker = MagicMock()
        self.coordinator.opt_worker = MagicMock()

    def test_surrogate_rank_seeds(self):
        # Setup seeds
        seeds = [{"id": 1, "params": {}}, {"id": 2, "params": {}}]

        # Setup surrogate predictions
        pred1 = SurrogatePrediction(
            expected_value=0.8,
            prob_feasible=0.9,
            predicted_objective=0.5,
            minimize_objective=True,
            metadata=seeds[0],
        )
        pred2 = SurrogatePrediction(
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
