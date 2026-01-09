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
            "torch.utils.checkpoint": MagicMock(),
        }

        # Start patcher
        self.modules_patcher = patch.dict(sys.modules, self.mock_modules)
        self.modules_patcher.start()

        # Import modules under test (now using mocks)
        # We need to reload them to ensure they pick up the mocks
        import ai_scientist.coordinator
        import ai_scientist.optim.alm_bridge
        import ai_scientist.optim.surrogate
        import importlib

        if "ai_scientist.optim.alm_bridge" in sys.modules:
            mod = sys.modules["ai_scientist.optim.alm_bridge"]
            importlib.reload(mod)
        else:
            import ai_scientist.optim.alm_bridge

        if "ai_scientist.optim.surrogate" in sys.modules:
            mod = sys.modules["ai_scientist.optim.surrogate"]
            importlib.reload(mod)
        else:
            import ai_scientist.optim.surrogate

        if "ai_scientist.coordinator" in sys.modules:
            mod = sys.modules["ai_scientist.coordinator"]
            importlib.reload(mod)
            self.Coordinator = mod.Coordinator
        else:
            import ai_scientist.coordinator

            self.Coordinator = ai_scientist.coordinator.Coordinator

        self.SurrogatePrediction = sys.modules[
            "ai_scientist.optim.surrogate"
        ].SurrogatePrediction

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

        # Clean up modules that were explicitly reloaded during setUp.
        # Do NOT include ai_scientist.forward_model here - it was never mocked,
        # and deleting it causes pickle identity issues for ProcessPoolExecutor
        # in subsequent tests (the _process_worker_initializer function reference
        # becomes stale when the module is reloaded).
        #
        # Include parent modules (ai_scientist.optim) because they may have
        # cached references to the reloaded submodules. Clean in reverse
        # dependency order (children first) to avoid dangling references.
        modules_to_clean = [
            "ai_scientist.coordinator",
            "ai_scientist.optim.alm_bridge",
            "ai_scientist.optim.surrogate",
            "ai_scientist.optim",  # Parent may cache references to surrogate
        ]

        for mod_name in modules_to_clean:
            if mod_name in sys.modules:
                del sys.modules[mod_name]

        # Force garbage collection of the mock objects to prevent leaks
        self.mock_modules = None
        self.modules_patcher = None

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

    def test_produce_candidates_skips_rl_when_disabled(self):
        # Disable RL refinement via config toggle
        self.cfg.proposal_mix.rl_refinement_enabled = False

        # Ensure HYBRID path (cycle < 5 -> HYBRID)
        self.mock_world_model.average_recent_hv_delta.return_value = None

        seeds = [
            {"id": 1, "params": {"r_cos": [[1.0]], "z_sin": [[0.0]]}},
            {"id": 2, "params": {"r_cos": [[1.0]], "z_sin": [[0.0]]}},
        ]

        self.coordinator.explore_worker.run.return_value = {"candidates": seeds}

        # PreRelax + Geometer pass-through
        self.coordinator.prerelax_worker = MagicMock()
        self.coordinator.prerelax_worker.run.return_value = {"candidates": seeds}
        self.coordinator.geo_worker.run.return_value = {"candidates": seeds}

        # RL worker should not be called at all
        self.coordinator.rl_worker = MagicMock()

        # Optimization worker returns candidates
        self.coordinator.opt_worker.run.return_value = {"candidates": seeds}

        out = self.coordinator.produce_candidates(
            cycle=1,
            experiment_id=1,
            n_candidates=2,
            template=self.cfg.boundary_template,
        )

        self.assertEqual(len(out), 2)
        self.coordinator.rl_worker.run.assert_not_called()
        self.coordinator.opt_worker.run.assert_called_once()

    @patch("ai_scientist.coordinator.TrajectoryState")
    @patch("ai_scientist.coordinator.create_alm_context")
    @patch("ai_scientist.optim.alm_bridge.create_alm_context")
    @patch("ai_scientist.coordinator.step_alm")
    def test_produce_candidates_aso_uses_surrogate(
        self,
        mock_step,
        mock_create_alm_bridge,
        mock_create_alm,
        mock_traj_cls,
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
        mock_create_alm_bridge.return_value = mock_create_alm.return_value
        # Avoid deep ALM execution in this unit test while capturing inputs.
        captured = {}

        def _capture_traj(*args, **kwargs):
            if "traj" in kwargs:
                captured["traj"] = kwargs["traj"]
            elif args:
                captured["traj"] = args[0]
            else:
                captured["traj"] = None
            return []

        self.coordinator._run_trajectory_aso = MagicMock(side_effect=_capture_traj)

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

        # 3. Check best seed selected (id=9)
        # Verify the trajectory passed into the ASO runner uses the top-ranked seed.
        self.assertIn("traj", captured)
        self.assertIsNotNone(captured["traj"])
        self.assertEqual(captured["traj"].seed["id"], 9)


if __name__ == "__main__":
    unittest.main()
