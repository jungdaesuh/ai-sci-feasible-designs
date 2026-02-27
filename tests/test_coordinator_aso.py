# ruff: noqa: E402
import sys
from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest
import jax.numpy as jnp
import numpy as np

# Removed top-level imports to prevent stale references
# from ai_scientist.config import ALMConfig, ASOConfig, ExperimentConfig
# from ai_scientist.coordinator import Coordinator, TrajectoryState
# from ai_scientist.optim.alm_bridge import ALMContext, ALMStepResult
# from ai_scientist.planner import DirectiveAction, DirectiveSource, OptimizationDirective


# Configure mocks to be JSON serializable (behave like Pydantic models)
class MockPydanticModel(MagicMock):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._mock_data = kwargs

    def model_dump(self, *args, **kwargs):
        return self._mock_data

    def dict(self, *args, **kwargs):
        return self._mock_data

    def model_copy(self, *args, **kwargs):
        return self

    def __iter__(self):
        # Prevent Pydantic from treating this as an iterator
        raise TypeError("MockPydanticModel is not iterable")

    def __repr__(self):
        # Stable repr for caching
        return f"MockPydanticModel({self._mock_data})"

    def __getattr__(self, name):
        try:
            data = object.__getattribute__(self, "_mock_data")
            if name in data:
                return data[name]
        except AttributeError:
            pass
        return super().__getattr__(name)


# Helper to create a mock ALM state
def create_mock_alm_state(**kwargs):
    defaults = {
        "x": jnp.zeros(10),
        "multipliers": jnp.zeros(3),
        "penalty_parameters": jnp.ones(3),
        "objective": jnp.array(1.0),
        "constraints": jnp.zeros(3),
        "bounds": jnp.ones(10),
    }
    defaults.update(kwargs)

    # Create a MockPydanticModel instead of raw MagicMock
    state = MockPydanticModel(**defaults)

    # Mock model_copy method explicitly if needed, but MockPydanticModel handles it
    def mock_copy(update=None):
        new_kwargs = defaults.copy()
        if update:
            new_kwargs.update(update)
        return create_mock_alm_state(**new_kwargs)

    state.model_copy = MagicMock(side_effect=mock_copy)
    state.copy = MagicMock(side_effect=mock_copy)
    return state


def _valid_seed(scale: float = 1.0) -> dict[str, object]:
    return {
        "params": {
            "r_cos": [[1.0 * scale]],
            "z_sin": [[0.0]],
            "n_field_periods": 1,
            "is_stellarator_symmetric": True,
        },
        "seed": 123,
    }


class TestCoordinatorASO:
    @pytest.fixture(autouse=True)
    def mock_dependencies(self):
        """Mock external dependencies (vmecpp, constellaration) for this test class."""
        mock_vmecpp = MagicMock()
        mock_constellaration = MagicMock()

        # Configure pytree mock to return proper 2-tuple for mask_and_ravel
        mock_pytree = MagicMock()
        mock_pytree.mask_and_ravel.return_value = (
            MagicMock(),  # flat array (initial_guess)
            lambda x: MagicMock(),  # unravel function
        )

        # Configure ALM module with the State class as MockPydanticModel
        # Pydantic validation requires the class to be consistent with instance
        mock_alm_module = MagicMock()
        mock_alm_module.AugmentedLagrangianState = MockPydanticModel

        mock_modules = {
            "vmecpp": mock_vmecpp,
            "vmecpp.cpp": MagicMock(),
            "vmecpp.cpp._vmecpp": MagicMock(),
            "constellaration": mock_constellaration,
            "constellaration.forward_model": MagicMock(),
            "constellaration.boozer": MagicMock(),
            "constellaration.mhd": MagicMock(),
            "constellaration.geometry": MagicMock(),
            "constellaration.geometry.surface_rz_fourier": MagicMock(),
            "constellaration.optimization": MagicMock(),
            "constellaration.optimization.augmented_lagrangian": mock_alm_module,
            "constellaration.optimization.settings": MagicMock(),
            "constellaration.utils": MagicMock(),
            "constellaration.utils.pytree": mock_pytree,
            "constellaration.problems": MagicMock(),
            "constellaration.initial_guess": MagicMock(),
        }

        # Use patch.dict to safely mock sys.modules for the duration of the test
        with patch.dict(sys.modules, mock_modules):
            # We must reload/re-import coordinator to pick up the mocks
            # If it was already loaded, remove it first
            if "ai_scientist.coordinator" in sys.modules:
                del sys.modules["ai_scientist.coordinator"]
            if "ai_scientist.optim.alm_bridge" in sys.modules:
                del sys.modules["ai_scientist.optim.alm_bridge"]

            yield

            # Cleanup: ensuring polluted modules are removed so they don't affect other tests
            modules_to_clean = [
                "ai_scientist.coordinator",
                "ai_scientist.optim.alm_bridge",
            ]
            for mod in modules_to_clean:
                if mod in sys.modules:
                    del sys.modules[mod]

    @pytest.fixture
    def mock_cfg(self):
        from ai_scientist.config import ALMConfig, ASOConfig, ExperimentConfig

        cfg = MagicMock(spec=ExperimentConfig)
        cfg.aso = ASOConfig(enabled=True)
        cfg.alm = ALMConfig()
        cfg.problem = "p3"
        cfg.random_seed = 123
        cfg.reporting_dir = "reports"
        # Add proposal_mix for produce_candidates_aso
        cfg.proposal_mix = MagicMock()
        cfg.proposal_mix.surrogate_pool_multiplier = 2.0
        return cfg

    @pytest.fixture
    def coordinator(self, mock_cfg):
        # Must import inside the test/fixture so that it happens INSIDE the patch.dict context
        from ai_scientist.coordinator import Coordinator

        with (
            patch("ai_scientist.coordinator.OptimizationWorker"),
            patch("ai_scientist.coordinator.ExplorationWorker"),
            patch("ai_scientist.coordinator.GeometerWorker"),
        ):
            wm = MagicMock()
            wm.average_recent_hv_delta.return_value = None
            planner = MagicMock()
            coord = Coordinator(mock_cfg, wm, planner)
            # Setup constraint names for P3
            coord.constraint_names = ["c1", "c2", "c3"]
            return coord

    def test_generate_diagnostics_extracts_all_fields(self, coordinator):
        """Task 5.7: Verify _generate_diagnostics extracts all ALM fields correctly."""
        # Import inside test to ensure we use the reloaded classes that match the mocks
        from ai_scientist.coordinator import TrajectoryState
        from ai_scientist.optim.alm_bridge import ALMContext

        # Previous state (for delta calculation)
        prev_state = create_mock_alm_state(
            objective=jnp.array(1.1),
            constraints=jnp.array([0.05, 0.0, 0.0]),
        )

        # Current state
        current_state = create_mock_alm_state(
            objective=jnp.array(1.0),
            constraints=jnp.array([0.06, 0.0, 0.0]),  # Increasing violation
            multipliers=jnp.array([0.5, 0.0, 0.0]),
            penalty_parameters=jnp.array([10.0, 1.0, 1.0]),
            bounds=jnp.ones(10) * 0.5,
        )

        # Setup trajectory with history and alm_context (required by _generate_diagnostics)
        mock_alm_context = MagicMock(spec=ALMContext)
        traj = TrajectoryState(
            id=0,
            seed={},
            history=[prev_state, current_state],
            steps=1,
            alm_context=mock_alm_context,
        )

        # Run generation
        diag = coordinator._generate_diagnostics(current_state, traj)

        # Verify fields
        assert diag.objective == 1.0
        assert diag.objective_delta == pytest.approx(-0.1)
        assert diag.max_violation == pytest.approx(0.06)
        assert diag.bounds_norm == pytest.approx(np.linalg.norm(np.ones(10) * 0.5))
        assert diag.multipliers == [0.5, 0.0, 0.0]
        assert diag.penalty_parameters == [10.0, 1.0, 1.0]

        # Verify constraint analysis
        c1 = diag.constraint_diagnostics[0]
        assert c1.name == "c1"
        assert c1.violation == pytest.approx(0.06)
        assert c1.trend == "increasing_violation"  # 0.05 -> 0.06 is > 5% increase

    @patch("ai_scientist.coordinator.step_alm")
    @patch("ai_scientist.coordinator.create_alm_context")
    @patch("ai_scientist.coordinator.state_to_boundary_params")
    def test_aso_loop_integration_mock(
        self, mock_state_to_params, mock_create_context, mock_step_alm, coordinator
    ):
        """Task 5.8: Integration test of 3-step ASO loop with mock ALM."""
        from ai_scientist.config import ASOConfig
        from ai_scientist.optim.alm_bridge import ALMContext, ALMStepResult
        from ai_scientist.planner import (
            DirectiveAction,
            DirectiveSource,
            OptimizationDirective,
        )

        # Mock Config
        coordinator.cfg.aso = ASOConfig(
            enabled=True, max_stagnation_steps=5, steps_per_supervision=1
        )

        # Mock Dependencies
        coordinator._prepare_seeds = MagicMock(return_value=[_valid_seed()])
        coordinator._apply_seed_novelty_validity_gate = MagicMock(
            side_effect=lambda seeds, **_kwargs: list(seeds)
        )
        coordinator._seed_to_boundary = MagicMock()
        coordinator._get_problem = MagicMock()
        coordinator._build_optimization_settings = MagicMock()

        # Mock ALM Context/State
        mock_context = MagicMock(spec=ALMContext)
        initial_state = create_mock_alm_state()
        mock_create_context.return_value = (mock_context, initial_state)

        # Mock Step Results (Sequence of 3 steps)
        # Step 1: Improving
        state1 = create_mock_alm_state(objective=jnp.array(0.9))
        res1 = ALMStepResult(
            state=state1, n_evals=10, objective=0.9, max_violation=0.1, metrics=None
        )

        # Step 2: Stagnating
        state2 = create_mock_alm_state(objective=jnp.array(0.9))
        res2 = ALMStepResult(
            state=state2, n_evals=10, objective=0.9, max_violation=0.1, metrics=None
        )

        # Step 3: Converged (Feasible)
        state3 = create_mock_alm_state(
            objective=jnp.array(0.9), constraints=jnp.array([0.0, 0.0, 0.0])
        )
        res3 = ALMStepResult(
            state=state3, n_evals=10, objective=0.9, max_violation=0.0, metrics=None
        )

        mock_step_alm.side_effect = [res1, res2, res3]

        # Mock Planner Supervision
        # Step 1: Continue
        dir1 = OptimizationDirective(
            action=DirectiveAction.CONTINUE, source=DirectiveSource.HEURISTIC
        )
        # Step 2: Adjust (due to stagnation)
        dir2 = OptimizationDirective(
            action=DirectiveAction.ADJUST,
            alm_overrides={"penalty_parameters": [100.0, 1.0, 1.0]},
            source=DirectiveSource.HEURISTIC,
        )
        # Step 3: Stop (Converged)
        dir3 = OptimizationDirective(
            action=DirectiveAction.STOP,
            reasoning="Converged",
            source=DirectiveSource.HEURISTIC,
        )

        coordinator.planner.supervise.side_effect = [dir1, dir2, dir3]

        # Run Loop
        candidates = coordinator.produce_candidates_aso(
            cycle=1, experiment_id=1, eval_budget=100, template=MagicMock()
        )

        # Verifications
        # P3 ASO requires an aspect-ratio upper bound for constellaration's ALM objective_constraints.
        assert mock_create_context.call_count == 1
        assert (
            mock_create_context.call_args.kwargs.get("aspect_ratio_upper_bound")
            == coordinator.cfg.alm.aspect_ratio_upper_bound
        )
        assert mock_step_alm.call_count == 3
        assert coordinator.planner.supervise.call_count == 3
        assert len(candidates) == 1  # One final candidate
        assert candidates[0]["source"] == "aso"

        # Check that ADJUST override was applied in Step 2 -> used in Step 3
        # The mock_step_alm calls receive state from PREVIOUS step.
        # Step 1 call gets initial_state.
        # Step 2 call gets state1.
        # Step 3 call gets state2 (which should have been modified by ADJUST).

        # We can check if the state passed to step_alm(..., state=state2_modified) had the override
        # However, state is immutable (Pydantic/JAX), so it returns a new copy.
        # The coordinator updates traj.alm_state.

        # Let's verify that the penalty parameter override logic in Coordinator worked.
        # We can inspect the arguments passed to step_alm.

        # Call 1: initial_state
        # Call 2: state1
        # Call 3: state2 (should have modified penalties)

        call_args = mock_step_alm.call_args_list[2]
        # step_alm is called with keyword arguments in coordinator
        passed_state = call_args.kwargs.get("state")
        if passed_state is None:
            # Fallback if positional args were used (though unlikely given the code)
            passed_state = call_args.args[1]

        print(
            f"DEBUG: Type of passed_state.penalty_parameters: {type(passed_state.penalty_parameters)}"
        )
        print(
            f"DEBUG: passed_state.penalty_parameters: {passed_state.penalty_parameters}"
        )

        # In the test setup, we mock step_alm to return res2 (state2)
        # Then the coordinator receives dir2 (ADJUST)
        # Then the coordinator modifies traj.alm_state (which is state2)
        # Then next loop calls step_alm with modified state2.

        # Use numpy comparison to avoid JAX/Mock recursion issues
        assert np.allclose(passed_state.penalty_parameters, np.array([100.0, 1.0, 1.0]))

    @patch("ai_scientist.coordinator.step_alm")
    @patch("ai_scientist.coordinator.create_alm_context")
    @patch("ai_scientist.coordinator.state_to_boundary_params")
    def test_aso_emits_terminal_candidate_on_budget_exhaustion(
        self, mock_state_to_params, mock_create_context, mock_step_alm, coordinator
    ):
        from ai_scientist.optim.alm_bridge import ALMContext, ALMStepResult
        from ai_scientist.planner import (
            DirectiveAction,
            DirectiveSource,
            OptimizationDirective,
        )

        coordinator._prepare_seeds = MagicMock(return_value=[_valid_seed()])
        coordinator._apply_seed_novelty_validity_gate = MagicMock(
            side_effect=lambda seeds, **_kwargs: list(seeds)
        )
        coordinator._seed_to_boundary = MagicMock(return_value=MagicMock())
        coordinator._get_problem = MagicMock(return_value=MagicMock())
        coordinator._build_optimization_settings = MagicMock(return_value=MagicMock())

        mock_context = MagicMock(spec=ALMContext)
        state = create_mock_alm_state(
            objective=jnp.array(1.23),
            constraints=jnp.array([0.2, 0.1, 0.0]),
        )
        mock_create_context.return_value = (mock_context, state)
        mock_step_alm.return_value = ALMStepResult(
            state=state,
            n_evals=5,
            objective=1.23,
            max_violation=0.2,
            metrics=None,
        )
        coordinator.planner.supervise.return_value = OptimizationDirective(
            action=DirectiveAction.CONTINUE,
            source=DirectiveSource.HEURISTIC,
        )
        mock_state_to_params.return_value = {
            "r_cos": [[1.0]],
            "z_sin": [[0.0]],
            "n_field_periods": 1,
            "is_stellarator_symmetric": True,
        }

        candidates = coordinator.produce_candidates_aso(
            cycle=1,
            experiment_id=1,
            eval_budget=5,
            template=MagicMock(),
        )

        assert len(candidates) == 1
        assert candidates[0]["source"] == "aso_terminal_state"
        assert candidates[0]["objective"] == pytest.approx(1.23)
        assert candidates[0]["max_violation"] == pytest.approx(0.2)

    @patch("ai_scientist.coordinator.step_alm")
    @patch("ai_scientist.coordinator.create_alm_context")
    @patch("ai_scientist.coordinator.state_to_boundary_params")
    def test_aso_forces_restart_on_repeated_vmec_failure_sentinel(
        self, mock_state_to_params, mock_create_context, mock_step_alm, coordinator
    ):
        from ai_scientist.optim.alm_bridge import ALMContext, ALMStepResult
        from ai_scientist.planner import (
            DirectiveAction,
            DirectiveSource,
            OptimizationDirective,
        )

        seed_a = {
            "params": {
                "r_cos": [[1.0]],
                "z_sin": [[0.0]],
                "n_field_periods": 1,
                "is_stellarator_symmetric": True,
            }
        }
        seed_b = {
            "params": {
                "r_cos": [[1.1]],
                "z_sin": [[0.0]],
                "n_field_periods": 1,
                "is_stellarator_symmetric": True,
            }
        }

        coordinator._prepare_seeds = MagicMock(side_effect=[[seed_a], [seed_b]])
        coordinator._surrogate_rank_seeds = MagicMock(
            side_effect=lambda seeds, _cycle: seeds
        )
        coordinator._seed_to_boundary = MagicMock(return_value=MagicMock())
        coordinator._get_problem = MagicMock(return_value=MagicMock())
        coordinator._build_optimization_settings = MagicMock(return_value=MagicMock())
        coordinator._persist_telemetry = MagicMock()

        mock_context = MagicMock(spec=ALMContext)
        initial_state = create_mock_alm_state(
            objective=jnp.array(10.0),
            constraints=jnp.array([10.0, 10.0, 10.0]),
        )
        mock_create_context.return_value = (mock_context, initial_state)

        # Two sentinel VMEC-failure-like steps followed by a stop.
        fail_state_1 = create_mock_alm_state(
            objective=jnp.array(10.0),
            constraints=jnp.array([10.0, 10.0, 10.0]),
        )
        fail_state_2 = create_mock_alm_state(
            objective=jnp.array(10.0),
            constraints=jnp.array([10.0, 10.0, 10.0]),
        )
        post_restart_state = create_mock_alm_state(
            objective=jnp.array(1.0),
            constraints=jnp.array([0.0, 0.0, 0.0]),
        )
        mock_step_alm.side_effect = [
            ALMStepResult(
                state=fail_state_1,
                n_evals=1,
                objective=10.0,
                max_violation=10.0,
                metrics=None,
            ),
            ALMStepResult(
                state=fail_state_2,
                n_evals=1,
                objective=10.0,
                max_violation=10.0,
                metrics=None,
            ),
            ALMStepResult(
                state=post_restart_state,
                n_evals=1,
                objective=1.0,
                max_violation=0.0,
                metrics=MagicMock(),
            ),
        ]

        # Planner keeps asking CONTINUE, coordinator should force a RESTART on failure streak.
        coordinator.planner.supervise.side_effect = [
            OptimizationDirective(
                action=DirectiveAction.CONTINUE,
                source=DirectiveSource.HEURISTIC,
            ),
            OptimizationDirective(
                action=DirectiveAction.CONTINUE,
                source=DirectiveSource.HEURISTIC,
            ),
            OptimizationDirective(
                action=DirectiveAction.STOP,
                reasoning="Converged",
                source=DirectiveSource.HEURISTIC,
            ),
        ]
        mock_state_to_params.return_value = {
            "r_cos": [[1.0]],
            "z_sin": [[0.0]],
            "n_field_periods": 1,
            "is_stellarator_symmetric": True,
        }

        coordinator.produce_candidates_aso(
            cycle=1,
            experiment_id=1,
            eval_budget=3,
            template=MagicMock(),
        )

        assert coordinator._prepare_seeds.call_count >= 2
        assert mock_create_context.call_count >= 2
        assert any(event["aso_action"] == "RESTART" for event in coordinator.telemetry)
        assert any(event["vmec_failure_streak"] >= 2 for event in coordinator.telemetry)
        assert any(event["vmec_step_failed"] for event in coordinator.telemetry)

    @patch("ai_scientist.coordinator.make_boundary_from_params")
    @patch("ai_scientist.coordinator.step_alm")
    @patch("ai_scientist.coordinator.create_alm_context")
    @patch("ai_scientist.coordinator.state_to_boundary_params")
    def test_aso_drops_invalid_terminal_candidate(
        self,
        mock_state_to_params,
        mock_create_context,
        mock_step_alm,
        mock_make_boundary,
        coordinator,
    ):
        from ai_scientist.optim.alm_bridge import ALMContext, ALMStepResult
        from ai_scientist.planner import (
            DirectiveAction,
            DirectiveSource,
            OptimizationDirective,
        )

        coordinator._prepare_seeds = MagicMock(return_value=[_valid_seed()])
        coordinator._seed_to_boundary = MagicMock(return_value=MagicMock())
        coordinator._get_problem = MagicMock(return_value=MagicMock())
        coordinator._build_optimization_settings = MagicMock(return_value=MagicMock())

        mock_context = MagicMock(spec=ALMContext)
        state = create_mock_alm_state(
            objective=jnp.array(1.23),
            constraints=jnp.array([0.2, 0.1, 0.0]),
        )
        mock_create_context.return_value = (mock_context, state)
        mock_step_alm.return_value = ALMStepResult(
            state=state,
            n_evals=5,
            objective=1.23,
            max_violation=0.2,
            metrics=None,
        )
        coordinator.planner.supervise.return_value = OptimizationDirective(
            action=DirectiveAction.CONTINUE,
            source=DirectiveSource.HEURISTIC,
        )
        mock_state_to_params.return_value = {
            "r_cos": [[1.0]],
            "z_sin": [[0.0]],
            "n_field_periods": 1,
            "is_stellarator_symmetric": True,
        }
        mock_make_boundary.side_effect = ValueError("bad-boundary")

        candidates = coordinator.produce_candidates_aso(
            cycle=1,
            experiment_id=1,
            eval_budget=5,
            template=MagicMock(),
        )

        assert candidates == []
        mock_make_boundary.assert_called_once()

    def test_prepare_seeds_uses_stable_fallback_when_geometer_rejects_all(
        self, coordinator
    ):
        raw = [{"params": {"dummy": True}}]
        fallback = {
            "seed": 7,
            "params": {
                "r_cos": [[1.0]],
                "z_sin": [[0.0]],
                "n_field_periods": 1,
                "is_stellarator_symmetric": True,
            },
        }

        coordinator.explore_worker.run.return_value = {"candidates": raw}
        coordinator.geo_worker.run.return_value = {"candidates": []}
        coordinator._stable_fallback_seed = MagicMock(return_value=fallback)

        result = coordinator._prepare_seeds(initial_seeds=None, cycle=3, n_needed=2)

        assert result == [fallback]
        coordinator._stable_fallback_seed.assert_called_once_with(3)

    def test_prepare_seeds_strict_policy_skips_fallback_when_geometer_rejects_all(
        self, coordinator
    ):
        raw = [{"params": {"dummy": True}}]
        coordinator.explore_worker.run.return_value = {"candidates": raw}
        coordinator.geo_worker.run.return_value = {"candidates": []}
        coordinator._stable_fallback_seed = MagicMock(
            return_value={"seed": 7, "params": {"x": 1}}
        )

        result = coordinator._prepare_seeds(
            initial_seeds=None,
            cycle=3,
            n_needed=2,
            allow_fallbacks=False,
        )

        assert result == []
        coordinator._stable_fallback_seed.assert_not_called()

    def test_prepare_seeds_keeps_initial_seeds_without_forced_fallback(
        self, coordinator
    ):
        initial = [
            {
                "r_cos": [[1.0]],
                "z_sin": [[0.0]],
                "n_field_periods": 1,
                "is_stellarator_symmetric": True,
            }
        ]
        fallback = {"seed": 99, "params": {"x": 2}}
        coordinator._stable_fallback_seed = MagicMock(return_value=fallback)

        result = coordinator._prepare_seeds(initial_seeds=initial, cycle=2, n_needed=4)

        assert len(result) == 1
        assert result[0]["params"]["r_cos"] == [[1.0]]
        coordinator._stable_fallback_seed.assert_not_called()

    def test_produce_candidates_aso_strict_policy_raises_when_no_seed_survives(
        self, coordinator
    ):
        coordinator._prepare_seeds = MagicMock(return_value=[])
        coordinator._surrogate_rank_seeds = MagicMock(return_value=[])
        coordinator.cfg.aso = replace(
            coordinator.cfg.aso, seed_fallback_policy="forbid"
        )

        with pytest.raises(RuntimeError, match="seed_fallback_policy='forbid'"):
            coordinator.produce_candidates_aso(
                cycle=1,
                experiment_id=1,
                eval_budget=5,
                template=MagicMock(),
            )

    def test_apply_seed_novelty_validity_gate_dedupes_sources(self, coordinator):
        historical = {
            "design_hash": "hist-1",
            "params": _valid_seed()["params"],
        }
        coordinator.world_model.recent_candidate_snapshots = MagicMock(
            return_value=[historical]
        )
        with patch("ai_scientist.coordinator.make_boundary_from_params") as mock_make:
            mock_make.side_effect = lambda params: params
            accepted = coordinator._apply_seed_novelty_validity_gate(
                [_valid_seed(1.0), _valid_seed(1.0), _valid_seed(1.1)],
                experiment_id=1,
                problem="p3",
                limit=4,
            )

        assert len(accepted) == 1
        assert accepted[0]["params"]["r_cos"][0][0] == pytest.approx(1.1)
        assert mock_make.call_count == 3

    @patch("ai_scientist.coordinator.step_alm")
    @patch("ai_scientist.coordinator.create_alm_context")
    @patch("ai_scientist.coordinator.state_to_boundary_params")
    def test_restart_consumes_queue_before_fallback_generation(
        self, mock_state_to_params, mock_create_context, mock_step_alm, coordinator
    ):
        from ai_scientist.coordinator import TrajectoryState
        from ai_scientist.optim.alm_bridge import ALMContext, ALMStepResult
        from ai_scientist.planner import (
            DirectiveAction,
            DirectiveSource,
            OptimizationDirective,
        )

        coordinator._seed_to_boundary = MagicMock(return_value=MagicMock())
        coordinator._get_problem = MagicMock(return_value=MagicMock())
        coordinator._build_optimization_settings = MagicMock(return_value=MagicMock())
        coordinator._prepare_seeds = MagicMock(return_value=[_valid_seed(2.0)])
        coordinator.cfg.aso = replace(
            coordinator.cfg.aso, seed_fallback_policy="forbid"
        )

        mock_context = MagicMock(spec=ALMContext)
        state = create_mock_alm_state()
        mock_create_context.return_value = (mock_context, state)
        mock_step_alm.side_effect = [
            ALMStepResult(
                state=state,
                n_evals=1,
                objective=10.0,
                max_violation=10.0,
                metrics=None,
            ),
            ALMStepResult(
                state=state,
                n_evals=1,
                objective=1.0,
                max_violation=0.0,
                metrics=None,
            ),
        ]
        coordinator.planner.supervise.side_effect = [
            OptimizationDirective(
                action=DirectiveAction.RESTART,
                source=DirectiveSource.HEURISTIC,
            ),
            OptimizationDirective(
                action=DirectiveAction.STOP,
                source=DirectiveSource.HEURISTIC,
            ),
        ]
        mock_state_to_params.return_value = _valid_seed()["params"]

        restart_queue = [_valid_seed(1.2)]
        candidates = coordinator._run_trajectory_aso(
            traj=TrajectoryState(id=0, seed=_valid_seed()),
            eval_budget=2,
            cycle=1,
            experiment_id=1,
            config=coordinator.cfg,
            planner_intent=None,
            restart_seed_queue=restart_queue,
        )

        assert restart_queue == []
        coordinator._prepare_seeds.assert_not_called()
        assert len(candidates) >= 1

    @patch("ai_scientist.coordinator.build_staged_seed_plan_from_snapshots")
    def test_produce_candidates_aso_prepends_staged_governor_seeds(
        self, mock_build_staged, coordinator
    ):
        from ai_scientist.staged_governor import StagedSeedPlan

        user_seed = {
            "params": {
                "r_cos": [[1.0]],
                "z_sin": [[0.0]],
                "n_field_periods": 1,
                "is_stellarator_symmetric": True,
            }
        }
        staged_seed = {
            "params": {
                "r_cos": [[0.95]],
                "z_sin": [[0.01]],
                "n_field_periods": 1,
                "is_stellarator_symmetric": True,
            },
            "source": "staged_governor",
            "staged_governor": {"phase": "focus"},
        }
        mock_build_staged.return_value = StagedSeedPlan(
            seeds=[staged_seed],
            focus_hash="focus123",
            partner_hash="partner123",
            worst_constraint="mirror",
        )

        coordinator.world_model.recent_candidate_snapshots = MagicMock(
            return_value=[{"design_hash": "x"}]
        )
        coordinator._prepare_seeds = MagicMock(return_value=[_valid_seed()])
        coordinator._apply_seed_novelty_validity_gate = MagicMock(
            side_effect=lambda seeds, **_kwargs: list(seeds)
        )
        coordinator._surrogate_rank_seeds = MagicMock(
            return_value=[{**_valid_seed(), "surrogate_score": 1.0}]
        )
        coordinator._run_trajectory_aso = MagicMock(return_value=[])
        coordinator._persist_telemetry = MagicMock()

        coordinator.produce_candidates_aso(
            cycle=2,
            experiment_id=7,
            eval_budget=3,
            template=MagicMock(),
            initial_seeds=[user_seed],
        )

        coordinator.world_model.recent_candidate_snapshots.assert_called_once_with(
            experiment_id=7,
            problem=coordinator.cfg.problem,
            limit=coordinator.cfg.aso.staged_recent_limit,
        )
        mock_build_staged.assert_called_once()
        args, kwargs = coordinator._prepare_seeds.call_args
        merged = args[0]
        assert isinstance(merged, list)
        assert merged[0]["source"] == "staged_governor"
        assert merged[1] == user_seed
        assert kwargs["allow_fallbacks"] is True

    @patch("ai_scientist.coordinator.step_alm")
    @patch("ai_scientist.coordinator.create_alm_context")
    @patch("ai_scientist.coordinator.state_to_boundary_params")
    def test_aso_receives_and_logs_planner_intent(
        self, mock_state_to_params, mock_create_context, mock_step_alm, coordinator
    ):
        from ai_scientist.optim.alm_bridge import ALMContext, ALMStepResult
        from ai_scientist.planner import (
            DirectiveAction,
            DirectiveSource,
            OptimizationDirective,
        )

        coordinator._prepare_seeds = MagicMock(return_value=[_valid_seed()])
        coordinator._apply_seed_novelty_validity_gate = MagicMock(
            side_effect=lambda seeds, **_kwargs: list(seeds)
        )
        coordinator._seed_to_boundary = MagicMock(return_value=MagicMock())
        coordinator._get_problem = MagicMock(return_value=MagicMock())
        coordinator._build_optimization_settings = MagicMock(return_value=MagicMock())
        coordinator._persist_telemetry = MagicMock()

        mock_context = MagicMock(spec=ALMContext)
        state = create_mock_alm_state(
            objective=jnp.array(1.0),
            constraints=jnp.array([0.2, 0.1, 0.0]),
        )
        mock_create_context.return_value = (mock_context, state)
        mock_step_alm.return_value = ALMStepResult(
            state=state,
            n_evals=5,
            objective=1.0,
            max_violation=0.2,
            metrics=None,
        )
        coordinator.planner.supervise.return_value = OptimizationDirective(
            action=DirectiveAction.CONTINUE,
            source=DirectiveSource.HEURISTIC,
        )
        mock_state_to_params.return_value = {
            "r_cos": [[1.0]],
            "z_sin": [[0.0]],
            "n_field_periods": 1,
            "is_stellarator_symmetric": True,
        }

        planner_intent = {
            "primary_constraint_order": ["c1", "c2"],
            "penalty_focus_indices": [0],
            "confidence": 0.8,
        }
        coordinator.produce_candidates_aso(
            cycle=1,
            experiment_id=1,
            eval_budget=5,
            template=MagicMock(),
            planner_intent=planner_intent,
        )

        supervise_kwargs = coordinator.planner.supervise.call_args.kwargs
        assert supervise_kwargs["planner_intent"] == planner_intent
        assert coordinator.telemetry[0]["planner_intent"] == planner_intent
        assert coordinator.telemetry[0]["aso_action"] == "CONTINUE"
        assert "worst_constraint_before" in coordinator.telemetry[0]
        assert "worst_constraint_after" in coordinator.telemetry[0]
        assert "chosen_operator" in coordinator.telemetry[0]
        assert "parent_hashes" in coordinator.telemetry[0]
        assert "bridge_flag" in coordinator.telemetry[0]
        assert "feasibility_delta" in coordinator.telemetry[0]
        assert "hv_delta" in coordinator.telemetry[0]

    def test_produce_candidates_aso_uses_seed_specific_planner_intent(
        self, coordinator
    ):
        seed = {
            "params": {
                "r_cos": [[1.0]],
                "z_sin": [[0.0]],
                "n_field_periods": 1,
                "is_stellarator_symmetric": True,
            },
            "seed": 123,
        }
        cycle_intent = {
            "primary_constraint_order": ["c1", "c2"],
            "penalty_focus_indices": [0],
        }
        seed_intent = {
            "primary_constraint_order": ["c3", "c1"],
            "penalty_focus_indices": [2],
        }
        ranked_seed = {
            **seed,
            "planner_intent": seed_intent,
            "surrogate_score": 0.9,
        }

        coordinator._prepare_seeds = MagicMock(return_value=[ranked_seed])
        coordinator._surrogate_rank_seeds = MagicMock(return_value=[ranked_seed])
        coordinator._run_trajectory_aso = MagicMock(return_value=[])
        coordinator._persist_telemetry = MagicMock()

        coordinator.produce_candidates_aso(
            cycle=1,
            experiment_id=42,
            eval_budget=4,
            template=MagicMock(),
            initial_seeds=[seed],
            planner_intent=cycle_intent,
            planner_intents=[seed_intent],
        )

        call_kwargs = coordinator._run_trajectory_aso.call_args.kwargs
        assert call_kwargs["planner_intent"] == seed_intent

    def test_normalize_seed_entry_preserves_top_level_planner_intent(self, coordinator):
        seed = {
            "r_cos": [[1.0]],
            "z_sin": [[0.0]],
            "n_field_periods": 1,
            "is_stellarator_symmetric": True,
            "planner_intent": {
                "primary_constraint_order": ["c3", "c1"],
                "penalty_focus_indices": [2],
            },
        }

        normalized = coordinator._normalize_seed_entry(seed, cycle=0)

        assert normalized is not None
        assert normalized["params"]["r_cos"] == [[1.0]]
        assert normalized["planner_intent"] == seed["planner_intent"]
        assert "planner_intent" not in normalized["params"]

    def test_enqueue_seed_queue_requires_parent_hash_and_reason(self, coordinator):
        queue: list[dict[str, object]] = []

        coordinator._enqueue_seed_queue(
            queue=queue,
            new_items=[
                {
                    "params": _valid_seed()["params"],
                    "estimated_feasibility": 0.1,
                    "parent_feasibility": 0.2,
                }
            ],
            max_size=10,
            tie_epsilon=1e-6,
        )
        assert queue == []

        coordinator._enqueue_seed_queue(
            queue=queue,
            new_items=[
                {
                    "params": _valid_seed(1.1)["params"],
                    "lineage_parent_hashes": ["parent1"],
                    "improvement_reason": "constraint_targeted_repair",
                    "estimated_feasibility": 0.1,
                    "parent_feasibility": 0.2,
                }
            ],
            max_size=10,
            tie_epsilon=1e-6,
        )
        assert len(queue) == 1

    def test_select_runtime_ucb_arm_uses_eligible_history(self, coordinator):
        coordinator.world_model.model_router_reward_eligible_history = MagicMock(
            return_value=[
                {
                    "model_route": "codex-native",
                    "reward": 0.8,
                    "reward_components": {"operator_family": "staged_repair"},
                },
                {
                    "model_route": "codex-native",
                    "reward": 0.6,
                    "reward_components": {"operator_family": "staged_repair"},
                },
                {
                    "model_route": "baseline",
                    "reward": 0.1,
                    "reward_components": {"operator_family": "parent_group"},
                },
            ]
        )

        selected = coordinator._select_runtime_ucb_arm(
            experiment_id=1,
            problem="p3",
            min_eligible_events=2,
        )

        assert selected is not None
        assert selected["model_route"] == "codex-native"
        assert selected["operator_family"] == "staged_repair"

    def test_produce_candidates_aso_passes_leverage_context_to_parent_selector(
        self, coordinator
    ):
        coordinator.world_model.recent_candidate_snapshots = MagicMock(
            return_value=[
                {
                    "design_hash": "focus",
                    "feasibility": 0.2,
                    "constraint_margins": {"mirror": 0.3, "log10_qi": 0.1},
                }
            ]
        )
        coordinator.world_model.select_parent_group_performance_novelty = MagicMock(
            return_value=[]
        )
        coordinator.world_model.model_router_reward_eligible_history = MagicMock(
            return_value=[]
        )
        coordinator._prepare_seeds = MagicMock(return_value=[_valid_seed()])
        coordinator._apply_seed_novelty_validity_gate = MagicMock(
            side_effect=lambda seeds, **_kwargs: list(seeds)
        )
        coordinator._surrogate_rank_seeds = MagicMock(return_value=[_valid_seed()])
        coordinator._run_trajectory_aso = MagicMock(return_value=[])
        coordinator._persist_telemetry = MagicMock()

        coordinator.produce_candidates_aso(
            cycle=1,
            experiment_id=1,
            eval_budget=1,
            template=MagicMock(),
        )

        kwargs = coordinator.world_model.select_parent_group_performance_novelty.call_args.kwargs
        assert kwargs["worst_constraint"] == "mirror"
        assert kwargs["focus_constraint_margin"] == pytest.approx(0.3)
        assert kwargs["leverage_weight"] == pytest.approx(
            coordinator.cfg.aso.parent_selection_leverage_weight
        )
