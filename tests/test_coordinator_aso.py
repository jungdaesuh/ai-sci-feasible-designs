# ruff: noqa: E402
import sys
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
        coordinator._prepare_seeds = MagicMock(return_value=[{"params": {}}])
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
