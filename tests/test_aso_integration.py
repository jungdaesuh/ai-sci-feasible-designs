import shutil
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import pytest
import yaml

from ai_scientist import memory
from ai_scientist.config import load_experiment_config
from ai_scientist.coordinator import Coordinator
from ai_scientist.model_provider import ChatResponse
from ai_scientist.planner import (
    DirectiveAction,
    DirectiveSource,
    OptimizationDirective,
    PlanningAgent,
)


@pytest.mark.slow
@pytest.mark.integration
class TestASOIntegration:
    """End-to-end integration tests for ASO."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self, tmp_path):
        self.tmp_path = tmp_path
        # Ensure reporting directory exists and is clean for each test
        self.reporting_dir = self.tmp_path / "reports"
        self.reporting_dir.mkdir(exist_ok=True)
        yield
        # Clean up after test
        if self.reporting_dir.exists():
            shutil.rmtree(self.reporting_dir)

    def _create_minimal_config(self):
        # Create a minimal config for ASO
        config_dict = {
            "problem": "p3",
            "cycles": 3,
            "random_seed": 42,
            "experiment_tag": "aso_integration_test",
            "boundary_template": {
                "n_poloidal_modes": 2,
                "n_toroidal_modes": 3,  # Must be odd
                "n_field_periods": 1,
                "base_major_radius": 1.0,
                "base_minor_radius": 0.2,
                "perturbation_scale": 0.05,
            },
            "budgets": {
                "screen_evals_per_cycle": 5,
                "promote_top_k": 1,
                "max_high_fidelity_evals_per_cycle": 1,
                "wall_clock_minutes": 1,
                "n_workers": 1,
                "pool_type": "thread",
            },
            "adaptive_budgets": {"enabled": True},
            "fidelity_ladder": {"screen": "p3", "promote": "p3"},
            "governance": {"min_feasible_for_promotion": 1, "hv_lookback": 1},
            "proposal_mix": {
                "constraint_ratio": 0.7,
                "exploration_ratio": 0.3,
                "jitter_scale": 0.01,
                "surrogate_pool_multiplier": 2.0,
            },
            "alm": {
                "maxit": 3,
                "penalty_parameters_initial": 1.0,
                "bounds_initial": 2.0,
                "oracle_budget_initial": 2,  # Small for testing
                "oracle_budget_increment": 1,
                "oracle_budget_max": 5,
                "oracle_num_workers": 1,
                "penalty_parameters_increase_factor": 1.1,
                "constraint_violation_tolerance_reduction_factor": 0.9,
                "bounds_reduction_factor": 0.9,
                "penalty_parameters_max": 10.0,
                "bounds_min": 0.1,
            },
            "aso": {
                "enabled": True,
                "supervision_mode": "event_triggered",
                "supervision_interval": 1,
                "use_heuristic_fallback": True,
                "steps_per_supervision": 1,
                "stagnation_violation_threshold": 0.05,
                "max_penalty_boost": 4.0,
                "feasibility_threshold": 0.01,
                "stagnation_objective_threshold": 1e-4,
                "violation_increase_threshold": 0.1,
                "violation_decrease_threshold": 0.1,
                "max_stagnation_steps": 5,
            },
            "reporting_dir": str(self.reporting_dir),
            "reporting": {
                "prometheus_export_enabled": False,
            },
        }

        config_path = self.tmp_path / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f)

        return load_experiment_config(config_path)

    def _create_mock_seed(self):
        # A simple torus seed with correct shapes (2 poloidal, 3 toroidal)
        # shape: (2, 3)
        r_cos = jnp.array(
            [
                [
                    0.0,
                    1.0,
                    0.0,
                ],  # m=0, n=0 is index 1 (middle) if n_toroidal=3 (-1, 0, 1)
                [0.0, 0.2, 0.0],  # m=1, n=0
            ]
        )
        z_sin = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.2, 0.0]])

        return {
            "params": {
                "r_cos": r_cos,
                "z_sin": z_sin,
                "n_field_periods": 1,
                "is_stellarator_symmetric": True,
            }
        }

    def _create_mock_invoke_chat_completion(self):
        mock_response = ChatResponse(
            status_code=200,
            body={
                "choices": [
                    {
                        "message": {
                            "content": (
                                '```json\n{"action": "CONTINUE", "reasoning": "Test continues."}\n```'
                            )
                        }
                    }
                ]
            },
        )
        mock_invoke_chat_completion = MagicMock(return_value=mock_response)
        return mock_invoke_chat_completion

    def test_three_step_aso_loop(self):
        """Run 3-step ASO loop with toy problem."""
        # 1. Create minimal config
        config = self._create_minimal_config()

        # 2. Create mock/minimal boundary (via seed)
        seed = self._create_mock_seed()
        initial_seeds = [seed]

        # 3. Create Coordinator with ASO enabled
        mock_world_model = MagicMock(spec=memory.WorldModel)
        mock_world_model.average_recent_hv_delta.return_value = 0.1

        mock_planning_agent = MagicMock(spec=PlanningAgent)
        # Always continue
        mock_planning_agent.supervise.return_value = OptimizationDirective(
            action=DirectiveAction.CONTINUE,
            reasoning="Mock continue",
            source=DirectiveSource.HEURISTIC,
        )

        mock_invoke_chat_completion_func = self._create_mock_invoke_chat_completion()

        with (
            patch("ai_scientist.coordinator.step_alm") as mock_step_alm,
            patch(
                "ai_scientist.coordinator.create_alm_context"
            ) as mock_create_alm_context,
            patch(
                "ai_scientist.model_provider.invoke_chat_completion",
                new=mock_invoke_chat_completion_func,
            ),
        ):
            # Setup mock ALM context/state
            mock_state = MagicMock()
            mock_state.objective = 0.5
            mock_state.constraints = jnp.array([0.1, 0.1])
            mock_state.multipliers = jnp.array([1.0, 1.0])
            mock_state.penalty_parameters = jnp.array([10.0, 10.0])
            mock_state.bounds = jnp.array([0.5])
            mock_state.copy.return_value = mock_state

            mock_context = MagicMock()

            mock_create_alm_context.return_value = (mock_context, mock_state)

            # Setup mock step result
            mock_result = MagicMock()
            mock_result.state = mock_state
            mock_result.objective = 0.4  # improving
            mock_result.max_violation = 0.09
            mock_result.n_evals = 1

            mock_step_alm.return_value = mock_result

            coordinator = Coordinator(config, mock_world_model, mock_planning_agent)

            # 4. Call produce_candidates_aso with small budget
            # Mock supervisor to stop eventually
            mock_planning_agent.supervise.side_effect = [
                OptimizationDirective(
                    action=DirectiveAction.CONTINUE, reasoning="Go 1"
                ),
                OptimizationDirective(
                    action=DirectiveAction.CONTINUE, reasoning="Go 2"
                ),
                OptimizationDirective(action=DirectiveAction.STOP, reasoning="Done"),
            ]

            # Reset coordinator with side_effect
            coordinator = Coordinator(config, mock_world_model, mock_planning_agent)
            candidates = coordinator.produce_candidates_aso(
                cycle=0,
                experiment_id=0,
                eval_budget=5,
                template=config.boundary_template,
                initial_seeds=initial_seeds,
            )

            assert len(candidates) == 1
            assert candidates[0]["source"] == "aso"

            # 6. Verify telemetry file created
            telemetry_path = self.reporting_dir / "telemetry" / "aso_exp0.jsonl"
            assert telemetry_path.exists()
            assert telemetry_path.stat().st_size > 0

    def test_aso_respects_stop_directive(self):
        """Verify ASO stops when supervisor says STOP."""
        config = self._create_minimal_config()
        seed = self._create_mock_seed()
        initial_seeds = [seed]

        mock_world_model = MagicMock(spec=memory.WorldModel)
        mock_planning_agent = MagicMock(spec=PlanningAgent)

        # STOP immediately
        mock_planning_agent.supervise.return_value = OptimizationDirective(
            action=DirectiveAction.STOP,
            reasoning="Stop now",
            source=DirectiveSource.HEURISTIC,
        )

        with (
            patch("ai_scientist.coordinator.step_alm") as mock_step_alm,
            patch(
                "ai_scientist.coordinator.create_alm_context"
            ) as mock_create_alm_context,
        ):
            mock_state = MagicMock()
            mock_state.constraints = jnp.array([0.1])
            mock_state.multipliers = jnp.array([1.0])
            mock_state.penalty_parameters = jnp.array([1.0])
            mock_state.bounds = jnp.array([1.0])
            mock_state.copy.return_value = mock_state
            mock_create_alm_context.return_value = (MagicMock(), mock_state)

            mock_result = MagicMock()
            mock_result.state = mock_state
            mock_result.n_evals = 1
            mock_result.max_violation = 0.1
            mock_result.objective = 1.0
            mock_step_alm.return_value = mock_result

            coordinator = Coordinator(config, mock_world_model, mock_planning_agent)

            candidates = coordinator.produce_candidates_aso(
                cycle=0,
                experiment_id=1,
                eval_budget=10,
                template=config.boundary_template,
                initial_seeds=initial_seeds,
            )

            # Should stop after 1 step
            assert len(candidates) == 1
            assert candidates[0]["source"] == "aso"
            # step_alm called once
            assert mock_step_alm.call_count == 1
