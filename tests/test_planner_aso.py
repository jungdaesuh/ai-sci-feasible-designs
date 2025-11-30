import pytest

from ai_scientist.config import ASOConfig
from ai_scientist.planner import HeuristicSupervisor, DirectiveAction
from ai_scientist.planner import OptimizerDiagnostics, ConstraintDiagnostic


def create_mock_diagnostics(**kwargs) -> OptimizerDiagnostics:
    """Create mock diagnostics for testing."""
    defaults = {
        "step": 1,
        "trajectory_id": 0,
        "objective": 1.0,
        "objective_delta": 0.0,
        "max_violation": 0.1,
        "constraints_raw": [0.1, 0.0, 0.0],
        "multipliers": [1.0, 1.0, 1.0],
        "penalty_parameters": [1.0, 1.0, 1.0],
        "bounds_norm": 1.0,
        "status": "IN_PROGRESS",
        "constraint_diagnostics": [
            ConstraintDiagnostic(name="c1", violation=0.1, penalty=1.0, multiplier=1.0, trend="stable"),
            ConstraintDiagnostic(name="c2", violation=0.0, penalty=1.0, multiplier=1.0, trend="stable"),
            ConstraintDiagnostic(name="c3", violation=0.0, penalty=1.0, multiplier=1.0, trend="stable"),
        ],
        "narrative": ["test"],
        "steps_since_improvement": 0,
    }
    defaults.update(kwargs)
    return OptimizerDiagnostics(**defaults)


class TestHeuristicSupervisor:
    @pytest.fixture
    def aso_config(self):
        from ai_scientist.config import ASOConfig
        return ASOConfig()

    @pytest.fixture
    def supervisor(self, aso_config):
        from ai_scientist.planner import HeuristicSupervisor
        return HeuristicSupervisor(aso_config)

    def test_stop_on_feasible_found(self, supervisor):
        """FEASIBLE_FOUND + stable objective -> STOP"""
        diagnostics = create_mock_diagnostics(
            status="FEASIBLE_FOUND",
            objective_delta=1e-7,
            max_violation=1e-4,
        )
        directive = supervisor.analyze(diagnostics)
        assert directive.action == DirectiveAction.STOP

    def test_continue_on_feasible_improving(self, supervisor):
        """FEASIBLE_FOUND + improving -> CONTINUE"""
        diagnostics = create_mock_diagnostics(
            status="FEASIBLE_FOUND",
            objective_delta=-0.1,
            max_violation=1e-4,
        )
        directive = supervisor.analyze(diagnostics)
        assert directive.action == DirectiveAction.CONTINUE

    def test_adjust_on_stagnation_high_violation(self, supervisor):
        """STAGNATION + high violation -> ADJUST"""
        diagnostics = create_mock_diagnostics(
            status="STAGNATION",
            max_violation=0.5,
        )
        directive = supervisor.analyze(diagnostics)
        assert directive.action == DirectiveAction.ADJUST
        assert directive.alm_overrides is not None

    def test_stop_on_diverging(self, supervisor):
        """DIVERGING -> STOP"""
        diagnostics = create_mock_diagnostics(status="DIVERGING")
        directive = supervisor.analyze(diagnostics)
        assert directive.action == DirectiveAction.STOP
