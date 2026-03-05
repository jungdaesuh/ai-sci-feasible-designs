"""Tests for harness.problem_adapter — M1 acceptance."""

from __future__ import annotations


from ai_scientist.problem_profiles import P1_PROFILE, P2_PROFILE, P3_PROFILE

from harness.problem_adapter import ProblemAdapter
from harness.types import CycleSnapshot


def _snap(frontier: float | None = None) -> CycleSnapshot:
    return CycleSnapshot(
        frontier_value=frontier,
        pending_count=0,
        running_count=0,
        done_count=0,
        near_feasible_count=0,
        parent_paths=(),
    )


class TestObjectiveValue:
    def test_p2_objective_value_sign(self):
        """P2 maximizes lgradB — value should come through as-is."""
        adapter = ProblemAdapter(P2_PROFILE)
        val = adapter.objective_value({"lgradB": 8.5})
        assert val == 8.5

    def test_p1_objective_value_sign(self):
        """P1 minimizes max_elongation — value should be negated."""
        adapter = ProblemAdapter(P1_PROFILE)
        val = adapter.objective_value({"max_elongation": 2.1})
        assert val == -2.1


class TestFrontierDelta:
    def test_frontier_delta_positive_on_improvement(self):
        adapter = ProblemAdapter(P2_PROFILE)
        prev = _snap(frontier=8.0)
        now = _snap(frontier=8.5)
        delta = adapter.frontier_delta(prev, now)
        assert delta > 0.0
        assert delta == 0.5

    def test_frontier_delta_zero_when_none(self):
        adapter = ProblemAdapter(P2_PROFILE)
        prev = _snap(frontier=None)
        now = _snap(frontier=8.5)
        assert adapter.frontier_delta(prev, now) == 0.0

    def test_frontier_delta_p3_hv(self):
        adapter = ProblemAdapter(P3_PROFILE)
        prev = _snap(frontier=100.0)
        now = _snap(frontier=105.0)
        delta = adapter.frontier_delta(prev, now)
        assert delta == 5.0


class TestTargetReached:
    def test_target_reached_true_when_exceeded(self):
        """P2 target is 8.61 — frontier_value > 8.61 should be True."""
        adapter = ProblemAdapter(P2_PROFILE)
        snap = _snap(frontier=9.0)
        assert adapter.target_reached(snap) is True

    def test_target_not_reached_when_below(self):
        adapter = ProblemAdapter(P2_PROFILE)
        snap = _snap(frontier=8.0)
        assert adapter.target_reached(snap) is False

    def test_target_p1_minimization(self):
        """P1 target is 2.10 (minimize). Normalized = -2.10.
        frontier_value=-2.0 means elongation=2.0 < 2.10, so target reached."""
        adapter = ProblemAdapter(P1_PROFILE)
        snap = _snap(frontier=-2.0)
        assert adapter.target_reached(snap) is True

    def test_target_p1_not_reached(self):
        adapter = ProblemAdapter(P1_PROFILE)
        snap = _snap(frontier=-3.0)  # elongation=3.0 > 2.10
        assert adapter.target_reached(snap) is False

    def test_target_p3_always_false(self):
        """P3 has no single scalar target."""
        adapter = ProblemAdapter(P3_PROFILE)
        snap = _snap(frontier=999.0)
        assert adapter.target_reached(snap) is False

    def test_target_none_frontier(self):
        adapter = ProblemAdapter(P2_PROFILE)
        snap = _snap(frontier=None)
        assert adapter.target_reached(snap) is False


class TestBindingConstraints:
    def test_binding_constraints_returns_closest(self):
        # P2 constraints: aspect_ratio, iota_edge, log10_qi, mirror, max_elongation
        adapter = ProblemAdapter(P2_PROFILE)
        candidates = [
            {
                "constraint_margins": {
                    "log10_qi": 0.01,
                    "mirror": 0.05,
                    "max_elongation": 0.10,
                }
            },
            {
                "constraint_margins": {
                    "log10_qi": 0.02,
                    "mirror": 0.04,
                    "max_elongation": 0.08,
                }
            },
        ]
        result = adapter.binding_constraints(candidates)
        assert isinstance(result, tuple)
        # log10_qi avg=0.015, mirror avg=0.045, max_elongation avg=0.09
        assert result[0] == "log10_qi"
        assert "mirror" in result
        assert "max_elongation" in result

    def test_binding_constraints_empty(self):
        adapter = ProblemAdapter(P2_PROFILE)
        assert adapter.binding_constraints([]) == ()

    def test_binding_constraints_no_margins(self):
        adapter = ProblemAdapter(P2_PROFILE)
        candidates = [{"metrics": {"lgradB": 8.0}}]
        assert adapter.binding_constraints(candidates) == ()


class TestAdapter:
    def test_problem_property(self):
        adapter = ProblemAdapter(P2_PROFILE)
        assert adapter.problem == "p2"

    def test_profile_property(self):
        adapter = ProblemAdapter(P1_PROFILE)
        assert adapter.profile is P1_PROFILE
