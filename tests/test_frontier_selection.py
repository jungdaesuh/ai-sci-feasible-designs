from __future__ import annotations

from ai_scientist.problem_profiles import get_problem_profile
from scripts.p3_governor import (
    CandidateRow,
    _choose_frontier_focus,
    _choose_frontier_partner,
    _hv_gap_perturbation_scale,
)


def _candidate(
    *,
    candidate_id: int,
    is_feasible: bool,
    objective_utility: float | None,
    aspect: float | None,
    metrics: dict[str, float] | None = None,
) -> CandidateRow:
    return CandidateRow(
        candidate_id=candidate_id,
        design_hash=f"hash_{candidate_id}",
        seed=1000 + candidate_id,
        feasibility=0.05 if is_feasible else 0.5,
        is_feasible=is_feasible,
        lgradb=objective_utility,
        aspect=aspect,
        violations={},
        metrics={} if metrics is None else dict(metrics),
        meta={},
        lineage_parent_hashes=[],
        novelty_score=None,
        operator_family="test",
        model_route="test",
    )


def test_choose_frontier_focus_selects_best_feasible() -> None:
    profile = get_problem_profile("p3")
    candidate_a = _candidate(
        candidate_id=1,
        is_feasible=True,
        objective_utility=5.0,
        aspect=18.0,
        metrics={"aspect_ratio": 18.0},
    )
    candidate_b = _candidate(
        candidate_id=2,
        is_feasible=True,
        objective_utility=3.0,
        aspect=10.0,
        metrics={"aspect_ratio": 10.0},
    )
    focus = _choose_frontier_focus([candidate_a, candidate_b], profile=profile)
    assert focus is not None
    assert focus.candidate_id == 2


def test_choose_frontier_focus_returns_none_when_no_feasible() -> None:
    profile = get_problem_profile("p3")
    candidate = _candidate(
        candidate_id=1,
        is_feasible=False,
        objective_utility=2.0,
        aspect=12.0,
        metrics={"aspect_ratio": 12.0},
    )
    assert _choose_frontier_focus([candidate], profile=profile) is None


def test_choose_frontier_focus_uses_profile_objective_direction() -> None:
    profile = get_problem_profile("p1")
    better = _candidate(
        candidate_id=1,
        is_feasible=True,
        objective_utility=None,
        aspect=None,
        metrics={"max_elongation": 2.0},
    )
    worse = _candidate(
        candidate_id=2,
        is_feasible=True,
        objective_utility=None,
        aspect=None,
        metrics={"max_elongation": 3.0},
    )
    focus = _choose_frontier_focus([worse, better], profile=profile)
    assert focus is not None
    assert focus.candidate_id == 1


def test_choose_frontier_partner_picks_pareto_diverse_candidate() -> None:
    profile = get_problem_profile("p3")
    focus = _candidate(
        candidate_id=10,
        is_feasible=True,
        objective_utility=2.0,
        aspect=8.0,
        metrics={"aspect_ratio": 8.0},
    )
    close = _candidate(
        candidate_id=11,
        is_feasible=True,
        objective_utility=2.1,
        aspect=8.1,
        metrics={"aspect_ratio": 8.1},
    )
    far = _candidate(
        candidate_id=12,
        is_feasible=True,
        objective_utility=4.0,
        aspect=12.0,
        metrics={"aspect_ratio": 12.0},
    )
    medium = _candidate(
        candidate_id=13,
        is_feasible=True,
        objective_utility=1.0,
        aspect=7.5,
        metrics={"aspect_ratio": 7.5},
    )
    partner = _choose_frontier_partner(
        [focus, close, far, medium], focus=focus, profile=profile
    )
    assert partner is not None
    assert partner.candidate_id == 12


def test_choose_frontier_partner_returns_none_without_aspect_metric() -> None:
    profile = get_problem_profile("p1")
    focus = _candidate(
        candidate_id=1,
        is_feasible=True,
        objective_utility=-2.0,
        aspect=None,
        metrics={"max_elongation": 2.0},
    )
    other = _candidate(
        candidate_id=2,
        is_feasible=True,
        objective_utility=-2.5,
        aspect=None,
        metrics={"max_elongation": 2.5},
    )
    assert (
        _choose_frontier_partner([focus, other], focus=focus, profile=profile) is None
    )


def test_hv_gap_perturbation_scale_is_clamped() -> None:
    profile = get_problem_profile("p3")
    frontier_recipe = profile.frontier_recipe
    scale = _hv_gap_perturbation_scale(
        hv_value=0.0,
        record_hv=100.0,
        frontier_recipe=frontier_recipe,
    )
    assert scale == frontier_recipe.max_perturbation_scale


def test_hv_gap_perturbation_scale_uses_base_when_record_invalid() -> None:
    profile = get_problem_profile("p3")
    frontier_recipe = profile.frontier_recipe
    scale = _hv_gap_perturbation_scale(
        hv_value=5.0,
        record_hv=0.0,
        frontier_recipe=frontier_recipe,
    )
    assert scale == frontier_recipe.base_perturbation_scale
