from __future__ import annotations

import pytest

from ai_scientist.problem_profiles import (
    PROBLEM_PROFILES,
    get_problem_profile,
    profile_prompt_block,
)


def test_get_problem_profile_p3_contract() -> None:
    profile = get_problem_profile("p3")
    assert profile.problem == "p3"
    assert profile.allowed_actions == ("repair", "bridge", "global_restart")
    assert "log10_qi" in profile.allowed_constraint_names()


def test_get_problem_profile_p1_contract() -> None:
    profile = get_problem_profile("p1")
    assert profile.problem == "p1"
    assert "jump" in profile.allowed_actions
    assert "iota_edge" in profile.allowed_constraint_names()


def test_prompt_block_contains_constraints_and_objective() -> None:
    profile = get_problem_profile("p2")
    block = profile_prompt_block(profile)
    assert block["problem"] == "p2"
    assert block["objective"]["direction"] == "maximize"
    names = {item["name"] for item in block["constraints"]}
    assert {"aspect_ratio", "log10_qi", "max_elongation"}.issubset(names)


def test_invalid_problem_profile_raises() -> None:
    with pytest.raises(ValueError):
        get_problem_profile("invalid")


def test_frontier_recipe_present_on_all_profiles() -> None:
    for profile in PROBLEM_PROFILES.values():
        assert profile.frontier_recipe.objective_metric
        assert profile.frontier_recipe.objective_direction in {"minimize", "maximize"}
        assert profile.frontier_recipe.max_perturbation_scale >= (
            profile.frontier_recipe.min_perturbation_scale
        )
