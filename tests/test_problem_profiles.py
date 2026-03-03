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


def test_autonomy_policy_present_on_all_profiles() -> None:
    for profile in PROBLEM_PROFILES.values():
        assert profile.autonomy_policy.diversity_floor_min_candidates >= 1
        assert profile.autonomy_policy.anti_repeat_no_progress_cycles >= 1
        assert (
            profile.autonomy_policy.invalid_basin_max_consecutive_parent_failures >= 1
        )
        assert profile.autonomy_policy.stagnation_min_mutation_delta > 0.0
        assert profile.autonomy_policy.autonomous_feasibility_stall_eval_window >= 1
        assert profile.autonomy_policy.autonomous_objective_stall_eval_window >= 1


def test_run_surgery_policy_present_on_all_profiles() -> None:
    for profile in PROBLEM_PROFILES.values():
        policy = profile.autonomy_policy.run_surgery
        assert policy.eval_window >= 1
        assert policy.objective_stall_windows >= 1
        assert policy.feasibility_stall_windows >= 1
        assert policy.min_objective_delta >= 0.0
        assert policy.min_feasibility_delta >= 0.0
        assert policy.max_same_action_windows >= 1
        assert policy.backlog_prune_min_pending >= 1
        assert 0.0 < policy.backlog_prune_fraction <= 1.0
        assert policy.rebootstrap_top_feasible_k >= 1
        assert policy.rebootstrap_top_nearfeasible_k >= 1
        assert 0.0 <= policy.nearfeasible_min <= policy.nearfeasible_max
        assert policy.operator_shift_lock_cycles >= 1
        assert policy.invalid_basin_failure_limit >= 1
        assert policy.autoscale_min_workers >= 1
        assert policy.autoscale_max_workers >= policy.autoscale_min_workers
        assert policy.autoscale_step >= 1
        assert policy.autoscale_up_pending_ratio >= 0.0
        assert policy.autoscale_down_pending_ratio >= 0.0
        assert policy.autoscale_cooldown_cycles >= 0
