from __future__ import annotations

import pytest

from ai_scientist.staged_governor import (
    apply_delta_recipe,
    build_delta_replay_seeds,
    build_staged_seed_plan_from_snapshots,
    expand_parent_group_staged_offspring,
    worst_constraint_from_violations,
)


def _boundary(scale: float) -> dict[str, object]:
    return {
        "r_cos": [
            [1.0 * scale, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1 * scale, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.05 * scale, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.02 * scale, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        "z_sin": [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.05 * scale, 0.0, 0.05 * scale, 0.0, 0.0],
            [0.0, 0.0, 0.03 * scale, 0.0, 0.03 * scale, 0.0, 0.0],
            [0.0, 0.0, 0.02 * scale, 0.0, 0.02 * scale, 0.0, 0.0],
        ],
        "n_field_periods": 3,
        "is_stellarator_symmetric": True,
    }


def test_worst_constraint_from_violations():
    name, value = worst_constraint_from_violations(
        {"mirror": 0.12, "log10_qi": 0.03, "flux": 0.08}
    )
    assert name == "mirror"
    assert value == 0.12


def test_build_staged_seed_plan_from_snapshots_mirror_flow():
    snapshots = [
        {
            "design_hash": "focus",
            "params": _boundary(1.0),
            "feasibility": 0.09,
            "is_feasible": False,
            "objective": 5.0,
            "constraint_margins": {"mirror": 0.12, "log10_qi": 0.02},
            "metrics": {"mirror": 0.30, "log10_qi": -3.2, "aspect_ratio": 7.5},
        },
        {
            "design_hash": "partner",
            "params": _boundary(0.92),
            "feasibility": 0.002,
            "is_feasible": True,
            "objective": 6.5,
            "constraint_margins": {},
            "metrics": {"mirror": 0.24, "log10_qi": -3.7, "aspect_ratio": 8.1},
        },
    ]

    plan = build_staged_seed_plan_from_snapshots(
        snapshots=snapshots,
        problem="p3",
        near_feasibility_threshold=0.25,
        max_repair_candidates=3,
        bridge_blend_t=0.86,
    )

    assert plan is not None
    assert plan.focus_hash == "focus"
    assert plan.partner_hash == "partner"
    assert plan.worst_constraint == "mirror"
    assert len(plan.seeds) >= 3
    phases = [
        seed["staged_governor"]["phase"]
        for seed in plan.seeds
        if isinstance(seed.get("staged_governor"), dict)
    ]
    assert "focus" in phases
    assert "bridge" in phases
    assert "repair" in phases


def test_expand_parent_group_staged_offspring_sets_lineage_metadata():
    parent_group = [
        {
            "design_hash": "focus",
            "params": _boundary(1.0),
            "constraint_margins": {"mirror": 0.12},
        },
        {
            "design_hash": "partner",
            "params": _boundary(0.94),
            "constraint_margins": {},
        },
    ]

    seeds = expand_parent_group_staged_offspring(
        parent_group=parent_group,
        worst_constraint="mirror",
        max_repair_candidates=2,
        bridge_blend_t=0.86,
        offspring_per_parent=2,
    )

    assert len(seeds) >= 3
    assert all(seed.get("source") == "staged_governor" for seed in seeds)
    assert all("lineage_parent_hashes" in seed for seed in seeds)
    assert all("operator_family" in seed for seed in seeds)
    assert all("improvement_reason" in seed for seed in seeds)


def test_apply_delta_recipe_updates_sparse_coefficients():
    base = _boundary(1.0)
    updated = apply_delta_recipe(
        base,
        [{"key": "r_cos.0.0", "delta": 0.1}, {"key": "z_sin.1.2", "delta": -0.02}],
    )

    assert updated is not None
    assert updated["r_cos"][0][0] == pytest.approx(1.1)
    assert updated["z_sin"][1][2] == pytest.approx(0.03)


def test_build_delta_replay_seeds_uses_top_k_recipes():
    focus = _boundary(1.0)
    nearest = [
        {
            "design_hash": "case-a",
            "delta_recipe": [{"key": "r_cos.0.0", "delta": 0.05}],
        },
        {
            "design_hash": "case-b",
            "delta_recipe": [{"key": "z_sin.1.2", "delta": -0.01}],
        },
    ]

    seeds = build_delta_replay_seeds(
        focus_params=focus,
        case_deltas=nearest,
        top_k=1,
        focus_hash="focus",
        worst_constraint="mirror",
    )

    assert len(seeds) == 1
    staged = seeds[0]["staged_governor"]
    assert staged["phase"] == "delta_replay"
    assert seeds[0]["improvement_reason"] == "nearest_case_delta_replay"
