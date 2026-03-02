"""SSOT problem profiles for governor decision contracts.

This module centralizes per-problem objective/constraint definitions,
action allowlists, and controller policy defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ProblemId = Literal["p1", "p2", "p3"]
ActionName = Literal["repair", "bridge", "jump", "global_restart"]
ConstraintRelation = Literal["<=", ">="]


@dataclass(frozen=True)
class ConstraintSpec:
    name: str
    relation: ConstraintRelation
    threshold: float


@dataclass(frozen=True)
class ObjectiveSpec:
    expression: str
    direction: Literal["minimize", "maximize", "multi_objective"]


@dataclass(frozen=True)
class MutationBudget:
    max_candidates_per_cycle: int
    max_mutations_per_candidate: int
    max_mutation_groups_per_candidate: int
    repair_delta_cap: float
    bridge_delta_cap: float
    jump_delta_cap: float

    def action_delta_cap(self, action: str) -> float:
        if action == "repair":
            return float(self.repair_delta_cap)
        if action == "bridge":
            return float(self.bridge_delta_cap)
        if action == "jump":
            return float(self.jump_delta_cap)
        return 0.0


@dataclass(frozen=True)
class RestartThresholds:
    soft_retry_max_consecutive_failures: int
    degraded_restart_min_consecutive_failures: int
    degraded_restart_min_queue_desync_events_last20: int
    global_restart_min_stagnation_cycles: int
    circuit_break_min_invalid_outputs_last20: int


@dataclass(frozen=True)
class PhaseSwitchThresholds:
    improve_min_accepted_feasible_last20: int
    improve_max_dominant_violation_rate_last20: float
    revert_if_accepted_feasible_last10_eq: int


@dataclass(frozen=True)
class FrontierRecipeConfig:
    objective_metric: str
    objective_direction: Literal["minimize", "maximize"]
    aspect_metric: str | None
    base_perturbation_scale: float
    min_perturbation_scale: float
    max_perturbation_scale: float
    hv_gap_sensitivity: float
    frontier_move_families: tuple[str, ...]


@dataclass(frozen=True)
class ProblemProfile:
    problem: ProblemId
    objective: ObjectiveSpec
    constraints: tuple[ConstraintSpec, ...]
    allowed_actions: tuple[ActionName, ...]
    mutation_budget: MutationBudget
    restart_thresholds: RestartThresholds
    phase_switch_thresholds: PhaseSwitchThresholds
    frontier_recipe: FrontierRecipeConfig

    def allows_action(self, action: str) -> bool:
        return action in self.allowed_actions

    def allowed_constraint_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.constraints)


_DEFAULT_MUTATION_BUDGET = MutationBudget(
    max_candidates_per_cycle=8,
    max_mutations_per_candidate=6,
    max_mutation_groups_per_candidate=3,
    repair_delta_cap=0.15,
    bridge_delta_cap=0.30,
    jump_delta_cap=0.45,
)

_DEFAULT_RESTART_THRESHOLDS = RestartThresholds(
    soft_retry_max_consecutive_failures=2,
    degraded_restart_min_consecutive_failures=3,
    degraded_restart_min_queue_desync_events_last20=1,
    global_restart_min_stagnation_cycles=8,
    circuit_break_min_invalid_outputs_last20=3,
)

_DEFAULT_PHASE_THRESHOLDS = PhaseSwitchThresholds(
    improve_min_accepted_feasible_last20=3,
    improve_max_dominant_violation_rate_last20=0.20,
    revert_if_accepted_feasible_last10_eq=0,
)

_P3_FRONTIER_RECIPE = FrontierRecipeConfig(
    objective_metric="lgradB",
    objective_direction="maximize",
    aspect_metric="aspect_ratio",
    base_perturbation_scale=0.08,
    min_perturbation_scale=0.03,
    max_perturbation_scale=0.15,
    hv_gap_sensitivity=1.25,
    frontier_move_families=("scale_groups", "blend"),
)

_P2_FRONTIER_RECIPE = FrontierRecipeConfig(
    objective_metric="lgradB",
    objective_direction="maximize",
    aspect_metric=None,
    base_perturbation_scale=0.06,
    min_perturbation_scale=0.02,
    max_perturbation_scale=0.12,
    hv_gap_sensitivity=1.00,
    frontier_move_families=("scale_groups",),
)

_P1_FRONTIER_RECIPE = FrontierRecipeConfig(
    objective_metric="max_elongation",
    objective_direction="minimize",
    aspect_metric=None,
    base_perturbation_scale=0.05,
    min_perturbation_scale=0.02,
    max_perturbation_scale=0.10,
    hv_gap_sensitivity=1.00,
    frontier_move_families=("scale_groups",),
)


P3_PROFILE = ProblemProfile(
    problem="p3",
    objective=ObjectiveSpec(
        expression="max_hv(points=[(-lgradB, aspect_ratio)], ref=[1.0, 20.0])",
        direction="multi_objective",
    ),
    constraints=(
        ConstraintSpec(name="iota", relation=">=", threshold=0.25),
        ConstraintSpec(name="log10_qi", relation="<=", threshold=-3.5),
        ConstraintSpec(name="mirror", relation="<=", threshold=0.25),
        ConstraintSpec(name="flux", relation="<=", threshold=0.9),
        ConstraintSpec(name="vacuum", relation=">=", threshold=0.0),
    ),
    allowed_actions=("repair", "bridge", "global_restart"),
    mutation_budget=_DEFAULT_MUTATION_BUDGET,
    restart_thresholds=_DEFAULT_RESTART_THRESHOLDS,
    phase_switch_thresholds=_DEFAULT_PHASE_THRESHOLDS,
    frontier_recipe=_P3_FRONTIER_RECIPE,
)

# P1/P2 profiles are included for shared decision/prompt contracts.
P1_PROFILE = ProblemProfile(
    problem="p1",
    objective=ObjectiveSpec(
        expression="min(max_elongation)",
        direction="minimize",
    ),
    constraints=(
        ConstraintSpec(name="aspect_ratio", relation="<=", threshold=4.0),
        ConstraintSpec(name="average_triangularity", relation="<=", threshold=-0.5),
        ConstraintSpec(name="iota_edge", relation=">=", threshold=0.3),
    ),
    allowed_actions=("repair", "jump", "global_restart"),
    mutation_budget=_DEFAULT_MUTATION_BUDGET,
    restart_thresholds=_DEFAULT_RESTART_THRESHOLDS,
    phase_switch_thresholds=_DEFAULT_PHASE_THRESHOLDS,
    frontier_recipe=_P1_FRONTIER_RECIPE,
)

P2_PROFILE = ProblemProfile(
    problem="p2",
    objective=ObjectiveSpec(
        expression="max(minimum_normalized_magnetic_gradient_scale_length)",
        direction="maximize",
    ),
    constraints=(
        ConstraintSpec(name="aspect_ratio", relation="<=", threshold=10.0),
        ConstraintSpec(name="iota_edge", relation=">=", threshold=0.25),
        ConstraintSpec(name="log10_qi", relation="<=", threshold=-4.0),
        ConstraintSpec(name="mirror", relation="<=", threshold=0.2),
        ConstraintSpec(name="max_elongation", relation="<=", threshold=5.0),
    ),
    allowed_actions=("repair", "bridge", "global_restart"),
    mutation_budget=_DEFAULT_MUTATION_BUDGET,
    restart_thresholds=_DEFAULT_RESTART_THRESHOLDS,
    phase_switch_thresholds=_DEFAULT_PHASE_THRESHOLDS,
    frontier_recipe=_P2_FRONTIER_RECIPE,
)

PROBLEM_PROFILES: dict[ProblemId, ProblemProfile] = {
    "p1": P1_PROFILE,
    "p2": P2_PROFILE,
    "p3": P3_PROFILE,
}


def get_problem_profile(problem: str) -> ProblemProfile:
    key = str(problem).strip().lower()
    if key not in PROBLEM_PROFILES:
        raise ValueError(f"Unsupported problem profile: {problem!r}")
    return PROBLEM_PROFILES[key]  # type: ignore[index]


def profile_prompt_block(profile: ProblemProfile) -> dict:
    return {
        "problem": profile.problem,
        "objective": {
            "expression": profile.objective.expression,
            "direction": profile.objective.direction,
        },
        "constraints": [
            {
                "name": constraint.name,
                "relation": constraint.relation,
                "threshold": float(constraint.threshold),
            }
            for constraint in profile.constraints
        ],
        "allowed_actions": list(profile.allowed_actions),
        "mutation_budget": {
            "max_candidates_per_cycle": profile.mutation_budget.max_candidates_per_cycle,
            "max_mutations_per_candidate": profile.mutation_budget.max_mutations_per_candidate,
            "max_mutation_groups_per_candidate": profile.mutation_budget.max_mutation_groups_per_candidate,
            "repair_delta_cap": profile.mutation_budget.repair_delta_cap,
            "bridge_delta_cap": profile.mutation_budget.bridge_delta_cap,
            "jump_delta_cap": profile.mutation_budget.jump_delta_cap,
        },
        "frontier_recipe": {
            "objective_metric": profile.frontier_recipe.objective_metric,
            "objective_direction": profile.frontier_recipe.objective_direction,
            "aspect_metric": profile.frontier_recipe.aspect_metric,
            "base_perturbation_scale": profile.frontier_recipe.base_perturbation_scale,
            "min_perturbation_scale": profile.frontier_recipe.min_perturbation_scale,
            "max_perturbation_scale": profile.frontier_recipe.max_perturbation_scale,
            "hv_gap_sensitivity": profile.frontier_recipe.hv_gap_sensitivity,
            "frontier_move_families": list(
                profile.frontier_recipe.frontier_move_families
            ),
        },
    }
