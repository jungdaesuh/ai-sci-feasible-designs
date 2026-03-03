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
class RunSurgeryPolicy:
    eval_window: int
    objective_stall_windows: int
    feasibility_stall_windows: int
    min_objective_delta: float
    min_feasibility_delta: float
    max_same_action_windows: int
    backlog_prune_min_pending: int
    backlog_prune_fraction: float
    rebootstrap_top_feasible_k: int
    rebootstrap_top_nearfeasible_k: int
    nearfeasible_min: float
    nearfeasible_max: float
    operator_shift_lock_cycles: int
    invalid_basin_failure_limit: int
    autoscale_enabled: bool
    autoscale_min_workers: int
    autoscale_max_workers: int
    autoscale_step: int
    autoscale_up_pending_ratio: float
    autoscale_down_pending_ratio: float
    autoscale_cooldown_cycles: int


@dataclass(frozen=True)
class AutonomyPolicy:
    diversity_floor_min_candidates: int
    anti_repeat_no_progress_cycles: int
    invalid_basin_max_consecutive_parent_failures: int
    stagnation_min_mutation_delta: float
    autonomous_feasibility_stall_eval_window: int
    autonomous_objective_stall_eval_window: int
    run_surgery: RunSurgeryPolicy


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
    autonomy_policy: AutonomyPolicy

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

_DEFAULT_AUTONOMY_POLICY = AutonomyPolicy(
    diversity_floor_min_candidates=3,
    anti_repeat_no_progress_cycles=2,
    invalid_basin_max_consecutive_parent_failures=3,
    stagnation_min_mutation_delta=0.01,
    autonomous_feasibility_stall_eval_window=24,
    autonomous_objective_stall_eval_window=36,
    run_surgery=RunSurgeryPolicy(
        eval_window=24,
        objective_stall_windows=2,
        feasibility_stall_windows=2,
        min_objective_delta=0.002,
        min_feasibility_delta=0.0001,
        max_same_action_windows=2,
        backlog_prune_min_pending=24,
        backlog_prune_fraction=0.5,
        rebootstrap_top_feasible_k=40,
        rebootstrap_top_nearfeasible_k=80,
        nearfeasible_min=0.0100,
        nearfeasible_max=0.0500,
        operator_shift_lock_cycles=3,
        invalid_basin_failure_limit=3,
        autoscale_enabled=True,
        autoscale_min_workers=6,
        autoscale_max_workers=12,
        autoscale_step=2,
        autoscale_up_pending_ratio=1.5,
        autoscale_down_pending_ratio=0.5,
        autoscale_cooldown_cycles=2,
    ),
)

_P1_RUN_SURGERY_POLICY = RunSurgeryPolicy(
    eval_window=30,
    objective_stall_windows=2,
    feasibility_stall_windows=2,
    min_objective_delta=0.001,
    min_feasibility_delta=0.0001,
    max_same_action_windows=2,
    backlog_prune_min_pending=20,
    backlog_prune_fraction=0.4,
    rebootstrap_top_feasible_k=24,
    rebootstrap_top_nearfeasible_k=48,
    nearfeasible_min=0.0100,
    nearfeasible_max=0.0600,
    operator_shift_lock_cycles=2,
    invalid_basin_failure_limit=3,
    autoscale_enabled=True,
    autoscale_min_workers=4,
    autoscale_max_workers=10,
    autoscale_step=2,
    autoscale_up_pending_ratio=1.6,
    autoscale_down_pending_ratio=0.6,
    autoscale_cooldown_cycles=2,
)

_P2_RUN_SURGERY_POLICY = RunSurgeryPolicy(
    eval_window=28,
    objective_stall_windows=2,
    feasibility_stall_windows=2,
    min_objective_delta=0.0015,
    min_feasibility_delta=0.0001,
    max_same_action_windows=2,
    backlog_prune_min_pending=22,
    backlog_prune_fraction=0.45,
    rebootstrap_top_feasible_k=32,
    rebootstrap_top_nearfeasible_k=64,
    nearfeasible_min=0.0100,
    nearfeasible_max=0.0550,
    operator_shift_lock_cycles=3,
    invalid_basin_failure_limit=3,
    autoscale_enabled=True,
    autoscale_min_workers=5,
    autoscale_max_workers=10,
    autoscale_step=2,
    autoscale_up_pending_ratio=1.55,
    autoscale_down_pending_ratio=0.55,
    autoscale_cooldown_cycles=2,
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
    autonomy_policy=_DEFAULT_AUTONOMY_POLICY,
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
    autonomy_policy=AutonomyPolicy(
        diversity_floor_min_candidates=_DEFAULT_AUTONOMY_POLICY.diversity_floor_min_candidates,
        anti_repeat_no_progress_cycles=_DEFAULT_AUTONOMY_POLICY.anti_repeat_no_progress_cycles,
        invalid_basin_max_consecutive_parent_failures=_DEFAULT_AUTONOMY_POLICY.invalid_basin_max_consecutive_parent_failures,
        stagnation_min_mutation_delta=_DEFAULT_AUTONOMY_POLICY.stagnation_min_mutation_delta,
        autonomous_feasibility_stall_eval_window=_DEFAULT_AUTONOMY_POLICY.autonomous_feasibility_stall_eval_window,
        autonomous_objective_stall_eval_window=_DEFAULT_AUTONOMY_POLICY.autonomous_objective_stall_eval_window,
        run_surgery=_P1_RUN_SURGERY_POLICY,
    ),
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
    autonomy_policy=AutonomyPolicy(
        diversity_floor_min_candidates=_DEFAULT_AUTONOMY_POLICY.diversity_floor_min_candidates,
        anti_repeat_no_progress_cycles=_DEFAULT_AUTONOMY_POLICY.anti_repeat_no_progress_cycles,
        invalid_basin_max_consecutive_parent_failures=_DEFAULT_AUTONOMY_POLICY.invalid_basin_max_consecutive_parent_failures,
        stagnation_min_mutation_delta=_DEFAULT_AUTONOMY_POLICY.stagnation_min_mutation_delta,
        autonomous_feasibility_stall_eval_window=_DEFAULT_AUTONOMY_POLICY.autonomous_feasibility_stall_eval_window,
        autonomous_objective_stall_eval_window=_DEFAULT_AUTONOMY_POLICY.autonomous_objective_stall_eval_window,
        run_surgery=_P2_RUN_SURGERY_POLICY,
    ),
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
        "autonomy_policy": {
            "diversity_floor_min_candidates": profile.autonomy_policy.diversity_floor_min_candidates,
            "anti_repeat_no_progress_cycles": profile.autonomy_policy.anti_repeat_no_progress_cycles,
            "invalid_basin_max_consecutive_parent_failures": profile.autonomy_policy.invalid_basin_max_consecutive_parent_failures,
            "stagnation_min_mutation_delta": profile.autonomy_policy.stagnation_min_mutation_delta,
            "autonomous_feasibility_stall_eval_window": profile.autonomy_policy.autonomous_feasibility_stall_eval_window,
            "autonomous_objective_stall_eval_window": profile.autonomy_policy.autonomous_objective_stall_eval_window,
            "run_surgery": {
                "eval_window": profile.autonomy_policy.run_surgery.eval_window,
                "objective_stall_windows": profile.autonomy_policy.run_surgery.objective_stall_windows,
                "feasibility_stall_windows": profile.autonomy_policy.run_surgery.feasibility_stall_windows,
                "min_objective_delta": profile.autonomy_policy.run_surgery.min_objective_delta,
                "min_feasibility_delta": profile.autonomy_policy.run_surgery.min_feasibility_delta,
                "max_same_action_windows": profile.autonomy_policy.run_surgery.max_same_action_windows,
                "backlog_prune_min_pending": profile.autonomy_policy.run_surgery.backlog_prune_min_pending,
                "backlog_prune_fraction": profile.autonomy_policy.run_surgery.backlog_prune_fraction,
                "rebootstrap_top_feasible_k": profile.autonomy_policy.run_surgery.rebootstrap_top_feasible_k,
                "rebootstrap_top_nearfeasible_k": profile.autonomy_policy.run_surgery.rebootstrap_top_nearfeasible_k,
                "nearfeasible_min": profile.autonomy_policy.run_surgery.nearfeasible_min,
                "nearfeasible_max": profile.autonomy_policy.run_surgery.nearfeasible_max,
                "operator_shift_lock_cycles": profile.autonomy_policy.run_surgery.operator_shift_lock_cycles,
                "invalid_basin_failure_limit": profile.autonomy_policy.run_surgery.invalid_basin_failure_limit,
                "autoscale_enabled": profile.autonomy_policy.run_surgery.autoscale_enabled,
                "autoscale_min_workers": profile.autonomy_policy.run_surgery.autoscale_min_workers,
                "autoscale_max_workers": profile.autonomy_policy.run_surgery.autoscale_max_workers,
                "autoscale_step": profile.autonomy_policy.run_surgery.autoscale_step,
                "autoscale_up_pending_ratio": profile.autonomy_policy.run_surgery.autoscale_up_pending_ratio,
                "autoscale_down_pending_ratio": profile.autonomy_policy.run_surgery.autoscale_down_pending_ratio,
                "autoscale_cooldown_cycles": profile.autonomy_policy.run_surgery.autoscale_cooldown_cycles,
            },
        },
    }
