"""Planner prompts grounded in live problem specs + governance docs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from constellaration import problems as problem_module

_SOURCE_PATH = "constellaration/src/constellaration/problems.py"


@dataclass(frozen=True)
class ConstraintSpec:
    name: str
    operator: str
    value: float
    description: str


@dataclass(frozen=True)
class ObjectiveSpec:
    name: str
    direction: str  # "min" | "max"
    description: str


@dataclass(frozen=True)
class ProblemSpec:
    key: str
    constraints: tuple[ConstraintSpec, ...]
    objectives: tuple[ObjectiveSpec, ...]
    score_description: str
    source: str = _SOURCE_PATH

    def prompt_block(self) -> str:
        constraint_lines = "\n".join(
            f"- {c.name} {c.operator} {c.value:g} ({c.description})"
            for c in self.constraints
        )
        objective_lines = "\n".join(
            f"- {obj.direction.upper()} {obj.name}: {obj.description}"
            for obj in self.objectives
        )
        return (
            f"Constraints ({self.source}):\n{constraint_lines}\n"
            f"Objectives:\n{objective_lines}\n"
            f"Scoring: {self.score_description}\n"
        )


def _geometrical_spec() -> ProblemSpec:
    instance = problem_module.GeometricalProblem()
    return ProblemSpec(
        key="p1",
        constraints=(
            ConstraintSpec(
                "aspect_ratio",
                "<=",
                instance._aspect_ratio_upper_bound,
                "controls width vs. height",
            ),
            ConstraintSpec(
                "average_triangularity",
                "<=",
                instance._average_triangularity_upper_bound,
                "keeps indentations bounded toward circular shapes",
            ),
            ConstraintSpec(
                "edge_rotational_transform_over_n_field_periods",
                ">=",
                instance._edge_rotational_transform_over_n_field_periods_lower_bound,
                "ensures sufficient edge winding per field period",
            ),
        ),
        objectives=(
            ObjectiveSpec(
                "max_elongation",
                "min",
                "lower elongation yields more circular, feasible geometries",
            ),
        ),
        score_description=(
            "Normalized max elongation between 1.0 (ideal) and 10.0 (poor)."
        ),
    )


def _simple_qi_spec() -> ProblemSpec:
    instance = problem_module.SimpleToBuildQIStellarator()
    return ProblemSpec(
        key="p2",
        constraints=(
            ConstraintSpec(
                "aspect_ratio",
                "<=",
                instance._aspect_ratio_upper_bound,
                "limits width/height for coilability",
            ),
            ConstraintSpec(
                "edge_rotational_transform_over_n_field_periods",
                ">=",
                instance._edge_rotational_transform_over_n_field_periods_lower_bound,
                "preserves transform per field period",
            ),
            ConstraintSpec(
                "log10(QI residual)",
                "<=",
                instance._log10_qi_upper_bound,
                "bounds quasi-isodynamic residuals for transport",
            ),
            ConstraintSpec(
                "edge_magnetic_mirror_ratio",
                "<=",
                instance._edge_magnetic_mirror_ratio_upper_bound,
                "keeps boundary field variation manageable",
            ),
            ConstraintSpec(
                "max_elongation",
                "<=",
                instance._max_elongation_upper_bound,
                "prevents extreme vertical stretching",
            ),
        ),
        objectives=(
            ObjectiveSpec(
                "minimum_normalized_magnetic_gradient_scale_length",
                "max",
                "higher gradients correlate with easier-to-build coils",
            ),
        ),
        score_description=(
            "Linear score over the minimum normalized magnetic gradient scale length "
            "(0.0 poor → 1.0 optimal)."
        ),
    )


def _mhd_qi_spec() -> ProblemSpec:
    instance = problem_module.MHDStableQIStellarator()
    return ProblemSpec(
        key="p3",
        constraints=(
            ConstraintSpec(
                "edge_rotational_transform_over_n_field_periods",
                ">=",
                instance._edge_rotational_transform_over_n_field_periods_lower_bound,
                "ensures winding per period",
            ),
            ConstraintSpec(
                "log10(QI residual)",
                "<=",
                instance._log10_qi_upper_bound,
                "keeps QI residual small for confinement",
            ),
            ConstraintSpec(
                "edge_magnetic_mirror_ratio",
                "<=",
                instance._edge_magnetic_mirror_ratio_upper_bound,
                "limits mirror ratio at the edge",
            ),
            ConstraintSpec(
                "flux_compression_in_regions_of_bad_curvature",
                "<=",
                instance._flux_compression_in_regions_of_bad_curvature_upper_bound,
                "controls turbulent transport proxy",
            ),
            ConstraintSpec(
                "vacuum_well",
                ">=",
                instance._vacuum_well_lower_bound,
                "maintains ideal-MHD stability margin",
            ),
        ),
        objectives=(
            ObjectiveSpec(
                "minimum_normalized_magnetic_gradient_scale_length",
                "max",
                "larger gradients improve QI performance",
            ),
            ObjectiveSpec(
                "aspect_ratio",
                "min",
                "smaller aspect ratios favor compact machines",
            ),
        ),
        score_description=(
            "Hypervolume of feasible (-gradient, aspect_ratio) points relative to [1.0, 20.0]."
        ),
    )


_PROBLEM_SPECS: Mapping[str, ProblemSpec] = {
    "p1": _geometrical_spec(),
    "p2": _simple_qi_spec(),
    "p3": _mhd_qi_spec(),
}

PHASE_GUIDANCE = (
    "Phase 6 requires computing/archiving the Pareto front + hypervolume each cycle; "
    "Phase 9 acceptance demands feasible P1–P3 designs, reproducible logs, and cited claims "
    "(docs/MASTER_PLAN_AI_SCIENTIST.md)."
)

BUDGET_REMINDER = "Respect Wave B budget guardrails (docs/TASKS_CODEX_MINI.md) when narrating promotions."

TOOL_REMINDER = "Use ai_scientist.tools_api schemas (Wave 8) and cite repositories instead of line numbers."

REPRO_PROMPT = (
    "Each report needs deterministic reproduction steps: git SHAs (repo + constellaration), "
    "seed, fidelity, settings dump, and a rerun command for an archived design."
)


def get_problem_spec(problem: str) -> ProblemSpec:
    key = problem.lower()
    if key not in _PROBLEM_SPECS:
        raise KeyError(f"unknown problem spec '{problem}'")
    return _PROBLEM_SPECS[key]


def build_problem_prompt(problem: str, stage: str) -> str:
    spec = get_problem_spec(problem)
    return (
        f"Planning for {problem.upper()} at stage '{stage}'.\n"
        f"{spec.prompt_block()}"
        f"{PHASE_GUIDANCE}\n"
        f"{BUDGET_REMINDER}\n"
        f"{TOOL_REMINDER}\n"
        f"{REPRO_PROMPT}\n"
    )


def annotate_solution_summary(summary: str) -> str:
    return (
        f"Summary (tie back to constraints/objectives + cite {_SOURCE_PATH}): {summary}"
    )
