"""Problem definitions for AI Scientist optimization tasks."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np

from ai_scientist.constraints import get_constraint_names


class Problem(ABC):
    """Abstract base class for optimization problems."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Problem identifier (p1, p2, p3)."""
        ...

    @property
    @abstractmethod
    def constraint_names(self) -> List[str]:
        """Names of constraints for this problem."""
        ...

    @abstractmethod
    def _normalized_constraint_violations(self, metrics: Dict[str, Any]) -> np.ndarray:
        """
        Compute normalized constraint violations.

        Returns array where positive values indicate violation.
        """
        ...

    @abstractmethod
    def get_objective(self, metrics: Dict[str, Any]) -> float:
        """Extract objective value from metrics."""
        ...

    # Template methods
    def is_feasible(self, metrics: Dict[str, Any]) -> bool:
        """Check if metrics satisfy all constraints."""
        violations = self._normalized_constraint_violations(metrics)
        # Use strict tolerance to match constellaration exactly.
        # The 1e-9 threshold ensures no false acceptance at constraint boundaries
        # (e.g., vacuum_well = -0.0001 should NOT pass as feasible).
        return bool(np.all(violations <= 1e-9))

    def compute_feasibility(self, metrics: Dict[str, Any]) -> float:
        """Compute continuous feasibility metric (0 = feasible)."""
        violations = self._normalized_constraint_violations(metrics)
        return float(np.max(np.maximum(violations, 0)))

    def max_violation(self, metrics: Dict[str, Any]) -> float:
        """Get maximum constraint violation."""
        violations = self._normalized_constraint_violations(metrics)
        return float(np.max(np.maximum(violations, 0)))


class P1Problem(Problem):
    """Geometrical Problem - minimize elongation."""

    @property
    def name(self) -> str:
        return "p1"

    @property
    def constraint_names(self) -> List[str]:
        return get_constraint_names("p1")

    # Constraint constants (exposed for prompts)
    _aspect_ratio_upper_bound = 4.0
    _average_triangularity_upper_bound = -0.5
    _edge_rotational_transform_over_n_field_periods_lower_bound = 0.3

    def _normalized_constraint_violations(self, metrics: Dict[str, Any]) -> np.ndarray:
        # Constraints from constellaration.problems.GeometricalProblem
        # 1. aspect_ratio <= 4.0
        # 2. average_triangularity <= -0.5
        # 3. edge_rotational_transform_over_n_field_periods >= 0.3

        ar_limit = self._aspect_ratio_upper_bound
        tri_limit = self._average_triangularity_upper_bound
        iota_limit = self._edge_rotational_transform_over_n_field_periods_lower_bound

        ar = metrics.get("aspect_ratio", 0.0)
        tri = metrics.get("average_triangularity", 0.0)
        iota = metrics.get("edge_rotational_transform_over_n_field_periods", 0.0)

        violations = np.array(
            [
                (ar - ar_limit) / abs(ar_limit),
                (tri - tri_limit) / abs(tri_limit),
                (iota_limit - iota) / abs(iota_limit),
            ]
        )
        return violations

    def get_objective(self, metrics: Dict[str, Any]) -> float:
        return metrics.get("max_elongation", float("inf"))


class P2Problem(Problem):
    """Simple-to-Build QI Stellarator."""

    @property
    def name(self) -> str:
        return "p2"

    @property
    def constraint_names(self) -> List[str]:
        return get_constraint_names("p2")

    # Constraint constants (exposed for prompts)
    _aspect_ratio_upper_bound = 10.0
    _edge_rotational_transform_over_n_field_periods_lower_bound = 0.25
    _log10_qi_upper_bound = -4.0
    _edge_magnetic_mirror_ratio_upper_bound = 0.2
    _max_elongation_upper_bound = 5.0

    def _normalized_constraint_violations(self, metrics: Dict[str, Any]) -> np.ndarray:
        # Constraints from constellaration.problems.SimpleToBuildQIStellarator
        # 1. aspect_ratio <= 10.0
        # 2. edge_rotational_transform_over_n_field_periods >= 0.25
        # 3. log10(qi) <= -4.0
        # 4. edge_magnetic_mirror_ratio <= 0.2
        # 5. max_elongation <= 5.0

        ar_limit = self._aspect_ratio_upper_bound
        iota_limit = self._edge_rotational_transform_over_n_field_periods_lower_bound
        qi_limit = self._log10_qi_upper_bound
        mirror_limit = self._edge_magnetic_mirror_ratio_upper_bound
        elong_limit = self._max_elongation_upper_bound

        ar = metrics.get("aspect_ratio", 0.0)
        iota = metrics.get("edge_rotational_transform_over_n_field_periods", 0.0)
        qi = metrics.get(
            "qi", 1.0
        )  # Default to 1.0 to avoid log(0) issues if missing, though it should be there
        log_qi = np.log10(qi) if qi > 0 else 0.0  # Handle potential edge cases safely
        mirror = metrics.get("edge_magnetic_mirror_ratio", 0.0)
        elong = metrics.get("max_elongation", 0.0)

        violations = np.array(
            [
                (ar - ar_limit) / abs(ar_limit),
                (iota_limit - iota) / abs(iota_limit),
                (log_qi - qi_limit) / abs(qi_limit),
                (mirror - mirror_limit) / abs(mirror_limit),
                (elong - elong_limit) / abs(elong_limit),
            ]
        )
        return violations

    def get_objective(self, metrics: Dict[str, Any]) -> float:
        # Maximize minimum_normalized_magnetic_gradient_scale_length
        # We return negative because optimization usually minimizes
        return -metrics.get("minimum_normalized_magnetic_gradient_scale_length", 0.0)


class P3Problem(Problem):
    """MHD-Stable QI Stellarator (multi-objective)."""

    @property
    def name(self) -> str:
        return "p3"

    @property
    def constraint_names(self) -> List[str]:
        return get_constraint_names("p3")

    # Constraint constants (exposed for prompts)
    _edge_rotational_transform_over_n_field_periods_lower_bound = 0.25
    _log10_qi_upper_bound = -3.5
    _edge_magnetic_mirror_ratio_upper_bound = 0.25
    _flux_compression_in_regions_of_bad_curvature_upper_bound = 0.9
    _vacuum_well_lower_bound = 0.0
    # Normalization constant for vacuum_well when bound is 0 (avoids division by zero).
    # Value matches constellaration.problems.MHDStableQIStellarator implementation.
    _vacuum_well_normalization_fallback = 0.1

    def _normalized_constraint_violations(self, metrics: Dict[str, Any]) -> np.ndarray:
        # Constraints from constellaration.problems.MHDStableQIStellarator
        # 1. edge_rotational_transform_over_n_field_periods >= 0.25
        # 2. log10(qi) <= -3.5
        # 3. edge_magnetic_mirror_ratio <= 0.25
        # 4. flux_compression_in_regions_of_bad_curvature <= 0.9
        # 5. vacuum_well >= 0.0

        iota_limit = self._edge_rotational_transform_over_n_field_periods_lower_bound
        qi_limit = self._log10_qi_upper_bound
        mirror_limit = self._edge_magnetic_mirror_ratio_upper_bound
        flux_limit = self._flux_compression_in_regions_of_bad_curvature_upper_bound
        well_limit = self._vacuum_well_lower_bound
        well_norm = max(self._vacuum_well_normalization_fallback, abs(well_limit))

        iota = metrics.get("edge_rotational_transform_over_n_field_periods", 0.0)
        qi = metrics.get("qi", 1.0)
        log_qi = np.log10(qi) if qi > 0 else 0.0
        mirror = metrics.get("edge_magnetic_mirror_ratio", 0.0)
        flux = metrics.get("flux_compression_in_regions_of_bad_curvature", 0.0)
        well = metrics.get("vacuum_well", 0.0)

        violations = np.array(
            [
                (iota_limit - iota) / abs(iota_limit),
                (log_qi - qi_limit) / abs(qi_limit),
                (mirror - mirror_limit) / abs(mirror_limit),
                (flux - flux_limit) / abs(flux_limit),
                (well_limit - well) / abs(well_norm),
            ]
        )
        return violations

    def get_objective(self, metrics: Dict[str, Any]) -> float:
        # P3 is multi-objective, but for single-value context (like simple tracking)
        # we might return the primary objective or a scalarization.
        # However, Coordinator uses ALM which handles constraints.
        # The 'objective' in ALM state is usually the Lagrangian.
        # For simple tracking, let's return the gradient scale length (negated)
        # as it's the primary physics performance metric besides aspect ratio.
        return -metrics.get("minimum_normalized_magnetic_gradient_scale_length", 0.0)


def get_problem(name: str) -> Problem:
    """Get problem instance by name."""
    problems = {
        "p1": P1Problem(),
        "p2": P2Problem(),
        "p3": P3Problem(),
    }
    key = name.lower()
    if key not in problems:
        # Fallback for "p3_..." or similar variations if any
        if key.startswith("p1"):
            return problems["p1"]
        if key.startswith("p2"):
            return problems["p2"]
        return problems["p3"]

    return problems[key]
