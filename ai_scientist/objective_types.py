"""Objective and Score Type Definitions.

This module documents the THREE distinct objective concepts in the system:

1. **ALM Objective** (`alm_objective`):
   - What the Augmented Lagrangian Method actually minimizes
   - P1: max_elongation
   - P2/P3: 20.0 - minimum_normalized_magnetic_gradient_scale_length
   - Direction: ALWAYS minimize
   - Location: constellaration/optimization/augmented_lagrangian_runner.py

2. **Physics Objective** (`physics_objective`):
   - The primary physics goal per benchmark problem
   - P1: max_elongation (minimize)
   - P2: minimum_normalized_magnetic_gradient_scale_length (maximize)
   - P3: aspect_ratio (minimize) - but P3 is intrinsically multi-objective!
   - Location: ai_scientist/forward_model.py:compute_objective()

3. **Ranking Score** (`ranking_score`):
   - Unified scalar for candidate ranking and surrogate training
   - P1: -max_elongation (higher = better)
   - P2/P3: gradient / aspect (higher = better)
   - Direction: ALWAYS maximize
   - Location: ai_scientist/forward_model.py:compute_ranking_score()

CRITICAL: These are NOT interchangeable! Using alm_objective where ranking_score
is expected (or vice versa) will cause silent semantic bugs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal


# =============================================================================
# TargetKind: Surrogate Training Target (SSOT)
# =============================================================================


class TargetKind(str, Enum):
    """Surrogate training target kind.

    This enum is the Single Source of Truth (SSOT) for what the surrogate
    model predicts during gradient-based optimization.

    Values:
        OBJECTIVE: Problem-specific physics objective (elongation, gradient, aspect).
        HV: Hypervolume metric for Pareto optimization (P3 multi-objective).

    Usage:
        Always pass TargetKind to gradient_descent_on_inputs() and
        optimize_alm_inner_loop() to avoid semantic drift between training
        and optimization.
    """

    OBJECTIVE = "objective"
    # Backwards-compatible alias for the P3 Pareto proxy / hypervolume column.
    # The DB column is named "hv"; newer code may pass "gradient_proxy" and the
    # repository maps it to "hv".
    HV = "hv"
    GRADIENT_PROXY = "gradient_proxy"  # Per-candidate proxy, NOT hypervolume


def get_training_target(problem: str) -> TargetKind:
    """Get the canonical training target for a problem type.

    This is the SSOT for caller → optimizer target synchronization.
    Any change to this policy automatically propagates to all callers.

    Policy:
        P3 (multi-objective): Uses gradient_proxy as surrogate target for Pareto optimization.
        P1/P2 (single-objective): Uses physics objective directly.

    Args:
        problem: Problem identifier (p1, p2, p3).

    Returns:
        TargetKind enum value.
    """
    # H2 FIX: Use GRADIENT_PROXY (the canonical name) instead of HV
    return (
        TargetKind.GRADIENT_PROXY
        if problem.lower().startswith("p3")
        else TargetKind.OBJECTIVE
    )


@dataclass(frozen=True)
class ObjectiveSemantics:
    """Describes the semantics of an objective value."""

    kind: Literal["alm_objective", "physics_objective", "ranking_score"]
    minimize: bool
    description: str


# Canonical definitions per problem
P1_SEMANTICS = {
    "alm_objective": ObjectiveSemantics("alm_objective", True, "max_elongation"),
    "physics_objective": ObjectiveSemantics(
        "physics_objective", True, "max_elongation"
    ),
    "ranking_score": ObjectiveSemantics("ranking_score", False, "-max_elongation"),
}

P2_SEMANTICS = {
    "alm_objective": ObjectiveSemantics("alm_objective", True, "20 - gradient"),
    "physics_objective": ObjectiveSemantics("physics_objective", False, "gradient"),
    "ranking_score": ObjectiveSemantics("ranking_score", False, "gradient / aspect"),
}

P3_SEMANTICS = {
    "alm_objective": ObjectiveSemantics("alm_objective", True, "20 - gradient"),
    "physics_objective": ObjectiveSemantics("physics_objective", True, "aspect_ratio"),
    "ranking_score": ObjectiveSemantics("ranking_score", False, "gradient / aspect"),
}


def get_semantics(problem: str) -> dict[str, ObjectiveSemantics]:
    """Get objective semantics for a problem type.

    Args:
        problem: Problem identifier (p1, p2, p3).

    Returns:
        Dictionary mapping objective kind to its semantics.
    """
    problem_key = problem.lower()
    if problem_key.startswith("p1"):
        return P1_SEMANTICS
    elif problem_key.startswith("p2"):
        return P2_SEMANTICS
    else:
        return P3_SEMANTICS


def compute_ranking_score(metrics: Any, problem: str) -> float:
    """Compute unified ranking score (higher = better).

    This is the canonical scalar for:
    - Surrogate model training targets
    - Candidate ranking/selection
    - Pareto improvement tracking

    NOT the same as ALM objective or physics objective.

    Args:
        metrics: Metrics object or dict with physics values.
        problem: Problem identifier (p1, p2, p3).

    Returns:
        Ranking score where higher is always better.
    """

    # Handle both object and dict access
    def _get(key: str, default: float = 0.0) -> float:
        if hasattr(metrics, key):
            val = getattr(metrics, key)
        elif isinstance(metrics, dict):
            val = metrics.get(key)
        else:
            val = default
        return float(val) if val is not None else default

    problem_key = problem.lower()

    if problem_key.startswith("p1"):
        # P1: minimize elongation → negate for "higher is better"
        return -_get("max_elongation", float("inf"))
    elif problem_key.startswith("p2"):
        # P2: maximize gradient, normalized by aspect
        grad = _get("minimum_normalized_magnetic_gradient_scale_length", 0.0)
        aspect = _get("aspect_ratio", 1.0)
        return grad / max(1.0, aspect)
    else:  # P3
        # P3: multi-objective proxy (gradient / aspect)
        grad = _get("minimum_normalized_magnetic_gradient_scale_length", 0.0)
        aspect = _get("aspect_ratio", 1.0)
        return grad / max(1.0, aspect)


def compute_physics_objective(metrics: Any, problem: str) -> tuple[float, bool]:
    """Compute primary physics objective and optimization direction.

    This returns the benchmark-standard objective value, NOT what ALM minimizes.

    Args:
        metrics: Metrics object or dict with physics values.
        problem: Problem identifier (p1, p2, p3).

    Returns:
        (objective_value, minimize): The objective and whether to minimize it.
    """

    # Handle both object and dict access
    def _get(key: str, default: float = 0.0) -> float:
        if hasattr(metrics, key):
            val = getattr(metrics, key)
        elif isinstance(metrics, dict):
            val = metrics.get(key)
        else:
            val = default
        return float(val) if val is not None else default

    problem_key = problem.lower()

    if problem_key.startswith("p1"):
        return _get("max_elongation"), True  # minimize
    elif problem_key.startswith("p2"):
        return _get(
            "minimum_normalized_magnetic_gradient_scale_length"
        ), False  # maximize
    else:  # P3
        return _get("aspect_ratio"), True  # minimize (primary objective)
