"""Centralized constraint definitions for stellarator optimization problems.

This module is the Single Source of Truth (SSOT) for constraint names,
ensuring consistency across:
- problems.py (Problem.constraint_names)
- forward_model.py (compute_constraint_margins)
- coordinator.py (_alm_constraint_names)
- cycle_executor.py (inline constraint lists)
- rl_env.py (reward shaping)

The canonical names here match the constellaration benchmark definitions.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

# =============================================================================
# CANONICAL CONSTRAINT NAMES
# =============================================================================
# These are the API/telemetry/storage keys used in margins dicts and logs.
# They intentionally use short, readable names while the METRICS_KEY_MAP
# provides the mapping to full metrics field names.

P1_CONSTRAINT_NAMES: List[str] = [
    "aspect_ratio",
    "average_triangularity",
    "edge_rotational_transform",
]

P2_CONSTRAINT_NAMES: List[str] = [
    "aspect_ratio",
    "edge_rotational_transform",
    "qi",  # Constraint uses log10(qi) - threshold (see forward_model.compute_constraint_margins)
    "edge_magnetic_mirror_ratio",
    "max_elongation",
]

P3_CONSTRAINT_NAMES: List[str] = [
    "edge_rotational_transform",
    "qi",
    "edge_magnetic_mirror_ratio",
    "flux_compression",
    "vacuum_well",
]

# P3 ALM runner includes aspect_ratio as an additional constraint
P3_ALM_CONSTRAINT_NAMES: List[str] = [
    "aspect_ratio",
    "edge_rotational_transform",
    "qi",
    "edge_magnetic_mirror_ratio",
    "flux_compression",
    "vacuum_well",
]

# =============================================================================
# METRICS KEY MAPPING
# =============================================================================
# Maps canonical constraint names to the full field names in metrics objects.
# Use get_metrics_key() to retrieve.

METRICS_KEY_MAP: Dict[str, str] = {
    # Geometry
    "aspect_ratio": "aspect_ratio",
    "average_triangularity": "average_triangularity",
    "max_elongation": "max_elongation",
    # Rotational transform (iota) - note the long metrics name
    "edge_rotational_transform": "edge_rotational_transform_over_n_field_periods",
    # Magnetic properties
    "edge_magnetic_mirror_ratio": "edge_magnetic_mirror_ratio",
    # QI residual - stored as raw "qi" in metrics, but constraint is log10(qi) <= threshold
    "qi": "qi",
    # MHD stability
    "vacuum_well": "vacuum_well",
    "flux_compression": "flux_compression_in_regions_of_bad_curvature",
}

# =============================================================================
# CONSTRAINT BOUNDS (for reference/validation)
# =============================================================================

CONSTRAINT_BOUNDS: Dict[str, Dict[str, float]] = {
    "p1": {
        "aspect_ratio_upper": 4.0,
        "average_triangularity_upper": -0.5,
        "edge_rotational_transform_lower": 0.3,
    },
    "p2": {
        "aspect_ratio_upper": 10.0,
        "edge_rotational_transform_lower": 0.25,
        "qi_log10_upper": -4.0,
        "edge_magnetic_mirror_ratio_upper": 0.2,
        "max_elongation_upper": 5.0,
    },
    "p3": {
        "edge_rotational_transform_lower": 0.25,
        "qi_log10_upper": -3.5,
        "edge_magnetic_mirror_ratio_upper": 0.25,
        "flux_compression_upper": 0.9,
        "vacuum_well_lower": 0.0,
    },
}


# =============================================================================
# PHYSICS CONSTANTS (used in differentiable optimization)
# =============================================================================
# These constants are used in gradient-based optimization (differentiable.py)
# and should be kept consistent with CONSTRAINT_BOUNDS above.

# Maximum elongation bound (P2 constraint)
MAX_ELONGATION: float = 5.0

# Small epsilon for numerical stability in log computation
QI_EPS: float = 1e-12

# Log10(QI) thresholds by problem (constraint: log10(qi) <= threshold)
LOG10_QI_THRESHOLDS: Dict[str, float] = {
    "p2": -4.0,  # Stricter threshold for P2
    "p3": -3.5,  # Default threshold for P3
}

# Fourier decay regularization weight
LAMBDA_FOURIER_DECAY: float = 0.01

# L2 regularization weight for R₀₀ (major radius) toward dataset mean (~1.0)
# Strong regularization (10.0) prevents scale drift while allowing optimization
# Used when optimize_major_radius=True in ExperimentConfig
LAMBDA_R00_REGULARIZATION: float = 10.0

# H1 Fix: Default R₀₀ regularization target (dataset mean ~1.0)
# Override via R00_TARGET env var or by calling get_r00_target(custom_value)
R00_REGULARIZATION_TARGET: float = 1.0

# H2 Fix: P3 Hypervolume reference point thresholds (natural units)
# Points with gradient < HV_REFERENCE_GRADIENT contribute zero hypervolume
# Override via HV_REF_GRADIENT / HV_REF_ASPECT env vars
HV_REFERENCE_GRADIENT: float = 1.0  # Natural units (positive)
HV_REFERENCE_ASPECT_RATIO: float = 20.0

# Early stopping parameters for gradient descent
EARLY_STOPPING_PATIENCE: int = 10
EARLY_STOPPING_MIN_IMPROVEMENT: float = 1e-4


# =============================================================================
# API FUNCTIONS
# =============================================================================


def get_constraint_names(problem: str, *, for_alm: bool = False) -> List[str]:
    """Get canonical constraint names for a problem.

    Args:
        problem: Problem identifier ("p1", "p2", "p3").
        for_alm: If True and problem is P3, include aspect_ratio constraint
                 (used by constellaration ALM runner).

    Returns:
        List of canonical constraint names.
    """
    key = (problem or "p3").lower()

    if key.startswith("p1"):
        return list(P1_CONSTRAINT_NAMES)
    elif key.startswith("p2"):
        return list(P2_CONSTRAINT_NAMES)
    else:  # p3
        if for_alm:
            return list(P3_ALM_CONSTRAINT_NAMES)
        return list(P3_CONSTRAINT_NAMES)


def get_metrics_key(constraint_name: str) -> str:
    """Get the full metrics field name for a constraint.

    Args:
        constraint_name: Canonical constraint name (e.g., "edge_rotational_transform").

    Returns:
        Full metrics key (e.g., "edge_rotational_transform_over_n_field_periods").

    Raises:
        KeyError: If constraint_name is not in the registry.
    """
    if constraint_name not in METRICS_KEY_MAP:
        raise KeyError(
            f"Unknown constraint: {constraint_name!r}. "
            f"Valid names: {list(METRICS_KEY_MAP.keys())}"
        )
    return METRICS_KEY_MAP[constraint_name]


def get_constraint_bounds(problem: str) -> Dict[str, float]:
    """Get constraint bounds for a problem.

    Args:
        problem: Problem identifier ("p1", "p2", "p3").

    Returns:
        Dict mapping bound names to values.
    """
    key = (problem or "p3").lower()
    if key.startswith("p1"):
        return dict(CONSTRAINT_BOUNDS["p1"])
    elif key.startswith("p2"):
        return dict(CONSTRAINT_BOUNDS["p2"])
    else:
        return dict(CONSTRAINT_BOUNDS["p3"])


def is_physics_constraint(constraint_name: str) -> bool:
    """Check if a constraint requires physics evaluation (not just geometry).

    Physics constraints include QI, vacuum_well, flux_compression.
    Geometry constraints include aspect_ratio, elongation, triangularity, iota, mirror.
    """
    return constraint_name in {"qi", "vacuum_well", "flux_compression"}


def is_p3_only_constraint(constraint_name: str) -> bool:
    """Check if a constraint is specific to P3 (MHD stability).

    Returns True for vacuum_well and flux_compression, which are NOT
    part of P2 constraints but ARE part of P3.
    """
    return constraint_name in {"vacuum_well", "flux_compression"}


def get_log10_qi_threshold(problem: str) -> float:
    """Get the log10(qi) threshold for the given problem.

    Args:
        problem: Problem identifier ("p1", "p2", "p3").

    Returns:
        The log10(qi) upper bound threshold.
    """
    key = (problem or "p3").lower()
    if key.startswith("p2"):
        return LOG10_QI_THRESHOLDS["p2"]
    return LOG10_QI_THRESHOLDS["p3"]


def get_r00_target() -> float:
    """Get R₀₀ regularization target (H1 Fix).

    The target can be overridden via the R00_TARGET environment variable
    to match dataset statistics when the mean R₀₀ differs from 1.0.

    Returns:
        R₀₀ target value for regularization (default 1.0).
    """
    env_val = os.environ.get("R00_TARGET")
    if env_val is not None:
        return float(env_val)
    return R00_REGULARIZATION_TARGET


def get_hv_reference_point() -> Tuple[float, float]:
    """Get P3 hypervolume reference point in MINIMIZATION form (H1/H2 Fix).

    CONVENTION EQUIVALENCE (H1 Fix):
    The benchmark (constellaration/problems.py) uses:
        X = [(-1.0 * gradient), (+1.0 * aspect_ratio)]
        reference_point = [1.0, 20.0]

    This function returns:
        reference_point = (-1.0, 20.0) in minimization form

    These are mathematically equivalent because:
        -gradient > -1.0  iff  gradient < 1.0

    Both filter out points where gradient < 1.0 (poor performing designs).
    The sign difference is just a convention for minimization-form storage.

    The reference point can be overridden via environment variables:
    - HV_REF_GRADIENT: gradient threshold in natural units (default 1.0)
    - HV_REF_ASPECT: aspect ratio threshold (default 20.0)

    Returns:
        Tuple of (-gradient, aspect_ratio) in minimization form.
        Points with gradient < threshold contribute zero hypervolume.
    """
    grad = float(os.environ.get("HV_REF_GRADIENT", HV_REFERENCE_GRADIENT))
    aspect = float(os.environ.get("HV_REF_ASPECT", HV_REFERENCE_ASPECT_RATIO))
    return (-grad, aspect)  # Minimization form: negate gradient
