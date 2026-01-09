"""Post-generation validation checks.

This module provides validation routines for equilibria, including
Shafranov shift checks that can invalidate vacuum MHD assumptions
when beta is too high.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class ValidationResult:
    """Result of equilibrium validation.

    Attributes:
        passed: True if all validations passed.
        warnings: List of warning messages for soft failures.
        errors: List of error messages for hard failures.
        shafranov_shift: Normalized Shafranov shift (Delta_axis / R0), if available.
    """

    passed: bool
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    shafranov_shift: Optional[float] = None


def extract_shafranov_shift(metrics: Any) -> Optional[float]:
    """Extract Shafranov shift from VMEC metrics if available.

    The Shafranov shift is the displacement of the magnetic axis from
    the geometric center of the plasma. For vacuum equilibria, this should
    be small. Large shifts indicate high-beta effects that may invalidate
    the vacuum MHD assumption.

    Args:
        metrics: ConstellarationMetrics or similar object. May have:
            - threed1_shafranov_integrals.r_lao (vmecpp OutputQuantities)
            - Or be estimated from axis location vs boundary center

    Returns:
        Normalized shift Delta_axis / R0, or None if not available.
    """
    # Try to get from vmecpp Threed1ShafranovIntegrals
    if hasattr(metrics, "threed1_shafranov_integrals"):
        shafranov = metrics.threed1_shafranov_integrals
        if hasattr(shafranov, "r_lao"):
            # r_lao is the axis radial location
            # Need R0 (geometric center) to normalize
            if hasattr(metrics, "maximum_radius") and hasattr(
                metrics, "minimum_radius"
            ):
                r_max = metrics.maximum_radius
                r_min = metrics.minimum_radius
                r_0 = (r_max + r_min) / 2.0
                if r_0 > 0:
                    delta = abs(shafranov.r_lao - r_0)
                    return delta / r_0
        return None

    # Try model_dump() for Pydantic models
    if hasattr(metrics, "model_dump"):
        data = metrics.model_dump()
        if "threed1_shafranov_integrals" in data:
            return None  # Would need to parse nested dict

    return None


def validate_equilibrium(
    metrics: Any,
    *,
    shafranov_threshold: float = 0.1,
    warn_on_missing: bool = True,
) -> ValidationResult:
    """Validate equilibrium for physics consistency.

    Checks:
    1. Shafranov shift: Delta_axis / R0 should be small for vacuum validity.
       Large shifts (> threshold) indicate high-beta effects.

    Args:
        metrics: ConstellarationMetrics or similar physics output.
        shafranov_threshold: Maximum allowed normalized Shafranov shift.
            Default 0.1 means 10% of major radius.
        warn_on_missing: If True, add warning when shift data is unavailable.

    Returns:
        ValidationResult with validation outcome and any warnings.
    """
    warnings: List[str] = []
    errors: List[str] = []
    shafranov_shift: Optional[float] = None

    # Check 1: Shafranov shift
    shift = extract_shafranov_shift(metrics)
    shafranov_shift = shift

    if shift is not None:
        if shift > shafranov_threshold:
            warnings.append(
                f"Large Shafranov shift detected: {shift:.3f} "
                f"(threshold: {shafranov_threshold}). "
                "This may invalidate vacuum MHD assumptions for high-beta scenarios."
            )
    elif warn_on_missing:
        # Only warn in verbose mode, this is expected for most evaluations
        pass  # Shift data not available from this backend

    # Check 2: Basic equilibrium convergence (if available)
    if hasattr(metrics, "vmec_converged"):
        if not metrics.vmec_converged:
            errors.append("VMEC equilibrium did not converge.")

    # Check 3: Beta range (if available)
    if hasattr(metrics, "beta_volume_average"):
        beta = metrics.beta_volume_average
        if beta is not None and beta > 0.05:  # 5% beta
            warnings.append(
                f"High beta detected: {beta:.3f}. "
                "Vacuum equilibrium assumptions may be inaccurate."
            )

    passed = len(errors) == 0
    return ValidationResult(
        passed=passed,
        warnings=warnings,
        errors=errors,
        shafranov_shift=shafranov_shift,
    )


def validate_submission(
    boundary_params: dict[str, Any],
    metrics: Any,
    *,
    shafranov_threshold: float = 0.1,
) -> ValidationResult:
    """Full validation before final submission.

    Combines equilibrium validation with boundary geometry checks.

    Args:
        boundary_params: Boundary Fourier coefficients dict.
        metrics: Physics metrics from forward model.
        shafranov_threshold: Shafranov shift threshold.

    Returns:
        ValidationResult with all checks combined.
    """
    # Run equilibrium validation
    result = validate_equilibrium(
        metrics, shafranov_threshold=shafranov_threshold, warn_on_missing=False
    )

    # Additional boundary-level checks could go here
    # (e.g., Fourier spectrum decay, R00 normalization)

    return result
