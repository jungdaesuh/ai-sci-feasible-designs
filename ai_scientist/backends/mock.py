"""Mock physics backend for fast unit testing."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, List, Mapping

from ai_scientist.backends.base import PhysicsBackend


@dataclass
class MockMetrics:
    """Mock ConstellarationMetrics for testing.

    Matches the interface expected by forward_model.py without
    depending on the real constellaration package.
    """

    minimum_normalized_magnetic_gradient_scale_length: float = 1.5
    aspect_ratio: float = 8.0
    max_elongation: float = 1.2
    mean_iota: float = 0.42
    edge_rotational_transform_over_n_field_periods: float = 0.14
    mirror_ratio: float = 0.15
    edge_magnetic_mirror_ratio: float = 0.15
    average_triangularity: float = -0.6
    vacuum_well: float = 0.02
    relative_well_depth: float = 0.02
    flux_compression_in_regions_of_bad_curvature: float | None = 0.1
    qi: float | None = 1e-5
    beta_volume_average: float = 0.025
    vmec_converged: bool = True
    maximum_radius: float = 1.0
    minimum_radius: float = 0.1
    plasma_volume: float = 5.0
    surface_area: float = 25.0

    # Additional fields that may be accessed
    number_of_field_periods: int = 3
    is_quasihelical: bool = False
    max_dof_delta: float = 0.0
    vmec_exit_code: int = 0

    def model_dump(self) -> dict:
        """Pydantic-compatible serialization."""
        return {
            "minimum_normalized_magnetic_gradient_scale_length": self.minimum_normalized_magnetic_gradient_scale_length,
            "aspect_ratio": self.aspect_ratio,
            "max_elongation": self.max_elongation,
            "mean_iota": self.mean_iota,
            "edge_rotational_transform_over_n_field_periods": self.edge_rotational_transform_over_n_field_periods,
            "mirror_ratio": self.mirror_ratio,
            "edge_magnetic_mirror_ratio": self.edge_magnetic_mirror_ratio,
            "average_triangularity": self.average_triangularity,
            "vacuum_well": self.vacuum_well,
            "relative_well_depth": self.relative_well_depth,
            "beta_volume_average": self.beta_volume_average,
            "vmec_converged": self.vmec_converged,
            "maximum_radius": self.maximum_radius,
            "minimum_radius": self.minimum_radius,
            "plasma_volume": self.plasma_volume,
            "surface_area": self.surface_area,
            "number_of_field_periods": self.number_of_field_periods,
            "is_quasihelical": self.is_quasihelical,
            "max_dof_delta": self.max_dof_delta,
            "vmec_exit_code": self.vmec_exit_code,
            "flux_compression_in_regions_of_bad_curvature": self.flux_compression_in_regions_of_bad_curvature,
            "qi": self.qi,
        }


class MockPhysicsBackend(PhysicsBackend):
    """Fast mock backend for unit tests.

    Generates deterministic synthetic results based on the boundary hash,
    allowing reproducible tests without real physics calculations.

    Features:
    - Configurable default objective and feasibility
    - Optional delay to simulate slow physics
    - Call logging for test assertions
    - Deterministic results based on input hash

    Example:
        >>> backend = MockPhysicsBackend(default_objective=1.5)
        >>> result = backend.evaluate({"r_cos": [[1.0]], "z_sin": [[0.1]]}, settings)
        >>> assert backend.call_count == 1
        >>> assert result.objective >= 1.5
    """

    def __init__(
        self,
        *,
        default_objective: float = 1.0,
        default_feasibility: float = 0.0,
        delay_seconds: float = 0.0,
        metrics_override: MockMetrics | None = None,
    ):
        """Initialize the mock backend.

        Args:
            default_objective: Base objective value (varied by input hash).
            default_feasibility: Default feasibility (0.0 = feasible).
            delay_seconds: Artificial delay to simulate slow physics.
            metrics_override: Optional fixed metrics to return.
        """
        self.default_objective = default_objective
        self.default_feasibility = default_feasibility
        self.delay_seconds = delay_seconds
        self.metrics_override = metrics_override

        # Tracking for test assertions
        self.call_count: int = 0
        self.call_log: List[Mapping[str, Any]] = []
        self.last_boundary: Mapping[str, Any] | None = None
        self.last_settings: Any = None

    def evaluate(
        self,
        boundary: Mapping[str, Any],
        settings: Any,  # ForwardModelSettings
    ) -> Any:  # EvaluationResult
        """Generate synthetic evaluation result.

        The result is deterministic based on the boundary hash, allowing
        reproducible tests. Small variations are added based on the hash
        to simulate different physics outcomes.
        """
        # Import here to avoid circular imports
        from ai_scientist.forward_model import (
            EvaluationResult,
            compute_design_hash,
            compute_constraint_margins,
            compute_objective,
            max_violation,
        )

        # Track the call
        self.call_count += 1
        self.call_log.append(dict(boundary))
        self.last_boundary = boundary
        self.last_settings = settings

        # Optional delay
        if self.delay_seconds > 0:
            time.sleep(self.delay_seconds)

        # Generate deterministic seed from boundary
        try:
            boundary_str = json.dumps(boundary, sort_keys=True, default=str)
        except (TypeError, ValueError):
            boundary_str = str(boundary)
        seed = int(hashlib.md5(boundary_str.encode()).hexdigest()[:8], 16) % 10000

        # Compute design hash
        try:
            design_hash = compute_design_hash(boundary)
        except Exception:
            design_hash = hashlib.md5(boundary_str.encode()).hexdigest()[:16]

        # Generate varied metrics based on seed
        if self.metrics_override:
            metrics = self.metrics_override
        else:
            # Create metrics with small variations based on seed
            variation = (seed % 100) / 1000.0  # 0.0 to 0.099
            metrics = MockMetrics(
                minimum_normalized_magnetic_gradient_scale_length=1.5 + variation,
                aspect_ratio=8.0 + variation * 10,
                max_elongation=1.2 + variation * 0.5,
                mean_iota=0.42 + variation * 0.1,
                mirror_ratio=0.15 + variation * 0.05,
            )

        problem = getattr(settings, "problem", "p3")
        stage = getattr(settings, "stage", "high")
        margins = compute_constraint_margins(metrics, problem, stage=stage)
        feasibility = max_violation(margins)
        objective = compute_objective(metrics, problem)
        constraint_names = list(margins.keys())
        constraints = list(margins.values())
        is_feasible = feasibility <= 1e-2

        # Compute evaluation time (mock - instant)
        evaluation_time_sec = self.delay_seconds if self.delay_seconds > 0 else 0.001

        # Get fidelity from settings if available
        fidelity = getattr(settings, "fidelity", "low") if settings else "low"

        return EvaluationResult(
            metrics=metrics,
            objective=objective,
            constraints=constraints,
            constraint_names=constraint_names,
            feasibility=feasibility,
            is_feasible=is_feasible,
            cache_hit=False,
            design_hash=design_hash,
            evaluation_time_sec=evaluation_time_sec,
            fidelity=fidelity,
            settings=settings,
            equilibrium_converged=True,
            error_message=None,
        )

    def is_available(self) -> bool:
        """Mock backend is always available."""
        return True

    @property
    def name(self) -> str:
        return "mock"

    def reset(self) -> None:
        """Reset call tracking (useful between tests)."""
        self.call_count = 0
        self.call_log.clear()
        self.last_boundary = None
        self.last_settings = None
