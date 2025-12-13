"""Real physics backend using constellaration/vmecpp.

This backend wraps the actual constellaration forward_model for production
physics evaluations. It should only be used when the full physics stack
is available and needed.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Mapping

from ai_scientist.backends.base import PhysicsBackend

logger = logging.getLogger(__name__)

# Flag to track if we've already warned about unavailability
_WARNED_UNAVAILABLE = False


class RealPhysicsBackend(PhysicsBackend):
    """Production backend using constellaration/vmecpp physics.

    This backend provides full-fidelity physics evaluations using the
    constellaration library and VMEC equilibrium solver.

    Note:
        This backend requires constellaration and vmecpp to be installed.
        Use `is_available()` to check before calling `evaluate()`.

    Example:
        >>> backend = RealPhysicsBackend()
        >>> if backend.is_available():
        ...     result = backend.evaluate(boundary, settings)
    """

    def __init__(self):
        """Initialize the real physics backend."""
        self._available: bool | None = None  # Lazy check

    def evaluate(
        self,
        boundary: Mapping[str, Any],
        settings: Any,  # ForwardModelSettings
    ) -> Any:  # EvaluationResult
        """Evaluate using real constellaration physics.

        This calls the actual constellaration.forward_model.forward_model()
        with full VMEC equilibrium solving.

        Args:
            boundary: Dictionary with r_cos, z_sin, n_field_periods, etc.
            settings: ForwardModelSettings with problem, stage, fidelity config.

        Returns:
            EvaluationResult with real physics metrics.

        Raises:
            ImportError: If constellaration is not installed.
            RuntimeError: If physics evaluation fails.
        """
        # Import here to isolate the dependency
        from constellaration import forward_model as constellaration_forward

        from ai_scientist.forward_model import (
            EvaluationResult,
            compute_constraint_margins,
            compute_design_hash,
            compute_objective,
            make_boundary_from_params,
            max_violation,
        )
        from ai_scientist.optim.prerelax import prerelax_boundary

        start_time = time.time()

        # Compute design hash
        d_hash = compute_design_hash(boundary)

        # Optional pre-relaxation
        if settings.prerelax:
            try:
                nfp_val = boundary.get("n_field_periods") or boundary.get("nfp") or 1
                boundary, _ = prerelax_boundary(
                    dict(boundary),
                    steps=settings.prerelax_steps,
                    lr=settings.prerelax_lr,
                    nfp=int(nfp_val),
                )
            except Exception as e:
                logger.warning(f"Pre-relaxation failed: {e}")

        # Create boundary surface
        try:
            surf = make_boundary_from_params(boundary)
        except Exception as e:
            logger.error(f"Failed to create boundary: {e}")
            raise ValueError(f"Invalid boundary parameters: {e}") from e

        # Run real physics
        try:
            constellaration_settings = settings.constellaration_settings
            if constellaration_settings is None:
                constellaration_settings = (
                    constellaration_forward.ConstellarationSettings()
                )

            metrics, _ = constellaration_forward.forward_model(
                boundary=surf,
                settings=constellaration_settings,
            )
        except Exception as e:
            logger.error(f"Physics evaluation failed: {e}")
            raise RuntimeError(f"Physics evaluation failed: {e}") from e

        # Compute derived values
        objective = compute_objective(metrics, settings.problem)
        constraints_map = compute_constraint_margins(
            metrics, settings.problem, stage=settings.stage
        )
        feasibility = max_violation(constraints_map)
        is_feasible = feasibility <= 1e-2

        evaluation_time = time.time() - start_time

        return EvaluationResult(
            metrics=metrics,
            objective=objective,
            constraints=list(constraints_map.values()),
            constraint_names=list(constraints_map.keys()),
            feasibility=feasibility,
            is_feasible=is_feasible,
            cache_hit=False,
            design_hash=d_hash,
            evaluation_time_sec=evaluation_time,
            settings=settings,
            fidelity=settings.fidelity,
            equilibrium_converged=True,
            error_message=None,
        )

    def is_available(self) -> bool:
        """Check if constellaration is installed and working."""
        global _WARNED_UNAVAILABLE

        if self._available is not None:
            return self._available

        try:
            import importlib.util

            spec = importlib.util.find_spec("constellaration.forward_model")
            self._available = spec is not None
        except (ImportError, ModuleNotFoundError):
            self._available = False

        if not self._available and not _WARNED_UNAVAILABLE:
            logger.info(
                "RealPhysicsBackend: constellaration not available, "
                "falling back to mock backend for tests"
            )
            _WARNED_UNAVAILABLE = True

        return self._available

    @property
    def name(self) -> str:
        return "constellaration"
