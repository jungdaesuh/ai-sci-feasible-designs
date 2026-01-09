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

        Implements split VMEC/QI evaluation:
        - Phase A: VMEC-only (always returns metrics)
        - Phase B: QI (optional, stage-gated, caught if fails)

        QI failure becomes qi=None (constraint violation) instead of full
        evaluation failure, preserving VMEC results.

        Args:
            boundary: Dictionary with r_cos, z_sin, n_field_periods, etc.
            settings: ForwardModelSettings with problem, stage, fidelity config.

        Returns:
            EvaluationResult with real physics metrics.

        Raises:
            ImportError: If constellaration is not installed.
            RuntimeError: If VMEC evaluation fails (QI failures are caught).
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

        # Determine if QI should be computed based on stage
        stage_key = (settings.stage or "").lower()
        should_compute_qi = stage_key in ("high", "p2", "p3", "high_fidelity")

        # Get constellaration settings
        constellaration_settings = settings.constellaration_settings
        if constellaration_settings is None:
            constellaration_settings = constellaration_forward.ConstellarationSettings()

        # ═══════════════════════════════════════════════════════════════
        # PHASE A: VMEC + geometry (no QI) - always succeeds or throws
        # ═══════════════════════════════════════════════════════════════
        # Create settings without QI for Phase A
        settings_no_qi = constellaration_settings.model_copy(
            update={
                "boozer_preset_settings": None,
                "qi_settings": None,
            }
        )

        try:
            metrics_base, equilibrium = constellaration_forward.forward_model(
                boundary=surf,
                settings=settings_no_qi,
            )
        except Exception as e:
            logger.error(f"VMEC evaluation failed: {e}")
            raise RuntimeError(f"Physics evaluation failed: {e}") from e

        # ═══════════════════════════════════════════════════════════════
        # PHASE B: Boozer + QI (optional, stage-gated, caught if fails)
        # ═══════════════════════════════════════════════════════════════
        qi_value = None
        qi_error = None

        if should_compute_qi and constellaration_settings.qi_settings is not None:
            try:
                # Import QI modules
                from constellaration.boozer import boozer as boozer_module
                from constellaration.omnigeneity import qi

                # Build boozer settings from equilibrium resolution
                boozer_settings = (
                    boozer_module.create_boozer_settings_from_equilibrium_resolution(
                        mhd_equilibrium=equilibrium,
                        settings=constellaration_settings.boozer_preset_settings,
                    )
                )

                # Run Boozer transform
                boozer_result = boozer_module.run_boozer(
                    equilibrium=equilibrium,
                    settings=boozer_settings,
                )

                # Run QI computation
                import numpy as np

                qi_metrics = qi.quasi_isodynamicity_residual(
                    boozer=boozer_result,
                    settings=constellaration_settings.qi_settings,
                )
                qi_value = float(np.sum(qi_metrics.residuals**2))

            except ValueError as e:
                # Expected failure: "Not enough crossings found" from _find_bounce_points
                logger.warning(f"QI computation failed (expected edge case): {e}")
                qi_error = str(e)
            except Exception as e:
                # Unexpected failure - log but don't crash
                logger.warning(f"QI computation failed (unexpected): {e}")
                qi_error = str(e)

        # Update metrics with QI result (if computed)
        if qi_value is not None:
            # Create updated metrics with QI value
            metrics = metrics_base.model_copy(update={"qi": qi_value})
        else:
            # Keep original metrics with qi=None (will trigger constraint violation)
            metrics = metrics_base

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
            error_message=qi_error,
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
