"""Base physics backend abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from ai_scientist.forward_model import EvaluationResult, ForwardModelSettings


class PhysicsBackend(ABC):
    """Abstract interface for physics evaluation backends.

    This abstraction allows the ai_scientist orchestration layer to work
    with different physics implementations:

    - MockPhysicsBackend: Fast synthetic results for unit tests
    - RealPhysicsBackend: Full constellaration/vmecpp physics
    - (Future) SurrogateBackend: Neural network approximation

    Example:
        >>> backend = MockPhysicsBackend()
        >>> result = backend.evaluate(boundary_params, settings)
        >>> print(result.objective, result.feasibility)
    """

    @abstractmethod
    def evaluate(
        self,
        boundary: Mapping[str, Any],
        settings: "ForwardModelSettings",
    ) -> "EvaluationResult":
        """Evaluate a stellarator boundary configuration.

        Args:
            boundary: Dictionary with r_cos, z_sin, n_field_periods, etc.
            settings: ForwardModelSettings with problem, stage, fidelity config.

        Returns:
            EvaluationResult with metrics, objective, constraints, feasibility.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available.

        For MockPhysicsBackend: Always True.
        For RealPhysicsBackend: True if constellaration is installed.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend name for logging."""
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r} available={self.is_available()}>"
