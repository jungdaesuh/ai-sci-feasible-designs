"""Curriculum learning scheduler for progressive constraint satisfaction (Issue #12).

Curriculum learning starts with easier problems and progressively adds constraints:
- P1 (Warm-up): Geometry optimization only (minimize elongation)
- P2 (Intermediate): Add QI + vacuum well constraints
- P3 (Full): Add flux compression + aspect ratio multi-objective

The scheduler tracks progress and advances to the next stage when feasibility
thresholds are met.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass  # For future type hints


class CurriculumStage(str, Enum):
    """Problem difficulty stages in curriculum."""

    P1 = "p1"  # Geometry only
    P2 = "p2"  # + QI + vacuum well
    P3 = "p3"  # Full multi-objective


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning.

    Attributes:
        enabled: Whether curriculum learning is enabled. Default False.
        advancement_threshold: Feasibility rate threshold to advance to next stage.
        min_cycles_per_stage: Minimum cycles at each stage before advancing.
        initial_stage: Starting stage (can skip P1 if already good at geometry).
    """

    enabled: bool = False
    advancement_threshold: float = 0.3
    min_cycles_per_stage: int = 5
    initial_stage: str = "p1"


@dataclass
class CurriculumState:
    """Mutable state tracking curriculum progress.

    Attributes:
        current_stage: Current problem difficulty stage.
        cycles_in_stage: Number of cycles spent in current stage.
        best_seeds_from_stage: Best candidate seeds to warm-start next stage.
    """

    current_stage: CurriculumStage = CurriculumStage.P1
    cycles_in_stage: int = 0
    best_seeds_from_stage: list = field(default_factory=list)


class CurriculumScheduler:
    """Manages curriculum stage transitions.

    Tracks progress through difficulty stages (P1 → P2 → P3) and
    determines when to advance based on feasibility rates.

    Example:
        >>> config = CurriculumConfig(enabled=True, advancement_threshold=0.3)
        >>> scheduler = CurriculumScheduler(config)
        >>> scheduler.current_stage
        <CurriculumStage.P1: 'p1'>
        >>> scheduler.tick()  # Call each cycle
        >>> scheduler.should_advance(feasibility_rate=0.5)
        False  # Not enough cycles yet
    """

    def __init__(self, config: CurriculumConfig) -> None:
        """Initialize curriculum scheduler.

        Args:
            config: Curriculum learning configuration.
        """
        self.config = config
        self._state = CurriculumState(
            current_stage=CurriculumStage(config.initial_stage)
        )

    @property
    def current_stage(self) -> CurriculumStage:
        """Get current problem difficulty stage."""
        return self._state.current_stage

    @property
    def cycles_in_stage(self) -> int:
        """Get number of cycles in current stage."""
        return self._state.cycles_in_stage

    def should_advance(self, feasibility_rate: float) -> bool:
        """Check if should advance to next stage.

        Args:
            feasibility_rate: Current feasibility rate (0.0 to 1.0).

        Returns:
            True if conditions are met to advance to next stage.
        """
        if not self.config.enabled:
            return False
        if self._state.current_stage == CurriculumStage.P3:
            return False  # Already at final stage
        if self._state.cycles_in_stage < self.config.min_cycles_per_stage:
            return False  # Not enough cycles in current stage
        return feasibility_rate >= self.config.advancement_threshold

    def advance(
        self, best_seeds: list[dict[str, Any]] | None = None
    ) -> CurriculumStage:
        """Advance to next stage and cache best seeds for warm-start.

        Args:
            best_seeds: Optional list of best candidate designs to warm-start next stage.

        Returns:
            The new current stage after advancement.
        """
        if best_seeds is not None:
            self._state.best_seeds_from_stage = best_seeds
        self._state.cycles_in_stage = 0

        if self._state.current_stage == CurriculumStage.P1:
            self._state.current_stage = CurriculumStage.P2
        elif self._state.current_stage == CurriculumStage.P2:
            self._state.current_stage = CurriculumStage.P3
        # P3 stays at P3

        return self._state.current_stage

    def tick(self) -> None:
        """Increment cycle counter. Call once per cycle."""
        self._state.cycles_in_stage += 1

    def get_effective_problem(self) -> str:
        """Get the effective problem string for current stage.

        Returns:
            Problem string (e.g., "p1", "p2", "p3").
        """
        return self._state.current_stage.value

    def get_warm_start_seeds(self) -> list[dict[str, Any]]:
        """Get cached seeds from previous stage for warm-starting.

        Returns:
            List of candidate designs from previous stage.
        """
        return self._state.best_seeds_from_stage
