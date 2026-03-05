"""Shared frozen dataclasses for the harness package.

Leaf module with no internal dependencies. Prevents circular imports
between state_reader, diagnosis, and observation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal

ExplorationMode = Literal["exploit", "explore"]


@dataclass(frozen=True)
class CycleSnapshot:
    """DB-derived state at the start of a cycle."""

    frontier_value: float | None
    pending_count: int
    running_count: int
    done_count: int
    near_feasible_count: int
    parent_paths: tuple[Path, ...]


@dataclass(frozen=True)
class CycleDiagnosis:
    """Analysis of the previous cycle's outcomes."""

    exploration_mode: ExplorationMode
    binding_constraints: tuple[str, ...]
    feasible_yield: float
    objective_delta: float


@dataclass(frozen=True)
class ProposalScript:
    """LLM-generated Python source for candidate generation."""

    source: str
    model: str
    latency_ms: int


@dataclass(frozen=True)
class CandidateBundle:
    """A validated candidate boundary with metadata.

    Both fields are wrapped in MappingProxyType for deep immutability.
    Construct via CandidateBundle.of(boundary_dict, metadata_dict).
    """

    boundary: MappingProxyType
    metadata: MappingProxyType

    @staticmethod
    def of(
        boundary: dict,
        metadata: dict,
    ) -> CandidateBundle:
        return CandidateBundle(
            boundary=MappingProxyType(boundary),
            metadata=MappingProxyType(metadata),
        )


@dataclass(frozen=True)
class EnqueueResult:
    """Summary of enqueue outcomes for a cycle."""

    inserted: int
    skipped: int


@dataclass(frozen=True)
class StopDecision:
    """Governor stop controller output."""

    should_stop: bool
    reason: str | None
