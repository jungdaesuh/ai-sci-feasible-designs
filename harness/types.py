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


def _deep_freeze(obj: object) -> object:
    """Recursively freeze a nested structure of dicts/lists/scalars.

    - dict  → MappingProxyType (keys/values recursively frozen)
    - list  → tuple            (elements recursively frozen)
    - other → returned as-is   (ints, floats, strings are already immutable)
    """
    if isinstance(obj, dict):
        return MappingProxyType({k: _deep_freeze(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return tuple(_deep_freeze(item) for item in obj)
    return obj


@dataclass(frozen=True)
class CandidateBundle:
    """A validated candidate boundary with metadata.

    Both fields are deeply frozen (dicts → MappingProxyType, lists → tuples).
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
            boundary=_deep_freeze(boundary),  # type: ignore[arg-type]
            metadata=_deep_freeze(metadata),  # type: ignore[arg-type]
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
