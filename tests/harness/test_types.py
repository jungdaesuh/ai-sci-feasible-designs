"""Tests for harness.types — M0 acceptance."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import pytest

from harness.types import (
    CandidateBundle,
    CycleDiagnosis,
    CycleSnapshot,
    EnqueueResult,
    ProposalScript,
    StopDecision,
)


def test_cycle_snapshot_frozen():
    snap = CycleSnapshot(
        frontier_value=8.5,
        pending_count=10,
        running_count=3,
        done_count=50,
        near_feasible_count=5,
        parent_paths=(Path("/a"), Path("/b")),
    )
    with pytest.raises(AttributeError):
        snap.frontier_value = 9.0  # type: ignore[misc]


def test_all_types_instantiate():
    snap = CycleSnapshot(
        frontier_value=None,
        pending_count=0,
        running_count=0,
        done_count=0,
        near_feasible_count=0,
        parent_paths=(),
    )
    assert repr(snap).startswith("CycleSnapshot(")

    diag = CycleDiagnosis(
        exploration_mode="exploit",
        binding_constraints=("qi",),
        feasible_yield=0.5,
        objective_delta=0.1,
    )
    assert diag.exploration_mode == "exploit"

    prop = ProposalScript(source="print(1)", model="test", latency_ms=100)
    assert prop.source == "print(1)"

    bundle = CandidateBundle.of(boundary={"r_cos": [[1]]}, metadata={"k": "v"})
    assert bundle.boundary["r_cos"] == [[1]]
    assert isinstance(bundle.boundary, MappingProxyType)
    assert isinstance(bundle.metadata, MappingProxyType)

    enq = EnqueueResult(inserted=3, skipped=1)
    assert enq.inserted + enq.skipped == 4

    stop = StopDecision(should_stop=False, reason=None)
    assert not stop.should_stop

    stop2 = StopDecision(should_stop=True, reason="target_reached")
    assert stop2.reason == "target_reached"


def test_candidate_bundle_deeply_immutable():
    bundle = CandidateBundle.of(boundary={"r_cos": [[1, 2]]}, metadata={"key": "val"})
    with pytest.raises(TypeError):
        bundle.boundary["r_cos"] = [[9]]  # type: ignore[index]
