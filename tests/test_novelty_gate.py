from __future__ import annotations

import pytest

from ai_scientist.novelty_gate import NoveltyCandidate, apply_two_stage_novelty_gate


def test_two_stage_novelty_gate_prefilter_and_judge_paths() -> None:
    candidates = [
        NoveltyCandidate(label="far", embedding_distance=0.2, feasibility=0.0),
        NoveltyCandidate(label="near", embedding_distance=0.06, feasibility=0.0),
        NoveltyCandidate(label="reject", embedding_distance=0.01, feasibility=0.0),
    ]

    selected, diagnostics = apply_two_stage_novelty_gate(
        candidates,
        embedding_prefilter_min_distance=0.05,
        near_duplicate_distance=0.08,
        feasibility_max=float("inf"),
        judge=lambda candidate: candidate.label == "near",
        judge_label="test-judge",
        fallback_to_ungated=False,
    )

    assert [candidate.label for candidate in selected] == ["far", "near"]
    assert diagnostics["judge_call_count"] == 1
    assert diagnostics["judge_accept_count"] == 1
    assert diagnostics["rejected_count"] == 1
    rejected_row = next(row for row in diagnostics["rows"] if row["label"] == "reject")
    assert rejected_row["pass_distance"] is False


def test_two_stage_novelty_gate_falls_back_to_ungated_when_all_rejected() -> None:
    candidates = [
        NoveltyCandidate(label="a", embedding_distance=0.0, feasibility=0.0),
        NoveltyCandidate(label="b", embedding_distance=0.0, feasibility=0.0),
    ]
    selected, diagnostics = apply_two_stage_novelty_gate(
        candidates,
        embedding_prefilter_min_distance=0.1,
        near_duplicate_distance=0.1,
        fallback_to_ungated=True,
    )
    assert [candidate.label for candidate in selected] == ["a", "b"]
    assert diagnostics["fallback_to_ungated"] is True


def test_two_stage_novelty_gate_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValueError, match="near_duplicate_distance"):
        apply_two_stage_novelty_gate(
            [NoveltyCandidate(label="a", embedding_distance=0.1)],
            embedding_prefilter_min_distance=0.1,
            near_duplicate_distance=0.09,
            fallback_to_ungated=False,
        )


def test_two_stage_novelty_gate_rejects_near_duplicates_without_judge() -> None:
    candidates = [
        NoveltyCandidate(label="near", embedding_distance=0.06, feasibility=0.0),
        NoveltyCandidate(label="far", embedding_distance=0.2, feasibility=0.0),
    ]

    selected, diagnostics = apply_two_stage_novelty_gate(
        candidates,
        embedding_prefilter_min_distance=0.05,
        near_duplicate_distance=0.08,
        fallback_to_ungated=False,
    )

    assert [candidate.label for candidate in selected] == ["far"]
    near_row = next(row for row in diagnostics["rows"] if row["label"] == "near")
    assert near_row["near_duplicate"] is True
    assert near_row["judge_called"] is False
    assert near_row["accepted"] is False
