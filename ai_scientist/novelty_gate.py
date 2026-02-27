"""Shared two-stage novelty gate primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable


@dataclass(frozen=True)
class NoveltyCandidate:
    label: str
    embedding_distance: float | None
    feasibility: float | None = None
    novelty_score: float | None = None


NoveltyJudge = Callable[[NoveltyCandidate], bool]


def _safe_optional_float(value: float | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _passes_embedding_prefilter(*, distance: float | None, min_distance: float) -> bool:
    if min_distance <= 0.0:
        return True
    return distance is not None and distance >= min_distance


def _passes_feasibility_cap(
    *, feasibility: float | None, feasibility_max: float
) -> bool:
    if not math.isfinite(feasibility_max):
        return True
    return feasibility is not None and feasibility <= feasibility_max


def apply_two_stage_novelty_gate(
    candidates: Iterable[NoveltyCandidate],
    *,
    embedding_prefilter_min_distance: float,
    near_duplicate_distance: float,
    feasibility_max: float = float("inf"),
    judge: NoveltyJudge | None = None,
    judge_label: str = "disabled",
    fallback_to_ungated: bool,
) -> tuple[list[NoveltyCandidate], dict]:
    """Apply embedding prefilter then near-duplicate adjudication.

    Stage 1:
    - Reject candidates under `embedding_prefilter_min_distance`.
    - Reject candidates above optional feasibility cap.

    Stage 2:
    - For Stage-1 accepted candidates with distance <= `near_duplicate_distance`,
      invoke `judge` when provided.
    - If no judge is provided, near-duplicates are rejected in strict mode.
    """
    min_distance = float(embedding_prefilter_min_distance)
    near_distance = float(near_duplicate_distance)
    cap = float(feasibility_max)
    if min_distance < 0.0:
        raise ValueError("embedding_prefilter_min_distance must be >= 0.")
    if near_distance < min_distance:
        raise ValueError(
            "near_duplicate_distance must be >= embedding_prefilter_min_distance."
        )

    source = list(candidates)
    kept: list[NoveltyCandidate] = []
    rows: list[dict] = []
    judge_calls = 0
    judge_accepts = 0
    near_duplicate_count = 0

    for candidate in source:
        distance = _safe_optional_float(candidate.embedding_distance)
        feasibility = _safe_optional_float(candidate.feasibility)
        pass_distance = _passes_embedding_prefilter(
            distance=distance,
            min_distance=min_distance,
        )
        pass_feasibility = _passes_feasibility_cap(
            feasibility=feasibility,
            feasibility_max=cap,
        )
        stage1_accept = pass_distance and pass_feasibility
        near_duplicate = (
            stage1_accept and distance is not None and distance <= near_distance
        )
        if near_duplicate:
            near_duplicate_count += 1

        judge_called = False
        judge_accept: bool | None = None
        if not stage1_accept:
            accepted = False
        elif not near_duplicate:
            accepted = True
        elif judge is None:
            judge_accept = False
            accepted = False
        else:
            judge_called = True
            judge_calls += 1
            judge_accept = bool(judge(candidate))
            accepted = bool(judge_accept)
            if accepted:
                judge_accepts += 1

        if accepted:
            kept.append(candidate)

        rows.append(
            {
                "label": candidate.label,
                "embedding_distance": distance,
                "distance_l2": distance,
                "feasibility": feasibility,
                "pass_embedding_prefilter": pass_distance,
                "pass_distance": pass_distance,
                "pass_feasibility": pass_feasibility,
                "stage1_accept": stage1_accept,
                "near_duplicate": near_duplicate,
                "judge_called": judge_called,
                "judge_label": str(judge_label),
                "judge_accept": judge_accept,
                "accepted": accepted,
            }
        )

    fallback = False
    selected = kept
    if not selected and fallback_to_ungated:
        selected = source
        fallback = True

    diagnostics = {
        "version": "m3.4_two_stage_novelty_v1",
        "enabled": bool(min_distance > 0.0 or math.isfinite(cap) or judge is not None),
        "embedding_prefilter_min_distance": min_distance,
        "near_duplicate_distance": near_distance,
        "feasibility_max": cap if math.isfinite(cap) else None,
        "evaluated_count": len(source),
        "kept_count": len(kept),
        "rejected_count": len(source) - len(kept),
        "near_duplicate_count": near_duplicate_count,
        "judge_label": str(judge_label),
        "judge_call_count": judge_calls,
        "judge_accept_count": judge_accepts,
        "fallback_to_ungated": fallback,
        "rows": rows,
    }
    return selected, diagnostics
