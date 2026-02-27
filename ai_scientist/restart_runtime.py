"""Runtime helpers for adaptive restart wiring in P1/P2 loops."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ai_scientist.novelty_gate import (
    NoveltyCandidate,
    NoveltyJudge,
    apply_two_stage_novelty_gate,
)
from ai_scientist.restart_seed_selector import RestartSeedCandidate, select_restart_seed

_SENTINEL_OBJECTIVE_MAGNITUDE = 1e8
_SENTINEL_FEASIBILITY_MAGNITUDE = 1e6


def _safe_l2_distance(x: np.ndarray, reference_x: np.ndarray) -> float | None:
    delta = np.asarray(x, dtype=float) - np.asarray(reference_x, dtype=float)
    distance = float(np.linalg.norm(delta))
    return distance if np.isfinite(distance) else None


def _build_restart_novelty_judge(mode: str) -> tuple[NoveltyJudge | None, str]:
    mode_normalized = str(mode).strip().lower()
    if mode_normalized in {"", "disabled"}:
        return None, "disabled"
    if mode_normalized == "heuristic":
        return (
            lambda candidate: (
                candidate.feasibility is not None and candidate.feasibility <= 0.0
            ),
            "heuristic",
        )
    raise ValueError("novelty_judge_mode must be one of: disabled, heuristic.")


def _apply_constrained_novelty_gate(
    *,
    candidates: list[RestartSeedCandidate],
    reference_x: np.ndarray,
    novelty_min_distance: float,
    novelty_feasibility_max: float,
    novelty_near_duplicate_distance: float,
    novelty_judge_mode: str,
) -> tuple[list[RestartSeedCandidate], dict]:
    threshold = float(novelty_min_distance)
    near_duplicate_distance = float(novelty_near_duplicate_distance)
    judge, judge_label = _build_restart_novelty_judge(novelty_judge_mode)

    candidate_by_label: dict[str, RestartSeedCandidate] = {}
    novelty_candidates: list[NoveltyCandidate] = []
    for candidate in candidates:
        distance = _safe_l2_distance(candidate.x, reference_x)
        novelty_candidates.append(
            NoveltyCandidate(
                label=candidate.label,
                embedding_distance=distance,
                feasibility=float(candidate.feasibility),
            )
        )
        candidate_by_label[candidate.label] = candidate

    selected_novelty, novelty_diag = apply_two_stage_novelty_gate(
        novelty_candidates,
        embedding_prefilter_min_distance=threshold,
        near_duplicate_distance=near_duplicate_distance,
        feasibility_max=float(novelty_feasibility_max),
        judge=judge,
        judge_label=judge_label,
        fallback_to_ungated=True,
    )
    selected_pool = [
        candidate_by_label[item.label]
        for item in selected_novelty
        if item.label in candidate_by_label
    ]
    diagnostics = {
        "enabled": bool(novelty_diag.get("enabled")),
        "version": str(novelty_diag.get("version", "")),
        "novelty_min_distance": threshold,
        "novelty_near_duplicate_distance": near_duplicate_distance,
        "novelty_feasibility_max": (
            float(novelty_feasibility_max)
            if np.isfinite(float(novelty_feasibility_max))
            else None
        ),
        "llm_judge_mode": str(novelty_diag.get("judge_label", judge_label)),
        "llm_judge_call_count": int(novelty_diag.get("judge_call_count", 0)),
        "near_duplicate_count": int(novelty_diag.get("near_duplicate_count", 0)),
        "evaluated_count": int(novelty_diag.get("evaluated_count", len(candidates))),
        "kept_count": int(novelty_diag.get("kept_count", len(selected_pool))),
        "rejected_count": int(
            novelty_diag.get("rejected_count", len(candidates) - len(selected_pool))
        ),
        "fallback_to_ungated": bool(novelty_diag.get("fallback_to_ungated")),
        "rows": list(novelty_diag.get("rows", [])),
    }
    return selected_pool, diagnostics


def _worst_objective(problem: str) -> float:
    return float("inf") if problem.lower().startswith("p1") else -float("inf")


def _sanitize_objective(problem: str, objective: float, feasibility: float) -> float:
    value = float(objective)
    feas = float(feasibility)
    if not np.isfinite(value):
        return _worst_objective(problem)
    if (
        abs(value) >= _SENTINEL_OBJECTIVE_MAGNITUDE
        and np.isfinite(feas)
        and feas >= _SENTINEL_FEASIBILITY_MAGNITUDE
    ):
        return _worst_objective(problem)
    return value


def _sanitize_feasibility(feasibility: float) -> float:
    value = float(feasibility)
    return value if np.isfinite(value) else float("inf")


def _make_candidate(
    *,
    problem: str,
    label: str,
    x: np.ndarray,
    objective: float,
    feasibility: float,
) -> RestartSeedCandidate:
    return RestartSeedCandidate(
        label=label,
        x=np.asarray(x, dtype=float),
        objective=_sanitize_objective(problem, objective, feasibility),
        feasibility=_sanitize_feasibility(feasibility),
    )


def _best_high_is_usable(
    *,
    best_high_x: np.ndarray | None,
    best_high_objective: float | None,
    best_high_feasibility: float | None,
) -> bool:
    return (
        best_high_x is not None
        and best_high_objective is not None
        and best_high_feasibility is not None
        and np.isfinite(float(best_high_objective))
        and np.isfinite(float(best_high_feasibility))
    )


def select_adaptive_restart_runtime(
    *,
    problem: str,
    state_x: np.ndarray,
    state_objective: float,
    state_feasibility: float,
    best_violation_x: np.ndarray,
    best_violation_objective: float,
    best_violation_feasibility: float,
    best_low_x: np.ndarray,
    best_low_objective: float,
    best_low_feasibility: float,
    best_high_x: np.ndarray | None,
    best_high_objective: float | None,
    best_high_feasibility: float | None,
    selection_counts: dict[str, int],
    feasibility_weight: float,
    objective_weight: float,
    diversity_weight: float,
    saturation_penalty: float,
    novelty_min_distance: float = 0.0,
    novelty_feasibility_max: float = float("inf"),
    novelty_near_duplicate_distance: float | None = None,
    novelty_judge_mode: str = "heuristic",
) -> tuple[np.ndarray, str, str, dict, dict[str, int]]:
    """Select a restart center from runtime state with explicit objective-space contract.

    Contract:
    - P1 inputs use minimize-space objective values.
    - P2 inputs use maximize-space objective values (for example `lgradb`).
    """
    reference_x = np.asarray(state_x, dtype=float)
    candidates: list[RestartSeedCandidate] = [
        _make_candidate(
            problem=problem,
            label="state",
            x=state_x,
            objective=state_objective,
            feasibility=state_feasibility,
        ),
        _make_candidate(
            problem=problem,
            label="best_violation",
            x=best_violation_x,
            objective=best_violation_objective,
            feasibility=best_violation_feasibility,
        ),
        _make_candidate(
            problem=problem,
            label="best_low",
            x=best_low_x,
            objective=best_low_objective,
            feasibility=best_low_feasibility,
        ),
    ]
    if _best_high_is_usable(
        best_high_x=best_high_x,
        best_high_objective=best_high_objective,
        best_high_feasibility=best_high_feasibility,
    ):
        assert best_high_x is not None
        assert best_high_objective is not None
        assert best_high_feasibility is not None
        candidates.append(
            _make_candidate(
                problem=problem,
                label="best_high",
                x=best_high_x,
                objective=best_high_objective,
                feasibility=best_high_feasibility,
            )
        )

    novelty_threshold = float(novelty_min_distance)
    if novelty_threshold < 0.0:
        raise ValueError("novelty_min_distance must be >= 0.")
    near_duplicate_distance = (
        novelty_threshold
        if novelty_near_duplicate_distance is None
        else float(novelty_near_duplicate_distance)
    )

    gated_candidates, novelty_gate = _apply_constrained_novelty_gate(
        candidates=candidates,
        reference_x=reference_x,
        novelty_min_distance=novelty_threshold,
        novelty_feasibility_max=float(novelty_feasibility_max),
        novelty_near_duplicate_distance=near_duplicate_distance,
        novelty_judge_mode=str(novelty_judge_mode),
    )
    selected_seed, decision = select_restart_seed(
        gated_candidates,
        problem=problem,
        selection_counts=selection_counts,
        reference_x=reference_x,
        feasibility_weight=float(feasibility_weight),
        objective_weight=float(objective_weight),
        diversity_weight=float(diversity_weight),
        saturation_penalty=float(saturation_penalty),
    )
    decision["novelty_gate"] = novelty_gate
    selected_identity = str(decision.get("selected_identity"))
    updated_counts = dict(selection_counts)
    updated_counts[selected_identity] = updated_counts.get(selected_identity, 0) + 1
    return (
        np.asarray(selected_seed.x, dtype=float),
        selected_seed.label,
        selected_identity,
        decision,
        updated_counts,
    )


def append_restart_history(
    path: Path,
    *,
    outer: int,
    selected_seed: str,
    selected_seed_identity: str,
    counts: dict[str, int],
    decision: dict,
) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "outer": int(outer),
                    "selected_seed": str(selected_seed),
                    "selected_seed_identity": str(selected_seed_identity),
                    "counts": dict(counts),
                    "decision": decision,
                }
            )
            + "\n"
        )
