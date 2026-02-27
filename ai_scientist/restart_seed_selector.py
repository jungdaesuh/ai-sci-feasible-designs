"""Adaptive restart seed selector shared by P1/P2 optimization loops."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class RestartSeedCandidate:
    label: str
    x: np.ndarray
    objective: float
    feasibility: float


def _candidate_identity(x: np.ndarray) -> str:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size == 0:
        return "empty"
    if not np.all(np.isfinite(arr)):
        return "non_finite"
    rounded = np.round(arr, decimals=8)
    rounded[rounded == 0.0] = 0.0
    digest = hashlib.sha256(rounded.tobytes()).hexdigest()
    return f"x:{digest}"


def _normalize_range(values: np.ndarray, *, maximize: bool) -> np.ndarray:
    if values.size == 0:
        return values
    scores = np.zeros_like(values, dtype=float)
    finite_mask = np.isfinite(values)
    if not np.any(finite_mask):
        return scores
    finite_values = values[finite_mask]
    lo = float(np.min(finite_values))
    hi = float(np.max(finite_values))
    if hi <= lo:
        scores[finite_mask] = 1.0
        return scores
    normalized = (finite_values - lo) / (hi - lo)
    scores[finite_mask] = normalized if maximize else 1.0 - normalized
    return scores


def select_restart_seed(
    candidates: list[RestartSeedCandidate],
    *,
    problem: str,
    selection_counts: Mapping[str, int],
    reference_x: np.ndarray | None,
    feasibility_weight: float = 0.55,
    objective_weight: float = 0.35,
    diversity_weight: float = 0.10,
    saturation_penalty: float = 0.15,
) -> tuple[RestartSeedCandidate, dict]:
    """Pick a restart candidate with feasibility-first scoring and diversity/penalty terms."""

    if not candidates:
        raise ValueError("candidates must not be empty")

    obj_values = np.array([float(c.objective) for c in candidates], dtype=float)
    feas_values = np.array([float(c.feasibility) for c in candidates], dtype=float)

    is_p1 = problem.lower().startswith("p1")
    objective_scores = _normalize_range(obj_values, maximize=not is_p1)
    feasibility_scores = _normalize_range(feas_values, maximize=False)
    candidate_identities = [
        _candidate_identity(candidate.x) for candidate in candidates
    ]

    diversity_scores = np.zeros(len(candidates), dtype=float)
    if reference_x is not None:
        distances: list[float] = []
        for candidate in candidates:
            delta = np.asarray(candidate.x, dtype=float) - reference_x
            distance = float(np.linalg.norm(delta))
            distances.append(distance if np.isfinite(distance) else float("nan"))
        distances_arr = np.array(distances, dtype=float)
        diversity_scores = _normalize_range(distances_arr, maximize=True)

    penalties = np.array(
        [
            float(selection_counts.get(identity, 0)) * float(saturation_penalty)
            for identity in candidate_identities
        ],
        dtype=float,
    )

    total_scores = (
        float(feasibility_weight) * feasibility_scores
        + float(objective_weight) * objective_scores
        + float(diversity_weight) * diversity_scores
        - penalties
    )

    best_index = int(np.argmax(total_scores))
    selected = candidates[best_index]
    selected_identity = candidate_identities[best_index]
    decision = {
        "selected_label": selected.label,
        "selected_identity": selected_identity,
        "scores": [
            {
                "label": candidate.label,
                "identity": candidate_identities[idx],
                "feasibility_score": float(feasibility_scores[idx]),
                "objective_score": float(objective_scores[idx]),
                "diversity_score": float(diversity_scores[idx]),
                "saturation_penalty": float(penalties[idx]),
                "total": float(total_scores[idx]),
            }
            for idx, candidate in enumerate(candidates)
        ],
        "weights": {
            "feasibility": float(feasibility_weight),
            "objective": float(objective_weight),
            "diversity": float(diversity_weight),
            "saturation_penalty": float(saturation_penalty),
        },
    }
    return selected, decision
