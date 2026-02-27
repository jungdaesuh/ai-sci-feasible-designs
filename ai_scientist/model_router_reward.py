"""Shared model-router reward contract helpers."""

from __future__ import annotations

import math


def relative_improvement(*, previous: float, current: float) -> float:
    prev = float(previous)
    curr = float(current)
    if not math.isfinite(prev):
        prev = 0.0
    if not math.isfinite(curr):
        curr = 0.0
    delta = curr - prev
    denom = abs(prev)
    if denom < 1e-9:
        if delta > 0.0:
            return 1.0
        if delta < 0.0:
            return -1.0
        return 0.0
    return float(delta / denom)


def compute_model_router_reward(
    *,
    previous_feasible_yield: float,
    current_feasible_yield: float,
    previous_hv: float,
    current_hv: float,
    feasible_weight: float = 0.5,
    hv_weight: float = 0.5,
) -> dict:
    """Compute weighted relative-improvement reward for router/bandit updates."""
    feasible_w = float(feasible_weight)
    hv_w = float(hv_weight)
    if feasible_w < 0.0 or hv_w < 0.0:
        raise ValueError("feasible_weight and hv_weight must be >= 0.")
    total_weight = feasible_w + hv_w
    if total_weight <= 0.0:
        raise ValueError("feasible_weight + hv_weight must be > 0.")

    feasible_rel = relative_improvement(
        previous=float(previous_feasible_yield),
        current=float(current_feasible_yield),
    )
    hv_rel = relative_improvement(
        previous=float(previous_hv),
        current=float(current_hv),
    )
    normalized_feasible_w = feasible_w / total_weight
    normalized_hv_w = hv_w / total_weight
    reward = (normalized_feasible_w * feasible_rel) + (normalized_hv_w * hv_rel)

    return {
        "previous_feasible_yield": float(previous_feasible_yield),
        "current_feasible_yield": float(current_feasible_yield),
        "delta_feasible_yield": float(current_feasible_yield - previous_feasible_yield),
        "relative_feasible_yield": feasible_rel,
        "previous_hv": float(previous_hv),
        "current_hv": float(current_hv),
        "delta_hv": float(current_hv - previous_hv),
        "relative_hv": hv_rel,
        "weights": {
            "feasible_weight": normalized_feasible_w,
            "hv_weight": normalized_hv_w,
        },
        "reward": float(reward),
    }
