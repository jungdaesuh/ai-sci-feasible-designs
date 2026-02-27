"""Shared model-router reward contract helpers."""

from __future__ import annotations

import math
from typing import Mapping


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


def compute_ucb_score(
    *,
    mean_reward: float,
    pulls: int,
    total_pulls: int,
    exploration_c: float,
) -> float:
    """Compute Upper Confidence Bound score for one arm."""
    mean = float(mean_reward)
    n = int(pulls)
    total = int(total_pulls)
    c = float(exploration_c)
    if n < 0 or total < 0:
        raise ValueError("pulls and total_pulls must be >= 0.")
    if c < 0.0:
        raise ValueError("exploration_c must be >= 0.")
    if n == 0:
        # Always try unseen arms first.
        return float("inf")
    return mean + (c * math.sqrt(math.log(float(max(total, 1))) / float(n)))


def select_ucb_arm(
    arm_stats: Mapping[str, Mapping[str, float | int]],
    *,
    exploration_c: float,
) -> str | None:
    """Select arm key with highest UCB score."""
    if not arm_stats:
        return None
    total_pulls = 0
    for stats in arm_stats.values():
        pulls_raw = stats.get("pulls", 0)
        pulls = int(pulls_raw) if isinstance(pulls_raw, (int, float)) else 0
        if pulls > 0:
            total_pulls += pulls
    total_pulls = max(total_pulls, 1)

    best_arm: str | None = None
    best_score = -float("inf")
    for arm, stats in arm_stats.items():
        pulls_raw = stats.get("pulls", 0)
        mean_raw = stats.get("mean_reward", 0.0)
        pulls = int(pulls_raw) if isinstance(pulls_raw, (int, float)) else 0
        mean_reward = float(mean_raw) if isinstance(mean_raw, (int, float)) else 0.0
        score = compute_ucb_score(
            mean_reward=mean_reward,
            pulls=pulls,
            total_pulls=total_pulls,
            exploration_c=exploration_c,
        )
        if score > best_score:
            best_score = score
            best_arm = arm
    return best_arm
