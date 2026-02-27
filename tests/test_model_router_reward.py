from __future__ import annotations

import pytest

from ai_scientist.model_router_reward import (
    compute_model_router_reward,
    relative_improvement,
)


def test_relative_improvement_handles_zero_baseline() -> None:
    assert relative_improvement(previous=0.0, current=2.0) == 1.0
    assert relative_improvement(previous=0.0, current=-2.0) == -1.0
    assert relative_improvement(previous=0.0, current=0.0) == 0.0


def test_compute_model_router_reward_uses_weighted_relative_deltas() -> None:
    payload = compute_model_router_reward(
        previous_feasible_yield=20.0,
        current_feasible_yield=30.0,
        previous_hv=10.0,
        current_hv=11.0,
        feasible_weight=0.75,
        hv_weight=0.25,
    )
    assert payload["relative_feasible_yield"] == pytest.approx(0.5)
    assert payload["relative_hv"] == pytest.approx(0.1)
    assert payload["reward"] == pytest.approx((0.75 * 0.5) + (0.25 * 0.1))


def test_compute_model_router_reward_rejects_invalid_weights() -> None:
    with pytest.raises(ValueError, match="must be >= 0"):
        compute_model_router_reward(
            previous_feasible_yield=0.0,
            current_feasible_yield=0.0,
            previous_hv=0.0,
            current_hv=0.0,
            feasible_weight=-1.0,
            hv_weight=1.0,
        )
    with pytest.raises(ValueError, match="must be > 0"):
        compute_model_router_reward(
            previous_feasible_yield=0.0,
            current_feasible_yield=0.0,
            previous_hv=0.0,
            current_hv=0.0,
            feasible_weight=0.0,
            hv_weight=0.0,
        )
