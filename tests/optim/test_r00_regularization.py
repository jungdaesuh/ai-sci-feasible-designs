"""Test R₀₀ (major radius) regularization for stellarator optimization.

This module tests the H1 fix that allows optional R₀₀ optimization with
strong L2 regularization toward the dataset mean (~1.0).
"""

import pytest

try:
    import torch

    from ai_scientist.constraints import LAMBDA_R00_REGULARIZATION

except ImportError:
    pytest.skip("PyTorch not available", allow_module_level=True)


def test_r00_regularization_weight_is_strong():
    """Verify regularization weight is strong enough to prevent scale drift."""
    # The weight should be at least 1.0 to be effective
    assert LAMBDA_R00_REGULARIZATION >= 1.0, (
        f"R₀₀ regularization weight ({LAMBDA_R00_REGULARIZATION}) should be >= 1.0 "
        "to prevent scale drift during optimization"
    )

    # The weight should be reasonably bounded (not so large it dominates)
    assert LAMBDA_R00_REGULARIZATION <= 100.0, (
        f"R₀₀ regularization weight ({LAMBDA_R00_REGULARIZATION}) should be <= 100.0 "
        "to allow some flexibility in major radius"
    )


def test_r00_regularization_converges_to_target():
    """Verify R₀₀ regularization pulls values toward target (1.0)."""
    # Initial R₀₀ far from target
    r00_initial = torch.tensor(1.5, requires_grad=True)
    target = 1.0

    optimizer = torch.optim.Adam([r00_initial], lr=0.1)

    for _ in range(100):
        optimizer.zero_grad()
        loss = LAMBDA_R00_REGULARIZATION * (r00_initial - target) ** 2
        loss.backward()
        optimizer.step()

    # Should converge close to target
    assert abs(r00_initial.item() - target) < 0.05, (
        f"R₀₀ should converge toward {target}, got {r00_initial.item():.4f}"
    )


def test_r00_regularization_from_low_value():
    """Verify R₀₀ regularization works when starting below target."""
    # Initial R₀₀ below target
    r00_initial = torch.tensor(0.5, requires_grad=True)
    target = 1.0

    optimizer = torch.optim.Adam([r00_initial], lr=0.1)

    for _ in range(100):
        optimizer.zero_grad()
        loss = LAMBDA_R00_REGULARIZATION * (r00_initial - target) ** 2
        loss.backward()
        optimizer.step()

    # Should converge close to target
    assert abs(r00_initial.item() - target) < 0.05, (
        f"R₀₀ should converge toward {target}, got {r00_initial.item():.4f}"
    )


def test_r00_regularization_gradient_direction():
    """Verify gradient direction is correct for R₀₀ regularization."""
    r00 = torch.tensor(1.5, requires_grad=True)
    target = 1.0

    loss = LAMBDA_R00_REGULARIZATION * (r00 - target) ** 2
    loss.backward()

    # When r00 > target, gradient should be positive (to push r00 down)
    assert r00.grad > 0, "Gradient should be positive when r00 > target"

    # Reset and test other direction
    r00 = torch.tensor(0.5, requires_grad=True)
    loss = LAMBDA_R00_REGULARIZATION * (r00 - target) ** 2
    loss.backward()

    # When r00 < target, gradient should be negative (to push r00 up)
    assert r00.grad < 0, "Gradient should be negative when r00 < target"


if __name__ == "__main__":
    test_r00_regularization_weight_is_strong()
    test_r00_regularization_converges_to_target()
    test_r00_regularization_from_low_value()
    test_r00_regularization_gradient_direction()
    print("All R₀₀ regularization tests passed!")
