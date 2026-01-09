import pytest

from ai_scientist.forward_model import compute_p3_objectives


def test_compute_p3_objectives_accepts_dict_metrics() -> None:
    """compute_p3_objectives should work with dict metrics (not just pydantic objects)."""
    metrics = {
        "aspect_ratio": 7.0,
        "minimum_normalized_magnetic_gradient_scale_length": 3.5,
    }
    aspect, gradient = compute_p3_objectives(metrics)
    assert aspect == pytest.approx(7.0)
    assert gradient == pytest.approx(3.5)
