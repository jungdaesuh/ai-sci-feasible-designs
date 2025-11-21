import numpy as np
import pytest

from ai_scientist import tools
from ai_scientist.test_helpers import base_params, dummy_metrics


def test_make_boundary_from_params_constructs_surface() -> None:
    boundary = tools.make_boundary_from_params(base_params())
    assert boundary.r_cos.shape == (2, 5)
    np.testing.assert_allclose(boundary.r_cos[0, 4], 0.2, atol=1e-2)


def test_evaluate_p1_caches_by_stage(monkeypatch) -> None:
    metrics = dummy_metrics()
    call_counter = {"count": 0}

    def fake_forward(boundary, settings):
        call_counter["count"] += 1
        return metrics, None

    monkeypatch.setattr("constellaration.forward_model.forward_model", fake_forward)
    tools.clear_evaluation_cache()
    params = base_params()

    result = tools.evaluate_p1(params, stage="screen")
    assert result["objective"] == pytest.approx(metrics.max_elongation)
    assert "metrics" in result
    assert call_counter["count"] == 1

    cached = tools.evaluate_p1(params, stage="screen")
    assert cached["objective"] == result["objective"]
    assert call_counter["count"] == 1
    stats = tools.get_cache_stats("screen")
    assert stats["hits"] >= 1

    tools.evaluate_p1(params, stage="promote")
    assert call_counter["count"] == 2
    stats = tools.get_cache_stats("promote")
    assert stats["misses"] == 1
    tools.evaluate_p1(params, stage="promote")
    assert tools.get_cache_stats("promote")["hits"] >= 1


def test_evaluate_p1_can_bypass_cache(monkeypatch) -> None:
    metrics = dummy_metrics()
    call_counter = {"count": 0}

    def fake_forward(boundary, settings):
        call_counter["count"] += 1
        return metrics, None

    monkeypatch.setattr("constellaration.forward_model.forward_model", fake_forward)
    tools.clear_evaluation_cache()
    params = base_params()

    tools.evaluate_p1(params, stage="screen")
    assert call_counter["count"] == 1

    tools.evaluate_p1(params, stage="screen")
    assert call_counter["count"] == 1

    tools.evaluate_p1(params, stage="screen", use_cache=False)
    assert call_counter["count"] == 2
