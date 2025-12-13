import pytest

from ai_scientist import tools
from ai_scientist.test_helpers import base_params


def test_evaluate_p2_uses_high_fidelity_cache(mock_backend):
    tools.clear_evaluation_cache()
    params = base_params()

    result = tools.evaluate_p2(params, stage="p2")
    assert result["objective"] == pytest.approx(
        float(result["metrics"]["minimum_normalized_magnetic_gradient_scale_length"])
    )
    assert result["hv"] >= 0
    assert mock_backend.call_count == 1

    cached = tools.evaluate_p2(params)
    assert cached["objective"] == result["objective"]
    assert mock_backend.call_count == 1
    assert tools.get_cache_stats("p2")["hits"] >= 1
