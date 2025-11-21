import pytest

from ai_scientist import tools
from ai_scientist.test_helpers import base_params, dummy_metrics
from constellaration.forward_model import ConstellarationSettings


def _assert_high_fidelity_settings(settings: ConstellarationSettings) -> None:
    assert settings.vmec_preset_settings.fidelity == "high_fidelity"
    assert settings.qi_settings is not None


def test_evaluate_p2_uses_high_fidelity_cache(monkeypatch):
    metrics = dummy_metrics()

    def fake_forward(boundary, settings):
        _assert_high_fidelity_settings(settings)
        return metrics, None

    monkeypatch.setattr("constellaration.forward_model.forward_model", fake_forward)
    tools.clear_evaluation_cache()
    params = base_params()

    result = tools.evaluate_p2(params)
    assert result["stage"] == "p2"
    assert result["objective"] == pytest.approx(
        metrics.minimum_normalized_magnetic_gradient_scale_length
    )
    assert result["hv"] >= 0

    cached = tools.evaluate_p2(params)
    assert cached["objective"] == result["objective"]
    assert tools.get_cache_stats("p2")["hits"] >= 1
