import pytest

from ai_scientist import tools
from ai_scientist.test_helpers import base_params, dummy_metrics_with
from constellaration.forward_model import ConstellarationSettings


def _assert_high_fidelity_settings(settings: ConstellarationSettings) -> None:
    assert settings.vmec_preset_settings.fidelity == "high_fidelity"
    assert settings.qi_settings is not None


def test_evaluate_p3_records_high_fidelity(monkeypatch):
    metrics = dummy_metrics_with()

    call_stats = {"count": 0}

    def fake_forward(boundary, settings):
        call_stats["count"] += 1
        _assert_high_fidelity_settings(settings)
        return metrics, None

    monkeypatch.setattr("constellaration.forward_model.forward_model", fake_forward)
    tools.clear_evaluation_cache()
    params = base_params()

    result = tools.evaluate_p3(params)
    assert result["objective"] == pytest.approx(metrics.aspect_ratio)
    assert result["score"] > 0
    assert "hv" in result
    assert call_stats["count"] == 1

    tools.evaluate_p3(params)
    assert tools.get_cache_stats("p3")["hits"] >= 1


def test_evaluate_p3_set_computes_hypervolume(monkeypatch):
    metrics_sequence = [
        dummy_metrics_with(
            aspect_ratio=3.5,
            minimum_normalized_magnetic_gradient_scale_length=1.5,
            edge_magnetic_mirror_ratio=0.2,
            edge_rotational_transform_over_n_field_periods=0.4,
            vacuum_well=0.1,
            qi=1e-5,
            flux_compression_in_regions_of_bad_curvature=0.2,
        ),
        dummy_metrics_with(
            aspect_ratio=2.4,
            minimum_normalized_magnetic_gradient_scale_length=1.3,
            edge_magnetic_mirror_ratio=0.2,
            edge_rotational_transform_over_n_field_periods=0.4,
            vacuum_well=0.1,
            qi=1e-5,
            flux_compression_in_regions_of_bad_curvature=0.2,
        ),
        dummy_metrics_with(
            aspect_ratio=18.0,
            minimum_normalized_magnetic_gradient_scale_length=0.5,
            edge_magnetic_mirror_ratio=0.4,
            edge_rotational_transform_over_n_field_periods=0.4,
            vacuum_well=0.1,
            qi=1e-5,
            flux_compression_in_regions_of_bad_curvature=0.2,
        ),
    ]

    call_state = {"index": 0}

    def fake_forward(boundary, settings):
        idx = call_state["index"]
        call_state["index"] += 1
        _assert_high_fidelity_settings(settings)
        return metrics_sequence[idx], None

    monkeypatch.setattr("constellaration.forward_model.forward_model", fake_forward)
    tools.clear_evaluation_cache()

    candidates = [base_params() for _ in range(len(metrics_sequence))]
    result = tools.evaluate_p3_set(candidates)

    assert result["hv_score"] > 0
    assert len(result["metrics_list"]) == len(metrics_sequence)
    assert len(result["objectives"]) == len(metrics_sequence)
    assert len(result["feasibilities"]) == len(metrics_sequence)
