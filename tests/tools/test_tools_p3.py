import pytest

from ai_scientist import tools
from ai_scientist.test_helpers import base_params


def test_evaluate_p3_records_high_fidelity(mock_backend):
    tools.clear_evaluation_cache()
    params = base_params()

    result = tools.evaluate_p3(params)
    assert result["objective"] == pytest.approx(
        float(result["metrics"]["aspect_ratio"])
    )
    assert result["score"] > 0
    assert "hv" in result
    assert mock_backend.call_count == 1

    tools.evaluate_p3(params)
    assert tools.get_cache_stats("p3")["hits"] >= 1


def test_evaluate_p3_set_computes_hypervolume():
    from ai_scientist import forward_model as fm
    from ai_scientist.backends.base import PhysicsBackend
    from ai_scientist.backends.mock import MockMetrics

    metrics_sequence = [
        MockMetrics(
            aspect_ratio=3.5,
            minimum_normalized_magnetic_gradient_scale_length=1.5,
            edge_magnetic_mirror_ratio=0.2,
            edge_rotational_transform_over_n_field_periods=0.4,
            vacuum_well=0.1,
            qi=1e-5,
            flux_compression_in_regions_of_bad_curvature=0.2,
        ),
        MockMetrics(
            aspect_ratio=2.4,
            minimum_normalized_magnetic_gradient_scale_length=1.3,
            edge_magnetic_mirror_ratio=0.2,
            edge_rotational_transform_over_n_field_periods=0.4,
            vacuum_well=0.1,
            qi=1e-5,
            flux_compression_in_regions_of_bad_curvature=0.2,
        ),
        MockMetrics(
            aspect_ratio=18.0,
            minimum_normalized_magnetic_gradient_scale_length=0.5,
            edge_magnetic_mirror_ratio=0.4,
            edge_rotational_transform_over_n_field_periods=0.4,
            vacuum_well=0.1,
            qi=1e-5,
            flux_compression_in_regions_of_bad_curvature=0.2,
        ),
    ]

    class _SequenceBackend(PhysicsBackend):
        def __init__(self, sequence: list[MockMetrics]):
            self._sequence = sequence
            self._idx = 0

        def evaluate(self, boundary, settings):
            metrics = self._sequence[self._idx]
            self._idx += 1
            margins = fm.compute_constraint_margins(metrics, settings.problem)
            feasibility = fm.max_violation(margins)
            return fm.EvaluationResult(
                metrics=metrics,
                objective=fm.compute_objective(metrics, settings.problem),
                constraints=list(margins.values()),
                constraint_names=list(margins.keys()),
                feasibility=feasibility,
                is_feasible=feasibility <= 1e-2,
                cache_hit=False,
                design_hash=fm.compute_design_hash(boundary),
                evaluation_time_sec=0.0,
                fidelity=settings.fidelity,
                settings=settings,
                equilibrium_converged=True,
                error_message=None,
            )

        def is_available(self) -> bool:
            return True

        @property
        def name(self) -> str:
            return "sequence"

    tools.clear_evaluation_cache()
    fm.set_backend(_SequenceBackend(metrics_sequence))
    tools.clear_evaluation_cache()

    candidates = [base_params() for _ in range(len(metrics_sequence))]
    result = tools.evaluate_p3_set(candidates)

    assert result["hv_score"] > 0
    assert len(result["metrics_list"]) == len(metrics_sequence)
    assert len(result["objectives"]) == len(metrics_sequence)
    assert len(result["feasibilities"]) == len(metrics_sequence)
