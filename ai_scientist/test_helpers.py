"""Shared fixtures and helpers reused between ai_scientist tests."""

from ai_scientist.backends.mock import MockMetrics


def base_params() -> dict[str, list[list[float]] | int | bool]:
    return {
        "r_cos": [[0.0, 0.0, 1.5, 0.0, 0.2], [0.0, 0.0, 0.05, 0.0, 0.1]],
        "z_sin": [[0.0, 0.0, 0.0, 0.05, 0.0], [0.0, 0.0, 0.02, 0.0, 0.0]],
        "n_field_periods": 1,
        "is_stellarator_symmetric": True,
    }


def dummy_metrics() -> MockMetrics:
    return MockMetrics(
        aspect_ratio=3.0,
        max_elongation=1.2,
        edge_rotational_transform_over_n_field_periods=0.4,
        edge_magnetic_mirror_ratio=0.6,
        average_triangularity=-0.3,
        vacuum_well=0.1,
        minimum_normalized_magnetic_gradient_scale_length=0.2,
        qi=1e-5,
        flux_compression_in_regions_of_bad_curvature=0.1,
    )


def dummy_metrics_with(**overrides: float) -> MockMetrics:
    data = dummy_metrics().model_dump()
    data.update(overrides)
    return MockMetrics(**data)
