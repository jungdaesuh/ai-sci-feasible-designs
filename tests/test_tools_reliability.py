import math

import pytest

from ai_scientist import tools
from ai_scientist.test_helpers import base_params


def test_safe_evaluate_penalizes_nan_minimize() -> None:
    def compute() -> dict[str, float | dict[str, float]]:
        return {
            "objective": math.nan,
            "feasibility": 0.0,
            "metrics": {"aspect_ratio": math.nan},
        }

    result = tools._safe_evaluate(compute, stage="screen", maximize=False)

    assert result["objective"] == pytest.approx(1e9)
    assert result["minimize_objective"] is True
    assert result.get("penalized") is True
    assert math.isinf(result["feasibility"])


def test_safe_evaluate_respects_maximize_penalty() -> None:
    def compute() -> dict[str, float]:
        raise RuntimeError("boom")

    result = tools._safe_evaluate(compute, stage="p2", maximize=True)

    assert result["objective"] == pytest.approx(-1e9)
    assert result["minimize_objective"] is False
    assert result["stage"] == "p2"


def test_structured_flatten_keeps_rcos_index_stable() -> None:
    schema = tools.FlattenSchema(mpol=3, ntor=2, schema_version=1)
    params_small = base_params()
    params_small["r_cos"][1][2] = 0.77

    params_large = base_params()
    params_large["r_cos"].append([0.0, 0.0, 0.0, 0.0, 0.0])
    params_large["r_cos"][1][2] = 0.77

    vector_small, _ = tools.structured_flatten(params_small, schema=schema)
    vector_large, _ = tools.structured_flatten(params_large, schema=schema)

    width = 2 * schema.ntor + 1
    rcos_index = (1 * width) + schema.ntor

    assert vector_small.shape[0] == (schema.mpol + 1) * width * 2
    assert vector_large.shape[0] == vector_small.shape[0]
    assert vector_small[rcos_index] == pytest.approx(vector_large[rcos_index])


def test_structured_flatten_zero_pads_when_schema_ntor_exceeds_matrix() -> None:
    # Matrix has ntor=1 (3 columns) while schema requests ntor=2; padding must preserve n=0.
    schema = tools.FlattenSchema(mpol=0, ntor=2, schema_version=1)
    params = {
        "r_cos": [[1.0, 2.0, 3.0]],  # columns correspond to n=-1,0,+1
        "z_sin": [[0.0, 0.0, 0.0]],
        "n_field_periods": 1,
        "is_stellarator_symmetric": True,
    }

    vector, _ = tools.structured_flatten(params, schema=schema)

    width = 2 * schema.ntor + 1  # 5
    n0_index = schema.ntor  # offset inside first block

    assert vector.shape[0] == width * 2  # r_cos + z_sin blocks
    assert vector[n0_index] == pytest.approx(2.0)  # preserves true n=0 coefficient
    assert vector[n0_index - 2] == 0.0  # padded n=-2
    assert vector[n0_index + 2] == 0.0  # padded n=+2
