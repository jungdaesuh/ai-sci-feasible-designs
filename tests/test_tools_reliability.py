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
