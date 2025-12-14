import pytest

from ai_scientist.forward_model import (
    EvaluationResult,
    ForwardModelSettings,
    clear_cache,
    compute_design_hash,
)
from ai_scientist.forward_model import forward_model as run_forward_model
from ai_scientist.forward_model import forward_model_batch

MOCK_BOUNDARY_PARAMS = {
    "r_cos": [[0.0, 1.0, 0.1], [0.1, 0.1, 0.0]],
    "z_sin": [[0.0, 0.0, 0.1], [0.1, 0.1, 0.0]],
    "nfp": 3,
    "is_stellarator_symmetric": True,
}


@pytest.fixture(autouse=True)
def clear_cache_fixture():
    clear_cache()
    yield


def test_forward_model_basic(mock_backend):
    settings = ForwardModelSettings(problem="p1")
    result = run_forward_model(MOCK_BOUNDARY_PARAMS, settings)

    assert isinstance(result, EvaluationResult)
    assert result.cache_hit is False
    assert mock_backend.call_count == 1
    assert result.objective == pytest.approx(float(result.metrics.max_elongation))


def test_forward_model_caching(mock_backend):
    settings = ForwardModelSettings(problem="p1")

    # First call
    result1 = run_forward_model(MOCK_BOUNDARY_PARAMS, settings)
    assert result1.cache_hit is False
    assert mock_backend.call_count == 1

    # Second call
    result2 = run_forward_model(MOCK_BOUNDARY_PARAMS, settings)
    assert result2.cache_hit is True
    assert mock_backend.call_count == 1  # No new call

    assert result1.design_hash == result2.design_hash


def test_forward_model_different_settings(mock_backend):
    # Same boundary, different settings -> different cache key?
    # The current implementation keys by hash(boundary) + hash(settings).

    settings_p1 = ForwardModelSettings(problem="p1")
    settings_p2 = ForwardModelSettings(problem="p2")

    run_forward_model(MOCK_BOUNDARY_PARAMS, settings_p1)
    run_forward_model(MOCK_BOUNDARY_PARAMS, settings_p2)

    assert mock_backend.call_count == 2


def test_batch_evaluation(mock_backend):
    settings = ForwardModelSettings(problem="p1")
    boundaries = [
        MOCK_BOUNDARY_PARAMS,
        MOCK_BOUNDARY_PARAMS,
    ]  # same params for simplicity

    results = forward_model_batch(boundaries, settings, n_workers=2)  # pyright: ignore[reportArgumentType]

    assert len(results) == 2
    assert results[0].design_hash == results[1].design_hash
    # Since they are identical, the second one might hit cache depending on race conditions,
    # but forward_model logic is inside the thread.

    # If we run sequentially inside batch or if they finish fast enough, one might cache hit.
    # But mock is called.
    # Actually, since it's threaded, both might start before cache is populated.
    # So call_count could be 1 or 2.
    assert mock_backend.call_count >= 1


def test_problem_p2_metrics(mock_backend):
    settings = ForwardModelSettings(problem="p2")
    result = run_forward_model(MOCK_BOUNDARY_PARAMS, settings)

    assert result.objective == pytest.approx(
        float(result.metrics.minimum_normalized_magnetic_gradient_scale_length)
    )
    assert result.constraints_map["aspect_ratio"] == pytest.approx(
        (float(result.metrics.aspect_ratio) - 10.0) / 10.0
    )


def test_design_hash_stability():
    params1 = {"r_cos": [[1.0]], "z_sin": [[0.0]]}
    params2 = {"r_cos": [[1.0]], "z_sin": [[0.0]]}

    h1 = compute_design_hash(params1)
    h2 = compute_design_hash(params2)

    assert h1 == h2

    # With 1e-9 rounding precision, 1e-10 difference should still hash the same
    params3 = {
        "r_cos": [[1.0000000001]],
        "z_sin": [[0.0]],
    }  # 1e-10 diff, within tolerance
    h3 = compute_design_hash(params3)
    assert h1 == h3

    # But 1e-8 difference should produce different hash (Issue #8 fix)
    params3b = {
        "r_cos": [[1.00000001]],
        "z_sin": [[0.0]],
    }  # 1e-8 diff, beyond tolerance
    h3b = compute_design_hash(params3b)
    assert h1 != h3b  # This is the fix for gradient optimization collisions

    params4 = {"r_cos": [[1.1]], "z_sin": [[0.0]]}
    h4 = compute_design_hash(params4)
    assert h1 != h4


def test_evaluation_result_structure(mock_backend):
    settings = ForwardModelSettings(problem="p1", fidelity="medium")
    result = run_forward_model(MOCK_BOUNDARY_PARAMS, settings)

    assert result.fidelity == "medium"
    assert result.equilibrium_converged is True
    assert result.error_message is None

    assert isinstance(result.constraint_names, list)
    assert len(result.constraint_names) > 0
    assert isinstance(result.constraints, list)

    # Check to_pareto_point
    pareto = result.to_pareto_point()
    assert isinstance(pareto, tuple)
    assert len(pareto) == 2
    assert pareto[0] == result.objective
    assert pareto[1] == result.feasibility

    # Check dominates (basic check)
    assert result.dominates(result) is False  # Cannot dominate itself
