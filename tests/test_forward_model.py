import unittest.mock
import pytest
from ai_scientist import forward_model
from constellaration import forward_model as constellaration_forward

# Mock Data
MOCK_METRICS_P1 = constellaration_forward.ConstellarationMetrics(
    aspect_ratio=5.0,
    aspect_ratio_over_edge_rotational_transform=20.0,
    max_elongation=1.5,
    axis_rotational_transform_over_n_field_periods=0.1,
    edge_rotational_transform_over_n_field_periods=0.25,
    axis_magnetic_mirror_ratio=0.1,
    edge_magnetic_mirror_ratio=0.1,
    average_triangularity=0.1,
    vacuum_well=0.05,
    minimum_normalized_magnetic_gradient_scale_length=1.2,
    qi=1e-5,
    flux_compression_in_regions_of_bad_curvature=0.95,
)

MOCK_BOUNDARY_PARAMS = {
    "r_cos": [[0.0, 1.0, 0.1], [0.1, 0.1, 0.0]],
    "z_sin": [[0.0, 0.0, 0.1], [0.1, 0.1, 0.0]],
    "nfp": 3,
    "is_stellarator_symmetric": True
}

@pytest.fixture
def mock_constellaration_forward():
    with unittest.mock.patch(
        "constellaration.forward_model.forward_model", 
        return_value=(MOCK_METRICS_P1, None)
    ) as mock:
        yield mock

@pytest.fixture(autouse=True)
def clear_cache():
    forward_model.clear_cache()
    yield

def test_forward_model_basic(mock_constellaration_forward):
    settings = forward_model.ForwardModelSettings(problem="p1")
    result = forward_model.forward_model(MOCK_BOUNDARY_PARAMS, settings)
    
    assert isinstance(result, forward_model.EvaluationResult)
    assert result.metrics == MOCK_METRICS_P1
    assert result.cache_hit is False
    assert result.objective == 1.5  # P1 objective is max_elongation
    
    mock_constellaration_forward.assert_called_once()

def test_forward_model_caching(mock_constellaration_forward):
    settings = forward_model.ForwardModelSettings(problem="p1")
    
    # First call
    result1 = forward_model.forward_model(MOCK_BOUNDARY_PARAMS, settings)
    assert result1.cache_hit is False
    assert mock_constellaration_forward.call_count == 1
    
    # Second call
    result2 = forward_model.forward_model(MOCK_BOUNDARY_PARAMS, settings)
    assert result2.cache_hit is True
    assert mock_constellaration_forward.call_count == 1  # No new call
    
    assert result1.design_hash == result2.design_hash

def test_forward_model_different_settings(mock_constellaration_forward):
    # Same boundary, different settings -> different cache key?
    # The current implementation keys by hash(boundary) + hash(settings).
    
    settings_p1 = forward_model.ForwardModelSettings(problem="p1")
    settings_p2 = forward_model.ForwardModelSettings(problem="p2")
    
    forward_model.forward_model(MOCK_BOUNDARY_PARAMS, settings_p1)
    forward_model.forward_model(MOCK_BOUNDARY_PARAMS, settings_p2)
    
    assert mock_constellaration_forward.call_count == 2

def test_batch_evaluation(mock_constellaration_forward):
    settings = forward_model.ForwardModelSettings(problem="p1")
    boundaries = [MOCK_BOUNDARY_PARAMS, MOCK_BOUNDARY_PARAMS] # same params for simplicity
    
    results = forward_model.forward_model_batch(boundaries, settings, n_workers=2)
    
    assert len(results) == 2
    assert results[0].design_hash == results[1].design_hash
    # Since they are identical, the second one might hit cache depending on race conditions, 
    # but forward_model logic is inside the thread.
    
    # If we run sequentially inside batch or if they finish fast enough, one might cache hit.
    # But mock is called.
    # Actually, since it's threaded, both might start before cache is populated.
    # So call_count could be 1 or 2. 
    
def test_problem_p2_metrics(mock_constellaration_forward):
    settings = forward_model.ForwardModelSettings(problem="p2")
    result = forward_model.forward_model(MOCK_BOUNDARY_PARAMS, settings)
    
    # P2 objective: minimum_normalized_magnetic_gradient_scale_length (maximize? No, P2 usually maximizes gradient)
    # Wait, in evaluation.py evaluate_p2:
    # objective = gradient
    # maximize_objective = True
    # But in forward_model.py compute_objective:
    # return metrics.minimum_normalized_magnetic_gradient_scale_length
    
    assert result.objective == 1.2
    
    # P2 constraints check
    # Aspect ratio - 10.0 <= 0
    # 5.0 - 10.0 = -5.0 (Satisfied)
    assert result.constraints["aspect_ratio"] == -5.0

def test_design_hash_stability():
    params1 = {"r_cos": [[1.0]], "z_sin": [[0.0]]}
    params2 = {"r_cos": [[1.0]], "z_sin": [[0.0]]}
    
    h1 = forward_model.compute_design_hash(params1)
    h2 = forward_model.compute_design_hash(params2)
    
    assert h1 == h2
    
    params3 = {"r_cos": [[1.000000001]], "z_sin": [[0.0]]} # Within rounding tolerance?
    # Default rounding is 1e-6
    h3 = forward_model.compute_design_hash(params3)
    assert h1 == h3
    
    params4 = {"r_cos": [[1.1]], "z_sin": [[0.0]]}
    h4 = forward_model.compute_design_hash(params4)
    assert h1 != h4
