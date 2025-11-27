import pytest
import numpy as np
try:
    import torch
    from ai_scientist.optim.surrogate_v2 import NeuralOperatorSurrogate, StellaratorNeuralOp
except ImportError as e:
    pytest.skip(f"PyTorch not available: {e}", allow_module_level=True)

def _make_params(scale: float, mpol: int = 3, ntor: int = 3) -> dict:
    matrix = np.full((mpol + 1, 2 * ntor + 1), scale, dtype=float)
    return {
        "r_cos": matrix.tolist(),
        "z_sin": (matrix * 0.1).tolist(),
        "n_field_periods": 3,
        "is_stellarator_symmetric": True,
    }

def _training_history(rows: int) -> tuple[list[dict], list[float]]:
    metrics_list: list[dict] = []
    targets: list[float] = []
    for idx in range(rows):
        params = _make_params(float(idx) / 10.0)
        metrics_payload = {
            "vacuum_well": 0.1 * idx,
            "qi": 1.0 / (idx + 1.0),
            "max_elongation": 2.0 + 0.1 * idx
        }
        metrics_list.append({
            "candidate_params": params, 
            "metrics": metrics_payload
        })
        targets.append(float(idx))
    return metrics_list, targets

def test_surrogate_ensemble_init_and_fit():
    """Test that we can initialize and train an ensemble of models."""
    n_ensembles = 3
    surrogate = NeuralOperatorSurrogate(
        min_samples=4,
        epochs=2, # short training
        batch_size=4,
        n_ensembles=n_ensembles,
        hidden_dim=16
    )
    
    assert surrogate._n_ensembles == n_ensembles
    assert len(surrogate._models) == 0 # Before fit
    
    metrics, targets = _training_history(10)
    
    surrogate.fit(metrics, targets, minimize_objective=False)
    
    assert surrogate._trained
    assert len(surrogate._models) == n_ensembles
    for model in surrogate._models:
        assert isinstance(model, StellaratorNeuralOp)
        # Check if on correct device (CPU default)
        assert next(model.parameters()).device.type == "cpu"

def test_surrogate_ensemble_predict_mean():
    """Test that predict_torch returns the mean of the ensemble."""
    n_ensembles = 2
    surrogate = NeuralOperatorSurrogate(
        min_samples=4, epochs=2, n_ensembles=n_ensembles, hidden_dim=16
    )
    metrics, targets = _training_history(10)
    surrogate.fit(metrics, targets, minimize_objective=False)
    
    # Create a dummy input tensor
    # Schema needs to be captured first via fit, which we did.
    # Let's grab schema from surrogate
    mpol = surrogate._schema.mpol
    ntor = surrogate._schema.ntor
    input_dim = 2 * (mpol + 1) * (2 * ntor + 1) + 1
    
    x = torch.randn(1, input_dim)
    # Set nfp to 3
    x[0, -1] = 3.0
    
    # Get individual predictions manually
    obj_preds = []
    for model in surrogate._models:
        model.eval()
        with torch.no_grad():
            o, _, _ = model(x)
            obj_preds.append(o.item())
            
    mean_expected = np.mean(obj_preds)
    
    # Get ensemble prediction
    with torch.no_grad():
        obj, _, _, _, _, _ = surrogate.predict_torch(x)
        
    assert np.isclose(obj.item(), mean_expected, atol=1e-5)

def test_surrogate_uncertainty_ranking():
    """Test that ranking produces uncertainty metrics."""
    surrogate = NeuralOperatorSurrogate(
        min_samples=4, epochs=2, n_ensembles=3, hidden_dim=16
    )
    metrics, targets = _training_history(10)
    surrogate.fit(metrics, targets, minimize_objective=False)
    
    candidates = [
        {"params": _make_params(0.5), "constraint_distance": 0.0},
        {"params": _make_params(1.5), "constraint_distance": 0.1},
    ]
    
    # Rank with exploration
    ranked = surrogate.rank_candidates(candidates, minimize_objective=False, exploration_ratio=1.0)
    
    assert len(ranked) == 2
    # We can't easily assert exact values because of random init, but we can check structure
    best = ranked[0]
    assert best.expected_value is not None
    # Check that it ran without error
