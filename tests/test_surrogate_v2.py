import pytest
import numpy as np
try:
    import torch
    from ai_scientist.optim.surrogate_v2 import NeuralOperatorSurrogate, StellaratorNeuralOp
except ImportError as e:
    print(f"DEBUG: Import failed: {e}")
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
        params = _make_params(float(idx))
        # Ensure metrics structure matches expectation in fit()
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

def test_neural_surrogate_init():
    surrogate = NeuralOperatorSurrogate(min_samples=2, epochs=1)
    assert not surrogate._trained
    assert surrogate._model is None

def test_neural_surrogate_fit_and_predict():
    surrogate = NeuralOperatorSurrogate(
        min_samples=4, 
        epochs=5, 
        batch_size=2,
        learning_rate=0.01
    )
    metrics, targets = _training_history(8)
    
    # Test fit
    surrogate.fit(metrics, targets, minimize_objective=False, cycle=1)
    assert surrogate._trained
    assert surrogate._model is not None
    assert isinstance(surrogate._model, StellaratorNeuralOp)
    
    # Check schema capture
    assert surrogate._schema is not None
    assert surrogate._schema.mpol >= 0
    
    # Test rank/predict
    candidates = [
        {"params": _make_params(0.5)},
        {"params": _make_params(2.0)},
    ]
    
    ranked = surrogate.rank_candidates(candidates, minimize_objective=False)
    assert len(ranked) == 2
    assert ranked[0].predicted_objective is not None
    
    # Check prediction output structure
    pred = ranked[0]
    assert isinstance(pred.predicted_mhd, float)
    assert isinstance(pred.predicted_qi, float)
    assert isinstance(pred.predicted_elongation, float)

def test_neural_surrogate_cold_start():
    surrogate = NeuralOperatorSurrogate(min_samples=100)
    metrics, targets = _training_history(10)
    surrogate.fit(metrics, targets, minimize_objective=False)
    assert not surrogate._trained
    
    candidates = [{"params": _make_params(1.0)}]
    ranked = surrogate.rank_candidates(candidates, minimize_objective=False)
    assert len(ranked) == 1
    # Should return 0.0 scores
    assert ranked[0].predicted_objective == 0.0

def test_model_forward_shape():
    # Test the model module directly
    mpol, ntor = 3, 3
    model = StellaratorNeuralOp(mpol=mpol, ntor=ntor)
    
    input_dim = 2 * (mpol + 1) * (2 * ntor + 1)
    batch_size = 4
    x = torch.randn(batch_size, input_dim)
    
    obj, mhd, qi, elong = model(x)
    
    assert obj.shape == (batch_size,)
    assert mhd.shape == (batch_size,)
    assert qi.shape == (batch_size,)
    assert elong.shape == (batch_size,)

if __name__ == "__main__":
    # Manual run
    test_neural_surrogate_fit_and_predict()
    print("Tests passed!")
