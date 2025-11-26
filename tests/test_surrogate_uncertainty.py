
import pytest
import torch
import numpy as np
from ai_scientist.optim.surrogate_v2 import NeuralOperatorSurrogate
from ai_scientist.optim.surrogate import SurrogatePrediction

def test_surrogate_uncertainty_execution():
    # 1. Setup Dummy Data
    # mpol=1, ntor=1 -> (1+1)*(2*1+1) = 2*3 = 6 coeffs per surface
    # 2 surfaces (r, z) -> 12 coeffs
    # +1 for nfp -> 13 input dim
    
    surrogate = NeuralOperatorSurrogate(
        min_samples=5,
        epochs=5, # quick train
        n_ensembles=5
    )
    
    metrics_list = []
    targets = []
    
    for i in range(10):
        # Create dummy params
        params = {
            "n_field_periods": 3,
            "rc00": 1.0 + np.random.normal(0, 0.1),
            # Add other dummy coeffs to match flattening if needed, 
            # but structured_flatten handles incomplete dicts by schema or robustly?
            # We need to match the schema expectation. 
            # Let's just rely on structured_flatten to create a schema from the first one.
            # To be safe, let's use a fixed set of keys.
            "coeffs": [0.1] * 12 # This might not work if structured_flatten expects specific keys
        }
        # Actually tools.structured_flatten works on specific keys usually.
        # Let's look at how it's used.
        # For the test, we can mock tools.structured_flatten or provide a dense dict.
        
        # Let's construct a proper detailed dict
        detailed_params = {"n_field_periods": 3}
        for m in range(2): # mpol=1
            for n in range(-1, 2): # ntor=1
                detailed_params[f"rc{m}{n}"] = np.random.randn()
                detailed_params[f"zs{m}{n}"] = np.random.randn()
                
        metrics_list.append({
            "candidate_params": detailed_params,
            "metrics": {"vacuum_well": 0.1, "qi": 0.5, "max_elongation": 2.0}
        })
        targets.append(float(i))

    # 2. Train
    surrogate.fit(metrics_list, targets, minimize_objective=True)
    
    assert surrogate._trained
    
    # 3. Rank
    # Create a candidate
    candidate_params = metrics_list[0]["candidate_params"]
    candidates = [{"candidate_params": candidate_params, "constraint_distance": 0.0}]
    
    # Run with exploration
    preds = surrogate.rank_candidates(candidates, minimize_objective=True, exploration_ratio=1.0)
    
    assert len(preds) == 1
    pred = preds[0]
    
    print(f"Predicted Objective: {pred.predicted_objective}")
    print(f"Expected Value (Score): {pred.expected_value}")
    
    # Check that we didn't crash and got numbers
    assert isinstance(pred.predicted_objective, float)
    
    # With dropout, if we run multiple times, the result of rank_candidates (which is an average) 
    # should be stable, but the *std dev* inside (uncertainty) contributes to the score.
    
    # To verify uncertainty is actually being added:
    # Run with exploration_ratio=0.0
    preds_no_exp = surrogate.rank_candidates(candidates, minimize_objective=True, exploration_ratio=0.0)
    score_no_exp = preds_no_exp[0].expected_value
    
    # Run with exploration_ratio=10.0
    preds_exp = surrogate.rank_candidates(candidates, minimize_objective=True, exploration_ratio=100.0)
    score_exp = preds_exp[0].expected_value
    
    # The score should be different (and likely higher if uncertainty > 0)
    # Note: we subtract objective if minimize=True. 
    # Score = (Prob - Dist) + Weight * Uncertainty - Objective
    # If Uncertainty > 0, Score_exp should be > Score_no_exp + (difference in weight * uncertainty)
    # Since objective is same (average of same MC process approx), 
    # we just check they are different.
    
    assert score_exp != score_no_exp, "Exploration ratio should affect score via uncertainty"

if __name__ == "__main__":
    test_surrogate_uncertainty_execution()
