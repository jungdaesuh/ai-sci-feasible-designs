import unittest
from unittest.mock import MagicMock
import numpy as np
import torch

from ai_scientist.optim import differentiable
from ai_scientist.optim.surrogate_v2 import NeuralOperatorSurrogate
from ai_scientist import config as ai_config
from constellaration.geometry import surface_rz_fourier

class TestDifferentiableOptim(unittest.TestCase):

    def setUp(self):
        self.surrogate = NeuralOperatorSurrogate(device="cpu")
        # Mark as trained so it uses the mock model
        self.surrogate._trained = True
        
        # Mock Config
        self.cfg = MagicMock(spec=ai_config.ExperimentConfig)
        self.cfg.boundary_template = ai_config.BoundaryTemplateConfig(
            n_poloidal_modes=3,
            n_toroidal_modes=3,
            n_field_periods=1,
            base_major_radius=1.0,
            base_minor_radius=0.1,
            perturbation_scale=0.0,
        )
        self.cfg.constraint_weights = ai_config.ConstraintWeightsConfig(
            mhd=1.0, qi=1.0, elongation=1.0
        )

    def test_gradient_descent_updates_params(self):
        # Create a dummy candidate
        # r_cos: [pol, tor] -> [3, 3]
        # Only center index of tor is usually non-zero for axis, but let's just make a valid shape
        r_cos = np.zeros((3, 3))
        r_cos[0, 1] = 1.0 # Major radius
        z_sin = np.zeros((3, 3))
        z_sin[1, 1] = 0.1 # Minor radius
        
        params = {
            "r_cos": r_cos.tolist(),
            "z_sin": z_sin.tolist(),
            "n_field_periods": 1,
            "is_stellarator_symmetric": True
        }
        
        candidates = [{"params": params, "design_hash": "test_hash"}]
        
        # Run optimization
        optimized = differentiable.gradient_descent_on_inputs(
            candidates,
            self.surrogate,
            self.cfg,
            steps=10,
            lr=0.1
        )
        
        self.assertEqual(len(optimized), 1)
        new_params = optimized[0]["params"]
        
        # Check if parameters changed
        r_cos_new = np.array(new_params["r_cos"])
        
        # Since we used a large LR and the mock model provides gradients, 
        # the values should have shifted.
        self.assertFalse(np.allclose(r_cos, r_cos_new), "Parameters should have updated")
        
        # Ensure source is tagged
        self.assertEqual(optimized[0]["source"], "gradient_descent")

    def test_optimize_alm_inner_loop(self):
        # Setup inputs
        x_initial = np.ones(10) # Dummy scaled params
        scale = np.ones(10)
        
        alm_state = {
            "multipliers": np.zeros(3),
            "penalty_parameters": np.ones(3),
            "constraints": np.zeros(3)
        }
        
        # Run optimization
        x_new = differentiable.optimize_alm_inner_loop(
            x_initial=x_initial,
            scale=scale,
            surrogate=self.surrogate,
            alm_state=alm_state,
            steps=5,
            lr=0.1
        )
        
        # Check shape
        self.assertEqual(x_new.shape, x_initial.shape)
        # Check that it moved (Mock model gradients should push it)
        self.assertFalse(np.allclose(x_new, x_initial), "ALM inner loop should update params")

if __name__ == '__main__':
    unittest.main()
