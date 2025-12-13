import unittest
from unittest.mock import MagicMock

import numpy as np
import torch.nn as nn

from ai_scientist import config as ai_config
from ai_scientist.optim import differentiable
from ai_scientist.optim.surrogate_v2 import NeuralOperatorSurrogate


class TestDifferentiableOptim(unittest.TestCase):
    def setUp(self):
        self.surrogate = NeuralOperatorSurrogate(device="cpu")
        self.surrogate._trained = True

        # Mock Model
        # We need a model that accepts input and returns 4 values
        # Input shape is (Batch, DenseSize).
        # DenseSize depends on mpol/ntor.
        mpol = 3
        ntor = 3

        # Create a schema mock
        self.schema = MagicMock()
        self.schema.mpol = mpol
        self.schema.ntor = ntor
        self.surrogate._schema = self.schema

        # Create a simple dummy model
        # B3 FIX: Now returns 4 values: obj, mhd, qi, iota
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.mpol = mpol
                self.ntor = ntor
                # Just some learnable params to allow gradients
                self.layer = nn.Linear(1, 4)  # Now 4 outputs

            def forward(self, x):
                # x is (Batch, DenseSize)
                # We ignore x content for shape, but use it for gradient flow
                # Collapse x to 1 dim to feed linear
                val = x.mean(dim=1, keepdim=True)
                out = self.layer(val)
                # return obj, mhd, qi, iota (B3 fix)
                return out[:, 0], out[:, 1], out[:, 2], out[:, 3]

        self.surrogate._models = [DummyModel(), DummyModel()]

        # Mock Config
        self.cfg = MagicMock(spec=ai_config.ExperimentConfig)
        self.cfg.boundary_template = ai_config.BoundaryTemplateConfig(
            n_poloidal_modes=mpol + 1,  # Template uses num modes, not max index
            n_toroidal_modes=ntor * 2 + 1,
            n_field_periods=1,
            base_major_radius=1.0,
            base_minor_radius=0.1,
            perturbation_scale=0.0,
        )
        self.cfg.constraint_weights = ai_config.ConstraintWeightsConfig(
            mhd=1.0, qi=1.0, elongation=1.0
        )
        self.cfg.problem = "p1"  # Add problem attribute for optimization direction

    def test_gradient_descent_updates_params(self):
        # Create a dummy candidate
        # r_cos: [pol, tor] -> [4, 4] (mpol=3, ntor=3, so size is 4, 7?)
        # Template n_pol = 4, n_tor = 7.

        r_cos = np.zeros((4, 7))
        r_cos[0, 3] = 1.0  # Major radius at center
        z_sin = np.zeros((4, 7))
        z_sin[1, 3] = 0.1  # Minor radius

        params = {
            "r_cos": r_cos.tolist(),
            "z_sin": z_sin.tolist(),
            "n_field_periods": 1,
            "is_stellarator_symmetric": True,
        }

        candidates = [{"params": params, "design_hash": "test_hash"}]

        # Run optimization
        optimized = differentiable.gradient_descent_on_inputs(
            candidates, self.surrogate, self.cfg, steps=10, lr=0.1
        )

        self.assertEqual(len(optimized), 1)
        new_params = optimized[0]["params"]

        # Check if parameters changed
        r_cos_new = np.array(new_params["r_cos"])

        # Since we used a large LR and the mock model provides gradients,
        # the values should have shifted.
        self.assertFalse(
            np.allclose(r_cos, r_cos_new), "Parameters should have updated"
        )

        # Ensure source is tagged
        self.assertEqual(optimized[0]["source"], "gradient_descent")

    def test_optimize_alm_inner_loop(self):
        # Setup inputs
        # We need correct size for x_initial.
        # We can use _compute_index_mapping to find size or just guess large enough.
        # But wait, optimize_alm_inner_loop uses _compute_index_mapping inside.
        # We should ensure x_initial matches the expected compact size.

        # Let's use the helper from differentiable to get size
        dense_indices, dense_size = differentiable._compute_index_mapping(
            self.cfg.boundary_template, 3, 3, "cpu"
        )
        # dense_indices is for mapping compact->dense.
        # x_initial is compact vector.
        compact_size = dense_indices.shape[0]

        x_initial = np.ones(compact_size)
        scale = np.ones(compact_size)

        alm_state = {
            "multipliers": np.zeros(3),
            "penalty_parameters": np.ones(3),
            "constraints": np.zeros(3),
        }

        # Run optimization
        x_new = differentiable.optimize_alm_inner_loop(
            x_initial=x_initial,
            scale=scale,
            surrogate=self.surrogate,
            alm_state=alm_state,
            n_field_periods_val=1,
            steps=5,
            lr=0.1,
        )

        # Check shape
        self.assertEqual(x_new.shape, x_initial.shape)
        # Check that it moved
        self.assertFalse(
            np.allclose(x_new, x_initial), "ALM inner loop should update params"
        )

    def test_p3_uses_maximization_direction(self):
        """P3 with HV target should maximize (not minimize)."""
        # P1: minimize (physics objective = elongation)
        self.assertFalse(differentiable._is_maximization_problem("p1"))
        self.assertFalse(
            differentiable._is_maximization_problem("p1", target="objective")
        )

        # P2: maximize (physics objective = gradient)
        self.assertTrue(differentiable._is_maximization_problem("p2"))
        self.assertTrue(
            differentiable._is_maximization_problem("p2", target="objective")
        )

        # P3 with physics objective: minimize (aspect_ratio)
        self.assertFalse(
            differentiable._is_maximization_problem("p3", target="objective")
        )

        # P3 with HV target: MAXIMIZE (key fix!)
        self.assertTrue(differentiable._is_maximization_problem("p3", target="hv"))


if __name__ == "__main__":
    unittest.main()
