import unittest
import numpy as np
from ai_scientist.prefilter import FeasibilityPrefilter


class TestFeasibilityPrefilter(unittest.TestCase):
    def test_initialization(self):
        pf = FeasibilityPrefilter()
        self.assertFalse(pf.is_trained)
        self.assertIsNone(pf.model)

    def test_train_and_predict(self):
        pf = FeasibilityPrefilter()

        # Create dummy data
        # Feature 1: feasible if > 0.5
        X = np.random.rand(100, 2)
        y = (X[:, 0] > 0.5).astype(bool)

        pf.train(X, y)
        self.assertTrue(pf.is_trained)

        # Test prediction
        X_test = np.array([[0.8, 0.1], [0.2, 0.1]])
        preds = pf.predict_feasible(X_test)

        self.assertTrue(preds[0])  # Should be feasible
        self.assertFalse(preds[1])  # Should be infeasible

    def test_filter_candidates(self):
        pf = FeasibilityPrefilter()

        # Train simple model
        X = np.random.rand(50, 2)
        y = (X[:, 0] > 0.5).astype(bool)
        pf.train(X, y)

        # Candidates
        candidates = [
            {"params": {"r_cos": [0.8], "z_sin": [0.1]}, "id": 1},  # Feasible
            {"params": {"r_cos": [0.2], "z_sin": [0.1]}, "id": 2},  # Infeasible
        ]

        filtered = pf.filter_candidates(candidates)

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["id"], 1)

    def test_untrained_pass_through(self):
        pf = FeasibilityPrefilter()
        candidates = [{"id": 1}, {"id": 2}]

        # Should return all if not trained
        filtered = pf.filter_candidates(candidates)
        self.assertEqual(len(filtered), 2)
