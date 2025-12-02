import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from ai_scientist.prefilter import FeasibilityPrefilter
from ai_scientist.datasets.sampler import load_constellaration_dataset


class TestPrefilterIntegration(unittest.TestCase):
    def test_prefilter_features(self):
        """Test that prefilter uses the correct reduced feature set."""
        pf = FeasibilityPrefilter()
        self.assertEqual(pf.feature_keys, ["aspect_ratio", "max_elongation"])

    def test_prefilter_training_and_prediction(self):
        """Test that prefilter trains and predicts with reduced features."""
        pf = FeasibilityPrefilter()
        # Create dummy data (N=20, features=2)
        X = np.random.rand(20, 2)
        y = np.array([True] * 10 + [False] * 10)

        pf.train(X, y)
        self.assertTrue(pf.is_trained)

        # Predict
        candidates = [
            {"params": {}, "aspect_ratio": 0.5, "max_elongation": 0.5},
            {"params": {}, "aspect_ratio": 0.1, "max_elongation": 0.1},
        ]
        filtered = pf.filter_candidates(
            candidates, threshold=0.0
        )  # Should keep all if threshold 0
        self.assertEqual(len(filtered), 2)


class TestDatasetSamplerP1(unittest.TestCase):
    @patch("ai_scientist.datasets.sampler.load_dataset")
    def test_p1_filter_iota(self, mock_load_dataset):
        """Test that P1 filter enforces iota >= 0.3."""
        # Create a dummy dataset object with a filter method
        mock_ds = MagicMock()
        mock_load_dataset.return_value = mock_ds

        # When filter is called, we capture the function and test it
        def capture_filter(func):
            # Test cases
            # 1. Valid: AR=4.0, Tri=-0.6, Iota=0.4
            c1 = {
                "aspect_ratio": 4.0,
                "average_triangularity": -0.6,
                "edge_rotational_transform_over_n_field_periods": 0.4,
            }
            # 2. Invalid Iota: AR=4.0, Tri=-0.6, Iota=0.2
            c2 = {
                "aspect_ratio": 4.0,
                "average_triangularity": -0.6,
                "edge_rotational_transform_over_n_field_periods": 0.2,
            }
            # 3. Missing Iota
            c3 = {"aspect_ratio": 4.0, "average_triangularity": -0.6}

            assert func(c1) is True, "Should keep valid P1 candidate"
            assert func(c2) is False, "Should drop candidate with iota < 0.3"
            assert func(c3) is False, "Should drop candidate without iota"
            return mock_ds

        mock_ds.filter.side_effect = capture_filter

        load_constellaration_dataset(problem="p1")

        mock_ds.filter.assert_called_once()


if __name__ == "__main__":
    unittest.main()
