import unittest
from unittest.mock import MagicMock, patch

from ai_scientist.datasets.sampler import load_constellaration_dataset


class TestDatasetSampler(unittest.TestCase):
    @patch("ai_scientist.datasets.sampler.load_dataset")
    def test_load_dataset_basic(self, mock_load_dataset):
        """Test basic dataset loading without filtering."""
        mock_ds = MagicMock()
        mock_load_dataset.return_value = mock_ds

        ds = load_constellaration_dataset(split="train")

        mock_load_dataset.assert_called_once_with(
            "proxima-fusion/constellaration", split="train"
        )
        self.assertEqual(ds, mock_ds)

    @patch("ai_scientist.datasets.sampler.load_dataset")
    def test_load_dataset_p1_filtering(self, mock_load_dataset):
        """Test P1 filtering logic."""
        # Create a mock dataset that behaves like a list of dicts for filtering
        mock_data = [
            # Valid P1
            {"aspect_ratio": 4.0, "average_triangularity": -0.6},
            # Invalid: AR too high
            {"aspect_ratio": 4.2, "average_triangularity": -0.6},
            # Invalid: Triangularity too high (not negative enough)
            {"aspect_ratio": 4.0, "average_triangularity": -0.4},
        ]

        # Mock the filter method to actually run the lambda
        def mock_filter(callback):
            return [x for x in mock_data if callback(x)]

        mock_ds = MagicMock()
        mock_ds.filter.side_effect = mock_filter
        mock_load_dataset.return_value = mock_ds

        filtered_ds = load_constellaration_dataset(split="train", problem="p1")

        self.assertEqual(len(filtered_ds), 1)
        self.assertEqual(filtered_ds[0]["aspect_ratio"], 4.0)

    @patch("ai_scientist.datasets.sampler.load_dataset")
    def test_load_dataset_p2_filtering(self, mock_load_dataset):
        """Test P2 filtering logic."""
        mock_data = [
            # Valid P2
            {
                "aspect_ratio": 9.0,
                "edge_rotational_transform_over_n_field_periods": 0.3,
                "edge_magnetic_mirror_ratio": 0.1,
                "max_elongation": 4.0,
            },
            # Invalid: AR too high
            {
                "aspect_ratio": 11.0,
                "edge_rotational_transform_over_n_field_periods": 0.3,
                "edge_magnetic_mirror_ratio": 0.1,
                "max_elongation": 4.0,
            },
            # Invalid: Iota too low
            {
                "aspect_ratio": 9.0,
                "edge_rotational_transform_over_n_field_periods": 0.2,
                "edge_magnetic_mirror_ratio": 0.1,
                "max_elongation": 4.0,
            },
        ]

        def mock_filter(callback):
            return [x for x in mock_data if callback(x)]

        mock_ds = MagicMock()
        mock_ds.filter.side_effect = mock_filter
        mock_load_dataset.return_value = mock_ds

        filtered_ds = load_constellaration_dataset(split="train", problem="p2")

        self.assertEqual(len(filtered_ds), 1)
        self.assertEqual(filtered_ds[0]["aspect_ratio"], 9.0)
