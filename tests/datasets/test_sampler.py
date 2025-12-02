import pytest
from unittest.mock import MagicMock, patch
from ai_scientist.datasets.sampler import load_constellaration_dataset


@pytest.fixture
def mock_load_dataset():
    with patch("ai_scientist.datasets.sampler.load_dataset") as mock:
        yield mock


def test_load_dataset_no_filter(mock_load_dataset):
    """Test loading dataset without filtering."""
    mock_ds = MagicMock()
    mock_load_dataset.return_value = mock_ds

    ds = load_constellaration_dataset(split="train")

    mock_load_dataset.assert_called_once_with(
        "proxima-fusion/constellaration", split="train"
    )
    assert ds == mock_ds
    mock_ds.filter.assert_not_called()


def test_load_dataset_p1_filter(mock_load_dataset):
    """Test loading dataset with P1 filtering."""
    mock_ds = MagicMock()
    mock_filtered_ds = MagicMock()
    mock_ds.filter.return_value = mock_filtered_ds
    mock_load_dataset.return_value = mock_ds

    ds = load_constellaration_dataset(split="test", problem="p1")

    mock_load_dataset.assert_called_once_with(
        "proxima-fusion/constellaration", split="test"
    )
    mock_ds.filter.assert_called_once()
    assert ds == mock_filtered_ds

    # Verify filter lambda logic
    filter_func = mock_ds.filter.call_args[0][0]

    # Valid P1 example
    valid_ex = {
        "aspect_ratio": 4.05,
        "average_triangularity": -0.65,
        "edge_rotational_transform_over_n_field_periods": 0.35,
    }
    assert filter_func(valid_ex) is True

    # Invalid aspect ratio
    invalid_ar = {"aspect_ratio": 5.0, "average_triangularity": -0.65}
    assert filter_func(invalid_ar) is False

    # Invalid triangularity
    invalid_tri = {"aspect_ratio": 4.05, "average_triangularity": -0.1}
    assert filter_func(invalid_tri) is False


def test_load_dataset_p2_filter(mock_load_dataset):
    """Test loading dataset with P2 filtering."""
    mock_ds = MagicMock()
    mock_filtered_ds = MagicMock()
    mock_ds.filter.return_value = mock_filtered_ds
    mock_load_dataset.return_value = mock_ds

    ds = load_constellaration_dataset(split="train", problem="p2")

    mock_load_dataset.assert_called_once_with(
        "proxima-fusion/constellaration", split="train"
    )
    # P2 filtering should now be applied
    mock_ds.filter.assert_called_once()
    assert ds == mock_filtered_ds

    # Verify filter lambda logic
    filter_func = mock_ds.filter.call_args[0][0]

    # Valid P2 example - all constraints satisfied
    valid_ex = {
        "aspect_ratio": 8.0,
        "edge_rotational_transform_over_n_field_periods": 0.3,
        "max_elongation": 4.0,
        "edge_magnetic_mirror_ratio": 0.15,
    }
    assert filter_func(valid_ex) is True

    # Invalid aspect ratio (too high)
    invalid_ar = {
        "aspect_ratio": 12.0,
        "edge_rotational_transform_over_n_field_periods": 0.3,
        "max_elongation": 4.0,
    }
    assert filter_func(invalid_ar) is False

    # Invalid edge rotational transform (too low)
    invalid_iota = {
        "aspect_ratio": 8.0,
        "edge_rotational_transform_over_n_field_periods": 0.2,
        "max_elongation": 4.0,
    }
    assert filter_func(invalid_iota) is False

    # Invalid max elongation (too high)
    invalid_elong = {
        "aspect_ratio": 8.0,
        "edge_rotational_transform_over_n_field_periods": 0.3,
        "max_elongation": 6.0,
    }
    assert filter_func(invalid_elong) is False

    # Missing fields should be kept (conservative)
    incomplete_ex = {"aspect_ratio": 8.0}
    assert filter_func(incomplete_ex) is True


def test_load_dataset_p1_missing_fields(mock_load_dataset):
    """Test P1 filter handles missing fields correctly."""
    mock_ds = MagicMock()
    mock_filtered_ds = MagicMock()
    mock_ds.filter.return_value = mock_filtered_ds
    mock_load_dataset.return_value = mock_ds

    load_constellaration_dataset(split="train", problem="p1")

    mock_load_dataset.assert_called_once_with(
        "proxima-fusion/constellaration", split="train"
    )
    mock_ds.filter.assert_called_once()

    filter_func = mock_ds.filter.call_args[0][0]

    # Missing aspect_ratio should be rejected
    missing_ar = {"average_triangularity": -0.65}
    assert filter_func(missing_ar) is False

    # Missing triangularity should be rejected
    missing_tri = {"aspect_ratio": 4.05}
    assert filter_func(missing_tri) is False

    # Empty dict should be rejected
    empty = {}
    assert filter_func(empty) is False
