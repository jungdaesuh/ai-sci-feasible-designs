import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from ai_scientist.prefilter import FeasibilityPrefilter


@pytest.fixture
def prefilter():
    return FeasibilityPrefilter()


def test_initialization(prefilter):
    assert prefilter.model is None
    assert prefilter.is_trained is False
    assert "aspect_ratio" in prefilter.feature_keys


def test_train_insufficient_data(prefilter):
    X = np.random.rand(5, 4)
    y = np.array([0, 1, 0, 1, 0])
    prefilter.train(X, y)
    assert prefilter.model is None
    assert prefilter.is_trained is False


def test_train_single_class(prefilter):
    X = np.random.rand(20, 4)
    y = np.zeros(20)
    prefilter.train(X, y)
    assert prefilter.model is None
    assert prefilter.is_trained is False


def test_train_success(prefilter):
    X = np.random.rand(20, 4)
    y = np.array([0] * 10 + [1] * 10)

    with patch("ai_scientist.prefilter.RandomForestClassifier") as mock_rf:
        prefilter.train(X, y)
        mock_rf.assert_called_once()
        mock_rf.return_value.fit.assert_called_once()
        assert prefilter.model is not None
        assert prefilter.is_trained is True


def test_predict_feasible_untrained(prefilter):
    X = np.random.rand(5, 4)
    preds = prefilter.predict_feasible(X)
    assert np.all(preds)  # Should return all True if untrained


def test_predict_feasible_trained(prefilter):
    prefilter.model = MagicMock()
    # Mock predict_proba to return high prob for first sample, low for second
    # Probs are [prob_0, prob_1]
    prefilter.model.predict_proba.return_value = np.array([[0.1, 0.9], [0.8, 0.2]])

    X = np.random.rand(2, 4)
    preds = prefilter.predict_feasible(X, threshold=0.5)

    assert preds[0]
    assert not preds[1]


def test_filter_candidates(prefilter):
    candidates = [
        {
            "aspect_ratio": 1.0,
            "edge_rotational_transform_over_n_field_periods": 0.5,
            "edge_magnetic_mirror_ratio": 0.1,
            "max_elongation": 2.0,
            "id": 1,
        },
        {
            "aspect_ratio": 1.0,
            "edge_rotational_transform_over_n_field_periods": 0.5,
            "edge_magnetic_mirror_ratio": 0.1,
            "max_elongation": 2.0,
            "id": 2,
        },
        {"incomplete": True, "id": 3},  # Should be kept
    ]

    # Untrained -> Keep all
    filtered = prefilter.filter_candidates(candidates)
    assert len(filtered) == 3

    # Trained
    prefilter.model = MagicMock()
    # 2 valid candidates. Let's say 1st is feasible, 2nd is not.
    prefilter.model.predict_proba.return_value = np.array([[0.1, 0.9], [0.8, 0.2]])

    filtered = prefilter.filter_candidates(candidates)

    # Expect: id=1 (feasible), id=3 (incomplete/skipped filtering)
    assert len(filtered) == 2
    ids = [c["id"] for c in filtered]
    assert 1 in ids
    assert 3 in ids
    assert 2 not in ids
