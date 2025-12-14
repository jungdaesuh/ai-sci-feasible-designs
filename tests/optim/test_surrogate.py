import hypothesis
from hypothesis import strategies as st
import numpy as np
import torch
from unittest.mock import MagicMock
from ai_scientist.optim.surrogate_v2 import NeuralOperatorSurrogate


@hypothesis.given(
    predictions=st.lists(
        st.lists(st.floats(-10, 10, allow_nan=False), min_size=3, max_size=3),
        min_size=2,
        max_size=5,
    ),
)
def test_ensemble_uncertainty_positive(predictions):
    """Ensemble uncertainty should be non-negative when using actual surrogate logic.

    This test exercises the actual NeuralOperatorSurrogate.predict_torch() method
    by mocking individual model predictions and verifying that the ensemble
    aggregation (mean/std computation) maintains the invariant: std >= 0.
    """
    # Create surrogate and mock ensemble models to return specific predictions
    surrogate = NeuralOperatorSurrogate()
    surrogate._models = [MagicMock() for _ in range(len(predictions))]

    # Setup mocks: each model returns (obj, mhd, qi, iota) tensors
    for i, model in enumerate(surrogate._models):
        preds_tensor = torch.tensor(predictions[i], dtype=torch.float32)
        # H6 FIX: Model now returns 6 outputs (obj, mhd, qi, iota, mirror, flux)
        model.return_value = (
            preds_tensor,
            torch.zeros_like(preds_tensor),
            torch.zeros_like(preds_tensor),
            torch.zeros_like(preds_tensor),
            torch.zeros_like(preds_tensor),
            torch.zeros_like(preds_tensor),
        )

    # Call the actual predict_torch method to aggregate ensemble predictions
    x = torch.zeros((len(predictions[0]), 10))  # Batch size = len(predictions[0])
    # H6 FIX: predict_torch now returns 12 values (6 metrics × mean/std)
    obj_mean, obj_std, _, _, _, _, _, _, _, _, _, _ = surrogate.predict_torch(x)

    # Verify uncertainty is non-negative
    assert torch.all(obj_std >= 0)

    # Verify calculation matches expected ensemble statistics
    preds_np = np.array(predictions, dtype=np.float32)
    expected_std = np.std(preds_np, axis=0, ddof=1)  # Bessel correction
    calculated_std = obj_std.detach().cpu().numpy()

    assert np.allclose(calculated_std, expected_std, atol=1e-5)


@hypothesis.given(
    predictions=st.lists(
        st.lists(st.floats(-10, 10, allow_nan=False), min_size=3, max_size=3),
        min_size=3,
        max_size=10,
    ),
)
def test_ensemble_variance_decreases_with_agreement(predictions):
    """When ensemble models agree, variance should be low.

    This test exercises the actual NeuralOperatorSurrogate.predict_torch() method
    to verify the property: when all models produce similar predictions
    (range < 0.1), the ensemble variance should be small (< 0.01).
    """
    # Create surrogate and mock ensemble models
    surrogate = NeuralOperatorSurrogate()
    surrogate._models = [MagicMock() for _ in range(len(predictions))]

    # Setup mocks: each model returns 6 outputs (obj, mhd, qi, iota, mirror, flux)
    for i, model in enumerate(surrogate._models):
        preds_tensor = torch.tensor(predictions[i], dtype=torch.float32)
        # H6 FIX: Model now returns 6 outputs (obj, mhd, qi, iota, mirror, flux)
        model.return_value = (
            preds_tensor,
            torch.zeros_like(preds_tensor),
            torch.zeros_like(preds_tensor),
            torch.zeros_like(preds_tensor),
            torch.zeros_like(preds_tensor),
            torch.zeros_like(preds_tensor),
        )

    # Call the actual predict_torch method to aggregate ensemble predictions
    x = torch.zeros((len(predictions[0]), 10))
    # H6 FIX: predict_torch now returns 12 values (6 metrics × mean/std)
    obj_mean, obj_std, _, _, _, _, _, _, _, _, _, _ = surrogate.predict_torch(x)

    variance = obj_std.detach().cpu().numpy() ** 2
    predictions_np = np.array(predictions, dtype=np.float32)

    # When all predictions are similar (range < 0.1), variance should be small
    range_per_output = np.ptp(predictions_np, axis=0)
    for i, (rng, var) in enumerate(zip(range_per_output, variance)):
        if rng < 0.1:
            assert var < 0.01, f"Output {i}: range={rng}, variance={var}"
