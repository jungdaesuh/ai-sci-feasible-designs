import hypothesis
from hypothesis import strategies as st
import numpy as np


@hypothesis.given(
    predictions=st.lists(
        st.lists(st.floats(-10, 10, allow_nan=False), min_size=3, max_size=3),
        min_size=2,
        max_size=5,
    ),
)
def test_ensemble_uncertainty_positive(predictions):
    """Ensemble uncertainty should be non-negative."""
    predictions = np.array(predictions)  # shape: (n_models, n_outputs)
    variance = predictions.var(axis=0)
    std = np.sqrt(variance)

    assert np.all(std >= 0)


@hypothesis.given(
    predictions=st.lists(
        st.lists(st.floats(-10, 10, allow_nan=False), min_size=3, max_size=3),
        min_size=3,
        max_size=10,
    ),
)
def test_ensemble_variance_decreases_with_agreement(predictions):
    """When models agree, variance should be low."""
    predictions = np.array(predictions)
    variance = predictions.var(axis=0)

    # If all predictions are similar (within 0.1), variance should be small
    range_per_output = np.ptp(predictions, axis=0)
    for i, (rng, var) in enumerate(zip(range_per_output, variance)):
        if rng < 0.1:
            assert var < 0.01, f"Output {i}: range={rng}, variance={var}"
