import time

import numpy as np
import pytest

from ai_scientist.optim.surrogate import SurrogateBundle


def _make_params(scale: float) -> dict:
    size = 6
    matrix = np.full((size, size), scale, dtype=float)
    return {
        "r_cos": matrix.tolist(),
        "z_sin": (matrix * 0.1).tolist(),
        "n_field_periods": 3,
        "is_stellarator_symmetric": True,
    }


def _training_history(rows: int) -> tuple[list[dict], list[float]]:
    metrics: list[dict] = []
    targets: list[float] = []
    for idx in range(rows):
        params = _make_params(float(idx))
        feasible = 0.0 if idx % 2 == 0 else 1.0
        metrics.append({"candidate_params": params, "feasibility": feasible})
        targets.append(float(idx))
    return metrics, targets


def test_rank_prefers_feasible_in_maximize():
    bundle = SurrogateBundle()
    metrics, targets = _training_history(8)
    bundle.fit(metrics, targets, minimize_objective=False, cycle=1)

    class _StubClassifier:
        def predict_proba(self, matrix):
            return np.array([[0.1, 0.9], [0.9, 0.1]], dtype=float)

    class _StubRegressor:
        def predict(self, matrix):
            return np.array([3.0, 5.0], dtype=float)

    class _IdentityScaler:
        def transform(self, matrix):
            return np.asarray(matrix, dtype=float)

    bundle._classifier = _StubClassifier()
    bundle._regressor = _StubRegressor()
    bundle._scaler = _IdentityScaler()

    candidates = [
        {"params": _make_params(0.5)},
        {"params": _make_params(2.0)},
    ]

    ranked = bundle.rank_candidates(candidates, minimize_objective=False)
    assert ranked[0].metadata["params"]["r_cos"][0][0] == 0.5
    assert ranked[0].expected_value > ranked[1].expected_value


def test_rank_prefers_feasible_in_minimize():
    bundle = SurrogateBundle()
    metrics, targets = _training_history(8)
    bundle.fit(metrics, targets, minimize_objective=True, cycle=1)

    class _StubClassifier:
        def predict_proba(self, matrix):
            return np.array([[0.1, 0.9], [0.8, 0.2]], dtype=float)

    class _StubRegressor:
        def predict(self, matrix):
            return np.array([2.0, 20.0], dtype=float)

    class _IdentityScaler:
        def transform(self, matrix):
            return np.asarray(matrix, dtype=float)

    bundle._classifier = _StubClassifier()
    bundle._regressor = _StubRegressor()
    bundle._scaler = _IdentityScaler()

    candidates = [
        {"params": _make_params(0.25)},
        {"params": _make_params(4.0)},
    ]

    ranked = bundle.rank_candidates(candidates, minimize_objective=True)
    assert ranked[0].metadata["params"]["r_cos"][0][0] == 0.25
    assert ranked[0].expected_value > ranked[1].expected_value


@pytest.mark.slow
def test_fit_and_predict_timing_budget():
    bundle = SurrogateBundle(timeout_seconds=1.0)
    metrics, targets = _training_history(200)

    start = time.perf_counter()
    bundle.fit(metrics, targets, minimize_objective=False, cycle=1)
    predictions = bundle.rank_candidates(
        [{"params": _make_params(float(i))} for i in range(10)],
        minimize_objective=False,
    )
    elapsed = time.perf_counter() - start

    assert predictions, "ranking should return at least one entry"
    # NOTE: Timing budget increased from 0.2s to 0.5s to accommodate RF capacity
    # increase (n_estimators 12→100, max_depth 6→12) for better accuracy
    assert elapsed < 0.5, f"fit+predict should stay under 0.5s, got {elapsed:.3f}s"
