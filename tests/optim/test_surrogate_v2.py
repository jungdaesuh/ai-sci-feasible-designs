import numpy as np
import pytest

try:
    import torch

    from ai_scientist.optim.surrogate_v2 import (
        NeuralOperatorSurrogate,
        StellaratorNeuralOp,
    )
except ImportError as e:
    print(f"DEBUG: Import failed: {e}")
    pytest.skip(f"PyTorch not available: {e}", allow_module_level=True)


def _make_params(scale: float, mpol: int = 3, ntor: int = 3) -> dict:
    matrix = np.full((mpol + 1, 2 * ntor + 1), scale, dtype=float)
    return {
        "r_cos": matrix.tolist(),
        "z_sin": (matrix * 0.1).tolist(),
        "n_field_periods": 3,
        "is_stellarator_symmetric": True,
    }


def _training_history(rows: int) -> tuple[list[dict], list[float]]:
    metrics_list: list[dict] = []
    targets: list[float] = []
    for idx in range(rows):
        params = _make_params(float(idx))
        # Ensure metrics structure matches expectation in fit()
        metrics_payload = {
            "vacuum_well": 0.1 * idx,
            "qi": 1.0 / (idx + 1.0),
            "max_elongation": 2.0 + 0.1 * idx,
        }
        metrics_list.append({"candidate_params": params, "metrics": metrics_payload})
        targets.append(float(idx))
    return metrics_list, targets


def test_neural_surrogate_init():
    surrogate = NeuralOperatorSurrogate(min_samples=2, epochs=1)
    assert not surrogate._trained
    assert len(surrogate._models) == 0


def test_neural_surrogate_fit_and_predict():
    surrogate = NeuralOperatorSurrogate(
        min_samples=4, epochs=5, batch_size=2, learning_rate=0.01
    )
    metrics, targets = _training_history(8)

    # Test fit
    surrogate.fit(metrics, targets, minimize_objective=False, cycle=1)
    assert surrogate._trained
    assert len(surrogate._models) > 0
    assert isinstance(surrogate._models[0], StellaratorNeuralOp)

    # Check schema capture
    assert surrogate._schema is not None
    assert surrogate._schema.mpol >= 0

    # Test rank/predict
    candidates = [
        {"params": _make_params(0.5)},
        {"params": _make_params(2.0)},
    ]

    ranked = surrogate.rank_candidates(candidates, minimize_objective=False)
    assert len(ranked) == 2
    assert ranked[0].predicted_objective is not None

    # Check prediction output structure
    pred = ranked[0]
    assert isinstance(pred.predicted_mhd, float)
    assert isinstance(pred.predicted_qi, float)
    assert isinstance(pred.predicted_elongation, float)


def test_neural_surrogate_cold_start():
    surrogate = NeuralOperatorSurrogate(min_samples=100)
    metrics, targets = _training_history(10)
    surrogate.fit(metrics, targets, minimize_objective=False)
    assert not surrogate._trained

    candidates = [{"params": _make_params(1.0)}]
    ranked = surrogate.rank_candidates(candidates, minimize_objective=False)
    assert len(ranked) == 1
    # Should return 0.0 scores
    assert ranked[0].predicted_objective == 0.0


def test_model_forward_shape():
    # Test the model module directly
    mpol, ntor = 3, 3
    model = StellaratorNeuralOp(mpol=mpol, ntor=ntor)

    # Input dim includes +1 for n_field_periods
    spectral_dim = 2 * (mpol + 1) * (2 * ntor + 1)
    input_dim = spectral_dim + 1
    batch_size = 4

    x = torch.randn(batch_size, input_dim)
    # Set nfp (last column) to valid positive integers
    x[:, -1] = torch.randint(1, 5, (batch_size,)).float()

    # Issue #1 FIX: Model now returns 6 outputs (obj, mhd, qi, iota, mirror_ratio, flux_compression)
    obj, mhd, qi, iota, mirror, flux = model(x)

    assert obj.shape == (batch_size,)
    assert mhd.shape == (batch_size,)
    assert qi.shape == (batch_size,)
    assert iota.shape == (batch_size,)
    assert mirror.shape == (batch_size,)
    assert flux.shape == (batch_size,)


class TestMultiConstraintFeasibility:
    """Tests for B7 fix: multi-constraint feasibility check."""

    def test_soft_feasibility_all_satisfied(self):
        """Test feasibility when all constraints are satisfied."""
        surrogate = NeuralOperatorSurrogate(min_samples=4, epochs=2)
        metrics, targets = _training_history(8)
        surrogate.fit(metrics, targets, minimize_objective=False)

        # Feasible values
        prob = surrogate._compute_soft_feasibility(
            mhd_val=0.05,  # vacuum_well >= 0
            qi_val=1e-5,  # log10(1e-5) = -5 < -3.5 (P3 threshold)
            elongation_val=3.0,  # < 5.0 (P2 threshold)
            problem="p3",
        )

        # Should be high probability
        assert prob > 0.5, f"Expected prob > 0.5, got {prob}"

    def test_soft_feasibility_vacuum_well_violated(self):
        """Test feasibility when vacuum_well is violated."""
        surrogate = NeuralOperatorSurrogate(min_samples=4, epochs=2)
        metrics, targets = _training_history(8)
        surrogate.fit(metrics, targets, minimize_objective=False)

        # Negative vacuum_well (MHD unstable)
        prob = surrogate._compute_soft_feasibility(
            mhd_val=-0.5,  # vacuum_well < 0 (violated)
            qi_val=1e-5,
            elongation_val=3.0,
            problem="p3",
        )

        # Should be lower probability due to violation
        prob_satisfied = surrogate._compute_soft_feasibility(
            mhd_val=0.05,
            qi_val=1e-5,
            elongation_val=3.0,
            problem="p3",
        )
        assert prob < prob_satisfied, "Violated should have lower prob"

    def test_soft_feasibility_qi_violated(self):
        """Test feasibility when QI residual is violated."""
        surrogate = NeuralOperatorSurrogate(min_samples=4, epochs=2)
        metrics, targets = _training_history(8)
        surrogate.fit(metrics, targets, minimize_objective=False)

        # High QI residual (violated for P3: threshold is log10(qi) <= -3.5)
        prob_bad_qi = surrogate._compute_soft_feasibility(
            mhd_val=0.05,
            qi_val=0.1,  # log10(0.1) = -1 > -3.5 (violated)
            elongation_val=3.0,
            problem="p3",
        )

        prob_good_qi = surrogate._compute_soft_feasibility(
            mhd_val=0.05,
            qi_val=1e-5,  # log10(1e-5) = -5 < -3.5 (satisfied)
            elongation_val=3.0,
            problem="p3",
        )

        assert prob_bad_qi < prob_good_qi, "Bad QI should have lower prob"

    def test_soft_feasibility_elongation_p2_only(self):
        """Test that elongation constraint only applies to P2."""
        surrogate = NeuralOperatorSurrogate(min_samples=4, epochs=2)
        metrics, targets = _training_history(8)
        surrogate.fit(metrics, targets, minimize_objective=False)

        # High elongation (violated for P2: threshold is 5.0)
        prob_p2 = surrogate._compute_soft_feasibility(
            mhd_val=0.05,
            qi_val=1e-5,
            elongation_val=6.0,  # > 5.0 (violated for P2)
            problem="p2",
        )

        prob_p3 = surrogate._compute_soft_feasibility(
            mhd_val=0.05,
            qi_val=1e-5,
            elongation_val=6.0,  # P3 has no elongation constraint
            problem="p3",
        )

        # P2 should have lower prob due to elongation violation
        assert prob_p2 < prob_p3, "P2 should penalize high elongation"

    def test_rank_candidates_with_problem_parameter(self):
        """Test that rank_candidates accepts problem parameter."""
        surrogate = NeuralOperatorSurrogate(min_samples=4, epochs=5, batch_size=2)
        metrics, targets = _training_history(8)
        surrogate.fit(metrics, targets, minimize_objective=False)

        candidates = [
            {"params": _make_params(0.5)},
            {"params": _make_params(2.0)},
        ]

        # Test with different problem types
        ranked_p2 = surrogate.rank_candidates(
            candidates, minimize_objective=False, problem="p2"
        )
        ranked_p3 = surrogate.rank_candidates(
            candidates, minimize_objective=False, problem="p3"
        )

        assert len(ranked_p2) == 2
        assert len(ranked_p3) == 2

        # Both should have prob_feasible set
        assert 0 <= ranked_p2[0].prob_feasible <= 1
        assert 0 <= ranked_p3[0].prob_feasible <= 1


class TestElongationIsoperimetricUsed:
    """Tests for B5 fix: verify elongation_isoperimetric is used in ranking."""

    def test_rank_candidates_uses_isoperimetric_elongation(self):
        """Verify rank_candidates uses elongation_isoperimetric."""
        from ai_scientist.optim import geometry

        surrogate = NeuralOperatorSurrogate(min_samples=4, epochs=5, batch_size=2)
        metrics, targets = _training_history(8)
        surrogate.fit(metrics, targets, minimize_objective=False)

        # Create candidate with known elongation
        mpol, ntor = 3, 3
        R_major = 10.0
        kappa = 2.0  # Elongation

        r_cos = np.zeros((mpol + 1, 2 * ntor + 1))
        z_sin = np.zeros((mpol + 1, 2 * ntor + 1))
        r_cos[0, ntor] = R_major
        r_cos[1, ntor] = 1.0
        z_sin[1, ntor] = kappa

        params = {
            "r_cos": r_cos.tolist(),
            "z_sin": z_sin.tolist(),
            "n_field_periods": 3,
        }

        candidates = [{"params": params}]
        ranked = surrogate.rank_candidates(candidates, minimize_objective=False)

        # Compute expected elongation using isoperimetric method
        r_cos_t = torch.tensor(r_cos).unsqueeze(0).float()
        z_sin_t = torch.tensor(z_sin).unsqueeze(0).float()
        expected_elo = geometry.elongation_isoperimetric(r_cos_t, z_sin_t, 3)

        # The predicted elongation should match the isoperimetric calculation
        assert np.isclose(
            ranked[0].predicted_elongation, expected_elo.item(), rtol=0.01
        ), f"Expected {expected_elo.item()}, got {ranked[0].predicted_elongation}"


if __name__ == "__main__":
    # Manual run
    test_neural_surrogate_fit_and_predict()
    print("Tests passed!")
