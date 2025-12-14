import json
from ai_scientist.memory.repository import WorldModel
from ai_scientist.memory.schema import SCHEMA
from ai_scientist.optim.surrogate import SurrogateBundle


def verify_a3_data_segregation():
    print("Verifying A3: Data Segregation...")
    # Setup in-memory DB by not passing db_path (if supported) or using a temp file
    # WorldModel(db_path: str | Path)
    # If we pass ":memory:", it might work if the class uses it as sqlite path
    repo = WorldModel(":memory:")
    repo._conn.executescript(SCHEMA)

    # Insert experiments
    repo._conn.execute(
        "INSERT INTO experiments (id, started_at, config_json, git_sha, constellaration_sha) VALUES (?, ?, ?, ?, ?)",
        (1, "now", "{}", "sha", "sha"),
    )
    repo._conn.execute(
        "INSERT INTO experiments (id, started_at, config_json, git_sha, constellaration_sha) VALUES (?, ?, ?, ?, ?)",
        (2, "now", "{}", "sha", "sha"),
    )

    # Insert candidates for exp 1
    # Schema: id, experiment_id, problem, params_json, seed, status, design_hash
    params1 = json.dumps({"r_cos": [[1.0]], "nfp": 3, "n_field_periods": 3})
    repo._conn.execute(
        "INSERT INTO candidates (id, experiment_id, problem, params_json, seed, status, design_hash) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (1, 1, "test", params1, 1, "screen", "hash1"),
    )
    # Schema: id, candidate_id, raw_json, feasibility, objective, hv, is_feasible
    repo._conn.execute(
        "INSERT INTO metrics (candidate_id, raw_json, feasibility, objective, hv, is_feasible) VALUES (?, ?, ?, ?, ?, ?)",
        (1, json.dumps({"metrics": {"qi": 1.0}}), 0.0, 1.0, 0.5, 1),
    )

    # Insert candidates for exp 2
    params2 = json.dumps({"r_cos": [[2.0]], "nfp": 5, "n_field_periods": 5})
    repo._conn.execute(
        "INSERT INTO candidates (id, experiment_id, problem, params_json, seed, status, design_hash) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (2, 2, "test", params2, 2, "promote", "hash2"),
    )
    repo._conn.execute(
        "INSERT INTO metrics (candidate_id, raw_json, feasibility, objective, hv, is_feasible) VALUES (?, ?, ?, ?, ?, ?)",
        (2, json.dumps({"metrics": {"qi": 0.5}}), 0.0, 2.0, 0.8, 1),
    )
    repo._conn.commit()

    # Check filtering
    data_exp1 = repo.surrogate_training_data(experiment_id=1)
    data_exp2 = repo.surrogate_training_data(experiment_id=2)
    data_all = repo.surrogate_training_data(experiment_id=None)

    assert len(data_exp1) == 1, f"Exp1 should have 1 item, got {len(data_exp1)}"
    # params is "candidate_params" or "params_json" parsed
    # surrogate_training_data returns (metrics_payload, target)
    # metrics_payload has "candidate_params" injected
    assert data_exp1[0][0]["candidate_params"]["n_field_periods"] == 3

    assert len(data_exp2) == 1, f"Exp2 should have 1 item, got {len(data_exp2)}"
    assert data_exp2[0][0]["candidate_params"]["n_field_periods"] == 5

    assert len(data_all) == 2, f"All data should have 2 items, got {len(data_all)}"

    print("A3 Verified: Data strictly segregated by experiment_id.")


def verify_a4_2_rf_fixes():
    print("\nVerifying A4.2: Random Forest Fixes (Nfp, QI Scaling)...")

    # 1. Check Feature Vector (Nfp inclusion)
    params = {"r_cos": [[1.0, 0.0], [0.0, 1.0]], "n_field_periods": 3}
    import ai_scientist.optim.surrogate as surrogate_mod

    vec = surrogate_mod._params_feature_vector(params)
    # Expect 3 elements: sum, size, nfp
    assert len(vec) == 3, f"Feature vector length should be 3, got {len(vec)}"
    assert vec[2] == 3.0, f"Nfp (last element) should be 3.0, got {vec[2]}"
    print("A4.2 Part 1 Verified: Nfp included in features.")

    # 2. Check QI Log Scaling
    # Monkeypatch to avoid threading issues during verification
    SurrogateBundle._with_timeout = lambda self, func: func()
    bundle = SurrogateBundle(timeout_seconds=10.0)

    # Create training data
    # Point A: QI=0.1 (log10 = -1.0)
    # Point B: QI=0.01 (log10 = -2.0)
    history = [
        (
            {"candidate_params": params, "metrics": {"qi": 0.1, "vacuum_well": 0.1}},
            0.5,
        ),  # Target doesn't matter much for aux
        (
            {"candidate_params": params, "metrics": {"qi": 0.01, "vacuum_well": 0.1}},
            0.6,
        ),
    ]
    # Replicate to ensure split allows training
    history = history * 20

    metrics_list, targets = zip(*history)
    bundle.fit(metrics_list, targets, minimize_objective=False)

    # We construct features using the bundle's vectorizer
    # Prior to fix, it returned 4 (flattened params). Now should be 5 (flattened + nfp).
    # Note: validation set usage requires internal method access
    feats = bundle._vectorize(params)
    print(f"Bundle vectorized shape: {feats.shape}")
    assert feats.shape[0] == 5, (
        f"Expected 5 features (4 coeffs + 1 nfp), got {feats.shape[0]}"
    )

    feats = feats.reshape(1, -1)

    # Verify denormalization in rank_candidates
    candidates = [{"candidate_params": params}]
    preds = bundle.rank_candidates(candidates, minimize_objective=False)
    pred_qi = preds[0].predicted_qi
    assert pred_qi is not None, "predicted_qi should not be None"

    print(f"Predicted QI: {pred_qi}")

    # If logic is correct:
    # Target was log10(qi). Avrg = -1.5.
    # Prediction (log) = -1.5.
    # Denorm = 10^(-1.5) = 10^(-1) * 10^(-0.5) = 0.1 * 0.316 = 0.0316.

    # If logic was OLD (linear training, no denorm):
    # Target 0.055. Pred = 0.055.

    # 0.0316 vs 0.055 is distinguishable.
    # Let's assert it is < 0.04 to confirm log scaling usage.

    assert pred_qi is not None, "predicted_qi should not be None"
    assert pred_qi < 0.045, (
        f"Predicted QI {pred_qi} suggests linear averaging (approx 0.055). Log averaging should be approx 0.0316."
    )
    print("A4.2 Part 2 Verified: QI is log-scaled and denormalized.")


def verify_a4_3_neural_op_fixes():
    print(
        "\nVerifying A4.3: NeuralOperatorSurrogate Fixes (Fidelity, Soft Feasibility)..."
    )
    from ai_scientist.optim.surrogate_v2 import NeuralOperatorSurrogate

    # 1. Check Soft Feasibility Logic
    # P3 SHOULD have vacuum_well constraint (it IS in P3_CONSTRAINT_NAMES).
    # P1 should NOT have vacuum_well constraint (it is NOT in P1_CONSTRAINT_NAMES).
    surrogate = NeuralOperatorSurrogate(min_samples=2, epochs=1)

    # Violate vacuum_well (< 0)
    # Ensure other constraints are met to isolate vacuum_well check

    # P3 with vacuum_well violation - SHOULD be penalized (P3 checks vacuum_well)
    feas_p3_violated = surrogate._compute_soft_feasibility(
        mhd_val=-0.5,  # vacuum_well < 0 (violated)
        qi_val=1e-5,  # Good (log10 = -5 < -3.5)
        elongation_val=3.0,
        problem="p3",
    )

    # P3 with vacuum_well satisfied
    feas_p3_satisfied = surrogate._compute_soft_feasibility(
        mhd_val=0.05,  # vacuum_well > 0 (satisfied)
        qi_val=1e-5,
        elongation_val=3.0,
        problem="p3",
    )

    print(f"P3 Feasibility (vacuum_well=-0.5, violated): {feas_p3_violated:.2f}")
    print(f"P3 Feasibility (vacuum_well=0.05, satisfied): {feas_p3_satisfied:.2f}")

    # P3 should penalize vacuum_well violation
    assert feas_p3_violated < feas_p3_satisfied, (
        f"P3 should penalize vacuum_well violation: got {feas_p3_violated:.2f} >= {feas_p3_satisfied:.2f}"
    )

    # P1 should NOT penalize vacuum_well (it's not a P1 constraint)
    feas_p1_vw_violated = surrogate._compute_soft_feasibility(
        mhd_val=-0.5,  # vacuum_well < 0 - P1 should NOT check this
        qi_val=0.1,  # Dummy (P1 doesn't check QI either)
        elongation_val=3.0,
        problem="p1",
    )
    print(f"P1 Feasibility (vacuum_well=-0.5): {feas_p1_vw_violated:.2f}")

    # P1 should have high feasibility because it doesn't check vacuum_well
    assert feas_p1_vw_violated > 0.7, (
        f"P1 should NOT penalize vacuum_well, but got feasibility {feas_p1_vw_violated:.2f}"
    )

    print("A4.3 Part 1 Verified: Conditional soft feasibility constraints.")

    # 2. Check Fit Execution (Fidelity)
    # Just ensure fit runs with _stage
    # We can't easily verify internal usage of fidelity without deeper inspection,
    # but running without crash is a good sign given we changed training loop.
    surrogate._trained = False

    # Create fake training data
    # NeuralOp expects tensor features.
    # fit(self, metrics_list, target_values, minimize_objective, cycle=None)
    # It extracts params from metrics_list and converts to tensor.
    # candidate_params must be appropriate shape.

    params = {"r_cos": [[1.0, 0.0], [0.0, 1.0]], "nfp": 3, "n_field_periods": 3}
    sample_metrics = {"vacuum_well": 0.1, "qi": 1e-4, "max_elongation": 3.0}
    # _stage determines fidelity: "screen" -> 0, "promote" -> 1
    history = [
        (
            {
                "candidate_params": params,
                "metrics": sample_metrics,
                "feasibility": 0.0,
                "_stage": "screen",
            },
            0.5,
        ),
        (
            {
                "candidate_params": params,
                "metrics": sample_metrics,
                "feasibility": 0.0,
                "_stage": "promote",
            },
            0.6,
        ),
    ]
    metrics_list, targets = zip(*history)

    try:
        surrogate.fit(metrics_list, targets, minimize_objective=False)
        print("A4.3 Part 2 Verified: fit() runs with _stage presence.")
    except Exception as e:
        print(f"A4.3 Part 2 Failed: fit() crashed: {e}")
        raise e


if __name__ == "__main__":
    verify_a3_data_segregation()
    verify_a4_2_rf_fixes()
    verify_a4_3_neural_op_fixes()
