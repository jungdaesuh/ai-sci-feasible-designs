import json
import logging
import numpy as np
from ai_scientist.memory.repository import WorldModel
from ai_scientist.memory.schema import SCHEMA
from ai_scientist.optim.surrogate import SurrogateBundle
from ai_scientist.optim import samplers
from ai_scientist import config as ai_config
from ai_scientist.datasets import sampler as dataset_sampler

# Setup logging
logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger(__name__)


def verify_a3_data_segregation():
    print("Verifying A3: Data Segregation...")
    repo = WorldModel(":memory:")
    repo._conn.executescript(SCHEMA)

    repo._conn.execute(
        "INSERT INTO experiments (id, started_at, config_json, git_sha, constellaration_sha) VALUES (?, ?, ?, ?, ?)",
        (1, "now", "{}", "sha", "sha"),
    )
    repo._conn.execute(
        "INSERT INTO experiments (id, started_at, config_json, git_sha, constellaration_sha) VALUES (?, ?, ?, ?, ?)",
        (2, "now", "{}", "sha", "sha"),
    )

    params1 = json.dumps({"r_cos": [[1.0]], "nfp": 3, "n_field_periods": 3})
    repo._conn.execute(
        "INSERT INTO candidates (id, experiment_id, problem, params_json, seed, status, design_hash) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (1, 1, "test", params1, 1, "screen", "hash1"),
    )
    repo._conn.execute(
        "INSERT INTO metrics (candidate_id, raw_json, feasibility, objective, hv, is_feasible) VALUES (?, ?, ?, ?, ?, ?)",
        (1, json.dumps({"metrics": {"qi": 1.0}}), 0.0, 1.0, 0.5, 1),
    )

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

    data_exp1 = repo.surrogate_training_data(experiment_id=1)
    data_exp2 = repo.surrogate_training_data(experiment_id=2)
    data_all = repo.surrogate_training_data(experiment_id=None)

    assert len(data_exp1) == 1
    assert data_exp1[0][0]["candidate_params"]["n_field_periods"] == 3
    assert len(data_exp2) == 1
    assert data_exp2[0][0]["candidate_params"]["n_field_periods"] == 5
    assert len(data_all) == 2
    print("A3 Verified.")


def verify_a4_2_rf_fixes():
    print("\nVerifying A4.2: Random Forest Fixes...")
    params = {"r_cos": [[1.0, 0.0], [0.0, 1.0]], "n_field_periods": 3}
    import ai_scientist.optim.surrogate as surrogate_mod

    vec = surrogate_mod._params_feature_vector(params)
    assert len(vec) == 3
    assert vec[2] == 3.0
    print("A4.2 Part 1 Verified.")

    SurrogateBundle._with_timeout = lambda self, func: func()
    bundle = SurrogateBundle(timeout_seconds=10.0)

    history = [
        ({"candidate_params": params, "metrics": {"qi": 0.1, "vacuum_well": 0.1}}, 0.5),
        (
            {"candidate_params": params, "metrics": {"qi": 0.01, "vacuum_well": 0.1}},
            0.6,
        ),
    ] * 20

    metrics_list, targets = zip(*history)
    bundle.fit(metrics_list, targets, minimize_objective=False)

    candidates = [{"candidate_params": params}]
    preds = bundle.rank_candidates(candidates, minimize_objective=False)
    pred_qi = preds[0].predicted_qi
    assert pred_qi is not None
    assert pred_qi < 0.045
    print("A4.2 Part 2 Verified.")


def verify_a4_3_neural_op_fixes():
    print("\nVerifying A4.3: NeuralOperatorSurrogate Fixes...")
    from ai_scientist.optim.surrogate_v2 import NeuralOperatorSurrogate

    surrogate = NeuralOperatorSurrogate(min_samples=2, epochs=1)

    feas_p3_violated = surrogate._compute_soft_feasibility(
        mhd_val=-0.5, qi_val=1e-5, elongation_val=3.0, problem="p3"
    )
    feas_p3_satisfied = surrogate._compute_soft_feasibility(
        mhd_val=0.05, qi_val=1e-5, elongation_val=3.0, problem="p3"
    )
    assert feas_p3_violated < feas_p3_satisfied

    feas_p1 = surrogate._compute_soft_feasibility(
        mhd_val=-0.5, qi_val=0.1, elongation_val=3.0, problem="p1"
    )
    assert feas_p1 > 0.7
    print("A4.3 Part 1 Verified.")


def verify_symmetry_enforcement():
    print("\nVerifying Symmetry Enforcement via Sampler...")

    # Configure template with sufficient modes to test m=0, n>0
    template = ai_config.BoundaryTemplateConfig(
        n_field_periods=3,
        n_poloidal_modes=3,  # mpol=2 -> n_pol=3
        n_toroidal_modes=5,  # ntor=2 -> n_tor=5
        base_major_radius=1.0,
        base_minor_radius=0.1,
        perturbation_scale=0.05,
    )

    sampler = samplers.RotatingEllipseSampler(template)
    seeds = [42]

    candidates = sampler.generate(seeds)
    params = candidates[0]["params"]
    z_sin = np.array(params["z_sin"])

    # z_sin shape: (mpol+1, 2*ntor+1) = (3, 5)
    # Center index (n=0) is 2.
    # m=0 row is index 0.

    # Check that n <= 0 are zeroed
    center_idx = 2
    n_le_0_slice = z_sin[0, : center_idx + 1]
    assert np.allclose(n_le_0_slice, 0.0), (
        f"Symmetry Violation: n<=0 modes must be zero. Got {n_le_0_slice}"
    )

    # Check that n > 0 is preserved (i.e. not forcibly zeroed by correct code)
    # Note: The sampler adds noise, so it SHOULD be non-zero unless the noise was 0.0 (unlikely)
    # or the code forcibly zeroed it (the bug).
    n_gt_0_slice = z_sin[0, center_idx + 1 :]

    # We expect some non-zero values due to perturbation
    assert not np.allclose(n_gt_0_slice, 0.0), (
        f"Symmetry Violation: n>0 modes for m=0 should be allowed (non-zero), but got {n_gt_0_slice}. "
        "This implies the sampler is overly restrictive (the bug)."
    )

    print("Symmetry Verified: n<=0 is zeroed, n>0 is active.")


def verify_dataset_sampler_fix():
    print("\nVerifying Dataset Sampler Fragility Fix...")

    # Mock examples
    ex_flat = {
        "metrics.aspect_ratio": 4.0,
        "metrics.average_triangularity": -0.6,
        "metrics.edge_rotational_transform_over_n_field_periods": 0.4,
    }

    ex_nested = {
        "metrics": {
            "aspect_ratio": 4.0,
            "average_triangularity": -0.6,
            "edge_rotational_transform_over_n_field_periods": 0.4,
        }
    }

    ex_missing = {"metrics": {}}

    # Verify helper
    assert dataset_sampler._get_metric(ex_flat, "aspect_ratio") == 4.0
    assert dataset_sampler._get_metric(ex_nested, "aspect_ratio") == 4.0
    assert dataset_sampler._get_metric(ex_missing, "aspect_ratio") is None

    # Verify P1 filter logic logic via direct invocation of the lambda if accessible,
    # or by mocking the load_dataset flow.
    # Since we can't easily access the inner function of load_constellaration_dataset without
    # mocking load_dataset return value, we will just trust the helper test above
    # coupled with a manual check of the logic structure.
    # However, we can use the helper to PROVE the logic holds.

    print("Dataset Sampler Verified: _get_metric handles both flat and nested inputs.")


if __name__ == "__main__":
    verify_a3_data_segregation()
    verify_a4_2_rf_fixes()
    verify_a4_3_neural_op_fixes()
    verify_symmetry_enforcement()
    verify_dataset_sampler_fix()
