import numpy as np

from ai_scientist import tools


def test_propose_boundary_exports_from_tools_package() -> None:
    assert callable(tools.propose_boundary)
    assert callable(tools.recombine_designs)


def test_propose_boundary_preserves_m0_positive_toroidal_modes_for_z_sin() -> None:
    # Construct an explicit surface where m=0, n>0 entries are non-zero in z_sin.
    # Under stellarator symmetry conventions used by constellaration masks, those
    # modes are allowed (only m=0, n<=0 are redundant/masked).
    params = {
        "r_cos": [
            # n = -2, -1, 0, +1, +2
            [0.1, 0.2, 1.0, 0.3, 0.4],  # m=0
            [0.0, 0.0, 0.0, 0.0, 0.0],  # m=1
        ],
        "z_sin": [
            [0.01, 0.02, 0.0, 0.03, 0.04],  # m=0
            [0.0, 0.0, 0.1, 0.0, 0.0],  # m=1
        ],
        "n_field_periods": 3,
        "is_stellarator_symmetric": True,
    }

    out = tools.propose_boundary(params, perturbation_scale=0.0, seed=0)

    r_cos = np.asarray(out["r_cos"], dtype=float)
    z_sin = np.asarray(out["z_sin"], dtype=float)
    center_idx = r_cos.shape[1] // 2

    # m=0, n<0 r_cos should be zeroed (redundant under the canonical convention).
    assert np.allclose(r_cos[0, :center_idx], 0.0)
    # m=0, n=0 should be preserved (major radius term).
    assert float(r_cos[0, center_idx]) == float(params["r_cos"][0][center_idx])
    # m=0, n>0 should be preserved.
    assert np.allclose(
        r_cos[0, center_idx + 1 :], np.asarray(params["r_cos"])[0, center_idx + 1 :]
    )

    # m=0, n<=0 z_sin should be zeroed (n=0 term is identically zero; n<0 redundant).
    assert np.allclose(z_sin[0, : center_idx + 1], 0.0)
    # m=0, n>0 should be preserved (these modes are allowed).
    assert np.allclose(
        z_sin[0, center_idx + 1 :], np.asarray(params["z_sin"])[0, center_idx + 1 :]
    )
