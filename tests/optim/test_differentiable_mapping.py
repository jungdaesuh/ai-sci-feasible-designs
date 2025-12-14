import pytest

try:
    import torch

    from ai_scientist.optim import differentiable

except ImportError:
    pytest.skip("PyTorch not available", allow_module_level=True)


def test_compute_index_mapping():
    # Define a small template
    class MockTemplate:
        n_poloidal_modes = 2  # m=0,1
        n_toroidal_modes = 3  # n=-1,0,1 -> size 3. max mode=1.
        n_field_periods = 1

    template = MockTemplate()
    mpol = 1
    ntor = 1

    device = "cpu"

    # Call mapping
    indices, dense_size = differentiable._compute_index_mapping(
        template, mpol, ntor, device
    )

    # Verify sizes
    # Compact size:
    # m=0, n=0..1 (cos) -> 2
    # m=1, n=-1..1 (cos) -> 3
    # m=0, n=0 (sin) -> 0 (fixed 0)
    # m=1, n=-1..1 (sin) -> 3
    # Wait, mask logic depends on symmetry.
    # R_cos: m=0, n>=0. m>0, all n.
    # Z_sin: m=0, n>0?? No, usually Z_sin(0,0)=0.
    # Let's check dense size.
    # Dense: 2 * (mpol+1) * (2*ntor+1) = 2 * 2 * 3 = 12.

    assert dense_size == 12

    # Compact size should be less than 12.
    assert indices.numel() < 12

    # Test scattering
    compact_vec = torch.arange(1, indices.numel() + 1, dtype=torch.float)
    dense_vec = torch.zeros(dense_size)
    dense_vec[indices] = compact_vec

    # Check if values are placed
    assert torch.sum(dense_vec > 0) == indices.numel()

    # Check ordering
    # The mapping should be sorted by compact index.
    # So indices[0] is where compact[0] goes.
    # dense_vec[indices[0]] should be compact_vec[0] (which is 1)
    assert dense_vec[indices[0]] == 1.0
    assert dense_vec[indices[-1]] == float(indices.numel())


def test_build_local_mask_excludes_r00_by_default():
    """Test that R₀₀ is excluded from mask by default (backward compatibility)."""
    max_poloidal = 4
    max_toroidal = 4
    ntor = max_toroidal

    mask = differentiable._build_local_mask(max_poloidal, max_toroidal)

    # R₀₀ position is at (0, ntor) in the (mpol+1, 2*ntor+1) grid
    assert not mask[0, ntor], "R₀₀ should be excluded by default"

    # Verify that n >= 1 modes for m=0 are included
    assert mask[0, ntor + 1], "R_{0,1} should be included"

    # Verify m > 0 modes are included
    assert mask[1, ntor], "R_{1,0} should be included"


def test_build_local_mask_includes_r00_when_enabled():
    """Test that R₀₀ is included in mask when include_r00=True."""
    max_poloidal = 4
    max_toroidal = 4
    ntor = max_toroidal

    # Default: R₀₀ excluded
    mask_default = differentiable._build_local_mask(
        max_poloidal, max_toroidal, include_r00=False
    )
    assert not mask_default[0, ntor], "R₀₀ should be excluded by default"

    # With include_r00=True: R₀₀ included
    mask_with_r00 = differentiable._build_local_mask(
        max_poloidal, max_toroidal, include_r00=True
    )
    assert mask_with_r00[0, ntor], "R₀₀ should be included when flag is set"

    # Check that exactly one more element is active
    assert mask_with_r00.sum() == mask_default.sum() + 1


def test_compute_index_mapping_with_include_r00():
    """Test that _compute_index_mapping includes R₀₀ when include_r00=True."""

    class MockTemplate:
        n_poloidal_modes = 3
        n_toroidal_modes = 5
        n_field_periods = 3

    template = MockTemplate()
    mpol = 2
    ntor = 2
    device = "cpu"

    # Get indices without R₀₀
    indices_no_r00, dense_size = differentiable._compute_index_mapping(
        template, mpol, ntor, device, include_r00=False
    )

    # Get indices with R₀₀
    indices_with_r00, dense_size_r00 = differentiable._compute_index_mapping(
        template, mpol, ntor, device, include_r00=True
    )

    # Dense size should be the same
    assert dense_size == dense_size_r00

    # With R₀₀ included, we should have 2 more indices (one for r_cos, one for z_sin)
    # Actually, z_sin[0,0] = 0 by stellarator symmetry, so only 1 more for r_cos
    # Wait - Z_sin mask follows same pattern as R_cos... let me think.
    # The mask is applied to both r_cos and z_sin equally.
    # So if we add (m=0, n=0), we add 1 to R portion and 1 to Z portion = 2 total.
    assert indices_with_r00.numel() == indices_no_r00.numel() + 2


if __name__ == "__main__":
    test_compute_index_mapping()
    test_build_local_mask_excludes_r00_by_default()
    test_build_local_mask_includes_r00_when_enabled()
    test_compute_index_mapping_with_include_r00()
    print("All tests passed!")
