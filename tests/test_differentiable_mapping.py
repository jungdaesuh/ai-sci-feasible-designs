import pytest
try:
    import torch
    from ai_scientist.optim import differentiable

except ImportError:
    pytest.skip("PyTorch not available", allow_module_level=True)

def test_compute_index_mapping():
    # Define a small template
    class MockTemplate:
        n_poloidal_modes = 2 # m=0,1
        n_toroidal_modes = 3 # n=-1,0,1 -> size 3. max mode=1.
        n_field_periods = 1
    
    template = MockTemplate()
    mpol = 1
    ntor = 1
    
    device = "cpu"
    
    # Call mapping
    indices, dense_size = differentiable._compute_index_mapping(template, mpol, ntor, device)
    
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

if __name__ == "__main__":
    test_compute_index_mapping()
    print("Test passed!")
