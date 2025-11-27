
import torch
import pytest
import numpy as np
from ai_scientist.optim import geometry

def test_geometry_metrics_run_and_differentiate():
    """
    Test that elongation, curvature, aspect_ratio run and have gradients.
    """
    batch_size = 2
    mpol = 3
    ntor = 3
    
    # Create random coefficients
    r_cos = torch.randn(batch_size, mpol+1, 2*ntor+1, requires_grad=True, dtype=torch.float32)
    z_sin = torch.randn(batch_size, mpol+1, 2*ntor+1, requires_grad=True, dtype=torch.float32)
    
    # Make them somewhat realistic to avoid NaNs (Major radius > minor radius)
    with torch.no_grad():
        r_cos[:, 0, ntor] = 5.0 # Major radius
        r_cos[:, 1, ntor] = 1.0 # Minor radius
        z_sin[:, 1, ntor] = 1.0
    
    nfp = 3
    
    # 1. Elongation
    elo = geometry.elongation(r_cos, z_sin, n_field_periods=nfp)
    assert elo.shape == (batch_size,)
    assert torch.all(elo > 0)
    
    loss_elo = torch.sum(elo)
    loss_elo.backward()
    
    assert r_cos.grad is not None
    assert z_sin.grad is not None
    
    # Reset grads
    r_cos.grad.zero_()
    z_sin.grad.zero_()
    
    # 2. Aspect Ratio
    ar = geometry.aspect_ratio(r_cos, z_sin, n_field_periods=nfp)
    assert ar.shape == (batch_size,)
    
    loss_ar = torch.sum(ar)
    loss_ar.backward()
    assert r_cos.grad is not None
    
    r_cos.grad.zero_()
    z_sin.grad.zero_()
    
    # 3. Mean Curvature
    # We expect curvature to be non-negative (mean abs curvature)
    curv = geometry.mean_curvature(r_cos, z_sin, n_field_periods=nfp)
    assert curv.shape == (batch_size,)
    assert torch.all(curv >= 0)
    
    loss_curv = torch.sum(curv)
    loss_curv.backward()
    assert r_cos.grad is not None
    
    r_cos.grad.zero_()
    
    # 4. Surface Area
    area = geometry.surface_area(r_cos, z_sin, n_field_periods=nfp)
    assert area.shape == (batch_size,)
    assert torch.all(area > 0)
    
    loss_area = torch.sum(area)
    loss_area.backward()
    assert r_cos.grad is not None

def test_geometry_metrics_batched_nfp():
    """Test with variable nfp per batch item."""
    batch_size = 3
    mpol = 2
    ntor = 2
    r_cos = torch.randn(batch_size, mpol+1, 2*ntor+1)
    z_sin = torch.randn(batch_size, mpol+1, 2*ntor+1)
    nfp = torch.tensor([1.0, 3.0, 5.0])
    
    elo = geometry.elongation(r_cos, z_sin, n_field_periods=nfp)
    assert elo.shape == (batch_size,)
