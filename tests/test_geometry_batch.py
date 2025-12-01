import torch

from ai_scientist.optim import geometry


def test_batch_fourier_to_real_space():
    """Verify that batch_fourier_to_real_space matches single-item processing."""
    batch_size = 4
    mpol = 2
    ntor = 2
    nfp = 3
    n_theta = 8
    n_zeta = 8

    # Create random coefficients
    # Shape: (B, mpol+1, 2*ntor+1)
    r_cos = torch.randn(batch_size, mpol + 1, 2 * ntor + 1)
    z_sin = torch.randn(batch_size, mpol + 1, 2 * ntor + 1)

    # Run batched
    R_batch, Z_batch, Phi_batch = geometry.batch_fourier_to_real_space(
        r_cos, z_sin, n_field_periods=nfp, n_theta=n_theta, n_zeta=n_zeta
    )

    assert R_batch.shape == (batch_size, n_theta, n_zeta * nfp)
    assert Z_batch.shape == (batch_size, n_theta, n_zeta * nfp)
    assert Phi_batch.shape == (batch_size, n_theta, n_zeta * nfp)

    # Verify against single-item loop
    for i in range(batch_size):
        # Extract single item
        rc = r_cos[i]
        zs = z_sin[i]

        R_single, Z_single, Phi_single = geometry.fourier_to_real_space(
            rc, zs, n_field_periods=nfp, n_theta=n_theta, n_zeta=n_zeta
        )

        # Compare
        assert torch.allclose(R_batch[i], R_single, atol=1e-5)
        assert torch.allclose(Z_batch[i], Z_single, atol=1e-5)
        assert torch.allclose(Phi_batch[i], Phi_single, atol=1e-5)


def test_batch_gradients():
    """Verify that gradients flow through the batched operation."""
    mpol = 1
    ntor = 1
    r_cos = torch.randn(2, mpol + 1, 2 * ntor + 1, requires_grad=True)
    z_sin = torch.randn(2, mpol + 1, 2 * ntor + 1, requires_grad=True)

    R, Z, Phi = geometry.batch_fourier_to_real_space(
        r_cos, z_sin, n_field_periods=1, n_theta=4, n_zeta=4
    )

    loss = R.sum() + Z.sum()
    loss.backward()

    assert r_cos.grad is not None
    assert z_sin.grad is not None
    assert torch.any(r_cos.grad != 0)
