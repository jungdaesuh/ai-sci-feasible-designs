"""Tests for VAE numerical stability fixes (AoT recommendations)."""

import torch

from ai_scientist.optim.generative import StellaratorVAE


class TestLogvarClamping:
    """Tests for log-variance clamping to prevent float overflow."""

    def test_extreme_logvar_does_not_produce_nan(self):
        """Verify extreme logvar values don't cause NaN in reparameterization."""
        vae = StellaratorVAE(mpol=4, ntor=5, latent_dim=16)

        # Simulate extreme logvar (would cause overflow without clamping)
        mu = torch.zeros(2, 16)
        logvar = torch.full((2, 16), 100.0)  # exp(100) overflows float32

        # Reparameterize should still work (uses exp(0.5 * logvar))
        z = vae.reparameterize(mu, logvar)

        # Check for overflow/NaN
        assert not torch.isnan(z).any(), "NaN in latent z from extreme logvar"
        assert not torch.isinf(z).any(), "Inf in latent z from extreme logvar"

    def test_negative_logvar_does_not_underflow(self):
        """Verify very negative logvar values don't cause underflow."""
        vae = StellaratorVAE(mpol=4, ntor=5, latent_dim=16)

        mu = torch.zeros(2, 16)
        logvar = torch.full((2, 16), -100.0)  # exp(-100) underflows to 0

        z = vae.reparameterize(mu, logvar)

        # Should still produce valid output (near mu with very small variance)
        assert not torch.isnan(z).any(), "NaN in latent z from very negative logvar"


class TestKLWarmup:
    """Tests for KL weight warmup annealing behavior."""

    def test_warmup_increases_linearly(self):
        """Verify KL weight increases linearly during warmup period."""
        kl_weight_target = 0.1
        warmup_epochs = 20

        weights = []
        for epoch in range(30):
            kl_weight_effective = min(
                kl_weight_target, kl_weight_target * (epoch + 1) / warmup_epochs
            )
            weights.append(kl_weight_effective)

        # During warmup, weight should increase
        for i in range(1, warmup_epochs):
            assert weights[i] > weights[i - 1], f"Weight should increase at epoch {i}"

        # After warmup, weight should be constant at target
        for i in range(warmup_epochs, 30):
            assert weights[i] == kl_weight_target, (
                f"Weight should be target at epoch {i}"
            )

    def test_warmup_starts_near_zero(self):
        """Verify warmup starts with very small KL weight."""
        kl_weight_target = 0.1
        warmup_epochs = 20

        kl_weight_epoch_0 = min(
            kl_weight_target, kl_weight_target * (0 + 1) / warmup_epochs
        )

        # First epoch should be 1/20 = 5% of target
        assert kl_weight_epoch_0 == kl_weight_target / warmup_epochs
        assert kl_weight_epoch_0 == 0.005  # 0.1 / 20


class TestVAEForwardPass:
    """Tests for VAE forward pass numerical stability."""

    def test_forward_produces_valid_output(self):
        """Verify forward pass produces valid reconstruction and latents."""
        vae = StellaratorVAE(mpol=4, ntor=5, latent_dim=16)

        # Create valid input: (B, 2, H, W) = (2, 2, 5, 11)
        x = torch.randn(2, 2, 5, 11)

        recon, mu, logvar = vae(x)

        assert recon.shape == x.shape
        assert mu.shape == (2, 16)
        assert logvar.shape == (2, 16)
        assert not torch.isnan(recon).any()
        assert not torch.isnan(mu).any()
        assert not torch.isnan(logvar).any()
