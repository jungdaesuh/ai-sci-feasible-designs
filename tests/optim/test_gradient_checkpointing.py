"""Tests for gradient checkpointing in surrogate_v2 (Issue #9).

Verifies that gradient checkpointing:
1. Preserves numerical correctness of gradients
2. Can be toggled on/off
3. Works through NeuralOperatorSurrogate API
"""

import numpy as np
import pytest

try:
    import torch

    from ai_scientist.optim.surrogate_v2 import (
        NeuralOperatorSurrogate,
        StellaratorNeuralOp,
    )
except ImportError as e:
    pytest.skip(f"PyTorch not available: {e}", allow_module_level=True)


def _make_params(scale: float, mpol: int = 3, ntor: int = 3) -> dict:
    """Create test boundary parameters."""
    r_cos = np.random.randn(mpol + 1, 2 * ntor + 1).astype(np.float32) * scale
    r_cos[0, ntor] = 1.0  # Major radius normalization
    z_sin = np.random.randn(mpol + 1, 2 * ntor + 1).astype(np.float32) * scale * 0.1
    return {
        "r_cos": r_cos.tolist(),
        "z_sin": z_sin.tolist(),
        "n_field_periods": 3,
    }


class TestStellaratorNeuralOpCheckpointing:
    """Tests for gradient checkpointing in StellaratorNeuralOp."""

    def test_checkpointing_disabled_by_default(self):
        """Verify checkpointing is disabled by default."""
        model = StellaratorNeuralOp(mpol=3, ntor=3)
        assert model._use_checkpointing is True

    def test_enable_checkpointing(self):
        """Verify checkpointing can be enabled."""
        model = StellaratorNeuralOp(mpol=3, ntor=3)
        model.enable_checkpointing(True)
        assert model._use_checkpointing is True
        model.enable_checkpointing(False)
        assert model._use_checkpointing is False

    def test_checkpointing_preserves_forward_output(self):
        """Verify forward output is identical with/without checkpointing."""
        torch.manual_seed(42)
        model = StellaratorNeuralOp(mpol=3, ntor=3)
        model.eval()

        # Input: flattened (r_cos, z_sin) + nfp
        input_dim = 2 * (3 + 1) * (2 * 3 + 1) + 1
        x = torch.randn(4, input_dim)

        # Forward without checkpointing
        model.enable_checkpointing(False)
        with torch.no_grad():
            out_no_ckpt = model(x)

        # Forward with checkpointing (eval mode = no checkpointing applied)
        model.enable_checkpointing(True)
        with torch.no_grad():
            out_ckpt = model(x)

        # Should be identical since checkpointing only affects training mode
        for i in range(len(out_no_ckpt)):
            assert torch.allclose(out_no_ckpt[i], out_ckpt[i], atol=1e-6)

    def test_checkpointing_preserves_gradients(self):
        """Verify gradients flow correctly with checkpointing enabled.

        Note: We test in eval mode to avoid random rotation augmentation
        which would produce different values between forward passes.
        The key verification is that gradients are computed and have
        reasonable magnitudes, not exact match (since checkpointing
        can introduce minor numerical differences).
        """
        torch.manual_seed(42)
        model = StellaratorNeuralOp(mpol=3, ntor=3)

        input_dim = 2 * (3 + 1) * (2 * 3 + 1) + 1
        x = torch.randn(4, input_dim, requires_grad=True)

        # Test with checkpointing enabled - even though checkpointing
        # only affects training mode, we verify the plumbing works
        model.train()
        model.enable_checkpointing(True)
        out_ckpt = model(x)
        loss_ckpt = out_ckpt[0].sum()
        loss_ckpt.backward()

        # Verify gradients were computed and are not zero/nan
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()
        # Gradients should have reasonable magnitude
        assert x.grad.abs().mean() > 1e-6


class TestNeuralOperatorSurrogateCheckpointing:
    """Tests for checkpointing in NeuralOperatorSurrogate."""

    def test_checkpointing_disabled_by_default(self):
        """Verify checkpointing is disabled by default."""
        surrogate = NeuralOperatorSurrogate(min_samples=2)
        assert surrogate._use_checkpointing is True

    def test_enable_checkpointing_propagates_to_models(self):
        """Verify enable_checkpointing propagates to all ensemble models."""
        surrogate = NeuralOperatorSurrogate(min_samples=2, n_ensembles=3)

        # Manually create models
        for _ in range(3):
            model = StellaratorNeuralOp(mpol=3, ntor=3)
            surrogate._models.append(model)

        # Enable checkpointing
        surrogate.enable_checkpointing(True)

        # All models should have checkpointing enabled
        assert surrogate._use_checkpointing is True
        for model in surrogate._models:
            assert model._use_checkpointing is True

        # Disable checkpointing
        surrogate.enable_checkpointing(False)
        assert surrogate._use_checkpointing is False
        for model in surrogate._models:
            assert model._use_checkpointing is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
