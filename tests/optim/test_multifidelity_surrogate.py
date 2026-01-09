"""Tests for multi-fidelity surrogate (Issue #10).

Verifies that fidelity embeddings:
1. Are correctly included in the model architecture
2. Produce different outputs for different fidelity levels
3. Default to high-fidelity when not specified
"""

import pytest

try:
    import torch

    from ai_scientist.optim.surrogate_v2 import (
        StellaratorNeuralOp,
    )
except ImportError as e:
    pytest.skip(f"PyTorch not available: {e}", allow_module_level=True)


def _make_input(batch_size: int, mpol: int = 3, ntor: int = 3) -> torch.Tensor:
    """Create test input tensor."""
    input_dim = 2 * (mpol + 1) * (2 * ntor + 1) + 1
    return torch.randn(batch_size, input_dim)


class TestFidelityEmbedding:
    """Tests for fidelity embedding in StellaratorNeuralOp."""

    def test_fidelity_embedding_exists(self):
        """Verify fidelity embedding layer is created."""
        model = StellaratorNeuralOp(mpol=3, ntor=3)
        assert hasattr(model, "fidelity_embedding")
        assert hasattr(model, "fidelity_embedding_dim")
        assert model.fidelity_embedding_dim == 8

    def test_fidelity_embedding_shape(self):
        """Verify fidelity embedding has correct dimensions."""
        model = StellaratorNeuralOp(mpol=3, ntor=3)
        # 2 levels (screen=0, promote=1), embedding dim=8
        assert model.fidelity_embedding.num_embeddings == 2
        assert model.fidelity_embedding.embedding_dim == 8

    def test_forward_accepts_fidelity_parameter(self):
        """Verify forward() accepts fidelity parameter."""
        model = StellaratorNeuralOp(mpol=3, ntor=3)
        model.eval()

        x = _make_input(batch_size=4)
        fidelity = torch.tensor([0, 0, 1, 1], dtype=torch.long)

        # Should not raise
        outputs = model(x, fidelity=fidelity)
        assert len(outputs) == 6  # obj, mhd, qi, iota, mirror, flux

    def test_forward_defaults_to_high_fidelity(self):
        """Verify forward() defaults to high-fidelity when fidelity=None."""
        torch.manual_seed(42)
        model = StellaratorNeuralOp(mpol=3, ntor=3)
        model.eval()

        x = _make_input(batch_size=4)

        # Without fidelity parameter (should default to 1 = high-fidelity)
        with torch.no_grad():
            out_default = model(x)

        # Explicitly pass high-fidelity
        with torch.no_grad():
            out_explicit = model(x, fidelity=torch.ones(4, dtype=torch.long))

        # Should be identical
        for i in range(len(out_default)):
            assert torch.allclose(out_default[i], out_explicit[i], atol=1e-6)

    def test_different_fidelity_produces_different_output(self):
        """Verify different fidelity levels produce different outputs."""
        torch.manual_seed(42)
        model = StellaratorNeuralOp(mpol=3, ntor=3)
        model.eval()

        x = _make_input(batch_size=4)

        # Low fidelity (screen = 0)
        with torch.no_grad():
            out_low = model(x, fidelity=torch.zeros(4, dtype=torch.long))

        # High fidelity (promote = 1)
        with torch.no_grad():
            out_high = model(x, fidelity=torch.ones(4, dtype=torch.long))

        # Outputs should be different
        # At least one output head should differ significantly
        any_different = False
        for i in range(len(out_low)):
            if not torch.allclose(out_low[i], out_high[i], atol=0.01):
                any_different = True
                break

        assert any_different, "Fidelity levels should produce different outputs"

    def test_fidelity_embedding_is_learnable(self):
        """Verify fidelity embedding parameters are learnable."""
        model = StellaratorNeuralOp(mpol=3, ntor=3)

        # Check that fidelity embedding requires gradients
        assert model.fidelity_embedding.weight.requires_grad is True


class TestMultiFidelityBackwardCompat:
    """Tests ensuring backward compatibility with existing code."""

    def test_existing_code_works_without_fidelity(self):
        """Verify model works when called without fidelity parameter."""
        model = StellaratorNeuralOp(mpol=3, ntor=3)
        model.eval()

        x = _make_input(batch_size=4)

        # Old-style call without fidelity should work
        with torch.no_grad():
            outputs = model(x)

        assert len(outputs) == 6
        for out in outputs:
            assert out.shape == (4,)
            assert not torch.isnan(out).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
