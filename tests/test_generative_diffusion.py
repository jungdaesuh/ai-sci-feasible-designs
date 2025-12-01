import torch

from ai_scientist import test_helpers
from ai_scientist.optim import generative


def test_diffusion_model_training_and_sampling():
    # 1. Create mock data
    candidates = []
    for i in range(5):
        params = test_helpers.base_params()
        # Add some noise to params so they aren't identical
        params["r_cos"][0][2] += i * 0.1

        metrics = test_helpers.dummy_metrics_with(
            aspect_ratio=4.0 + i * 0.1,
            max_elongation=1.5 + i * 0.1,
            minimum_normalized_magnetic_gradient_scale_length=20.0 + i,
            edge_rotational_transform_over_n_field_periods=0.4,
        )

        candidates.append(
            {"params": params, "metrics": metrics, "design_hash": f"hash_{i}"}
        )

    # 2. Initialize Model
    # Use small epochs and batch size for speed
    model = generative.DiffusionDesignModel(
        min_samples=2,
        learning_rate=1e-3,
        epochs=2,
        batch_size=2,
        timesteps=10,  # Reduce timesteps for test speed
        device="cpu",
    )

    # 3. Fit
    model.fit(candidates)
    assert model._trained
    assert model._model is not None

    # 4. Sample
    target_metrics = {
        "aspect_ratio": 4.5,
        "minimum_normalized_magnetic_gradient_scale_length": 22.0,
        "max_elongation": 1.6,
        "edge_rotational_transform_over_n_field_periods": 0.4,
    }

    generated = model.sample(n_samples=2, target_metrics=target_metrics, seed=42)

    assert len(generated) == 2
    assert "params" in generated[0]
    assert generated[0]["source"] == "diffusion_conditional"

    # Verify output shape/validity via tools (implicit)
    # structured_unflatten checks schema consistency


def test_stellarator_diffusion_shapes():
    mpol = 4
    ntor = 4
    metric_dim = 5
    B = 2

    model = generative.StellaratorDiffusion(mpol, ntor, metric_dim)

    x = torch.randn(B, 2, mpol + 1, 2 * ntor + 1)
    t = torch.randint(0, 10, (B,))
    metrics = torch.randn(B, metric_dim)

    out = model(x, t, metrics)

    assert out.shape == x.shape
