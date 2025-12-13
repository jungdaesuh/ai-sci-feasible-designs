import torch

from ai_scientist import test_helpers
from ai_scientist.optim import generative


def test_diffusion_model_training_and_sampling():
    # 1. Create mock data
    candidates = []
    for i in range(5):
        params = test_helpers.base_params()
        # Add some noise to params so they aren't identical
        params["r_cos"][0][2] += i * 0.1  # pyright: ignore[reportIndexIssue]

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
        pca_components=4,  # Small value for test data
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
    """Test StellaratorDiffusion model shape handling.

    StellaratorDiffusion expects:
    - x: (B, input_dim) - 2D PCA-compressed latent vector
    - metrics: (B, 4) - Conditioning metrics (iota, A, nfp, N)
    """
    input_dim = 50  # PCA latent dimension (paper default)
    metric_dim = 4  # (iota, A, nfp, N) - fixed by model architecture
    B = 2

    model = generative.StellaratorDiffusion(input_dim=input_dim)

    x = torch.randn(B, input_dim)  # 2D: PCA-compressed latent
    t = torch.randint(0, 10, (B,))
    metrics = torch.randn(B, metric_dim)

    out = model(x, t, metrics)

    assert out.shape == x.shape  # (B, input_dim)


def test_diffusion_fine_tune_on_elites():
    """Test fine_tune_on_elites preserves PCA and normalization."""
    from ai_scientist import test_helpers

    # 1. Create initial training data
    candidates = []
    for i in range(5):
        params = test_helpers.base_params()
        params["r_cos"][0][2] += i * 0.1  # pyright: ignore[reportIndexIssue]

        metrics = test_helpers.dummy_metrics_with(
            aspect_ratio=4.0 + i * 0.1,
            max_elongation=1.5 + i * 0.1,
            minimum_normalized_magnetic_gradient_scale_length=20.0 + i,
            edge_rotational_transform_over_n_field_periods=0.4,
        )

        candidates.append(
            {"params": params, "metrics": metrics, "design_hash": f"hash_{i}"}
        )

    # 2. Initialize and train model
    model = generative.DiffusionDesignModel(
        min_samples=2,
        learning_rate=1e-3,
        epochs=2,
        batch_size=2,
        timesteps=10,
        pca_components=4,  # Small value for test data
        device="cpu",
    )

    model.fit(candidates)
    assert model._trained

    # Store original PCA and normalization
    original_pca = model.pca
    original_m_mean = model.m_mean.clone()
    original_m_std = model.m_std.clone()

    # 3. Create elite candidates (new data with slightly different metrics)
    elites = []
    for i in range(3):
        params = test_helpers.base_params()
        params["r_cos"][0][3] += i * 0.2  # pyright: ignore[reportIndexIssue]

        metrics = test_helpers.dummy_metrics_with(
            aspect_ratio=5.0 + i * 0.1,  # Different range
            max_elongation=1.8 + i * 0.1,
            minimum_normalized_magnetic_gradient_scale_length=25.0 + i,
            edge_rotational_transform_over_n_field_periods=0.5,
        )

        elites.append(
            {"params": params, "metrics": metrics, "design_hash": f"elite_{i}"}
        )

    # 4. Fine-tune on elites
    model.fine_tune_on_elites(elites, epochs=2)

    # 5. Verify PCA and normalization are preserved (same object/values)
    assert model.pca is original_pca  # Same PCA object
    assert torch.allclose(model.m_mean, original_m_mean)  # Same normalization
    assert torch.allclose(model.m_std, original_m_std)

    # 6. Verify model still works
    target_metrics = {
        "aspect_ratio": 5.0,
        "minimum_normalized_magnetic_gradient_scale_length": 25.0,
        "max_elongation": 1.8,
        "edge_rotational_transform_over_n_field_periods": 0.5,
    }

    generated = model.sample(n_samples=1, target_metrics=target_metrics, seed=42)
    assert len(generated) == 1
    assert "params" in generated[0]
