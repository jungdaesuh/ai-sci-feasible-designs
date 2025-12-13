"""Generative Models for Phase 4 Exploration (VAE / Latent Optimization).

This module implements the Variational Autoencoder (VAE) for the "Generative Models"
phase of the AI Scientist V2 upgrade. It enables:
1.  Learning a smooth latent space for stellarator geometries.
2.  Sampling novel candidates from this latent space.
3.  (Future) Performing gradient-based optimization in the latent space.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA

from ai_scientist import tools

_LOGGER = logging.getLogger(__name__)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class StellaratorDiffusion(nn.Module):
    """Conditional Diffusion Model for Stellarator Geometries (MLP-based).

    Architecture follows Padidar et al. (2025):
    - Input: PCA-compressed latent vector (default 50 dims)
    - Architecture: 4 hidden layers, width 2048, GELU activation
    - Conditioning: Sinusoidal time embedding + Metric embedding
    """

    def __init__(
        self,
        input_dim: int = 50,  # PCA/Latent dimension
        hidden_dim: int = 2048,  # Paper: 2048
        n_layers: int = 4,  # Paper: 4
        condition_dim: int = 128,  # Paper: y -> 128
        time_dim: int = 128,  # Paper: t -> 128
        input_embed_dim: int = 64,  # Paper: x -> 64
    ):
        super().__init__()

        # 1. Embeddings
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.input_mlp = nn.Linear(input_dim, input_embed_dim)

        # Metric conditioning: (ι, A, nfp, N) -> 4 dims
        self.condition_mlp = nn.Sequential(
            nn.Linear(4, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
        )

        # 2. Main MLP Trunk
        # Input to trunk is concatenation of all embeddings
        concat_dim = input_embed_dim + time_dim + condition_dim

        layers = []
        in_d = concat_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_d, hidden_dim))
            layers.append(nn.GELU())  # Paper specifies GELU
            in_d = hidden_dim

        self.trunk = nn.Sequential(*layers)

        # 3. Output Head (predicts noise)
        self.head = nn.Linear(hidden_dim, input_dim)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, metrics: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) - Noisy latent state
            t: (B,) - Timesteps
            metrics: (B, 4) - Conditional metrics (ι, A, nfp, N)
        """
        x_emb = self.input_mlp(x)
        t_emb = self.time_mlp(t)
        c_emb = self.condition_mlp(metrics)

        # Concatenate: [x_emb, t_emb, c_emb]
        h = torch.cat([x_emb, t_emb, c_emb], dim=1)

        h = self.trunk(h)
        return self.head(h)


class DiffusionDesignModel:
    """Manager for Conditional Diffusion Model (StellarForge v1)."""

    # Metrics used for conditioning (Padidar et al.: ι, A, nfp, N)
    METRIC_KEYS = [
        "edge_rotational_transform_over_n_field_periods",  # iota
        "aspect_ratio",
        "number_of_field_periods",  # nfp, often in params not metrics
        "is_quasihelical",  # N (0=QA, 1=QH)
    ]

    def __init__(
        self,
        *,
        min_samples: int = 32,
        learning_rate: float = 1e-3,
        epochs: int = 200,
        batch_size: int = 4096,  # Default to paper spec
        device: str = "cpu",
        timesteps: int = 200,  # Paper: 200
        # Architecture args
        hidden_dim: int = 2048,
        n_layers: int = 4,
        pca_components: int = 50,
        log_interval: int = 10,
    ) -> None:
        self._min_samples = min_samples
        self._lr = learning_rate
        self._epochs = epochs
        self._batch_size = batch_size
        self._device = device if torch.cuda.is_available() or device == "mps" else "cpu"
        # Handle MPS explicitly
        if device == "mps" and not torch.backends.mps.is_available():
            _LOGGER.warning("MPS requested but not available. Falling back to CPU.")
            self._device = "cpu"

        self._timesteps = timesteps

        # Architecture params
        self._hidden_dim = hidden_dim
        self._n_layers = n_layers
        self._pca_components = pca_components
        self._log_interval = log_interval

        self.__model: StellaratorDiffusion | None = None  # Use property for access
        self._schema: tools.FlattenSchema | None = None
        self._optimizer: optim.Optimizer | None = None
        self.pca: PCA | None = None

        self._trained = False
        self.m_mean: torch.Tensor | None = None
        self.m_std: torch.Tensor | None = None

        self._build_noise_schedule()

    @property
    def _model(self) -> StellaratorDiffusion:
        """Access the diffusion model, raising if not initialized.

        Raises:
            RuntimeError: If model has not been created via fit() or load_checkpoint().
        """
        if self.__model is None:
            raise RuntimeError(
                "DiffusionDesignModel not initialized. "
                "Call fit() or load_checkpoint() first."
            )
        return self.__model

    @_model.setter
    def _model(self, value: StellaratorDiffusion | None) -> None:
        self.__model = value

    def _has_model(self) -> bool:
        """Check if model exists without raising."""
        return self.__model is not None

    def _build_noise_schedule(self) -> None:
        """Build the DDPM noise schedule with proper posterior variance.

        The posterior variance in DDPM is:
            beta_tilde_t = beta_t * (1 - alpha_hat_{t-1}) / (1 - alpha_hat_t)

        This is the variance of q(x_{t-1} | x_t, x_0), which is the correct
        variance to use during sampling (not simply beta_t).
        """
        self.beta = torch.linspace(1e-4, 0.02, self._timesteps, device=self._device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        # Compute alpha_hat_prev: alpha_hat shifted by 1 with alpha_hat[-1] = 1.0
        # alpha_hat_prev[t] = alpha_hat[t-1] for t > 0, else 1.0
        self.alpha_hat_prev = torch.cat(
            [torch.tensor([1.0], device=self._device), self.alpha_hat[:-1]]
        )

        # Posterior variance: beta_tilde = beta * (1 - alpha_hat_prev) / (1 - alpha_hat)
        # Clamp denominator to avoid division by zero at t=0
        self.beta_tilde = (
            self.beta
            * (1.0 - self.alpha_hat_prev)
            / (1.0 - self.alpha_hat).clamp(min=1e-8)
        )

    def _ensure_model(self) -> None:
        if self.__model is not None:
            return

        # Input dim is PCA latent dim (or raw if no PCA)
        input_dim = (
            self._pca_components
            if self.pca
            else self._schema.mpol * self._schema.ntor * 2
        )  # Fallback

        self._model = StellaratorDiffusion(
            input_dim=input_dim,
            hidden_dim=self._hidden_dim,
            n_layers=self._n_layers,
            # condition_dim, time_dim, input_embed_dim use defaults
        ).to(self._device)
        self._optimizer = optim.Adam(self._model.parameters(), lr=self._lr)

    @staticmethod
    def _tensor_to_list(value: torch.Tensor | None) -> list[float] | None:
        if value is None:
            return None
        return value.detach().cpu().tolist()

    def _tensor_from_list(self, data: Sequence[float] | None) -> torch.Tensor | None:
        if data is None:
            return None
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    def _extract_metrics(self, cand: Mapping[str, Any]) -> np.ndarray | None:
        """Extract conditioning metrics (iota, A, nfp, N)."""
        metrics = cand.get("metrics") or cand
        params = cand.get("params") or cand.get("candidate_params") or {}

        # Helper to find values in metrics or params
        def get_val(key):
            # Try metrics first
            if isinstance(metrics, dict):
                v = metrics.get(key)
                if v is not None:
                    return float(v)
            elif hasattr(metrics, "model_dump"):
                # Pydantic model - use model_dump()
                m_dict = metrics.model_dump()
                v = m_dict.get(key)
                if v is not None:
                    return float(v)
            elif hasattr(metrics, key):
                # Direct attribute access
                v = getattr(metrics, key, None)
                if v is not None:
                    return float(v)

            # Try params
            if isinstance(params, dict):
                v = params.get(key)
                if v is not None:
                    return float(v)
            return None

        # 1. Iota
        iota = get_val("edge_rotational_transform_over_n_field_periods")
        if iota is None:
            return None  # Strict requirement

        # 2. Aspect Ratio
        ar = get_val("aspect_ratio")
        if ar is None:
            return None

        # 3. NFP (Usually in params)
        nfp = get_val("number_of_field_periods") or get_val("nfp")
        if nfp is None:
            nfp = 3.0  # Fallback check? Or fail?

        # 4. N (QA vs QH)
        # Try to infer from params or metrics or config
        is_qh = get_val("is_quasihelical")
        if is_qh is None:
            # Fallback heuristic: check helicity in config if available?
            # For now, default to 0 (QA) if not found, to be safe?
            # Or try "quasisymmetry_type" which might be "QA" or "QH"
            qs_type = params.get("quasisymmetry_type")
            if qs_type == "QH":
                is_qh = 1.0
            else:
                is_qh = 0.0

        return np.array([iota, ar, nfp, is_qh], dtype=np.float32)

    def fit(self, candidates: Sequence[Mapping[str, Any]]) -> None:
        sample_count = len(candidates)
        if sample_count < self._min_samples:
            _LOGGER.info(
                "[diffusion] Skipping training: %d samples < %d",
                sample_count,
                self._min_samples,
            )
            return

        _LOGGER.info(
            "[diffusion] Training on %d samples (PCA=%d)...",
            sample_count,
            self._pca_components,
        )

        # 1. Prepare Data
        vectors = []
        metrics_list = []

        for cand in candidates:
            params = cand.get("params") or cand.get("candidate_params")
            if not params:
                continue

            # Get geometry
            vec, schema = tools.structured_flatten(params, schema=self._schema)
            if self._schema is None:
                self._schema = schema

            # Get metrics
            m_vec = self._extract_metrics(cand)
            if m_vec is None:
                continue

            vectors.append(vec)
            metrics_list.append(m_vec)

        if not vectors:
            return

        # Convert to numpy for PCA
        X_np = np.vstack(vectors)

        # Fit PCA
        if self._pca_components > 0:
            self.pca = PCA(n_components=self._pca_components)
            X_latent = self.pca.fit_transform(X_np)
        else:
            self.pca = None
            X_latent = X_np

        X = torch.tensor(X_latent, dtype=torch.float32).to(self._device)
        M = torch.tensor(np.vstack(metrics_list), dtype=torch.float32).to(self._device)

        # Normalize metrics
        self.m_mean = M.mean(dim=0)
        self.m_std = M.std(dim=0) + 1e-6
        M = (M - self.m_mean) / self.m_std

        dataset = TensorDataset(X, M)
        loader = DataLoader(dataset, batch_size=self._batch_size, shuffle=True)

        # Init Model
        self._ensure_model()
        self._model.train()  # Property guarantees non-None after _ensure_model

        for epoch in range(self._epochs):
            epoch_loss = 0.0
            for xb, mb in loader:
                self._optimizer.zero_grad()

                # xb is (B, latent_dim)
                B = xb.shape[0]

                # Sample t
                t = torch.randint(0, self._timesteps, (B,), device=self._device).long()

                # Noise
                noise = torch.randn_like(xb)

                # Forward diffusion q(x_t | x_0)
                alpha_hat_t = self.alpha_hat[t][:, None]  # (B, 1)
                xt = torch.sqrt(alpha_hat_t) * xb + torch.sqrt(1 - alpha_hat_t) * noise

                # Predict noise
                noise_pred = self._model(xt, t, mb)

                loss = F.mse_loss(noise_pred, noise)
                loss.backward()
                self._optimizer.step()

                epoch_loss += loss.item()

            if self._log_interval > 0 and epoch % self._log_interval == 0:
                _LOGGER.info(f"Epoch {epoch}/{self._epochs}: loss={epoch_loss:.4f}")

        self._trained = True
        _LOGGER.info("[diffusion] Training complete.")

    def fine_tune_on_elites(
        self,
        elites: Sequence[Mapping[str, Any]],
        epochs: int | None = None,
    ) -> None:
        """Incrementally fine-tune the model on elite candidates.

        Unlike fit(), this method:
        - Preserves existing PCA transform (no re-fitting)
        - Uses fewer epochs by default (20 vs 200)
        - Preserves metric normalization parameters

        Args:
            elites: Elite candidates to fine-tune on.
            epochs: Number of fine-tuning epochs (default: 20).
        """
        if not self._trained or not self._has_model():
            _LOGGER.warning(
                "[diffusion] Cannot fine-tune: model not trained. Use fit() first."
            )
            return

        if self.pca is None:
            _LOGGER.warning("[diffusion] Cannot fine-tune: PCA not initialized.")
            return

        if self.m_mean is None or self.m_std is None:
            _LOGGER.warning(
                "[diffusion] Cannot fine-tune: metric normalization not set."
            )
            return

        sample_count = len(elites)
        if sample_count < self._min_samples:
            _LOGGER.info(
                "[diffusion] Skipping fine-tune: %d elites < %d min_samples",
                sample_count,
                self._min_samples,
            )
            return

        fine_tune_epochs = epochs if epochs is not None else 20

        _LOGGER.info(
            "[diffusion] Fine-tuning on %d elites for %d epochs...",
            sample_count,
            fine_tune_epochs,
        )

        # 1. Prepare Data (reuse existing PCA and normalization)
        vectors = []
        metrics_list = []

        for cand in elites:
            params = cand.get("params") or cand.get("candidate_params")
            if not params:
                continue

            # Get geometry - use existing schema
            vec, _ = tools.structured_flatten(params, schema=self._schema)

            # Get metrics
            m_vec = self._extract_metrics(cand)
            if m_vec is None:
                continue

            vectors.append(vec)
            metrics_list.append(m_vec)

        if not vectors:
            _LOGGER.warning("[diffusion] No valid elites to fine-tune on.")
            return

        # Convert to numpy and apply existing PCA transform (not fit_transform!)
        X_np = np.vstack(vectors)
        X_latent = self.pca.transform(X_np)

        X = torch.tensor(X_latent, dtype=torch.float32).to(self._device)
        M = torch.tensor(np.vstack(metrics_list), dtype=torch.float32).to(self._device)

        # Apply existing normalization (not recalculating!)
        M = (M - self.m_mean) / self.m_std

        dataset = TensorDataset(X, M)
        loader = DataLoader(dataset, batch_size=self._batch_size, shuffle=True)

        # Fine-tune the model
        self._model.train()

        for epoch in range(fine_tune_epochs):
            epoch_loss = 0.0
            for xb, mb in loader:
                self._optimizer.zero_grad()

                B = xb.shape[0]
                t = torch.randint(0, self._timesteps, (B,), device=self._device).long()
                noise = torch.randn_like(xb)

                alpha_hat_t = self.alpha_hat[t][:, None]
                xt = torch.sqrt(alpha_hat_t) * xb + torch.sqrt(1 - alpha_hat_t) * noise

                noise_pred = self._model(xt, t, mb)
                loss = F.mse_loss(noise_pred, noise)
                loss.backward()
                self._optimizer.step()

                epoch_loss += loss.item()

            if self._log_interval > 0 and epoch % self._log_interval == 0:
                _LOGGER.info(
                    f"[fine-tune] Epoch {epoch}/{fine_tune_epochs}: loss={epoch_loss:.4f}"
                )

        _LOGGER.info("[diffusion] Fine-tuning complete.")

    def load_checkpoint(self, path: str | Path) -> None:
        """Load model state and PCA from checkpoint.

        Note: Uses weights_only=False because checkpoint contains sklearn PCA
        which has nested numpy dependencies. This is safe for trusted checkpoints
        (our own training output). See PyTorch docs on serialization security.
        """
        _LOGGER.info(f"[diffusion] Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location=self._device, weights_only=False)
        self.load_state_dict(checkpoint)

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": asdict(self._schema) if self._schema else None,
            "model_state": self._model.state_dict() if self._has_model() else None,
            "trained": self._trained,
            "m_mean": self._tensor_to_list(self.m_mean),
            "m_std": self._tensor_to_list(self.m_std),
            "pca": self.pca,  # Pickled PCA object
            "configs": {
                "hidden_dim": self._hidden_dim,
                "n_layers": self._n_layers,
                "pca_components": self._pca_components,
            },
        }

    def load_state_dict(self, checkpoint: Mapping[str, Any]) -> None:
        saved_schema = checkpoint.get("schema")
        if saved_schema:
            self._schema = tools.FlattenSchema(**saved_schema)

        # Load PCA
        self.pca = checkpoint.get("pca")
        if self.pca:
            self._pca_components = self.pca.n_components_

        # Load configs if present
        configs = checkpoint.get("configs", {})
        if "hidden_dim" in configs:
            self._hidden_dim = configs["hidden_dim"]
        if "n_layers" in configs:
            self._n_layers = configs["n_layers"]

        self._build_noise_schedule()  # Rebuild in case device changed

        if self._schema:
            self._ensure_model()
            model_state = checkpoint.get("model_state")
            if model_state and self._has_model():
                try:
                    self.__model.load_state_dict(
                        model_state
                    )  # Direct access for assignment
                except RuntimeError as e:
                    _LOGGER.warning(
                        f"Failed to load model state: {e}. Architecture mismatch?"
                    )

        self.m_mean = self._tensor_from_list(checkpoint.get("m_mean"))
        self.m_std = self._tensor_from_list(checkpoint.get("m_std"))
        self._trained = bool(checkpoint.get("trained", self._trained))

    def sample(
        self, n_samples: int, target_metrics: Mapping[str, float], seed: int
    ) -> list[Mapping[str, Any]]:
        """Generate candidates conditioned on target metrics."""
        if not self._trained or not self._has_model() or self._schema is None:
            _LOGGER.warning("[diffusion] Model not trained.")
            return []

        # Prepare target metrics tensor
        # Must match _extract_metrics logic
        # target_metrics usually comes from exploration worker which has good keys
        iota = target_metrics.get("iota_bar", 0.0)  # check keys
        ar = target_metrics.get("aspect_ratio", 0.0)
        nfp = target_metrics.get("nfp", 3.0)
        is_qh = target_metrics.get("N", 0.0)  # 0 for QA

        m_vec = [iota, ar, nfp, is_qh]

        M = torch.tensor([m_vec], dtype=torch.float32).to(self._device)

        if self.m_mean is not None and self.m_std is not None:
            M = (M - self.m_mean) / self.m_std

        M = M.repeat(n_samples, 1)  # (N, metric_dim)

        self._model.eval()
        rng = torch.Generator(device=self._device)
        rng.manual_seed(seed)

        with torch.no_grad():
            latent_dim = (
                self._pca_components
                if self.pca
                else (self._schema.mpol * self._schema.ntor * 2)
            )

            # Start from noise
            x = torch.randn(
                n_samples,
                latent_dim,
                device=self._device,
                generator=rng,
            )

            # DDPM reverse process: x_{t-1} = mu_theta(x_t, t) + sigma_t * z
            # where mu_theta = (1/sqrt(alpha_t)) * (x_t - beta_t/sqrt(1-alpha_hat_t) * eps_theta)
            # and sigma_t = sqrt(beta_tilde_t) is the POSTERIOR variance (not beta_t!)
            for i in reversed(range(self._timesteps)):
                t = torch.full((n_samples,), i, device=self._device, dtype=torch.long)
                predicted_noise = self._model(x, t, M)

                alpha = self.alpha[i]
                alpha_hat = self.alpha_hat[i]
                beta_tilde = self.beta_tilde[i]  # Use posterior variance, not beta!

                if i > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                # Compute mean: mu_theta(x_t, t)
                mu = (1 / torch.sqrt(alpha)) * (
                    x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise
                )

                # Add noise with correct posterior variance
                x = mu + torch.sqrt(beta_tilde) * noise

            # Move to CPU
            latent_samples = x.cpu().numpy()

            # Inverse PCA
            if self.pca:
                flattened = self.pca.inverse_transform(latent_samples)
            else:
                flattened = latent_samples

        candidates = []
        # Extract nfp from target_metrics - must match conditioning logic (line 568)
        # Check "nfp" first (used by conditioning), then "number_of_field_periods"
        nfp_value = target_metrics.get(
            "nfp", target_metrics.get("number_of_field_periods", 3.0)
        )
        nfp = int(nfp_value)

        for k in range(n_samples):
            try:
                params = tools.structured_unflatten(flattened[k], self._schema)
                # Add n_field_periods to params (required for downstream evaluation)
                params["n_field_periods"] = nfp
                candidates.append(
                    {
                        "seed": seed + k,
                        "params": params,
                        "design_hash": tools.design_hash(params),
                        "source": "diffusion_conditional",
                        "target_metrics": target_metrics,
                    }
                )
            except Exception as e:
                _LOGGER.warning(f"Unflatten error: {e}")

        return candidates


class StellaratorVAE(nn.Module):
    """Variational Autoencoder for Stellarator Geometries.

    Encodes (r_cos, z_sin) Fourier coefficients into a latent vector z,
    and reconstructs them.
    """

    def __init__(
        self, mpol: int, ntor: int, latent_dim: int = 16, hidden_dim: int = 64
    ):
        super().__init__()
        self.mpol = mpol
        self.ntor = ntor
        self.grid_h = mpol + 1
        self.grid_w = 2 * ntor + 1
        self.input_channels = 2  # r_cos, z_sin
        self.latent_dim = latent_dim

        # --- Encoder ---
        # Input: (B, 2, H, W)
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(self.input_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.SiLU(),
            nn.Conv2d(
                hidden_dim * 2, hidden_dim * 4, kernel_size=3, stride=2, padding=1
            ),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.SiLU(),
        )

        # Calculate flattened size after convolutions
        # H_out = ceil(H/4), W_out = ceil(W/4) roughly due to 2 strides of 2
        # We'll compute it dynamically in forward or pre-calc.
        # For robustness, we use AdaptiveAvgPool to fix the size before dense layers
        self.encoder_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_mu = nn.Linear(hidden_dim * 4, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 4, latent_dim)

        # --- Decoder ---
        # We need to upsample back to (H, W).
        # Strategy: Linear -> (hidden*4, H//4, W//4) -> Upsample -> Conv -> Upsample -> Conv

        # To keep it simple and robust to odd H/W, we'll use a dense projection
        # to the full grid size times channels, then refine with Conv.
        # This might be parameter heavy but safer for exact shape matching.
        # Or: Linear -> (Hidden, H, W) -> ResNet blocks -> (2, H, W)

        self.decoder_input = nn.Linear(
            latent_dim, hidden_dim * self.grid_h * self.grid_w
        )

        self.decoder_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.SiLU(),
            nn.Conv2d(hidden_dim // 2, self.input_channels, kernel_size=3, padding=1),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input to [mu, logvar]."""
        x = self.encoder_conv(x)
        x = self.encoder_pool(x).flatten(1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z from N(mu, var)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode z to reconstruction."""
        x = self.decoder_input(z)
        x = x.view(-1, 64, self.grid_h, self.grid_w)  # Reshape to (B, hidden, H, W)
        x = self.decoder_conv(x)
        return x

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


class GenerativeDesignModel:
    """Manager for the VAE-based generative model."""

    def __init__(
        self,
        *,
        min_samples: int = 32,
        latent_dim: int = 16,
        learning_rate: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 32,
        device: str = "cpu",
        kl_weight: float = 0.001,
    ) -> None:
        self._min_samples = min_samples
        self._latent_dim = latent_dim
        self._lr = learning_rate
        self._epochs = epochs
        self._batch_size = batch_size
        self._device = (
            device if torch.cuda.is_available() and device == "cuda" else "cpu"
        )
        self._kl_weight = kl_weight

        self.__model: StellaratorVAE | None = None  # Use property for access
        self._schema: tools.FlattenSchema | None = None
        self._optimizer: optim.Optimizer | None = None
        self._trained = False
        self._last_fit_count = 0

    @property
    def _model(self) -> StellaratorVAE:
        """Access the VAE model, raising if not initialized.

        Raises:
            RuntimeError: If model has not been created via fit().
        """
        if self.__model is None:
            raise RuntimeError(
                "GenerativeDesignModel not initialized. "
                "Call fit() with sufficient samples first."
            )
        return self.__model

    @_model.setter
    def _model(self, value: StellaratorVAE | None) -> None:
        self.__model = value

    def _has_model(self) -> bool:
        """Check if model exists without raising."""
        return self.__model is not None

    def fit(
        self,
        candidates: Sequence[Mapping[str, Any]],
        *,
        force_retrain: bool = False,
    ) -> None:
        """Train the VAE on a list of candidates."""
        sample_count = len(candidates)
        if sample_count < self._min_samples:
            _LOGGER.info(
                "[generative] Skipping training: %d samples (< %d)",
                sample_count,
                self._min_samples,
            )
            return

        if (
            self._trained
            and not force_retrain
            and (sample_count - self._last_fit_count < 50)
        ):
            # Avoid retraining too often if not much new data
            return

        _LOGGER.info("[generative] Training VAE on %d samples...", sample_count)

        # 1. Prepare Data
        vectors = []
        for cand in candidates:
            params = cand.get("params") or cand.get("candidate_params")
            if not params:
                continue
            vec, schema = tools.structured_flatten(params, schema=self._schema)
            if self._schema is None:
                self._schema = schema
            vectors.append(vec)

        if not vectors:
            return

        X = torch.tensor(np.vstack(vectors), dtype=torch.float32).to(self._device)

        dataset = TensorDataset(X)
        loader = DataLoader(dataset, batch_size=self._batch_size, shuffle=True)

        # 2. Initialize Model
        if not self._has_model():
            self._model = StellaratorVAE(
                mpol=self._schema.mpol,
                ntor=self._schema.ntor,
                latent_dim=self._latent_dim,
            ).to(self._device)
            self._optimizer = optim.Adam(self._model.parameters(), lr=self._lr)

        # 3. Train Loop
        self._model.train()  # Property guarantees non-None after init above

        grid_size = self._model.grid_h * self._model.grid_w
        half_size = grid_size  # for one channel in flattened vector

        for epoch in range(self._epochs):
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_kl = 0.0

            for (xb,) in loader:
                self._optimizer.zero_grad()

                # Reshape input flattened -> (B, 2, H, W)
                batch_size = xb.shape[0]
                r_cos = xb[:, :half_size].view(
                    batch_size, 1, self._model.grid_h, self._model.grid_w
                )
                z_sin = xb[:, half_size:].view(
                    batch_size, 1, self._model.grid_h, self._model.grid_w
                )
                x_img = torch.cat([r_cos, z_sin], dim=1)

                recon, mu, logvar = self._model(x_img)

                # Loss - using proper per-element normalization for both terms
                # This ensures consistent scaling regardless of input dimensions
                #
                # Reconstruction: MSE per element (mean over batch AND spatial dims)
                recon_loss = F.mse_loss(recon, x_img, reduction="mean")

                # KLD: -0.5 * mean(1 + log(sigma^2) - mu^2 - sigma^2)
                # Using mean over both batch and latent dims for consistent scaling
                kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

                # With proper normalization, kl_weight should be ~1.0
                # But we keep user-specified weight for backward compatibility
                loss = recon_loss + self._kl_weight * kld_loss

                loss.backward()
                self._optimizer.step()

                epoch_loss += loss.item()
                epoch_recon += recon_loss.item()
                epoch_kl += kld_loss.item()

            # if epoch % 20 == 0:
            #    _LOGGER.info(f"Epoch {epoch}: loss={epoch_loss:.4f} (Recon={epoch_recon:.4f}, KL={epoch_kl:.4f})")

        self._trained = True
        self._last_fit_count = sample_count
        _LOGGER.info("[generative] VAE training complete.")

    def sample(self, n_samples: int, seed: int) -> list[Mapping[str, Any]]:
        """Generate new candidates by sampling from the latent space."""
        if not self._trained or not self._has_model() or self._schema is None:
            _LOGGER.warning("[generative] Model not trained, cannot sample.")
            return []

        _LOGGER.info(
            "[generative] Sampling %d candidates from latent space...", n_samples
        )

        rng = torch.Generator(device=self._device)
        rng.manual_seed(seed)

        self._model.eval()
        with torch.no_grad():
            # Sample z ~ N(0, I)
            z = torch.randn(
                n_samples, self._latent_dim, device=self._device, generator=rng
            )

            # Decode
            recon = self._model.decode(z)  # (B, 2, H, W)

            # Flatten back to schema format
            # recon[:, 0] is r_cos, recon[:, 1] is z_sin
            r_cos = recon[:, 0].flatten(1)  # (B, H*W)
            z_sin = recon[:, 1].flatten(1)  # (B, H*W)

            flattened = torch.cat([r_cos, z_sin], dim=1).cpu().numpy()

        candidates = []
        for i in range(n_samples):
            vec = flattened[i]
            try:
                params = tools.structured_unflatten(vec, self._schema)
                # Add design hash and source
                candidates.append(
                    {
                        "seed": seed + i,
                        "params": params,
                        "design_hash": tools.design_hash(params),
                        "source": "vae_generative",
                        "constraint_distance": 0.0,  # Unknown, assumed good distribution
                    }
                )
            except Exception as e:
                _LOGGER.warning(f"Failed to unflatten generated vector: {e}")

        return candidates
