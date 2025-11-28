"""Generative Models for Phase 4 Exploration (VAE / Latent Optimization).

This module implements the Variational Autoencoder (VAE) for the "Generative Models"
phase of the AI Scientist V2 upgrade. It enables:
1.  Learning a smooth latent space for stellarator geometries.
2.  Sampling novel candidates from this latent space.
3.  (Future) Performing gradient-based optimization in the latent space.
"""

from __future__ import annotations

from dataclasses import asdict
import logging
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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


class Block(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, dropout: float = 0.1):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # First Conv
        h = self.conv1(x)
        h = self.relu(h)
        h = self.bnorm1(h)
        
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Broadcast time embedding
        h = h + time_emb[(..., ) + (None, ) * 2]
        
        # Second Conv
        h = self.conv2(h)
        h = self.relu(h)
        h = self.bnorm2(h)
        
        return self.dropout(h)


class StellaratorDiffusion(nn.Module):
    """Conditional Diffusion Model for Stellarator Geometries using a simple ResNet architecture."""
    def __init__(self, mpol: int, ntor: int, metric_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.mpol = mpol
        self.ntor = ntor
        self.grid_h = mpol + 1
        self.grid_w = 2 * ntor + 1
        self.input_channels = 2
        time_dim = 32

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        
        self.metric_mlp = nn.Sequential(
            nn.Linear(metric_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.inc = nn.Conv2d(self.input_channels, hidden_dim, 3, padding=1)
        
        self.blocks = nn.ModuleList([
            Block(hidden_dim, hidden_dim, time_dim) for _ in range(4)
        ])
        
        self.outc = nn.Conv2d(hidden_dim, self.input_channels, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, metrics: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        m_emb = self.metric_mlp(metrics)
        cond = t_emb + m_emb
        
        h = self.inc(x)
        for block in self.blocks:
            h = h + block(h, cond) # Residual connection
            
        return self.outc(h)


class DiffusionDesignModel:
    """Manager for Conditional Diffusion Model."""

    METRIC_KEYS = [
        "aspect_ratio",
        "minimum_normalized_magnetic_gradient_scale_length",
        "max_elongation",
        "edge_rotational_transform_over_n_field_periods"
    ]

    def __init__(
        self,
        *,
        min_samples: int = 32,
        learning_rate: float = 1e-3,
        epochs: int = 200,
        batch_size: int = 32,
        device: str = "cpu",
        timesteps: int = 300,
        ) -> None:
        self._min_samples = min_samples
        self._lr = learning_rate
        self._epochs = epochs
        self._batch_size = batch_size
        self._device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self._timesteps = timesteps

        self._model: StellaratorDiffusion | None = None
        self._schema: tools.FlattenSchema | None = None
        self._optimizer: optim.Optimizer | None = None
        self._trained = False
        self.m_mean: torch.Tensor | None = None
        self.m_std: torch.Tensor | None = None

        self._build_noise_schedule()

    def _build_noise_schedule(self) -> None:
        self.beta = torch.linspace(1e-4, 0.02, self._timesteps, device=self._device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def _ensure_model(self) -> None:
        if self._schema is None or self._model is not None:
            return
        self._model = StellaratorDiffusion(
            mpol=self._schema.mpol,
            ntor=self._schema.ntor,
            metric_dim=len(self.METRIC_KEYS),
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
        """Extract normalized metrics vector."""
        # Robust extraction: handle various locations
        metrics = cand.get("metrics") or cand
        if not isinstance(metrics, dict):
            # Try model_dump if pydantic
            if hasattr(metrics, "model_dump"):
                metrics = metrics.model_dump()
            else:
                return None
                
        vec = []
        for k in self.METRIC_KEYS:
            val = metrics.get(k)
            if val is None:
                return None # Skip incomplete data
            vec.append(float(val))
        return np.array(vec, dtype=np.float32)

    def fit(self, candidates: Sequence[Mapping[str, Any]]) -> None:
        sample_count = len(candidates)
        if sample_count < self._min_samples:
            _LOGGER.info("[diffusion] Skipping training: %d samples < %d", sample_count, self._min_samples)
            return
            
        _LOGGER.info("[diffusion] Training on %d samples...", sample_count)
        
        # Prepare Data
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
            
        X = torch.tensor(np.vstack(vectors), dtype=torch.float32).to(self._device)
        M = torch.tensor(np.vstack(metrics_list), dtype=torch.float32).to(self._device)
        
        # Normalize metrics (simple min-max or z-score?)
        # For now, just use raw. Ideally we should normalize.
        # Let's do simple standardization
        self.m_mean = M.mean(dim=0)
        self.m_std = M.std(dim=0) + 1e-6
        M = (M - self.m_mean) / self.m_std
        
        dataset = TensorDataset(X, M)
        loader = DataLoader(dataset, batch_size=self._batch_size, shuffle=True)
        
        # Init Model
        if self._schema is None:
            return

        self._ensure_model()
        if self._model is None:
            return

        self._model.train()
        
        grid_size = self._model.grid_h * self._model.grid_w
        half_size = grid_size
        
        for epoch in range(self._epochs):
            epoch_loss = 0.0
            for xb, mb in loader:
                self._optimizer.zero_grad()
                
                # Reshape geometry
                B = xb.shape[0]
                r_cos = xb[:, :half_size].view(B, 1, self._model.grid_h, self._model.grid_w)
                z_sin = xb[:, half_size:].view(B, 1, self._model.grid_h, self._model.grid_w)
                x0 = torch.cat([r_cos, z_sin], dim=1)
                
                # Sample t
                t = torch.randint(0, self._timesteps, (B,), device=self._device).long()
                
                # Noise
                noise = torch.randn_like(x0)
                
                # Forward diffusion q(x_t | x_0)
                alpha_hat_t = self.alpha_hat[t][:, None, None, None]
                xt = torch.sqrt(alpha_hat_t) * x0 + torch.sqrt(1 - alpha_hat_t) * noise
                
                # Predict noise
                noise_pred = self._model(xt, t, mb)
                
                loss = F.mse_loss(noise_pred, noise)
                loss.backward()
                self._optimizer.step()
                
                epoch_loss += loss.item()
                
            if epoch % 50 == 0:
                 _LOGGER.info(f"Epoch {epoch}: loss={epoch_loss:.4f}")
                 
        self._trained = True
        _LOGGER.info("[diffusion] Training complete.")

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": asdict(self._schema) if self._schema else None,
            "model_state": self._model.state_dict() if self._model else None,
            "trained": self._trained,
            "m_mean": self._tensor_to_list(self.m_mean),
            "m_std": self._tensor_to_list(self.m_std),
            "beta": self._tensor_to_list(self.beta),
            "alpha": self._tensor_to_list(self.alpha),
            "alpha_hat": self._tensor_to_list(self.alpha_hat),
            "timesteps": int(self._timesteps),
        }

    def load_state_dict(self, checkpoint: Mapping[str, Any]) -> None:
        saved_schema = checkpoint.get("schema")
        if saved_schema:
            self._schema = tools.FlattenSchema(**saved_schema)

        saved_timesteps = checkpoint.get("timesteps")
        if isinstance(saved_timesteps, int):
            self._timesteps = saved_timesteps
        self._build_noise_schedule()

        beta_data = checkpoint.get("beta")
        if beta_data:
            self.beta = self._tensor_from_list(beta_data)
        alpha_data = checkpoint.get("alpha")
        if alpha_data:
            self.alpha = self._tensor_from_list(alpha_data)
        alpha_hat_data = checkpoint.get("alpha_hat")
        if alpha_hat_data:
            self.alpha_hat = self._tensor_from_list(alpha_hat_data)

        if self._schema:
            self._ensure_model()
            if self._model is not None:
                model_state = checkpoint.get("model_state")
                if model_state:
                    self._model.load_state_dict(model_state)

        self.m_mean = self._tensor_from_list(checkpoint.get("m_mean"))
        self.m_std = self._tensor_from_list(checkpoint.get("m_std"))
        self._trained = bool(checkpoint.get("trained", self._trained))

    def sample(
        self,
        n_samples: int,
        target_metrics: Mapping[str, float],
        seed: int
    ) -> list[Mapping[str, Any]]:
        """Generate candidates conditioned on target metrics."""
        if not self._trained or self._model is None or self._schema is None:
            _LOGGER.warning("[diffusion] Model not trained.")
            return []
            
        # Prepare target metrics tensor
        m_vec = []
        for k in self.METRIC_KEYS:
            m_vec.append(target_metrics.get(k, 0.0))
        
        M = torch.tensor([m_vec], dtype=torch.float32).to(self._device)
        # Normalize
        M = (M - self.m_mean) / self.m_std
        M = M.repeat(n_samples, 1) # (N, metric_dim)
        
        self._model.eval()
        rng = torch.Generator(device=self._device)
        rng.manual_seed(seed)
        
        with torch.no_grad():
            # Start from noise
            x = torch.randn(
                n_samples, 2, self._model.grid_h, self._model.grid_w, 
                device=self._device, generator=rng
            )
            
            for i in reversed(range(self._timesteps)):
                t = torch.full((n_samples,), i, device=self._device, dtype=torch.long)
                predicted_noise = self._model(x, t, M)
                
                alpha = self.alpha[i]
                alpha_hat = self.alpha_hat[i]
                beta = self.beta[i]
                
                if i > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                    
                x = (1 / torch.sqrt(alpha)) * (
                    x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise
                ) + torch.sqrt(beta) * noise
                
            # Decode x -> params
            r_cos = x[:, 0].flatten(1)
            z_sin = x[:, 1].flatten(1)
            flattened = torch.cat([r_cos, z_sin], dim=1).cpu().numpy()
            
        candidates = []
        for k in range(n_samples):
            try:
                params = tools.structured_unflatten(flattened[k], self._schema)
                candidates.append({
                    "seed": seed + k,
                    "params": params,
                    "design_hash": tools.design_hash(params),
                    "source": "diffusion_conditional",
                    "target_metrics": target_metrics
                })
            except Exception as e:
                _LOGGER.warning(f"Unflatten error: {e}")
                
        return candidates


class StellaratorVAE(nn.Module):
    """Variational Autoencoder for Stellarator Geometries.
    
    Encodes (r_cos, z_sin) Fourier coefficients into a latent vector z,
    and reconstructs them.
    """

    def __init__(
        self, 
        mpol: int, 
        ntor: int, 
        latent_dim: int = 16, 
        hidden_dim: int = 64
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
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, stride=2, padding=1),
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
        
        self.decoder_input = nn.Linear(latent_dim, hidden_dim * self.grid_h * self.grid_w)
        
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
        x = x.view(-1, 64, self.grid_h, self.grid_w) # Reshape to (B, hidden, H, W)
        x = self.decoder_conv(x)
        return x

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        self._device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self._kl_weight = kl_weight

        self._model: StellaratorVAE | None = None
        self._schema: tools.FlattenSchema | None = None
        self._optimizer: optim.Optimizer | None = None
        self._trained = False
        self._last_fit_count = 0

    def fit(
        self,
        candidates: Sequence[Mapping[str, Any]],
        *,
        force_retrain: bool = False,
    ) -> None:
        """Train the VAE on a list of candidates."""
        sample_count = len(candidates)
        if sample_count < self._min_samples:
            _LOGGER.info("[generative] Skipping training: %d samples (< %d)", sample_count, self._min_samples)
            return

        if self._trained and not force_retrain and (sample_count - self._last_fit_count < 50):
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
        if self._model is None:
            self._model = StellaratorVAE(
                mpol=self._schema.mpol, 
                ntor=self._schema.ntor,
                latent_dim=self._latent_dim
            ).to(self._device)
            self._optimizer = optim.Adam(self._model.parameters(), lr=self._lr)

        # 3. Train Loop
        self._model.train()
        
        grid_size = self._model.grid_h * self._model.grid_w
        half_size = grid_size # for one channel in flattened vector
        
        for epoch in range(self._epochs):
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_kl = 0.0
            
            for (xb,) in loader:
                self._optimizer.zero_grad()
                
                # Reshape input flattened -> (B, 2, H, W)
                batch_size = xb.shape[0]
                r_cos = xb[:, :half_size].view(batch_size, 1, self._model.grid_h, self._model.grid_w)
                z_sin = xb[:, half_size:].view(batch_size, 1, self._model.grid_h, self._model.grid_w)
                x_img = torch.cat([r_cos, z_sin], dim=1)
                
                recon, mu, logvar = self._model(x_img)
                
                # Loss
                # MSE between recon and x_img
                recon_loss = F.mse_loss(recon, x_img, reduction='sum') / batch_size
                
                # KLD
                # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
                kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
                
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
        if not self._trained or self._model is None or self._schema is None:
            _LOGGER.warning("[generative] Model not trained, cannot sample.")
            return []
            
        _LOGGER.info("[generative] Sampling %d candidates from latent space...", n_samples)
        
        rng = torch.Generator(device=self._device)
        rng.manual_seed(seed)
        
        self._model.eval()
        with torch.no_grad():
            # Sample z ~ N(0, I)
            z = torch.randn(n_samples, self._latent_dim, device=self._device, generator=rng)
            
            # Decode
            recon = self._model.decode(z) # (B, 2, H, W)
            
            # Flatten back to schema format
            # recon[:, 0] is r_cos, recon[:, 1] is z_sin
            r_cos = recon[:, 0].flatten(1) # (B, H*W)
            z_sin = recon[:, 1].flatten(1) # (B, H*W)
            
            flattened = torch.cat([r_cos, z_sin], dim=1).cpu().numpy()
            
        candidates = []
        for i in range(n_samples):
            vec = flattened[i]
            try:
                params = tools.structured_unflatten(vec, self._schema)
                # Add design hash and source
                candidates.append({
                    "seed": seed + i,
                    "params": params,
                    "design_hash": tools.design_hash(params),
                    "source": "vae_generative",
                    "constraint_distance": 0.0 # Unknown, assumed good distribution
                })
            except Exception as e:
                _LOGGER.warning(f"Failed to unflatten generated vector: {e}")
                
        return candidates
