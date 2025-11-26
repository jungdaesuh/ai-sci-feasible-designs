"""Generative Models for Phase 4 Exploration (VAE / Latent Optimization).

This module implements the Variational Autoencoder (VAE) for the "Generative Models"
phase of the AI Scientist V2 upgrade. It enables:
1.  Learning a smooth latent space for stellarator geometries.
2.  Sampling novel candidates from this latent space.
3.  (Future) Performing gradient-based optimization in the latent space.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ai_scientist import tools

_LOGGER = logging.getLogger(__name__)


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
