"""Version 2.0 Surrogate: Neural Operator & Geometric Deep Learning.

This module implements the Physics-Informed Surrogate (Phase 2) of the AI Scientist V2 upgrade.
It uses a Deep Learning model (StellaratorNeuralOp) operating directly on the Fourier
boundary coefficients to predict objective values and feasibility metrics.

Implementation Details:
- Inputs: Flattened Fourier coefficients (r_cos, z_sin).
- Architecture: "Spectral" Convolutional Network.
  - Reshapes flattened inputs back to (2, mpol+1, 2*ntor+1) grids.
  - Applies 2D Convolutions (effectively mixing modes in frequency domain).
  - MLP heads for specific scalar outputs (Objective, MHD, QI, Elongation).
- Equivariance: Operates on the spectral representation which implicitly handles toroidal symmetry.
  Full SE(3) equivariance would require e3nn (currently unavailable), so we focus on
  spectral convolutions as the "Neural Operator" component.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ai_scientist import tools
from ai_scientist.optim import equivariance, geometry
from ai_scientist.optim.surrogate import BaseSurrogate, SurrogatePrediction


class StellaratorNeuralOp(nn.Module):
    """Spectral Convolutional Neural Operator with Geometric Equivariance.

    Hybrid architecture:
    1. Spectral Branch: 2D Convolutions on Fourier coefficient grid (r_cos, z_sin).
    2. Geometric Branch: PointNet with T-Net alignment operating on generated 3D point clouds.
       This provides approximate SE(3) invariance and physical grounding.
    """

    def __init__(self, mpol: int, ntor: int, nfp: int = 1, hidden_dim: int = 64):
        super().__init__()
        self.mpol = mpol
        self.ntor = ntor
        self.nfp = nfp
        self.grid_h = mpol + 1
        self.grid_w = 2 * ntor + 1
        self.input_channels = 2  # r_cos, z_sin

        # 1. Spectral Branch (Operating on coefficient grid)
        self.conv_net = nn.Sequential(
            nn.Conv2d(self.input_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # 2. Geometric Branch (Operating on 3D Point Cloud)
        # We use a modest grid for the surrogate to keep training fast.
        self.geo_n_theta = 16
        self.geo_n_zeta = 16
        self.geo_dim = 128
        self.geo_encoder = equivariance.PointNetEncoder(embedding_dim=self.geo_dim, align_input=True)

        # Multi-head output (Fusion of Spectral + Geometric)
        fusion_dim = hidden_dim + self.geo_dim
        
        self.head_base = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        self.head_objective = nn.Linear(hidden_dim, 1)
        self.head_mhd = nn.Linear(hidden_dim, 1)
        self.head_qi = nn.Linear(hidden_dim, 1)
        self.head_elongation = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Flattened input tensor of shape (Batch, 2 * (mpol+1) * (2*ntor+1))
        """
        batch_size = x.shape[0]
        
        # --- Spectral Branch ---
        half_size = self.grid_h * self.grid_w
        r_cos_flat = x[:, :half_size]
        z_sin_flat = x[:, half_size:]

        r_cos_grid = r_cos_flat.view(batch_size, 1, self.grid_h, self.grid_w)
        z_sin_grid = z_sin_flat.view(batch_size, 1, self.grid_h, self.grid_w)
        spectral_grid = torch.cat([r_cos_grid, z_sin_grid], dim=1) # (B, 2, H, W)

        spectral_feat = self.conv_net(spectral_grid) # (B, hidden, H, W)
        spectral_vec = self.global_pool(spectral_feat).view(batch_size, -1) # (B, hidden)
        
        # --- Geometric Branch ---
        # Recover (B, mpol+1, 2*ntor+1) for geometry tool
        r_cos_in = r_cos_grid.squeeze(1)
        z_sin_in = z_sin_grid.squeeze(1)
        
        # Generate Point Cloud (Differentiable)
        R, Z, Phi = geometry.batch_fourier_to_real_space(
            r_cos_in, z_sin_in, 
            n_field_periods=self.nfp,
            n_theta=self.geo_n_theta,
            n_zeta=self.geo_n_zeta
        )
        
        X, Y, Z_cart = geometry.to_cartesian(R, Z, Phi)
        
        # Stack to (Batch, 3, N_points)
        # R, Z, Phi are (Batch, T, ZetaTotal)
        # Flatten spatial dims
        X_flat = X.view(batch_size, -1)
        Y_flat = Y.view(batch_size, -1)
        Z_flat = Z_cart.view(batch_size, -1)
        
        points = torch.stack([X_flat, Y_flat, Z_flat], dim=1) # (B, 3, N)
        
        geo_vec = self.geo_encoder(points) # (B, geo_dim)
        
        # --- Fusion ---
        combined = torch.cat([spectral_vec, geo_vec], dim=1)
        
        base = self.head_base(combined)
        
        pred_obj = self.head_objective(base).squeeze(-1)
        pred_mhd = self.head_mhd(base).squeeze(-1)
        pred_qi = self.head_qi(base).squeeze(-1)
        pred_elong = self.head_elongation(base).squeeze(-1)

        return pred_obj, pred_mhd, pred_qi, pred_elong


class NeuralOperatorSurrogate(BaseSurrogate):
    """Deep Learning Surrogate (V2) using PyTorch.
    
    Uses StellaratorNeuralOp to predict metrics from boundary coefficients.
    """

    def __init__(
        self,
        *,
        min_samples: int = 32,
        points_cadence: int = 64,
        cycle_cadence: int = 5,
        device: str = "cpu",
        learning_rate: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 32,
    ) -> None:
        self._min_samples = min_samples
        self._points_cadence = points_cadence
        self._cycle_cadence = cycle_cadence
        self._device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self._lr = learning_rate
        self._epochs = epochs
        self._batch_size = batch_size
        
        self._trained = False
        self._last_fit_count = 0
        self._last_fit_cycle = 0
        
        self._model: StellaratorNeuralOp | None = None
        self._schema: tools.FlattenSchema | None = None
        self._optimizer: optim.Optimizer | None = None
        self._nfp: int | None = None

    def fit(
        self,
        metrics_list: Sequence[Mapping[str, Any]],
        target_values: Sequence[float],
        *,
        minimize_objective: bool,
        cycle: int | None = None,
    ) -> None:
        """Train the Neural Operator on the accumulated history."""
        sample_count = len(metrics_list)
        self._last_fit_count = sample_count
        if cycle is not None:
            self._last_fit_cycle = int(cycle)

        if sample_count < self._min_samples:
            logging.info(
                "[surrogate_v2] cold start: %d samples (< %d required for DL)", 
                sample_count, self._min_samples
            )
            self._trained = False
            return

        logging.info(
            "[surrogate_v2] Training Neural Operator on %d samples (Device: %s)...", 
            sample_count, self._device
        )

        # 1. Prepare Data
        features_list = []
        
        # Capture schema and nfp from the first item if not set
        for i, metrics in enumerate(metrics_list):
            params = metrics.get("candidate_params") or metrics.get("params", {})
            
            if i == 0:
                if self._nfp is None:
                    self._nfp = int(params.get("n_field_periods") or params.get("nfp", 1))

            vector, schema = tools.structured_flatten(params, schema=self._schema)
            if self._schema is None:
                self._schema = schema
            features_list.append(vector)
            
        if not features_list:
            return

        X = torch.tensor(np.vstack(features_list), dtype=torch.float32).to(self._device)
        y_obj = torch.tensor(target_values, dtype=torch.float32).to(self._device)
        
        # Auxiliary targets
        y_mhd_list, y_qi_list, y_elong_list = [], [], []
        for metrics in metrics_list:
            m_payload = metrics.get("metrics", metrics)
            y_mhd_list.append(float(m_payload.get("vacuum_well", -1.0)))
            y_qi_list.append(float(m_payload.get("qi", 1.0)))
            y_elong_list.append(float(m_payload.get("max_elongation", 10.0)))
            
        y_mhd = torch.tensor(y_mhd_list, dtype=torch.float32).to(self._device)
        y_qi = torch.tensor(y_qi_list, dtype=torch.float32).to(self._device)
        y_elong = torch.tensor(y_elong_list, dtype=torch.float32).to(self._device)

        dataset = TensorDataset(X, y_obj, y_mhd, y_qi, y_elong)
        loader = DataLoader(dataset, batch_size=self._batch_size, shuffle=True)

        # 2. Initialize Model if needed
        if self._model is None or self._model.mpol != self._schema.mpol or self._model.ntor != self._schema.ntor:
            logging.info("[surrogate_v2] Initializing StellaratorNeuralOp with mpol=%d, ntor=%d, nfp=%d", 
                         self._schema.mpol, self._schema.ntor, self._nfp)
            self._model = StellaratorNeuralOp(
                mpol=self._schema.mpol, 
                ntor=self._schema.ntor,
                nfp=self._nfp or 1
            ).to(self._device)
            self._optimizer = optim.Adam(self._model.parameters(), lr=self._lr)

        # 3. Train Loop
        self._model.train()
        criterion = nn.MSELoss()
        
        for epoch in range(self._epochs):
            epoch_loss = 0.0
            for xb, yb_obj, yb_mhd, yb_qi, yb_elong in loader:
                self._optimizer.zero_grad()
                pred_obj, pred_mhd, pred_qi, pred_elong = self._model(xb)
                
                # Multi-task loss
                loss = (
                    criterion(pred_obj, yb_obj) + 
                    0.5 * criterion(pred_mhd, yb_mhd) + 
                    0.5 * criterion(pred_qi, yb_qi) + 
                    0.5 * criterion(pred_elong, yb_elong)
                )
                
                loss.backward()
                self._optimizer.step()
                epoch_loss += loss.item()
            
            # Optional: Early stopping or logging
            # if epoch % 10 == 0:
            #    logging.debug(f"Epoch {epoch}: loss {epoch_loss:.4f}")

        self._trained = True

    def should_retrain(self, sample_count: int, cycle: int | None = None) -> bool:
        """Check if the deep surrogate should be retrained."""
        if not self._trained:
            return True
        
        delta_points = sample_count - self._last_fit_count
        if delta_points >= self._points_cadence:
            return True
        
        if cycle is None:
            return False
        return (cycle - self._last_fit_cycle) >= self._cycle_cadence

    def predict_torch(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Differentiable prediction for optimization loop."""
        if self._model is None:
            raise RuntimeError("NeuralOperatorSurrogate not initialized/trained")
        
        # Ensure we are on the right device
        if x.device != torch.device(self._device):
            x = x.to(self._device)
            
        return self._model(x)

    def rank_candidates(
        self,
        candidates: Sequence[Mapping[str, Any]],
        *,
        minimize_objective: bool,
        exploration_ratio: float = 0.0,
    ) -> list[SurrogatePrediction]:
        """Rank candidates using the Deep Surrogate."""
        if not candidates:
            return []
            
        if not self._trained or self._model is None:
            # Fallback to heuristic if not trained
            logging.info("[surrogate_v2] Model not trained, using heuristics")
            cold_ranks: list[SurrogatePrediction] = []
            for candidate in candidates:
                # Just give 0 scores
                cold_ranks.append(SurrogatePrediction(
                    expected_value=0.0, prob_feasible=0.0, predicted_objective=0.0,
                    minimize_objective=minimize_objective, metadata=candidate
                ))
            return cold_ranks

        # Vectorize candidates
        vectors = []
        for candidate in candidates:
            params = candidate.get("candidate_params") or candidate.get("params", {})
            # Use the stored schema to ensure consistency
            vec, _ = tools.structured_flatten(params, schema=self._schema)
            vectors.append(vec)
            
        X = torch.tensor(np.vstack(vectors), dtype=torch.float32).to(self._device)
        
        self._model.eval()
        with torch.no_grad():
            pred_obj, pred_mhd, pred_qi, pred_elong = self._model(X)
            
        # Convert to CPU/numpy
        pred_obj = pred_obj.cpu().numpy()
        pred_mhd = pred_mhd.cpu().numpy()
        pred_qi = pred_qi.cpu().numpy()
        pred_elong = pred_elong.cpu().numpy()
        
        predictions: list[SurrogatePrediction] = []
        for i, candidate in enumerate(candidates):
            # Approximate feasibility probability based on predicted margins?
            # For now, we don't have a classification head.
            # We can assume if predicted metrics are good, it's feasible.
            # Let's use a dummy 0.5 or maybe infer from MHD?
            # If vacuum_well (mhd) < 0, it's bad.
            
            # Simple logic: if mhd > 0 and qi < threshold, high prob.
            # This is just for ranking.
            is_likely_feasible = (pred_mhd[i] >= 0)
            prob_feasible = 0.8 if is_likely_feasible else 0.2
            
            obj_val = float(pred_obj[i])
            
            constraint_distance = float(candidate.get("constraint_distance", 0.0))
            constraint_distance = max(0.0, constraint_distance)
            
            oriented = -obj_val if minimize_objective else obj_val
            
            # Exploration bonus
            uncertainty = prob_feasible * (1.0 - prob_feasible)
            exploration_weight = max(0.0, float(exploration_ratio)) * 0.1
            
            score = (prob_feasible - constraint_distance) + exploration_weight * uncertainty
            
            predictions.append(
                SurrogatePrediction(
                    expected_value=score,
                    prob_feasible=prob_feasible,
                    predicted_objective=obj_val,
                    minimize_objective=minimize_objective,
                    metadata=candidate,
                    predicted_mhd=float(pred_mhd[i]),
                    predicted_qi=float(pred_qi[i]),
                    predicted_elongation=float(pred_elong[i]),
                )
            )
            
        return sorted(predictions, key=lambda item: item.expected_value, reverse=True)
