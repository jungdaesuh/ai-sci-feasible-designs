# ruff: noqa: F722, F821
# pyright: reportUndefinedVariable=false
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
- Equivariance: Uses a Hybrid approach.
  - Spectral Branch: Operates on the spectral grid (not SE(3) equivariant).
  - Geometric Branch: Uses PointNet with T-Net alignment and Random Rotation Augmentation
    to enforce SE(3) invariance on physical geometry features.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from jaxtyping import Float
from torch.utils.data import DataLoader, TensorDataset

from ai_scientist import tools
from ai_scientist.optim import equivariance, geometry
from ai_scientist.optim.surrogate import BaseSurrogate, SurrogatePrediction


class StellaratorNeuralOp(nn.Module):
    """Spectral Convolutional Neural Operator with Geometric Equivariance.

    Hybrid architecture:
    1. Spectral Branch: 2D Convolutions on Fourier coefficient grid (r_cos, z_sin).
    2. Geometric Branch: PointNet with T-Net alignment operating on generated 3D point clouds.
       We apply Random Rotation Augmentation during training to enforce SE(3) invariance.
    """

    def __init__(self, mpol: int, ntor: int, hidden_dim: int = 64):
        super().__init__()
        self.mpol = mpol
        self.ntor = ntor
        self.grid_h = mpol + 1
        self.grid_w = 2 * ntor + 1
        self.input_channels = 2  # r_cos, z_sin

        # 1. Spectral Branch (Operating on coefficient grid)
        # Dropout added after each SiLU for regularization (AoT recommendation)
        self.conv_net = nn.Sequential(
            nn.Conv2d(self.input_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # 2. Geometric Branch (Operating on 3D Point Cloud)
        # We use a modest grid for the surrogate to keep training fast.
        self.geo_n_theta = 16
        self.geo_n_zeta = 64
        self.geo_dim = 128

        # Cache theta grids and trig terms to avoid regenerating every forward pass
        # These are registered as buffers so they move with the model to correct device
        self._init_cached_grids()

        self.geo_encoder = equivariance.PointNetEncoder(
            embedding_dim=self.geo_dim, align_input=True
        )

        # Multi-head output (Fusion of Spectral + Geometric + Fidelity)
        # Fidelity embedding (Issue #10): 2 levels (screen=0, promote=1)
        self.fidelity_embedding_dim = 8
        self.fidelity_embedding = nn.Embedding(
            num_embeddings=2, embedding_dim=self.fidelity_embedding_dim
        )
        fusion_dim = hidden_dim + self.geo_dim + self.fidelity_embedding_dim

        # Dropout added for regularization (AoT recommendation)
        self.head_base = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=0.1),
        )

        self.head_objective = nn.Linear(hidden_dim, 1)
        self.head_mhd = nn.Linear(hidden_dim, 1)
        self.head_qi = nn.Linear(hidden_dim, 1)
        # Predict iota (edge rotational transform) as an auxiliary constraint proxy.
        self.head_iota = nn.Linear(hidden_dim, 1)
        # P3 constraint heads: mirror_ratio and flux_compression (Issue #1 fix)
        self.head_mirror_ratio = nn.Linear(hidden_dim, 1)
        self.head_flux_compression = nn.Linear(hidden_dim, 1)

        # Gradient checkpointing flag (Issue #9) - disabled by default
        self._use_checkpointing = False

    def enable_checkpointing(self, enabled: bool = True) -> None:
        """Enable/disable gradient checkpointing for memory efficiency (Issue #9).

        When enabled, intermediate activations in compute-heavy operations
        (_generate_point_cloud and geo_encoder) are recomputed during backward
        pass instead of being stored, reducing peak memory usage by ~30-50%.

        Args:
            enabled: Whether to enable checkpointing. Default True.
        """
        self._use_checkpointing = enabled

    def _init_cached_grids(self) -> None:
        """Pre-compute and cache theta grids and trig terms.

        These values are constant across forward passes:
        - theta: poloidal angle grid [0, 2π)
        - zeta: toroidal angle grid [0, 2π) (for variable nfp)
        - m_idx, n_vals: mode indices
        - cos_m_theta, sin_m_theta: trig terms for theta
        """
        pi = torch.pi
        n_theta = self.geo_n_theta
        n_zeta = self.geo_n_zeta

        # Poloidal angle grid (shared across all nfp values)
        theta = torch.linspace(0, 2 * pi, n_theta + 1)[:-1]  # (T,)

        # Toroidal angle grid (full torus for variable nfp)
        zeta = torch.linspace(0, 2 * pi, n_zeta + 1)[:-1]  # (Z,)

        # Mode indices
        m_idx = torch.arange(self.mpol + 1, dtype=torch.float32)  # (M,)
        n_idx_arr = torch.arange(2 * self.ntor + 1, dtype=torch.float32)  # (N,)
        n_vals = n_idx_arr - self.ntor  # Maps 0..2*ntor to -ntor..ntor

        # Precompute theta-dependent trig terms (shared across batch)
        m_theta = m_idx[:, None] * theta[None, :]  # (M, T)
        cos_m_theta = torch.cos(m_theta)  # (M, T)
        sin_m_theta = torch.sin(m_theta)  # (M, T)

        # Register as buffers (non-trainable, move with model)
        self.register_buffer("_cached_theta", theta)
        self.register_buffer("_cached_zeta", zeta)
        self.register_buffer("_cached_m_idx", m_idx)
        self.register_buffer("_cached_n_vals", n_vals)
        self.register_buffer("_cached_cos_m_theta", cos_m_theta)
        self.register_buffer("_cached_sin_m_theta", sin_m_theta)

    def _generate_point_cloud(
        self,
        r_cos: torch.Tensor,
        z_sin: torch.Tensor,
        nfp_batch: torch.Tensor,
    ) -> torch.Tensor:
        """Generate 3D point cloud using cached grids (Issue #6 fix).

        This avoids regenerating theta/zeta grids and trig terms every forward pass.
        Uses the same algorithm as geometry.batch_fourier_to_real_space() but with
        pre-computed cached values.

        Args:
            r_cos: (B, M, N) Fourier coefficients for R
            z_sin: (B, M, N) Fourier coefficients for Z
            nfp_batch: (B,) Number of field periods per sample

        Returns:
            points: (B, 3, T*Z) Point cloud in Cartesian coordinates
        """
        batch_size = r_cos.shape[0]

        # Use cached values (already on correct device via register_buffer)
        zeta = self._cached_zeta  # (Z,)
        n_vals = self._cached_n_vals  # (N,)
        cos_m_theta = self._cached_cos_m_theta  # (M, T)
        sin_m_theta = self._cached_sin_m_theta  # (M, T)

        # Variable Nfp: n*Nfp*zeta depends on batch
        nfp_tensor = nfp_batch.view(batch_size, 1)  # (B, 1)
        n_nfp_zeta = (
            n_vals[None, :, None] * nfp_tensor[:, :, None] * zeta[None, None, :]
        )  # (B, N, Z)
        cos_n_zeta = torch.cos(n_nfp_zeta)  # (B, N, Z)
        sin_n_zeta = torch.sin(n_nfp_zeta)  # (B, N, Z)

        # Stage 1: Contract over n -> (B, M, Z)
        A_rc = torch.einsum("bmn,bnz->bmz", r_cos, cos_n_zeta)
        B_rc = torch.einsum("bmn,bnz->bmz", r_cos, sin_n_zeta)
        A_zs = torch.einsum("bmn,bnz->bmz", z_sin, cos_n_zeta)
        B_zs = torch.einsum("bmn,bnz->bmz", z_sin, sin_n_zeta)

        # Stage 2: Contract over m -> (B, T, Z)
        R = torch.einsum("mt,bmz->btz", cos_m_theta, A_rc) + torch.einsum(
            "mt,bmz->btz", sin_m_theta, B_rc
        )
        Z = torch.einsum("mt,bmz->btz", sin_m_theta, A_zs) - torch.einsum(
            "mt,bmz->btz", cos_m_theta, B_zs
        )

        # Phi grid (broadcast zeta to match R shape)
        Phi = zeta[None, None, :].expand(batch_size, self.geo_n_theta, -1)

        # Convert to Cartesian
        X = R * torch.cos(Phi)
        Y = R * torch.sin(Phi)

        # Stack to (B, 3, N_points)
        X_flat = X.reshape(batch_size, -1)
        Y_flat = Y.reshape(batch_size, -1)
        Z_flat = Z.reshape(batch_size, -1)

        return torch.stack([X_flat, Y_flat, Z_flat], dim=1)

    def forward(
        self,
        x: Float[torch.Tensor, "batch input_dim"],  # pyright: ignore[reportUndefinedVariable]
        fidelity: torch.Tensor
        | None = None,  # (batch,) int: 0=screen, 1=promote (Issue #10)
    ) -> tuple[
        Float[torch.Tensor, "batch"],  # pyright: ignore[reportUndefinedVariable] obj
        Float[torch.Tensor, "batch"],  # pyright: ignore[reportUndefinedVariable] mhd
        Float[torch.Tensor, "batch"],  # pyright: ignore[reportUndefinedVariable] qi
        Float[torch.Tensor, "batch"],  # pyright: ignore[reportUndefinedVariable] iota
        Float[torch.Tensor, "batch"],  # pyright: ignore[reportUndefinedVariable] mirror_ratio
        Float[torch.Tensor, "batch"],  # pyright: ignore[reportUndefinedVariable] flux_compression
    ]:
        """
        Args:
            x: Flattened input tensor of shape (Batch, 2 * (mpol+1) * (2*ntor+1) + 1)
               The last element is n_field_periods.
            fidelity: Optional fidelity level tensor (Issue #10). Shape (Batch,).
                      Values: 0=screen (low-fidelity), 1=promote (high-fidelity).
                      If None, defaults to high-fidelity (1).
        """
        batch_size = x.shape[0]

        # Extract NFP (last column)
        nfp_batch = x[:, -1]
        x_spectral = x[:, :-1]

        # --- Spectral Branch ---
        half_size = self.grid_h * self.grid_w
        r_cos_flat = x_spectral[:, :half_size]
        z_sin_flat = x_spectral[:, half_size:]

        r_cos_grid = r_cos_flat.view(batch_size, 1, self.grid_h, self.grid_w)
        z_sin_grid = z_sin_flat.view(batch_size, 1, self.grid_h, self.grid_w)
        spectral_grid = torch.cat([r_cos_grid, z_sin_grid], dim=1)  # (B, 2, H, W)

        spectral_feat = self.conv_net(spectral_grid)  # (B, hidden, H, W)
        spectral_vec = self.global_pool(spectral_feat).view(
            batch_size, -1
        )  # (B, hidden)

        # --- Geometric Branch ---
        # Recover (B, mpol+1, 2*ntor+1) for geometry tool
        r_cos_in = r_cos_grid.squeeze(1)
        z_sin_in = z_sin_grid.squeeze(1)

        # Generate Point Cloud using cached grids (Issue #6 fix)
        # This avoids regenerating theta/zeta grids every forward pass
        # Gradient checkpointing (Issue #9): recompute activations during backward
        if self._use_checkpointing and self.training:
            # Checkpoint point cloud generation (memory-intensive)
            points = torch_checkpoint(
                self._generate_point_cloud,
                r_cos_in,
                z_sin_in,
                nfp_batch,
                use_reentrant=False,
            )
        else:
            points = self._generate_point_cloud(
                r_cos_in, z_sin_in, nfp_batch
            )  # (B, 3, N)

        # Augmentation: Random Rotation during training to enforce SE(3) invariance
        if self.training:
            rot_mat = equivariance.random_rotation_matrix(batch_size, device=x.device)
            points = torch.bmm(rot_mat, points)

        # Checkpoint PointNet encoder (Issue #9)
        if self._use_checkpointing and self.training:
            geo_vec = torch_checkpoint(self.geo_encoder, points, use_reentrant=False)
        else:
            geo_vec = self.geo_encoder(points)  # (B, geo_dim)

        # --- Fusion with Fidelity Conditioning (Issue #10) ---
        # Get fidelity embeddings
        if fidelity is not None:
            fid_embed = self.fidelity_embedding(fidelity)  # (B, 8)
        else:
            # Default to high-fidelity (1)
            fid_embed = self.fidelity_embedding(
                torch.ones(batch_size, dtype=torch.long, device=x.device)
            )
        combined = torch.cat([spectral_vec, geo_vec, fid_embed], dim=1)

        base = self.head_base(combined)

        pred_obj = self.head_objective(base).squeeze(-1)
        pred_mhd = self.head_mhd(base).squeeze(-1)
        pred_qi = self.head_qi(base).squeeze(-1)
        pred_iota = self.head_iota(base).squeeze(-1)
        pred_mirror_ratio = self.head_mirror_ratio(base).squeeze(-1)
        pred_flux_compression = self.head_flux_compression(base).squeeze(-1)

        return (
            pred_obj,
            pred_mhd,
            pred_qi,
            pred_iota,
            pred_mirror_ratio,
            pred_flux_compression,
        )


class NeuralOperatorSurrogate(BaseSurrogate):
    """Deep Learning Surrogate (V2) using PyTorch.

    Uses an ensemble of StellaratorNeuralOp models to predict metrics
    and quantify epistemic uncertainty (Deep Ensembles).
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
        n_ensembles: int = 5,
        hidden_dim: int = 64,
    ) -> None:
        self._min_samples = min_samples
        self._points_cadence = points_cadence
        self._cycle_cadence = cycle_cadence
        self._device = "cpu"
        if device == "cuda" and torch.cuda.is_available():
            self._device = "cuda"
        elif device == "mps" and torch.backends.mps.is_available():
            self._device = "mps"
        self._lr = learning_rate
        self._epochs = epochs
        self._batch_size = batch_size
        self._n_ensembles = max(1, n_ensembles)
        self._hidden_dim = hidden_dim

        self._trained = False
        self._last_fit_count = 0
        self._last_fit_cycle = 0

        self._models: list[StellaratorNeuralOp] = []
        self._optimizers: list[optim.Optimizer] = []
        self._schema: tools.FlattenSchema | None = None

        # Gradient checkpointing flag (Issue #9)
        self._use_checkpointing = False

    def load_checkpoint(self, path: str | Path) -> None:
        """Load model state from a checkpoint file (Task 3.2)."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        logging.info(f"[surrogate_v2] Loading checkpoint from {path}...")

        # Allow loading pickled objects (legacy/offline artifact support)
        # NOTE: We set weights_only=False to support loading full objects,
        # assuming the checkpoint is trusted (e.g. generated by scripts/train_offline.py).
        try:
            checkpoint = torch.load(path, map_location=self._device, weights_only=False)
        except TypeError:
            # Fallback for older PyTorch versions that don't support weights_only
            checkpoint = torch.load(path, map_location=self._device)

        # Handle full object checkpoint (from scripts/train_offline.py)
        if isinstance(checkpoint, NeuralOperatorSurrogate):
            logging.info(
                "[surrogate_v2] Detected full surrogate object checkpoint. Restoring state..."
            )
            self._models = checkpoint._models
            self._schema = checkpoint._schema
            self._n_ensembles = checkpoint._n_ensembles
            self._hidden_dim = checkpoint._hidden_dim
            self._trained = checkpoint._trained

            # Ensure models are on the correct device
            for model in self._models:
                model.to(self._device)
            return

        # Load Schema
        self._schema = checkpoint.get("schema")

        # Re-init models
        state_dicts = checkpoint.get("models")
        if not state_dicts:
            raise ValueError("Checkpoint does not contain 'models' state dicts")

        self._models = []
        self._n_ensembles = len(state_dicts)

        mpol = checkpoint.get("mpol")
        ntor = checkpoint.get("ntor")
        hidden_dim = checkpoint.get("hidden_dim", self._hidden_dim)

        if mpol is None or ntor is None:
            # Try to infer from schema
            if self._schema:
                mpol = self._schema.mpol
                ntor = self._schema.ntor
            else:
                raise ValueError("Checkpoint missing mpol/ntor and schema")

        logging.info(
            f"[surrogate_v2] Rehydrating {self._n_ensembles} models (mpol={mpol}, ntor={ntor})."
        )

        for i in range(self._n_ensembles):
            model = StellaratorNeuralOp(mpol=mpol, ntor=ntor, hidden_dim=hidden_dim).to(
                self._device
            )
            model.load_state_dict(state_dicts[i])
            model.eval()
            self._models.append(model)
            # No need to init optimizers for inference-only mode

        self._trained = True

    def enable_checkpointing(self, enabled: bool = True) -> None:
        """Enable gradient checkpointing for all ensemble models (Issue #9).

        Reduces peak memory usage by ~30-50% during training by recomputing
        intermediate activations during backward pass instead of storing them.

        Args:
            enabled: Whether to enable checkpointing. Default True.
        """
        self._use_checkpointing = enabled
        for model in self._models:
            model.enable_checkpointing(enabled)

    def fit(
        self,
        metrics_list: Sequence[Mapping[str, Any]],
        target_values: Sequence[float],
        *,
        minimize_objective: bool,
        cycle: int | None = None,
    ) -> None:
        """Train the Neural Operator Ensemble on the accumulated history."""
        sample_count = len(metrics_list)
        self._last_fit_count = sample_count
        if cycle is not None:
            self._last_fit_cycle = int(cycle)

        if sample_count < self._min_samples:
            logging.info(
                "[surrogate_v2] cold start: %d samples (< %d required for DL)",
                sample_count,
                self._min_samples,
            )
            self._trained = False
            return

        logging.info(
            "[surrogate_v2] Training Ensemble (N=%d) on %d samples (Device: %s)...",
            self._n_ensembles,
            sample_count,
            self._device,
        )

        # 1. Prepare Data
        features_list = []

        # Capture schema from the first item if not set
        for i, metrics in enumerate(metrics_list):
            params = metrics.get("candidate_params") or metrics.get("params", {})

            vector, schema = tools.structured_flatten(params, schema=self._schema)
            if self._schema is None:
                self._schema = schema

            # Append n_field_periods
            nfp = float(params.get("n_field_periods") or params.get("nfp", 1))
            vector_aug = np.append(vector, nfp)

            features_list.append(vector_aug)

        if not features_list:
            return

        X = torch.tensor(np.vstack(features_list), dtype=torch.float32).to(self._device)
        y_obj = torch.tensor(target_values, dtype=torch.float32).to(self._device)

        # Auxiliary targets
        y_mhd_list, y_qi_list, y_iota_list = [], [], []
        y_mirror_ratio_list, y_flux_compression_list = [], []
        for metrics in metrics_list:
            m_payload = metrics.get("metrics", metrics)
            y_mhd_list.append(float(m_payload.get("vacuum_well", -1.0)))
            y_qi_list.append(float(m_payload.get("qi", 1.0)))
            y_iota_list.append(
                float(
                    m_payload.get("edge_rotational_transform_over_n_field_periods", 0.3)
                )
            )
            # P3 constraint targets (Issue #1 fix)
            y_mirror_ratio_list.append(
                float(m_payload.get("edge_magnetic_mirror_ratio", 0.15))
            )
            y_flux_compression_list.append(
                float(
                    m_payload.get("flux_compression_in_regions_of_bad_curvature", 0.5)
                )
            )

        y_mhd = torch.tensor(y_mhd_list, dtype=torch.float32).to(self._device)
        y_qi = torch.tensor(y_qi_list, dtype=torch.float32).to(self._device)
        y_iota = torch.tensor(y_iota_list, dtype=torch.float32).to(self._device)
        y_mirror_ratio = torch.tensor(y_mirror_ratio_list, dtype=torch.float32).to(
            self._device
        )
        y_flux_compression = torch.tensor(
            y_flux_compression_list, dtype=torch.float32
        ).to(self._device)

        # Normalize targets for balanced multi-task learning
        # This ensures each loss term contributes proportionally regardless of scale
        # Store statistics for denormalization during inference
        self._y_obj_mean = y_obj.mean()
        self._y_obj_std = y_obj.std().clamp(min=1e-6)
        self._y_mhd_mean = y_mhd.mean()
        self._y_mhd_std = y_mhd.std().clamp(min=1e-6)
        self._y_qi_mean = y_qi.mean()
        self._y_qi_std = y_qi.std().clamp(min=1e-6)
        self._y_iota_mean = y_iota.mean()
        self._y_iota_std = y_iota.std().clamp(min=1e-6)
        # P3 constraint normalization stats (Issue #1 fix)
        self._y_mirror_ratio_mean = y_mirror_ratio.mean()
        self._y_mirror_ratio_std = y_mirror_ratio.std().clamp(min=1e-6)
        self._y_flux_compression_mean = y_flux_compression.mean()
        self._y_flux_compression_std = y_flux_compression.std().clamp(min=1e-6)

        # Apply log transform to QI before normalization (QI spans many orders of magnitude)
        y_qi_log = torch.log10(y_qi.clamp(min=1e-12))
        self._y_qi_log_mean = y_qi_log.mean()
        self._y_qi_log_std = y_qi_log.std().clamp(min=1e-6)

        y_obj_norm = (y_obj - self._y_obj_mean) / self._y_obj_std
        y_mhd_norm = (y_mhd - self._y_mhd_mean) / self._y_mhd_std
        y_qi_norm = (y_qi_log - self._y_qi_log_mean) / self._y_qi_log_std
        y_iota_norm = (y_iota - self._y_iota_mean) / self._y_iota_std
        y_mirror_ratio_norm = (
            y_mirror_ratio - self._y_mirror_ratio_mean
        ) / self._y_mirror_ratio_std
        y_flux_compression_norm = (
            y_flux_compression - self._y_flux_compression_mean
        ) / self._y_flux_compression_std

        dataset = TensorDataset(
            X,
            y_obj_norm,
            y_mhd_norm,
            y_qi_norm,
            y_iota_norm,
            y_mirror_ratio_norm,
            y_flux_compression_norm,
        )

        # 2. Initialize Models if needed or if schema changed
        # Check dimensions match schema (ignoring appended nfp)
        current_mpol = self._schema.mpol
        current_ntor = self._schema.ntor

        reinit = False
        if not self._models:
            reinit = True
        elif (
            self._models[0].mpol != current_mpol or self._models[0].ntor != current_ntor
        ):
            reinit = True

        if reinit:
            logging.info(
                "[surrogate_v2] Initializing %d models with mpol=%d, ntor=%d",
                self._n_ensembles,
                current_mpol,
                current_ntor,
            )
            self._models = []
            self._optimizers = []
            for _ in range(self._n_ensembles):
                model = StellaratorNeuralOp(
                    mpol=current_mpol, ntor=current_ntor, hidden_dim=self._hidden_dim
                ).to(self._device)
                self._models.append(model)
                # Enable checkpointing if configured (Issue #9)
                if self._use_checkpointing:
                    model.enable_checkpointing(True)
                # Weight decay added for regularization (AoT recommendation)
                self._optimizers.append(
                    optim.Adam(model.parameters(), lr=self._lr, weight_decay=1e-4)
                )

        # 3. Train Loop with Early Stopping (AoT recommendation)
        # Deep Ensembles: random init + bagging for better calibration
        # (Lakshminarayanan et al., 2017: "Simple and Scalable Predictive Uncertainty")
        criterion = nn.MSELoss()
        n_samples = len(dataset)

        # Early stopping parameters
        early_stopping_patience = 10
        min_val_improvement = 1e-4

        for idx, model in enumerate(self._models):
            model.train()
            optimizer = self._optimizers[idx]

            # P1 FIX: Bootstrap sampling (bagging) for each ensemble member
            # This improves uncertainty calibration by introducing data diversity
            rng = torch.Generator()
            rng.manual_seed(42 + idx)  # Reproducible but different per model
            bootstrap_indices = torch.randint(0, n_samples, (n_samples,), generator=rng)
            bootstrap_dataset = torch.utils.data.Subset(
                dataset, bootstrap_indices.tolist()
            )

            # Train/Validation split (80/20) for early stopping
            n_bootstrap = len(bootstrap_dataset)
            n_train = int(0.8 * n_bootstrap)
            n_val = n_bootstrap - n_train

            if n_val < 2:
                # Not enough samples for validation, skip early stopping
                train_dataset = bootstrap_dataset
                val_loader = None
            else:
                train_dataset, val_dataset = torch.utils.data.random_split(
                    bootstrap_dataset,
                    [n_train, n_val],
                    generator=torch.Generator().manual_seed(42 + idx),
                )
                val_loader = DataLoader(
                    val_dataset, batch_size=self._batch_size, shuffle=False
                )

            train_loader = DataLoader(
                train_dataset, batch_size=self._batch_size, shuffle=True
            )

            # Early stopping state
            best_val_loss = float("inf")
            patience_counter = 0
            best_state_dict = None

            for epoch in range(self._epochs):
                # Training phase
                model.train()
                for (
                    xb,
                    yb_obj,
                    yb_mhd,
                    yb_qi,
                    yb_iota,
                    yb_mirror,
                    yb_flux,
                ) in train_loader:
                    optimizer.zero_grad()
                    (
                        pred_obj,
                        pred_mhd,
                        pred_qi,
                        pred_iota,
                        pred_mirror,
                        pred_flux,
                    ) = model(xb)

                    # With normalized targets, all losses are on same scale
                    # Equal weighting is now appropriate
                    loss = (
                        criterion(pred_obj, yb_obj)
                        + criterion(pred_mhd, yb_mhd)
                        + criterion(pred_qi, yb_qi)
                        + criterion(pred_iota, yb_iota)
                        + criterion(pred_mirror, yb_mirror)
                        + criterion(pred_flux, yb_flux)
                    )

                    loss.backward()
                    optimizer.step()

                # Validation phase for early stopping
                if val_loader is not None:
                    model.eval()
                    val_loss_sum = 0.0
                    val_count = 0
                    with torch.no_grad():
                        for (
                            xb,
                            yb_obj,
                            yb_mhd,
                            yb_qi,
                            yb_iota,
                            yb_mirror,
                            yb_flux,
                        ) in val_loader:
                            (
                                pred_obj,
                                pred_mhd,
                                pred_qi,
                                pred_iota,
                                pred_mirror,
                                pred_flux,
                            ) = model(xb)
                            val_loss = (
                                criterion(pred_obj, yb_obj)
                                + criterion(pred_mhd, yb_mhd)
                                + criterion(pred_mirror, yb_mirror)
                                + criterion(pred_flux, yb_flux)
                                + criterion(pred_qi, yb_qi)
                                + criterion(pred_iota, yb_iota)
                            )
                            val_loss_sum += val_loss.item() * len(xb)
                            val_count += len(xb)

                    avg_val_loss = (
                        val_loss_sum / val_count if val_count > 0 else float("inf")
                    )

                    # Check for improvement
                    if avg_val_loss < best_val_loss - min_val_improvement:
                        best_val_loss = avg_val_loss
                        best_state_dict = {
                            k: v.clone() for k, v in model.state_dict().items()
                        }
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping_patience:
                            logging.debug(
                                f"[surrogate_v2] Model {idx}: early stopping at epoch {epoch + 1}"
                            )
                            break

            # Restore best model weights
            if best_state_dict is not None:
                model.load_state_dict(best_state_dict)

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

    def predict_torch(
        self,
        x: Float[torch.Tensor, "batch input_dim"],  # pyright: ignore[reportUndefinedVariable]
    ) -> tuple[
        Float[torch.Tensor, "batch"],  # pyright: ignore[reportUndefinedVariable] obj_mean
        Float[torch.Tensor, "batch"],  # pyright: ignore[reportUndefinedVariable] obj_std
        Float[torch.Tensor, "batch"],  # pyright: ignore[reportUndefinedVariable] mhd_mean
        Float[torch.Tensor, "batch"],  # pyright: ignore[reportUndefinedVariable] mhd_std
        Float[torch.Tensor, "batch"],  # pyright: ignore[reportUndefinedVariable] qi_mean
        Float[torch.Tensor, "batch"],  # pyright: ignore[reportUndefinedVariable] qi_std
        Float[torch.Tensor, "batch"],  # pyright: ignore[reportUndefinedVariable] iota_mean
        Float[torch.Tensor, "batch"],  # pyright: ignore[reportUndefinedVariable] iota_std
        Float[torch.Tensor, "batch"],  # pyright: ignore[reportUndefinedVariable] mirror_mean
        Float[torch.Tensor, "batch"],  # pyright: ignore[reportUndefinedVariable] mirror_std
        Float[torch.Tensor, "batch"],  # pyright: ignore[reportUndefinedVariable] flux_mean
        Float[torch.Tensor, "batch"],  # pyright: ignore[reportUndefinedVariable] flux_std
    ]:
        """Differentiable prediction for optimization loop (Mean + Std).

        Returns:
            (obj_mean, obj_std, mhd_mean, mhd_std, qi_mean, qi_std, iota_mean, iota_std,
             mirror_ratio_mean, mirror_ratio_std, flux_compression_mean, flux_compression_std)
        """
        if not self._models:
            raise RuntimeError("NeuralOperatorSurrogate not initialized/trained")

        if x.device != torch.device(self._device):
            x = x.to(self._device)

        # Collect predictions from all models
        preds_obj, preds_mhd, preds_qi, preds_iota = [], [], [], []
        preds_mirror, preds_flux = [], []

        for model in self._models:
            outputs = model(x)
            if isinstance(outputs, tuple) or isinstance(outputs, list):
                o, m, q, iota, mirror, flux = outputs[:6]
            else:
                o, m, q, iota, mirror, flux = outputs
            preds_obj.append(o)
            preds_mhd.append(m)
            preds_qi.append(q)
            preds_iota.append(iota)
            preds_mirror.append(mirror)
            preds_flux.append(flux)

        # Stack (Ensemble, Batch)
        stack_obj = torch.stack(preds_obj)
        stack_mhd = torch.stack(preds_mhd)
        stack_qi = torch.stack(preds_qi)
        stack_iota = torch.stack(preds_iota)
        stack_mirror = torch.stack(preds_mirror)
        stack_flux = torch.stack(preds_flux)

        # Compute Mean and Std in normalized space
        obj_mean_norm = torch.mean(stack_obj, dim=0)
        obj_std_norm = torch.std(stack_obj, dim=0)
        mhd_mean_norm = torch.mean(stack_mhd, dim=0)
        mhd_std_norm = torch.std(stack_mhd, dim=0)
        qi_mean_norm = torch.mean(stack_qi, dim=0)
        qi_std_norm = torch.std(stack_qi, dim=0)
        iota_mean_norm = torch.mean(stack_iota, dim=0)
        iota_std_norm = torch.std(stack_iota, dim=0)
        mirror_mean_norm = torch.mean(stack_mirror, dim=0)
        mirror_std_norm = torch.std(stack_mirror, dim=0)
        flux_mean_norm = torch.mean(stack_flux, dim=0)
        flux_std_norm = torch.std(stack_flux, dim=0)

        # Denormalize predictions to original scale
        # obj and mhd: linear denormalization
        # qi: model predicts log10(qi), convert back to raw qi
        if hasattr(self, "_y_obj_mean"):
            obj_mean = obj_mean_norm * self._y_obj_std + self._y_obj_mean
            obj_std = obj_std_norm * self._y_obj_std
            mhd_mean = mhd_mean_norm * self._y_mhd_std + self._y_mhd_mean
            mhd_std = mhd_std_norm * self._y_mhd_std
            # QI was trained in log10 space, denormalize to log10 then convert to raw
            qi_log_mean = qi_mean_norm * self._y_qi_log_std + self._y_qi_log_mean
            qi_log_std = qi_std_norm * self._y_qi_log_std
            # Convert log10(qi) back to raw qi: qi = 10^(log10_qi)
            qi_mean = torch.pow(10.0, qi_log_mean)
            # For std, use delta method: std(10^x) ≈ 10^x * ln(10) * std(x)
            qi_std = qi_mean * 2.302585 * qi_log_std
            iota_mean = iota_mean_norm * self._y_iota_std + self._y_iota_mean
            iota_std = iota_std_norm * self._y_iota_std
            # P3 constraint denormalization (Issue #1 fix)
            mirror_mean = (
                mirror_mean_norm * self._y_mirror_ratio_std + self._y_mirror_ratio_mean
            )
            mirror_std = mirror_std_norm * self._y_mirror_ratio_std
            flux_mean = (
                flux_mean_norm * self._y_flux_compression_std
                + self._y_flux_compression_mean
            )
            flux_std = flux_std_norm * self._y_flux_compression_std
        else:
            # Fallback if not trained (shouldn't happen in normal use)
            obj_mean, obj_std = obj_mean_norm, obj_std_norm
            mhd_mean, mhd_std = mhd_mean_norm, mhd_std_norm
            qi_mean, qi_std = qi_mean_norm, qi_std_norm
            iota_mean, iota_std = iota_mean_norm, iota_std_norm
            mirror_mean, mirror_std = mirror_mean_norm, mirror_std_norm
            flux_mean, flux_std = flux_mean_norm, flux_std_norm

        return (
            obj_mean,
            obj_std,
            mhd_mean,
            mhd_std,
            qi_mean,
            qi_std,
            iota_mean,
            iota_std,
            mirror_mean,
            mirror_std,
            flux_mean,
            flux_std,
        )

    def _compute_soft_feasibility(
        self,
        *,
        mhd_val: float,
        qi_val: float,
        elongation_val: float,
        iota_val: float = 0.3,
        mirror_ratio_val: float | None = None,
        flux_compression_val: float | None = None,
        problem: str = "p3",
    ) -> float:
        """Compute soft feasibility probability from surrogate predictions (B7 fix).

        This method checks ALL relevant constraints for the given problem,
        using actual predictions where available.

        Args:
            mhd_val: Predicted vacuum_well value.
            qi_val: Predicted raw QI residual (not log10).
            elongation_val: Predicted max elongation.
            iota_val: Predicted edge rotational transform.
            mirror_ratio_val: Predicted edge magnetic mirror ratio (Issue #1 fix).
            flux_compression_val: Predicted flux compression (Issue #1 fix).
            problem: Problem identifier ("p1", "p2", "p3").

        Returns:
            Soft feasibility score in [0, 1]. Higher = more likely feasible.
        """
        from ai_scientist.constraints import get_constraint_bounds

        bounds = get_constraint_bounds(problem)
        violations = 0.0
        total_checks = 0.0

        # Weight severe violations more heavily
        def soft_violation(val: float, bound: float, is_upper: bool) -> float:
            """Return 0 if satisfied, else normalized violation magnitude."""
            if is_upper:
                excess = val - bound
            else:  # lower bound
                excess = bound - val
            if excess <= 0:
                return 0.0
            # Normalize by bound magnitude (avoid div by zero)
            return min(excess / max(abs(bound), 0.1), 2.0)  # Cap at 2x violation

        # 1. Vacuum well (MHD stability) - P3 constraint, but good for all problems
        well_bound = bounds.get("vacuum_well_lower", 0.0)
        violations += soft_violation(mhd_val, well_bound, is_upper=False)
        total_checks += 1.0

        # 2. QI residual (log10 scale)
        qi_log = float(np.log10(max(qi_val, 1e-12)))
        qi_threshold = bounds.get("qi_log10_upper", -3.5)
        if "qi_log10_upper" in bounds:
            violations += soft_violation(qi_log, qi_threshold, is_upper=True)
            total_checks += 1.0

        # 3. Elongation (P2 has explicit bound)
        if problem.lower().startswith("p2"):
            elong_bound = bounds.get("max_elongation_upper", 5.0)
            violations += soft_violation(elongation_val, elong_bound, is_upper=True)
            total_checks += 1.0

        # 4. Iota constraint (edge rotational transform) when available.
        if "edge_rotational_transform_lower" in bounds:
            iota_bound = bounds.get("edge_rotational_transform_lower", 0.25)
            violations += soft_violation(iota_val, iota_bound, is_upper=False)
            total_checks += 1.0

        # 5. Mirror ratio constraint (P2, P3 - now predicted by surrogate, Issue #1 fix)
        if "edge_magnetic_mirror_ratio_upper" in bounds:
            mirror_bound = bounds.get("edge_magnetic_mirror_ratio_upper", 0.25)
            if mirror_ratio_val is not None:
                violations += soft_violation(
                    mirror_ratio_val, mirror_bound, is_upper=True
                )
            else:
                # Fallback: 20% penalty if prediction unavailable
                violations += 0.2
            total_checks += 1.0

        # 6. Flux compression constraint (P3 only - now predicted by surrogate, Issue #1 fix)
        if "flux_compression_upper" in bounds:
            flux_bound = bounds.get("flux_compression_upper", 0.9)
            if flux_compression_val is not None:
                violations += soft_violation(
                    flux_compression_val, flux_bound, is_upper=True
                )
            else:
                # Fallback: 20% penalty if prediction unavailable
                violations += 0.2
            total_checks += 1.0

        # 7. Aspect ratio constraint (still not predicted by surrogate)
        if "aspect_ratio_upper" in bounds:
            # Assume 20% chance of violation (aspect ratio computed analytically elsewhere)
            violations += 0.2
            total_checks += 1.0

        # Convert to probability: exp(-violations) clamped to [0.1, 0.95]
        # This gives soft gradients for optimization
        if total_checks == 0:
            return 0.5

        avg_violation = violations / total_checks
        prob = float(np.exp(-2.0 * avg_violation))  # Steeper penalty
        return max(0.1, min(0.95, prob))

    def rank_candidates(
        self,
        candidates: Sequence[Mapping[str, Any]],
        *,
        minimize_objective: bool,
        exploration_ratio: float = 0.0,
        problem: str = "p3",
    ) -> list[SurrogatePrediction]:
        """Rank candidates using the Deep Ensemble (Mean + Std for Exploration).

        Args:
            candidates: List of candidate configurations to rank.
            minimize_objective: Whether lower objective values are better.
            exploration_ratio: UCB exploration bonus weight (0 = pure exploitation).
            problem: Problem identifier ("p1", "p2", "p3") for constraint checking.

        Returns:
            Sorted list of SurrogatePrediction, best first.
        """
        if not candidates:
            return []

        if not self._trained or not self._models:
            logging.info("[surrogate_v2] Model not trained, using heuristics")
            cold_ranks: list[SurrogatePrediction] = []
            for candidate in candidates:
                cold_ranks.append(
                    SurrogatePrediction(
                        expected_value=0.0,
                        prob_feasible=0.0,
                        predicted_objective=0.0,
                        minimize_objective=minimize_objective,
                        metadata=candidate,
                        predicted_elongation=0.0,
                    )
                )
            return cold_ranks

        vectors = []
        for candidate in candidates:
            params = candidate.get("candidate_params") or candidate.get("params", {})
            vec, _ = tools.structured_flatten(params, schema=self._schema)
            nfp = float(params.get("n_field_periods") or params.get("nfp", 1))
            vec_aug = np.append(vec, nfp)
            vectors.append(vec_aug)

        X = torch.tensor(np.vstack(vectors), dtype=torch.float32).to(self._device)

        # Evaluation
        preds_obj, preds_mhd, preds_qi, preds_iota = [], [], [], []
        preds_mirror, preds_flux = [], []

        for model in self._models:
            model.eval()
            with torch.no_grad():
                outputs = model(X)
                if isinstance(outputs, tuple) or isinstance(outputs, list):
                    o, m, q, iota, mirror, flux = outputs[:6]
                else:
                    o, m, q, iota, mirror, flux = outputs
                preds_obj.append(o)
                preds_mhd.append(m)
                preds_qi.append(q)
                preds_iota.append(iota)
                preds_mirror.append(mirror)
                preds_flux.append(flux)

        # Convert to numpy (Ensemble, Batch)
        obj_stack = torch.stack(preds_obj).cpu().numpy()
        mhd_stack = torch.stack(preds_mhd).cpu().numpy()
        qi_stack = torch.stack(preds_qi).cpu().numpy()
        iota_stack = torch.stack(preds_iota).cpu().numpy()
        mirror_stack = torch.stack(preds_mirror).cpu().numpy()
        flux_stack = torch.stack(preds_flux).cpu().numpy()

        # Statistics
        obj_mean_norm = np.mean(obj_stack, axis=0)
        obj_std_norm = np.std(obj_stack, axis=0)

        mhd_mean_norm = np.mean(mhd_stack, axis=0)
        qi_mean_norm = np.mean(qi_stack, axis=0)
        iota_mean_norm = np.mean(iota_stack, axis=0)
        mirror_mean_norm = np.mean(mirror_stack, axis=0)
        flux_mean_norm = np.mean(flux_stack, axis=0)

        # Denormalize to original units so feasibility thresholds are meaningful.
        # This mirrors `predict_torch` but operates on numpy arrays.
        if hasattr(self, "_y_obj_mean"):
            y_obj_mean = float(self._y_obj_mean.detach().cpu().item())
            y_obj_std = float(self._y_obj_std.detach().cpu().item())
            y_mhd_mean = float(self._y_mhd_mean.detach().cpu().item())
            y_mhd_std = float(self._y_mhd_std.detach().cpu().item())
            y_qi_log_mean = float(self._y_qi_log_mean.detach().cpu().item())
            y_qi_log_std = float(self._y_qi_log_std.detach().cpu().item())
            y_iota_mean = float(self._y_iota_mean.detach().cpu().item())
            y_iota_std = float(self._y_iota_std.detach().cpu().item())
            # P3 constraint denorm stats (Issue #1 fix)
            y_mirror_mean = float(self._y_mirror_ratio_mean.detach().cpu().item())
            y_mirror_std = float(self._y_mirror_ratio_std.detach().cpu().item())
            y_flux_mean = float(self._y_flux_compression_mean.detach().cpu().item())
            y_flux_std = float(self._y_flux_compression_std.detach().cpu().item())

            obj_mean = obj_mean_norm * y_obj_std + y_obj_mean
            obj_std = obj_std_norm * y_obj_std

            mhd_mean = mhd_mean_norm * y_mhd_std + y_mhd_mean

            qi_log_mean = qi_mean_norm * y_qi_log_std + y_qi_log_mean
            qi_mean = np.power(10.0, qi_log_mean)
            iota_mean = iota_mean_norm * y_iota_std + y_iota_mean
            mirror_mean = mirror_mean_norm * y_mirror_std + y_mirror_mean
            flux_mean = flux_mean_norm * y_flux_std + y_flux_mean
        else:
            obj_mean, obj_std = obj_mean_norm, obj_std_norm
            mhd_mean = mhd_mean_norm
            qi_mean = qi_mean_norm
            iota_mean = iota_mean_norm
            mirror_mean = mirror_mean_norm
            flux_mean = flux_mean_norm

        # Analytically compute elongation using isoperimetric method (B5 fix).
        # The isoperimetric approach matches the benchmark's ellipse-fitting
        # definition better than covariance eigenvalues for non-elliptic shapes.
        with torch.no_grad():
            mpol = self._schema.mpol
            ntor = self._schema.ntor
            grid_h = mpol + 1
            grid_w = 2 * ntor + 1
            half_size = grid_h * grid_w

            nfp_batch = X[:, -1]
            x_spectral = X[:, :-1]

            r_cos_grid = x_spectral[:, :half_size].view(-1, grid_h, grid_w)
            z_sin_grid = x_spectral[:, half_size:].view(-1, grid_h, grid_w)

            # B5 FIX: Use elongation_isoperimetric instead of elongation (covariance)
            # This provides ~3.6% accuracy vs benchmark, compared to ~25% error with covariance
            elongations = (
                geometry.elongation_isoperimetric(r_cos_grid, z_sin_grid, nfp_batch)
                .cpu()
                .numpy()
            )

            predictions: list[SurrogatePrediction] = []
        for i, candidate in enumerate(candidates):
            # B7 FIX: Use multi-constraint feasibility check with all predictions
            prob_feasible = self._compute_soft_feasibility(
                mhd_val=float(mhd_mean[i]),
                qi_val=float(qi_mean[i]),
                elongation_val=float(elongations[i]),
                iota_val=float(iota_mean[i]),
                mirror_ratio_val=float(mirror_mean[i]),
                flux_compression_val=float(flux_mean[i]),
                problem=problem,
            )

            obj_val = float(obj_mean[i])
            uncertainty = float(obj_std[i])

            constraint_distance = float(candidate.get("constraint_distance", 0.0))
            constraint_distance = max(0.0, constraint_distance)

            # Score: Improvement + Exploration Bonus - Violations
            # For ranking, we usually maximize expected_value
            # If minimizing objective, we use -obj_val

            base_score = -obj_val if minimize_objective else obj_val

            # Active Learning: Add exploration bonus (UCB-like)
            # exploration_ratio scales the standard deviation contribution
            exploration_bonus = max(0.0, float(exploration_ratio)) * uncertainty

            # Combine: Feasibility * (Performance + Exploration)
            # Or simply additive?
            # Let's stick to a robust scoring:
            # Score: Improvement + Exploration Bonus - Violations
            # Formula: score = base_score + exploration_bonus - (10.0 * constraint_distance)
            # - base_score: The predicted objective value (negated if minimizing).
            # - exploration_bonus: Adds value for uncertain predictions to encourage exploration (UCB).
            # - constraint_distance: Penalty for geometric constraints (e.g. self-intersection).
            #   The weight 10.0 is a heuristic to strongly discourage invalid geometries.
            score = base_score + exploration_bonus - (10.0 * constraint_distance)

            predictions.append(
                SurrogatePrediction(
                    expected_value=score,
                    prob_feasible=prob_feasible,
                    predicted_objective=obj_val,
                    minimize_objective=minimize_objective,
                    metadata=candidate,
                    predicted_mhd=float(mhd_mean[i]),
                    predicted_qi=float(qi_mean[i]),
                    predicted_elongation=float(elongations[i]),
                )
            )

        return sorted(predictions, key=lambda item: item.expected_value, reverse=True)
