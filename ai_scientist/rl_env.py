"""RL Environment for Stellarator Design Optimization (StellarForge Phase 3).

This module implements a specific Gym environment, `StellaratorEnv`, that wraps the
`NeuralOperatorSurrogate` to provide a fast simulation loop for Reinforcement Learning agents.
The agent (The Engineer) observes the current design coefficients and takes actions to
perturb them, receiving rewards based on the surrogate's predictions of Objective, MHD stability, and Quasisymmetry.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from ai_scientist import tools
from ai_scientist.optim import geometry
from ai_scientist.optim.surrogate_v2 import NeuralOperatorSurrogate

_LOGGER = logging.getLogger(__name__)


class StellaratorEnv(gym.Env):
    """
    Gym Environment for refining stellarator designs using a surrogate model.

    State: Flattened Fourier coefficients (continuous vector).
    Action: Delta updates to coefficients (continuous vector).
    Reward: Improvement in composite objective (feasibility + performance).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        surrogate: NeuralOperatorSurrogate,
        initial_params: Dict[str, Any],
        target_metrics: Dict[str, float],
        max_steps: int = 20,
        action_scale: float = 1e-3,
        device: str = "cpu",
    ):
        """
        Initialize the RL environment.

        Args:
            surrogate: Pre-trained NeuralOperatorSurrogate instance.
            initial_params: Dictionary of initial boundary parameters.
            target_metrics: Target performance metrics (e.g. aspect_ratio=8.0).
            max_steps: Maximum steps per episode.
            action_scale: Scaling factor for actions (deltas).
            device: Compute device for surrogate calls.
        """
        self.surrogate = surrogate
        self.params = initial_params
        self.target_metrics = target_metrics
        self.max_steps = max_steps
        self.action_scale = action_scale
        self.device = device

        # Flatten params to create state vector
        # Note: We rely on surrogate's schema to ensure consistency
        if surrogate._schema is None:
            _, self.schema = tools.structured_flatten(self.params)
        else:
            self.schema = surrogate._schema

        self.vec, _ = tools.structured_flatten(self.params, schema=self.schema)
        self.dim = len(self.vec)

        # Action space: Continuous delta [-1, 1] scaled by action_scale
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.dim,), dtype=np.float32
        )

        # Observation space: The coefficients themselves
        # We allow unbounded values technically, though physically they are bounded
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.dim,), dtype=np.float32
        )

        self.current_step = 0
        self.current_vec = self.vec.copy()

        # Initial score
        self.initial_score = self._compute_score(self.current_vec)
        self.current_score = self.initial_score

    def _compute_score(self, vec: np.ndarray) -> float:
        """Compute reward score using surrogate predictions.

        Uses log-scale QI with continuous improvement shaping (Option C).
        Computes AR directly from params via geometry module (v3.2 fix).
        """
        # Preprocess for surrogate: Append nfp
        nfp = float(self.params.get("n_field_periods") or self.params.get("nfp", 1))

        # (Input Dim + 1)
        x_in = np.append(vec, nfp)

        x_tensor = (
            torch.tensor(x_in, dtype=torch.float32)
            .unsqueeze(0)
            .to(self.surrogate._device)
        )

        try:
            with torch.no_grad():
                # Predict (Mean + Std)
                # Returns: (obj_m, obj_s, mhd_m, mhd_s, qi_m, qi_s)
                obj_m, _, mhd_m, _, qi_m, _ = self.surrogate.predict_torch(x_tensor)

                # Move to CPU scalars
                _ = float(obj_m.item())  # obj_val unused after v3.2 (AR from geometry)
                mhd_val = float(mhd_m.item())
                qi_val = float(qi_m.item())
        except Exception as e:
            _LOGGER.warning(f"Surrogate predict failed: {e}")
            return -100.0

        # Reward Logic (v3.2 approved plan)
        # Option C: Strong feasibility penalty + mild continuous improvement
        cost = 0.0

        # 1. QI in log scale (consistent with benchmark constraints)
        QI_CLAMP_FLOOR = 1e-10
        # QI residual is non-negative by definition; guard against surrogate sign flips.
        qi_positive = abs(qi_val)
        qi_clamped = max(QI_CLAMP_FLOOR, qi_positive)
        log_qi = np.log10(qi_clamped)
        qi_target = self.target_metrics.get("log10_qi_threshold", -4.0)

        # Strong feasibility penalty + mild continuous improvement
        qi_feasibility_penalty = max(0.0, log_qi - qi_target)
        qi_continuous = log_qi - qi_target  # Can be negative (good)

        cost += 10.0 * qi_feasibility_penalty  # Strong feasibility push
        cost += 0.5 * qi_continuous  # Mild improvement push

        # 2. MHD Stability (Vacuum Well) - want > 0
        mhd_violation = max(0.0, -mhd_val)
        mhd_continuous = -mhd_val

        cost += 5.0 * mhd_violation  # Strong MHD feasibility push
        cost += 0.3 * mhd_continuous  # Mild MHD improvement

        # 3. Aspect Ratio Target (v3.2 FIX: compute from params, NOT obj_val)
        # After retraining change, obj_val becomes score (grad/aspect), not AR
        ar_target = self.target_metrics.get("aspect_ratio", 8.0)
        try:
            # Reconstruct params from current vec for AR computation
            mpol = self.schema.mpol
            ntor = self.schema.ntor
            grid_h = mpol + 1
            grid_w = 2 * ntor + 1
            half_size = grid_h * grid_w

            r_cos = torch.tensor(vec[:half_size], dtype=torch.float32).view(
                1, grid_h, grid_w
            )
            z_sin = torch.tensor(
                vec[half_size : 2 * half_size], dtype=torch.float32
            ).view(1, grid_h, grid_w)

            computed_ar = float(geometry.aspect_ratio(r_cos, z_sin, nfp).item())
        except Exception:
            computed_ar = ar_target  # Fallback: no AR penalty if computation fails

        ar_deviation = abs(computed_ar - ar_target)
        cost += 1.0 * ar_deviation

        # Reward is negative cost (Max Reward = Min Cost)
        return -cost

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state or new provided params."""
        super().reset(seed=seed)

        if options and "params" in options:
            self.params = options["params"]
            self.vec, _ = tools.structured_flatten(self.params, schema=self.schema)

        self.current_vec = self.vec.copy()
        self.current_step = 0
        self.current_score = self._compute_score(self.current_vec)

        return self.current_vec.astype(np.float32), {"score": self.current_score}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take one step in the environment."""
        self.current_step += 1

        # Apply action
        # Clip action to range [-1, 1] just in case
        act = np.clip(action, -1.0, 1.0)
        delta = act * self.action_scale

        self.current_vec += delta

        # Compute new score
        new_score = self._compute_score(self.current_vec)

        # Reward is the IMPROVEMENT
        reward = new_score - self.current_score

        # Update current score text
        self.current_score = new_score

        truncated = self.current_step >= self.max_steps
        terminated = False

        # Optional: Terminate if improvement stalls?

        return (
            self.current_vec.astype(np.float32),
            reward,
            terminated,
            truncated,
            {"score": self.current_score},
        )
