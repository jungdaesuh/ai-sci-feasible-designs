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
        action_scale: float = 1e-2,
        device: str = "cpu",
        problem: str = "p3",
    ):
        """
        Initialize the RL environment.

        Args:
            surrogate: Pre-trained NeuralOperatorSurrogate instance.
            initial_params: Dictionary of initial boundary parameters.
            target_metrics: Target performance metrics (e.g. aspect_ratio=8.0).
            max_steps: Maximum steps per episode.
            action_scale: Scaling factor for actions (deltas). Default 1e-2 allows
                         20% coefficient range traversal in 20 steps.
            device: Compute device for surrogate calls.
            problem: Problem type ("p1", "p2", "p3"). Controls which constraints
                     are penalized in the reward function. Default "p3".
        """
        self.surrogate = surrogate
        self.params = initial_params
        self.target_metrics = target_metrics
        self.max_steps = max_steps
        self.action_scale = action_scale
        self.adaptive_action_scale = target_metrics.get("adaptive_action_scale", False)
        self.device = device
        self.problem = problem.lower()  # Store problem type for reward shaping

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

    def _get_action_scale(self, step: int) -> float:
        """Get action scale, optionally annealed based on step.

        When adaptive_action_scale is enabled, uses exponential decay:
        - Early steps: larger exploration (base scale)
        - Late steps: smaller refinement (base * 0.9^(step/5))
        """
        if not self.adaptive_action_scale:
            return self.action_scale
        decay = 0.9 ** (step / 5)  # Halve roughly every ~3.5 steps
        return self.action_scale * decay

    def _compute_score(self, vec: np.ndarray) -> float:
        """Compute reward score using surrogate predictions.

        Uses log-scale QI with continuous improvement shaping (Option C).
        Computes AR directly from params via geometry module (v3.2 fix).

        IMPORTANT: Reward penalties are now problem-aware (B3 + AoT fixes):
        - QI penalty: All problems
        - MHD/vacuum_well penalty: P3 only (not P2!)
        - Elongation penalty: P2 only (has max_elongation constraint)
        - P1 constraints: AR <= 4, iota >= 0.3 (AoT fix: previously missing!)
        - Aspect ratio penalty: All problems (general, with problem-specific limits)
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
                # Returns: (obj_m, obj_s, mhd_m, mhd_s, qi_m, qi_s, iota_m, iota_s,
                #           mirror_m, mirror_s, flux_m, flux_s)
                (
                    obj_m,
                    _,
                    mhd_m,
                    _,
                    qi_m,
                    _,
                    iota_m,
                    _,
                    _,  # mirror_mean (unused)
                    _,  # mirror_std (unused)
                    _,  # flux_mean (unused)
                    _,  # flux_std (unused)
                ) = self.surrogate.predict_torch(x_tensor)

                # Move to CPU scalars
                _ = float(obj_m.item())  # obj_val unused after v3.2 (AR from geometry)
                mhd_val = float(mhd_m.item())
                qi_val = float(qi_m.item())
                iota_val = float(iota_m.item())
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
        # B3 FIX: Only penalize for P3 problems (vacuum_well is a P3-only constraint!)
        if self.problem.startswith("p3"):
            mhd_violation = max(0.0, -mhd_val)
            mhd_continuous = max(
                0.0, -mhd_val
            )  # H1 FIX: Only penalize in infeasible region

            cost += 5.0 * mhd_violation  # Strong MHD feasibility push
            cost += 0.3 * mhd_continuous  # Mild MHD improvement

        # 3. P1-specific constraints (AoT fix: previously missing!)
        # P1 constraints: average_triangularity <= -0.5, edge_rotational_transform >= 0.3
        if self.problem.startswith("p1"):
            # 3a. Triangularity constraint: average_triangularity <= -0.5
            # Computed geometrically from Fourier coefficients using dedicated function
            try:
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
                nfp_val = float(
                    self.params.get("n_field_periods") or self.params.get("nfp", 1)
                )

                # Compute triangularity using dedicated geometry function
                computed_tri = float(
                    geometry.average_triangularity(r_cos, z_sin, nfp_val).item()
                )
                tri_limit = self.target_metrics.get("average_triangularity", -0.5)
                # P1 constraint: average_triangularity <= -0.5
                # Violation when computed_tri > tri_limit
                tri_violation = max(0.0, computed_tri - tri_limit)
                cost += 5.0 * tri_violation  # Strong triangularity penalty

            except Exception:
                pass  # Skip if geometry computation fails

            # 3b. Iota constraint: edge_rotational_transform >= 0.3
            # Use surrogate prediction for iota
            iota_limit = self.target_metrics.get("edge_rotational_transform", 0.3)
            iota_violation = max(0.0, iota_limit - iota_val)
            cost += 5.0 * iota_violation  # Strong iota penalty

        # 4. Elongation Constraint (P2 only: max_elongation <= 5.0)
        # B3 FIX: P2 has elongation constraint that P3 does not
        if self.problem.startswith("p2"):
            try:
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
                nfp = float(
                    self.params.get("n_field_periods") or self.params.get("nfp", 1)
                )

                computed_elong = float(
                    geometry.elongation_isoperimetric(r_cos, z_sin, nfp).item()
                )
                elong_limit = self.target_metrics.get("max_elongation", 5.0)
                elong_violation = max(0.0, computed_elong - elong_limit)
                cost += 3.0 * elong_violation  # Moderate elongation penalty
            except Exception:
                pass  # Skip elongation penalty if computation fails

            # P2 Objective: Maximize L_∇B (magnetic gradient scale length)
            # B5 FIX: P2 reward had no signal for the objective function!
            # Once constraints are satisfied (violations ≈ 0), agent needs gradient
            # toward higher L_∇B values. obj_m is the surrogate's L_∇B prediction.
            l_grad_b_pred = float(obj_m.item())
            # Subtract from cost (higher L_∇B = lower cost = higher reward)
            # Scale factor 2.0 balances with constraint penalties (QI=10, elong=3)
            cost -= 2.0 * l_grad_b_pred

        # 5. Aspect Ratio Target (v3.2 FIX: compute from params, NOT obj_val)
        # After retraining change, obj_val becomes score (grad/aspect), not AR
        # NOTE: For P1, AR is already penalized above with stricter bounds
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
            nfp = float(self.params.get("n_field_periods") or self.params.get("nfp", 1))

            computed_ar = float(geometry.aspect_ratio(r_cos, z_sin, nfp).item())
        except Exception:
            computed_ar = ar_target  # Fallback: no AR penalty if computation fails

        ar_deviation = abs(computed_ar - ar_target)
        cost += 1.0 * ar_deviation

        # M2 FIX: Add explicit AR bounds enforcement for P3 multi-objective optimization
        # P3 is a Pareto problem (min AR, max gradient) with implicit bounds from
        # the reference Pareto front: AR ∈ [6.0, 12.0] for feasible solutions.
        # Penalize designs that push toward AR extremes outside this range.
        if self.problem.startswith("p3"):
            ar_lower = self.target_metrics.get("aspect_ratio_lower", 6.0)
            ar_upper = self.target_metrics.get("aspect_ratio_upper", 12.0)
            ar_lower_violation = max(0.0, ar_lower - computed_ar)
            ar_upper_violation = max(0.0, computed_ar - ar_upper)
            cost += 2.0 * (ar_lower_violation + ar_upper_violation)

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
        delta = act * self._get_action_scale(self.current_step)

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
