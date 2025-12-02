"""Worker agents for Phase 5 (Hierarchical Autonomy)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch

from ai_scientist import config as ai_config
from ai_scientist.optim import differentiable
from ai_scientist.optim.generative import DiffusionDesignModel, GenerativeDesignModel
from ai_scientist.optim.samplers import (
    NearAxisSampler,
    OfflineSeedSampler,
    RotatingEllipseSampler,
)
from ai_scientist.optim.surrogate_v2 import NeuralOperatorSurrogate


class Worker(ABC):
    """Abstract base class for specialized workers."""

    @abstractmethod
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the worker's task based on the provided context."""
        pass


class OptimizationWorker(Worker):
    """Worker responsible for exploiting the search space using differentiable optimization."""

    def __init__(
        self,
        cfg: ai_config.ExperimentConfig,
        surrogate: Optional[NeuralOperatorSurrogate],
    ):
        self.cfg = cfg
        self.surrogate = surrogate

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run gradient descent on the provided initial guesses.

        Context keys:
            initial_guesses: List[Dict[str, Any]] - List of boundary params to optimize.
        """
        initial_guesses = context.get("initial_guesses", [])
        if not initial_guesses:
            return {"candidates": []}

        if self.surrogate and self.surrogate._trained:
            print(
                f"[OptimizationWorker] Optimizing {len(initial_guesses)} candidates with Gradient Descent..."
            )
            try:
                optimized_candidates = differentiable.gradient_descent_on_inputs(
                    initial_guesses,
                    self.surrogate,
                    self.cfg,
                )
                return {"candidates": optimized_candidates, "status": "optimized"}
            except Exception as exc:
                print(f"[OptimizationWorker] Gradient Descent failed: {exc}")
                return {"candidates": initial_guesses, "status": "failed"}
        else:
            print(
                "[OptimizationWorker] Surrogate not ready or not provided. Skipping optimization."
            )
            return {"candidates": initial_guesses, "status": "skipped"}


class ExplorationWorker(Worker):
    """Worker responsible for exploring the search space using generative models."""

    def __init__(
        self,
        cfg: ai_config.ExperimentConfig,
        generative_model: GenerativeDesignModel | None,
        sampler: NearAxisSampler | OfflineSeedSampler | None = None,
    ):
        self.cfg = cfg
        self.generative_model = generative_model

        if sampler is not None:
            self.sampler = sampler
        else:
            # Initialize sampler based on config
            if cfg.proposal_mix.sampler_type == "near_axis":
                try:
                    self.sampler = NearAxisSampler(cfg.boundary_template)
                except Exception as exc:
                    print(f"[ExplorationWorker] Failed to init NearAxisSampler: {exc}")
                    self.sampler = None
            elif cfg.proposal_mix.sampler_type == "offline_seeds":
                try:
                    self.sampler = OfflineSeedSampler(cfg.problem)
                except Exception as exc:
                    print(
                        f"[ExplorationWorker] Failed to init OfflineSeedSampler: {exc}"
                    )
                    self.sampler = None
            elif cfg.proposal_mix.sampler_type == "rotating_ellipse":
                try:
                    self.sampler = RotatingEllipseSampler(cfg.boundary_template)
                except Exception as exc:
                    print(
                        f"[ExplorationWorker] Failed to init RotatingEllipseSampler: {exc}"
                    )
                    self.sampler = None
            else:
                self.sampler = None

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate new candidates using VAE or random sampling.

        Context keys:
            n_samples: int - Number of samples to generate.
        """
        n_samples = context.get("n_samples", 10)
        candidates = []

        # VAE Sampling
        if self.generative_model and self.cfg.generative.enabled:
            # Calculate VAE ratio (e.g., 40% as in runner.py, or override via context)
            vae_ratio = context.get("vae_ratio", 0.4)
            vae_count = int(n_samples * vae_ratio)
            if vae_count > 0:
                print(
                    f"[ExplorationWorker] Sampling {vae_count} from Generative Model..."
                )

                cycle_index = context.get("cycle", 0)
                seed_base = self.cfg.random_seed + cycle_index * 20000

                try:
                    if isinstance(self.generative_model, DiffusionDesignModel):
                        target_metrics = {
                            "aspect_ratio": 8.0,
                            "minimum_normalized_magnetic_gradient_scale_length": 20.0,
                            "max_elongation": 1.5,
                            "edge_rotational_transform_over_n_field_periods": 0.4,
                        }
                        vae_candidates = self.generative_model.sample(
                            vae_count, target_metrics=target_metrics, seed=seed_base
                        )
                    else:
                        # VAE
                        vae_candidates = self.generative_model.sample(
                            vae_count, seed=seed_base
                        )

                    candidates.extend(vae_candidates)
                except Exception as exc:
                    print(f"[ExplorationWorker] Generative sampling failed: {exc}")

        # Near Axis / Offline Seed Sampling (Fallback or mix)
        remaining = n_samples - len(candidates)
        if remaining > 0:
            if self.sampler:
                print(
                    f"[ExplorationWorker] Sampling {remaining} from {self.sampler.__class__.__name__}..."
                )
                # Generate seeds
                seeds = [
                    self.cfg.random_seed + i for i in range(remaining)
                ]  # Simple seeding for now
                try:
                    sampled = self.sampler.generate(seeds)
                    # Convert to candidate format if needed, NearAxisSampler.generate returns list of dicts
                    candidates.extend(sampled)
                except Exception as exc:
                    print(f"[ExplorationWorker] Sampler failed: {exc}")
            else:
                # Fallback to simple random or template?
                # For now just warn
                print(
                    "[ExplorationWorker] No sampler available for remaining candidates."
                )

        return {"candidates": candidates, "status": "explored"}


class GeometerWorker(Worker):
    """Worker responsible for validating geometric constraints (The Gatekeeper)."""

    def __init__(self, cfg: ai_config.ExperimentConfig):
        self.cfg = cfg

    def check_validity(self, candidate: Dict[str, Any]) -> bool:
        """
        Check if a candidate has valid geometry.

        Checks:
        1. Jacobian (Area Element) > 0 everywhere (no singularities).
        2. Reasonable Elongation (< 10).
        3. Positive Aspect Ratio.
        """
        params = candidate.get("params")
        if not params:
            return False

        try:
            # Convert to torch tensors for geometry utils
            r_cos = torch.tensor(params["r_cos"], dtype=torch.float32).unsqueeze(
                0
            )  # (1, m, n)
            z_sin = torch.tensor(params["z_sin"], dtype=torch.float32).unsqueeze(0)
            nfp = params.get("n_field_periods", 1)

            # 1. Check Jacobian (Normal Vector Magnitude)
            # We use a coarse grid for speed
            from ai_scientist.optim import geometry

            d = geometry._compute_derivatives(r_cos, z_sin, nfp, n_theta=32, n_zeta=32)

            R = d["R"]
            R_t, R_z = d["R_t"], d["R_z"]
            Z_t, Z_z = d["Z_t"], d["Z_z"]

            n_R = -R * Z_t
            n_phi = Z_t * R_z - R_t * Z_z
            n_Z = R * R_t

            norm_sq = n_R**2 + n_phi**2 + n_Z**2
            norm_n = torch.sqrt(torch.clamp(norm_sq, min=0.0))

            # If normal is too small, surface is singular/pinched
            if torch.min(norm_n) < 1e-4:
                return False

            # 2. Check Elongation
            elo = geometry.elongation(r_cos, z_sin, nfp, n_theta=32, n_zeta=32)
            if elo.item() > 10.0:
                return False

            # 3. Check Aspect Ratio
            ar = geometry.aspect_ratio(r_cos, z_sin, nfp, n_theta=32, n_zeta=32)
            if ar.item() <= 0:  # Should be positive
                return False

            return True

        except Exception as e:
            print(f"[GeometerWorker] Validation failed: {e}")
            return False

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter a list of candidates.

        Context keys:
            candidates: List[Dict[str, Any]]
        """
        candidates = context.get("candidates", [])
        valid_candidates = []

        for cand in candidates:
            if self.check_validity(cand):
                valid_candidates.append(cand)
            else:
                # Optionally log rejection
                pass

        print(
            f"[GeometerWorker] Retained {len(valid_candidates)}/{len(candidates)} candidates."
        )
        return {"candidates": valid_candidates, "status": "filtered"}
