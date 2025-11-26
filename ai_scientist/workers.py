"""Worker agents for Phase 5 (Hierarchical Autonomy)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping
import numpy as np
import jax.numpy as jnp
import torch

from ai_scientist import config as ai_config
from ai_scientist.optim import differentiable
from ai_scientist.optim.surrogate_v2 import NeuralOperatorSurrogate
from ai_scientist.optim.generative import GenerativeDesignModel
from ai_scientist.optim.samplers import NearAxisSampler

class Worker(ABC):
    """Abstract base class for specialized workers."""
    
    @abstractmethod
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the worker's task based on the provided context."""
        pass


class OptimizationWorker(Worker):
    """Worker responsible for exploiting the search space using differentiable optimization."""
    
    def __init__(self, cfg: ai_config.ExperimentConfig, surrogate: NeuralOperatorSurrogate):
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
            print(f"[OptimizationWorker] Optimizing {len(initial_guesses)} candidates with Gradient Descent...")
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
             print("[OptimizationWorker] Surrogate not ready or not provided. Skipping optimization.")
             return {"candidates": initial_guesses, "status": "skipped"}


class ExplorationWorker(Worker):
    """Worker responsible for exploring the search space using generative models."""
    
    def __init__(
        self, 
        cfg: ai_config.ExperimentConfig, 
        generative_model: GenerativeDesignModel | None,
        sampler: NearAxisSampler | None = None
    ):
        self.cfg = cfg
        self.generative_model = generative_model
        if sampler is None and cfg.proposal_mix.sampler_type == "near_axis":
            try:
                self.sampler = NearAxisSampler(cfg.boundary_template)
            except Exception as exc:
                print(f"[ExplorationWorker] Failed to init NearAxisSampler: {exc}")
                self.sampler = None
        else:
            self.sampler = sampler
        
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
            # Calculate VAE ratio (e.g., 40% as in runner.py)
            vae_count = int(n_samples * 0.4)
            if vae_count > 0:
                print(f"[ExplorationWorker] Sampling {vae_count} from VAE...")
                # Assuming generative_model has a sample method
                # Note: generative_model.sample returns list of dicts
                try:
                    vae_candidates = self.generative_model.sample(vae_count)
                    candidates.extend(vae_candidates)
                except Exception as exc:
                     print(f"[ExplorationWorker] VAE sampling failed: {exc}")

        # Near Axis Sampling (Fallback or mix)
        remaining = n_samples - len(candidates)
        if remaining > 0:
            if self.sampler:
                print(f"[ExplorationWorker] Sampling {remaining} from NearAxisSampler...")
                # Generate seeds
                seeds = [self.cfg.random_seed + i for i in range(remaining)] # Simple seeding for now
                try:
                    sampled = self.sampler.generate(seeds)
                    # Convert to candidate format if needed, NearAxisSampler.generate returns list of dicts
                    candidates.extend(sampled)
                except Exception as exc:
                    print(f"[ExplorationWorker] Sampler failed: {exc}")
            else:
                # Fallback to simple random or template?
                # For now just warn
                print("[ExplorationWorker] No sampler available for remaining candidates.")
            
        return {"candidates": candidates, "status": "explored"}
