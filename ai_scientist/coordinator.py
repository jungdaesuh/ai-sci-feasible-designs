"""Coordinator agent for Phase 5 (Hierarchical Autonomy).

The Coordinator manages the high-level strategy of the scientific process,
switching between Exploration (gathering new data/seeds) and Exploitation (optimizing candidates).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ai_scientist import config as ai_config
from ai_scientist import memory
from ai_scientist.planner import PlanningAgent
from ai_scientist.workers import OptimizationWorker, ExplorationWorker, GeometerWorker
from ai_scientist.optim.surrogate_v2 import NeuralOperatorSurrogate
from ai_scientist.optim.generative import GenerativeDesignModel

class Coordinator:
    """
    The central brain of the hierarchical agent system.
    Decides whether to Explore or Exploit and delegates to workers.
    """

    def __init__(
        self, 
        cfg: ai_config.ExperimentConfig, 
        world_model: memory.WorldModel,
        planner: PlanningAgent,
        surrogate: Optional[NeuralOperatorSurrogate] = None,
        generative_model: Optional[GenerativeDesignModel] = None
    ):
        self.cfg = cfg
        self.world_model = world_model
        self.planner = planner
        self.surrogate = surrogate
        self.generative_model = generative_model
        
        # Initialize Workers
        self.opt_worker = OptimizationWorker(cfg, self.surrogate)
        self.explore_worker = ExplorationWorker(cfg, self.generative_model)
        self.geo_worker = GeometerWorker(cfg)
        
        # State
        self.current_strategy = "HYBRID" # Default to doing both

    def decide_strategy(self, cycle: int, experiment_id: int) -> str:
        """
        Decide the strategy for the current cycle based on world model state.
        
        Returns:
            str: "EXPLORE", "EXPLOIT", or "HYBRID"
        """
        # 1. Early cycles: Hybrid (Bootstrap Phase)
        if cycle < 5:
            return "HYBRID"

        # 2. Check for stagnation (Adaptive Switching Phase 5.2)
        hv_delta = self.world_model.average_recent_hv_delta(experiment_id, lookback=3)
        
        if hv_delta is not None and hv_delta < 0.005:
            print(f"[Coordinator] Stagnation detected (HV delta={hv_delta:.4f}). Switching to EXPLORE.")
            return "EXPLORE"
            
        # 3. Default: Hybrid (standard evolutionary approach)
        return "HYBRID"

    def produce_candidates(
        self, 
        cycle: int, 
        experiment_id: int, 
        n_candidates: int,
        template: ai_config.BoundaryTemplateConfig
    ) -> List[Dict[str, Any]]:
        """
        Orchestrates the production of candidates for the current cycle.
        """
        strategy = self.decide_strategy(cycle, experiment_id)
        self.current_strategy = strategy
        
        candidates = []
        
        if strategy == "EXPLORE":
            # Pure exploration: Generate more samples, skip aggressive optimization
            # Increase VAE ratio to 80% to escape local minima
            explore_ctx = {"n_samples": n_candidates, "cycle": cycle, "vae_ratio": 0.8}
            res = self.explore_worker.run(explore_ctx)
            candidates = res.get("candidates", [])
            
        elif strategy == "EXPLOIT":
            # Pure exploitation: Take best previous, or generates seeds and heavily optimizes
            # For now, we treat "EXPLOIT" as "Generate seeds -> Optimize"
            explore_ctx = {"n_samples": n_candidates, "cycle": cycle}
            seeds = self.explore_worker.run(explore_ctx).get("candidates", [])
            
            # Filter seeds with Geometer
            geo_ctx = {"candidates": seeds}
            valid_seeds = self.geo_worker.run(geo_ctx).get("candidates", [])
            
            opt_ctx = {"initial_guesses": valid_seeds}
            res = self.opt_worker.run(opt_ctx)
            candidates = res.get("candidates", [])
            
        else: # HYBRID
            # Standard workflow: Generate seeds -> Optimize
            # But maybe we mix unoptimized seeds?
            explore_ctx = {"n_samples": n_candidates, "cycle": cycle}
            seeds = self.explore_worker.run(explore_ctx).get("candidates", [])
            
            # Filter seeds with Geometer
            geo_ctx = {"candidates": seeds}
            valid_seeds = self.geo_worker.run(geo_ctx).get("candidates", [])
            
            opt_ctx = {"initial_guesses": valid_seeds}
            res = self.opt_worker.run(opt_ctx)
            candidates = res.get("candidates", [])
            
        return candidates
