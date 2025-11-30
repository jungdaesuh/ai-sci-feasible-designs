"""Coordinator agent for Phase 5 (Hierarchical Autonomy).

The Coordinator manages the high-level strategy of the scientific process,
switching between Exploration (gathering new data/seeds) and Exploitation (optimizing candidates).
"""

from __future__ import annotations

from dataclasses import dataclass, field # Added
from typing import Any, Dict, List, Optional
import time # Added
import numpy as np # Added
import jax.numpy as jnp # Added
from constellaration.optimization.augmented_lagrangian import AugmentedLagrangianState # Added
from ai_scientist.optim.alm_bridge import ( # Added
    ALMContext, # Added
    ALMStepResult, # Added
    create_alm_context, # Added
    step_alm, # Added
    state_to_boundary_params, # Added
) # Added


from ai_scientist import config as ai_config
from ai_scientist import memory
from ai_scientist.planner import PlanningAgent, OptimizerDiagnostics, OptimizationDirective, ConstraintDiagnostic, DirectiveAction # Modified PlanningAgent import to include other ASO types
from ai_scientist.workers import OptimizationWorker, ExplorationWorker, GeometerWorker
from ai_scientist.optim.surrogate_v2 import NeuralOperatorSurrogate
from ai_scientist.optim.generative import GenerativeDesignModel


@dataclass # Added
class TrajectoryState: # Added
    """State for a single optimization trajectory.""" # Added
    id: int # Added
    seed: Dict[str, Any] # Added
    alm_context: Optional[ALMContext] = None # Added
    alm_state: Optional[AugmentedLagrangianState] = None # Added
    history: List[AugmentedLagrangianState] = field(default_factory=list) # Added
    evals_used: int = 0 # Added
    steps: int = 0 # Added
    status: str = "active" # Added
    best_objective: float = float("inf") # Added
    best_violation: float = float("inf") # Added
    stagnation_count: int = 0 # Added
    budget_used: int = 0 # Added

class Coordinator:
    """
    The central brain of the hierarchical agent system.
    Decides whether to Explore or Exploit and delegates to workers.
    """

    CONSTRAINT_NAMES = { # Added
        "p1": ["aspect_ratio", "average_triangularity", "edge_rotational_transform"], # Added
        "p2": ["aspect_ratio", "edge_rotational_transform", "edge_magnetic_mirror_ratio", # Added
               "max_elongation", "qi"], # Added
        "p3": ["edge_rotational_transform", "edge_magnetic_mirror_ratio", # Added
               "vacuum_well", "flux_compression", "qi"], # Added
    } # Added

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

        # ASO Initialization
        problem_key = (cfg.problem or "p3").lower()[:2]
        self.constraint_names = self.CONSTRAINT_NAMES.get(problem_key, self.CONSTRAINT_NAMES["p3"])
        self.telemetry: List[Dict[str, Any]] = []


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

    def produce_candidates_aso(
        self,
        cycle: int,
        experiment_id: int,
        eval_budget: int,
        template: ai_config.BoundaryTemplateConfig,
        initial_seeds: Optional[List[Dict[str, Any]]] = None,
        initial_config: Optional[ai_config.ExperimentConfig] = None,
    ) -> List[Dict[str, Any]]:
        """
        ASO loop with real ALM state supervision.
        """
        config = initial_config or self.cfg
        aso = config.aso
        alm = config.alm

        # 1. Prepare seeds
        seeds = self._prepare_seeds(initial_seeds, cycle, 1)
        if not seeds:
            print("[Coordinator] No valid seeds, returning empty")
            return []

        # 2. Run trajectory with real ALM
        traj = TrajectoryState(id=0, seed=seeds[0])
        candidates = self._run_trajectory_aso(
            traj=traj,
            eval_budget=eval_budget,
            cycle=cycle,
            experiment_id=experiment_id,
            config=config,
        )

        # 3. Persist telemetry
        self._persist_telemetry(experiment_id)

        return candidates

    def _run_trajectory_aso(
        self,
        traj: TrajectoryState,
        eval_budget: int,
        cycle: int,
        experiment_id: int,
        config: ai_config.ExperimentConfig,
    ) -> List[Dict[str, Any]]:
        """Run trajectory with real ALM state and supervision."""
        aso = config.aso
        alm = config.alm
        candidates = []

        # Initialize ALM context and state
        boundary = self._seed_to_boundary(traj.seed)
        problem = self._get_problem(config)
        settings = self._build_optimization_settings(config)

        traj.alm_context, traj.alm_state = create_alm_context(
            boundary=boundary,
            problem=problem,
            settings=settings,
        )

        oracle_budget = alm.oracle_budget_initial

        while traj.budget_used < eval_budget and traj.status == "active":
            traj.steps += 1
            step_start = time.perf_counter()

            # 1. Execute ALM step
            result = step_alm(
                context=traj.alm_context,
                state=traj.alm_state,
                budget=min(oracle_budget, eval_budget - traj.budget_used),
                num_workers=alm.oracle_num_workers,
            )

            traj.alm_state = result.state
            traj.budget_used += result.n_evals
            traj.history.append(result.state)

            # 2. Generate diagnostics from REAL ALM state
            diagnostics = self._generate_diagnostics(result.state, traj)

            # 3. Update trajectory tracking
            self._update_trajectory_best(traj, diagnostics)

            # 4. Get directive (tiered supervision)
            llm_called = diagnostics.requires_llm_supervision(aso)
            directive = self.planner.supervise(diagnostics, cycle, aso)

            # 5. Log telemetry
            wall_time_ms = (time.perf_counter() - step_start) * 1000
            self._log_telemetry(
                experiment_id, cycle, traj, diagnostics, directive, wall_time_ms, llm_called
            )

            # 6. Apply directive
            if directive.action == DirectiveAction.STOP:
                traj.status = "converged" if diagnostics.status == "FEASIBLE_FOUND" else "stagnated"
                print(f"[Coordinator] STOP: {directive.reasoning}")
                # Extract final candidate
                candidates.append({
                    "params": state_to_boundary_params(traj.alm_context, traj.alm_state),
                    "objective": result.objective,
                    "max_violation": result.max_violation,
                    "source": "aso",
                })
                break

            if directive.action == DirectiveAction.RESTART:
                # Try new seed
                new_seeds = self._prepare_seeds(None, cycle, 1)
                if new_seeds:
                    # Save current best before restart
                    candidates.append({
                        "params": state_to_boundary_params(traj.alm_context, traj.alm_state),
                        "objective": result.objective,
                        "max_violation": result.max_violation,
                        "source": "aso_pre_restart",
                    })

                    traj.seed = new_seeds[0]
                    traj.history = []
                    traj.stagnation_count = 0
                    boundary = self._seed_to_boundary(traj.seed)
                    traj.alm_context, traj.alm_state = create_alm_context(
                        boundary=boundary,
                        problem=problem,
                        settings=settings,
                    )
                    oracle_budget = alm.oracle_budget_initial
                    print(f"[Coordinator] RESTART with new seed")
                else:
                    traj.status = "abandoned"
                    print(f"[Coordinator] RESTART failed, no seeds")
                    break
                continue

            if directive.action == DirectiveAction.ADJUST:
                # Apply ALM overrides directly to state
                if directive.alm_overrides and "penalty_parameters" in directive.alm_overrides:
                    new_penalties = jnp.array(directive.alm_overrides["penalty_parameters"])
                    traj.alm_state = traj.alm_state.copy(
                        update={"penalty_parameters": new_penalties}
                    )
                    print(f"[Coordinator] ADJUST penalties: {directive.reasoning}")

            # Increase oracle budget for next iteration
            oracle_budget = min(alm.oracle_budget_max, oracle_budget + alm.oracle_budget_increment)

            # Auto-stop on excessive stagnation
            if traj.stagnation_count >= aso.max_stagnation_steps:
                traj.status = "stagnated"
                print(f"[Coordinator] Auto-STOP (stagnation limit)")
                candidates.append({
                    "params": state_to_boundary_params(traj.alm_context, traj.alm_state),
                    "objective": result.objective,
                    "max_violation": result.max_violation,
                    "source": "aso_stagnation",
                })
                break

        print(f"[Coordinator] Trajectory done: {traj.status}, {traj.steps} steps, "
              f"{traj.budget_used} evals, {len(candidates)} candidates")

        return candidates

    def _generate_diagnostics(
        self,
        alm_state: AugmentedLagrangianState,
        traj: TrajectoryState,
    ) -> OptimizerDiagnostics:
        """Generate rich diagnostics from REAL ALM state."""
        aso = self.cfg.aso
        prev = traj.history[-2] if len(traj.history) >= 2 else None

        # Extract all fields from ALM state
        objective = float(alm_state.objective)
        constraints = [float(c) for c in alm_state.constraints]
        multipliers = [float(m) for m in alm_state.multipliers]
        penalties = [float(p) for p in alm_state.penalty_parameters]
        bounds_norm = float(jnp.linalg.norm(alm_state.bounds))

        objective_delta = objective - float(prev.objective) if prev else 0.0
        max_violation = max(0.0, max(constraints)) if constraints else 0.0

        # Constraint analysis
        constraint_diagnostics = []
        diverging_count = 0

        for i, name in enumerate(self.constraint_names):
            if i >= len(constraints):
                continue

            violation = max(0.0, constraints[i])
            penalty = penalties[i] if i < len(penalties) else 1.0
            multiplier = multipliers[i] if i < len(multipliers) else 0.0
            trend = "stable"
            delta = 0.0

            if prev and i < len(prev.constraints):
                prev_violation = max(0.0, float(prev.constraints[i]))
                delta = violation - prev_violation

                if violation > prev_violation * (1 + aso.violation_increase_threshold) and violation > 1e-4:
                    trend = "increasing_violation"
                    diverging_count += 1
                elif violation < prev_violation * (1 - aso.violation_decrease_threshold):
                    trend = "decreasing_violation"

            constraint_diagnostics.append(ConstraintDiagnostic(
                name=name,
                violation=violation,
                penalty=penalty,
                multiplier=multiplier,
                trend=trend,
                delta=delta,
            ))

        # Status determination
        narrative = []
        if max_violation < aso.feasibility_threshold:
            status = "FEASIBLE_FOUND"
            narrative.append(f"Feasible region reached (max_violation={max_violation:.4f})")
        elif diverging_count >= len(self.constraint_names) // 2:
            status = "DIVERGING"
            narrative.append(f"{diverging_count} constraints diverging")
        elif prev and abs(objective_delta) < aso.stagnation_objective_threshold:
            if max_violation > aso.stagnation_violation_threshold:
                status = "STAGNATION"
                narrative.append(f"Stagnation: obj_delta={objective_delta:.6f}, violation={max_violation:.4f}")
            else:
                status = "IN_PROGRESS"
                narrative.append("Near convergence")
        else:
            status = "IN_PROGRESS"
            narrative.append("Normal progress")

        return OptimizerDiagnostics(
            step=traj.steps,
            trajectory_id=traj.id,
            objective=objective,
            objective_delta=objective_delta,
            max_violation=max_violation,
            constraints_raw=constraints,
            multipliers=multipliers,
            penalty_parameters=penalties,
            bounds_norm=bounds_norm,
            status=status,
            constraint_diagnostics=constraint_diagnostics,
            narrative=narrative,
            steps_since_improvement=traj.stagnation_count,
        )

    def _update_trajectory_best(self, traj: TrajectoryState, diag: OptimizerDiagnostics):
        """Update best values and stagnation counter."""
        improved = False
        if diag.max_violation < traj.best_violation:
            traj.best_violation = diag.max_violation
            improved = True
        if diag.objective < traj.best_objective and diag.max_violation <= traj.best_violation:
            traj.best_objective = diag.objective
            improved = True

        if improved:
            traj.stagnation_count = 0
        else:
            traj.stagnation_count += 1

    def _log_telemetry(
        self,
        experiment_id: int,
        cycle: int,
        traj: TrajectoryState,
        diag: OptimizerDiagnostics,
        directive: OptimizationDirective,
        wall_time_ms: float,
        llm_called: bool,
    ):
        """Record telemetry event with full ALM state."""
        from datetime import datetime, timezone
        self.telemetry.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "experiment_id": experiment_id,
            "cycle": cycle,
            "trajectory_id": traj.id,
            "step": traj.steps,
            "status": diag.status,
            "objective": diag.objective,
            "max_violation": diag.max_violation,
            "bounds_norm": diag.bounds_norm,
            "penalties": diag.penalty_parameters,
            "multipliers": diag.multipliers,
            "directive_action": directive.action.value,
            "directive_source": directive.source.value,
            "directive_reasoning": directive.reasoning,
            "evals_used": traj.budget_used,
            "wall_time_ms": wall_time_ms,
            "llm_called": llm_called,
        })

    def _persist_telemetry(self, experiment_id: int):
        """Persist telemetry to JSONL file."""
        if not self.telemetry:
            return

        from pathlib import Path
        import json

        telemetry_dir = Path(self.cfg.reporting_dir) / "telemetry"
        telemetry_dir.mkdir(parents=True, exist_ok=True)
        telemetry_file = telemetry_dir / f"aso_exp{experiment_id}.jsonl"

        with open(telemetry_file, "a") as f:
            for event in self.telemetry:
                f.write(json.dumps(event) + "\n")

        self.telemetry = []

    # Helper methods
    def _prepare_seeds(self, initial_seeds, cycle, n_needed):
        """Prepare and validate seeds using ExplorationWorker + GeometerWorker."""
        if initial_seeds:
            return initial_seeds

        # Generate using Explore worker
        # Generate a batch to ensure we have enough valid seeds
        explore_ctx = {"n_samples": max(n_needed, 10), "cycle": cycle}
        res = self.explore_worker.run(explore_ctx)
        candidates = res.get("candidates", [])

        # Filter with Geometer
        if candidates:
            geo_ctx = {"candidates": candidates}
            geo_res = self.geo_worker.run(geo_ctx)
            candidates = geo_res.get("candidates", [])

        if not candidates:
             return []

        return candidates[:n_needed]

    def _seed_to_boundary(self, seed):
        """Convert seed dict to SurfaceRZFourier."""
        from constellaration.geometry import surface_rz_fourier

        # Handle both "seed" dict and direct params dict
        params_map = seed.get("params", seed)

        return surface_rz_fourier.SurfaceRZFourier(
            r_cos=np.array(params_map["r_cos"]),
            z_sin=np.array(params_map["z_sin"]),
            n_field_periods=int(params_map.get("n_field_periods", 1)),
            is_stellarator_symmetric=bool(params_map.get("is_stellarator_symmetric", True)),
            r_sin=np.array(params_map["r_sin"]) if "r_sin" in params_map else None,
            z_cos=np.array(params_map["z_cos"]) if "z_cos" in params_map else None,
        )

    def _get_problem(self, config):
        """Get problem instance from config."""
        from constellaration import problems
        problem_key = (config.problem or "p3").lower()

        if problem_key.startswith("p1"):
            return problems.GeometricalProblem()
        elif problem_key.startswith("p2"):
            return problems.SimpleToBuildQIStellarator()
        else:
            # P3 defaults (MHD Stable QI)
            return problems.MHDStableQIStellarator()

    def _build_optimization_settings(self, config):
        """Build OptimizationSettings from config."""
        from constellaration.optimization.settings import (
            OptimizationSettings,
            AugmentedLagrangianMethodSettings,
            NevergradSettings
        )
        from constellaration.optimization.augmented_lagrangian import AugmentedLagrangianSettings
        from constellaration.forward_model import ConstellarationSettings

        # Determine forward model settings
        # For ASO we default to high fidelity (P3/P2 style)
        fm_settings = ConstellarationSettings.default_high_fidelity()

        return OptimizationSettings(
            max_poloidal_mode=config.boundary_template.n_poloidal_modes,
            max_toroidal_mode=config.boundary_template.n_toroidal_modes,
            infinity_norm_spectrum_scaling=0.0,
            forward_model_settings=fm_settings,
            optimizer_settings=AugmentedLagrangianMethodSettings(
                maxit=config.alm.maxit,
                penalty_parameters_initial=config.alm.penalty_parameters_initial,
                bounds_initial=config.alm.bounds_initial,
                augmented_lagrangian_settings=AugmentedLagrangianSettings(
                    penalty_parameters_increase_factor=config.alm.penalty_parameters_increase_factor,
                    constraint_violation_tolerance_reduction_factor=config.alm.constraint_violation_tolerance_reduction_factor,
                    bounds_reduction_factor=config.alm.bounds_reduction_factor,
                    penalty_parameters_max=config.alm.penalty_parameters_max,
                    bounds_min=config.alm.bounds_min,
                ),
                oracle_settings=NevergradSettings(
                    budget_initial=config.alm.oracle_budget_initial,
                    budget_increment=config.alm.oracle_budget_increment,
                    budget_max=config.alm.oracle_budget_max,
                    num_workers=config.alm.oracle_num_workers,
                    max_time=None,
                    batch_mode=True,
                ),
            )
        )

