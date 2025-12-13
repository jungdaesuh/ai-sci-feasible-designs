"""Coordinator agent for Phase 5 (Hierarchical Autonomy).

The Coordinator manages the high-level strategy of the scientific process,
switching between Exploration (gathering new data/seeds) and Exploitation (optimizing candidates).
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import jax.numpy as jnp
import numpy as np
import pydantic
from constellaration.optimization.augmented_lagrangian import AugmentedLagrangianState

from ai_scientist import config as ai_config
from ai_scientist import memory
from ai_scientist.optim.alm_bridge import (
    ALMContext,
    create_alm_context,
    state_to_boundary_params,
    step_alm,
)
from ai_scientist.optim.generative import DiffusionDesignModel, GenerativeDesignModel
from ai_scientist.optim.surrogate_v2 import NeuralOperatorSurrogate
from ai_scientist.planner import (
    ConstraintDiagnostic,
    DirectiveAction,
    OptimizationDirective,
    OptimizerDiagnostics,
    PlanningAgent,
)
from ai_scientist.problems import get_problem
from ai_scientist.workers import (
    ExplorationWorker,
    GeometerWorker,
    OptimizationWorker,
    PreRelaxWorker,
    RLRefinementWorker,
)


def _alm_constraint_names(problem: str) -> list[str]:
    """Return constraint names aligned with constellaration ALM `objective_constraints` order.

    Coordinator diagnostics consume `AugmentedLagrangianState.constraints`, which come
    from `constellaration.optimization.augmented_lagrangian_runner.objective_constraints`.
    Those constraints are *not* the same as the benchmark problem constraint lists
    (notably P3 includes an aspect-ratio upper-bound constraint injected by the runner).
    """
    key = (problem or "p3").lower()
    if key.startswith("p1"):
        # constellaration runner order: [aspect_ratio, average_triangularity, iota_lower_bound]
        return [
            "aspect_ratio",
            "average_triangularity",
            "edge_rotational_transform_over_n_field_periods",
        ]
    if key.startswith("p2"):
        # constellaration runner order: [aspect_ratio, iota_lower_bound, log10(qi), mirror, elong]
        return [
            "aspect_ratio",
            "edge_rotational_transform_over_n_field_periods",
            "log10_qi",
            "edge_magnetic_mirror_ratio",
            "max_elongation",
        ]
    # P3 runner order includes aspect_ratio upper bound as a constraint:
    # [aspect_ratio, iota_lower_bound, log10(qi), mirror, flux, vacuum_well_scaled]
    return [
        "aspect_ratio",
        "edge_rotational_transform_over_n_field_periods",
        "log10_qi",
        "edge_magnetic_mirror_ratio",
        "flux_compression_in_regions_of_bad_curvature",
        "vacuum_well",
    ]


class TrajectoryState(pydantic.BaseModel):
    """State for a single optimization trajectory."""

    model_config = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,  # Immutable like frozen dataclass
    )

    id: int
    seed: Dict[str, Any]
    alm_context: Optional[ALMContext] = None
    alm_state: Optional[AugmentedLagrangianState] = None
    history: List[AugmentedLagrangianState] = pydantic.Field(default_factory=list)
    evals_used: int = 0
    steps: int = 0
    status: str = "active"
    best_objective: float = float("inf")
    best_violation: float = float("inf")
    stagnation_count: int = 0
    budget_used: int = 0


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
        generative_model: Optional[GenerativeDesignModel | DiffusionDesignModel] = None,
    ):
        self.cfg = cfg
        self.world_model = world_model
        self.planner = planner
        self.surrogate = surrogate
        self.generative_model = generative_model

        self.opt_worker = OptimizationWorker(cfg, self.surrogate)
        self.explore_worker = ExplorationWorker(cfg, self.generative_model)
        self.geo_worker = GeometerWorker(cfg)
        self.rl_worker = RLRefinementWorker(cfg, self.surrogate)
        self.prerelax_worker = PreRelaxWorker(
            cfg,
            surrogate_schema=self.surrogate._schema if self.surrogate else None,
        )

        # State
        self.current_strategy = "HYBRID"  # Default to doing both

        # ASO Initialization
        self.problem = get_problem(cfg.problem or "p3")
        self.constraint_names = _alm_constraint_names(cfg.problem or "p3")
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
            print(
                f"[Coordinator] Stagnation detected (HV delta={hv_delta:.4f}). Switching to EXPLORE."
            )
            return "EXPLORE"

        # 3. Default: Hybrid (standard evolutionary approach)
        return "HYBRID"

    def produce_candidates(
        self,
        cycle: int,
        experiment_id: int,
        n_candidates: int,
        template: ai_config.BoundaryTemplateConfig,
    ) -> List[Dict[str, Any]]:
        """
        Orchestrates the production of candidates for the current cycle.
        """
        strategy = self.decide_strategy(cycle, experiment_id)
        self.current_strategy = strategy

        candidates = []

        if strategy == "EXPLORE":
            # ═══════════════════════════════════════════════════════════
            # QUAD-HYBRID PIPELINE (EXPLORE mode - lightweight)
            # Skip Surrogate ranking and RL refinement to focus on diversity
            # ═══════════════════════════════════════════════════════════

            # STAGE 1: Dream - Generate N seeds (increased VAE ratio for diversity)
            explore_ctx = {"n_samples": n_candidates, "cycle": cycle, "vae_ratio": 0.8}
            seeds = self.explore_worker.run(explore_ctx).get("candidates", [])
            print(f"[Coordinator] Dreamer generated {len(seeds)} seeds (EXPLORE mode)")

            # STAGE 2: Pre-relax - Fast geometric smoothing
            prerelax_ctx = {
                "candidates": seeds,
                "schema": self.surrogate._schema if self.surrogate else None,
            }
            prerelaxed = self.prerelax_worker.run(prerelax_ctx).get("candidates", [])
            print(f"[Coordinator] Pre-relaxer smoothed {len(prerelaxed)} candidates")

            # STAGE 3: Geometer - Validate geometric constraints
            geo_ctx = {"candidates": prerelaxed}
            candidates = self.geo_worker.run(geo_ctx).get("candidates", [])
            print(
                f"[Coordinator] Geometer passed {len(candidates)}/{len(prerelaxed)} candidates"
            )

            # EXPLORE mode: Skip Surrogate ranking and RL refinement
            # to maximize diversity and escape local minima
            print(
                f"[Coordinator] Explore pipeline complete: {len(candidates)} candidates "
                "(skipped Surrogate/RL for diversity)"
            )

        elif strategy == "EXPLOIT":
            # ═══════════════════════════════════════════════════════════
            # QUAD-HYBRID PIPELINE (EXPLOIT mode)
            # ═══════════════════════════════════════════════════════════

            # STAGE 1: Dream - Generate N seeds
            explore_ctx = {"n_samples": n_candidates, "cycle": cycle}
            seeds = self.explore_worker.run(explore_ctx).get("candidates", [])
            print(f"[Coordinator] Dreamer generated {len(seeds)} seeds")

            # STAGE 2: Pre-relax - Fast geometric smoothing
            prerelax_ctx = {
                "candidates": seeds,
                "schema": self.surrogate._schema if self.surrogate else None,
            }
            prerelaxed = self.prerelax_worker.run(prerelax_ctx).get("candidates", [])
            print(f"[Coordinator] Pre-relaxer smoothed {len(prerelaxed)} candidates")

            # STAGE 3: Geometer - Validate geometric constraints
            geo_ctx = {"candidates": prerelaxed}
            valid_seeds = self.geo_worker.run(geo_ctx).get("candidates", [])
            print(
                f"[Coordinator] Geometer passed {len(valid_seeds)}/{len(prerelaxed)} candidates"
            )

            # STAGE 4: Surrogate Rank - Select top-K
            if valid_seeds and self.surrogate and self.surrogate._trained:
                ranked_seeds = self._surrogate_rank_seeds(valid_seeds, cycle)
                k = min(100, len(ranked_seeds))
                top_k = ranked_seeds[:k]
                print(
                    f"[Coordinator] Surrogate selected top-{k} candidates for RL refinement"
                )
            else:
                top_k = valid_seeds[:100]

            # STAGE 5: RL Refine - Micro-surgery on top-K only
            rl_ctx = {
                "candidates": top_k,
                "target_metrics": explore_ctx.get("target_metrics"),
            }
            refined = self.rl_worker.run(rl_ctx).get("candidates", [])
            print(f"[Coordinator] RL Agent refined {len(refined)} candidates")

            # STAGE 6: Optimize - Final gradient descent
            opt_ctx = {"initial_guesses": refined}
            res = self.opt_worker.run(opt_ctx)
            candidates = res.get("candidates", [])
            print(
                f"[Coordinator] Quad-Hybrid pipeline complete: {len(candidates)} final candidates"
            )

        else:  # HYBRID
            # ═══════════════════════════════════════════════════════════
            # QUAD-HYBRID PIPELINE (HYBRID mode - standard)
            # ═══════════════════════════════════════════════════════════

            # STAGE 1: Dream - Generate N seeds
            explore_ctx = {"n_samples": n_candidates, "cycle": cycle}
            seeds = self.explore_worker.run(explore_ctx).get("candidates", [])
            print(f"[Coordinator] Dreamer generated {len(seeds)} seeds")

            # STAGE 2: Pre-relax - Fast geometric smoothing
            prerelax_ctx = {
                "candidates": seeds,
                "schema": self.surrogate._schema if self.surrogate else None,
            }
            prerelaxed = self.prerelax_worker.run(prerelax_ctx).get("candidates", [])
            print(f"[Coordinator] Pre-relaxer smoothed {len(prerelaxed)} candidates")

            # STAGE 3: Geometer - Validate geometric constraints
            geo_ctx = {"candidates": prerelaxed}
            valid_seeds = self.geo_worker.run(geo_ctx).get("candidates", [])
            print(
                f"[Coordinator] Geometer passed {len(valid_seeds)}/{len(prerelaxed)} candidates"
            )

            # STAGE 4: Surrogate Rank - Select top-K
            if valid_seeds and self.surrogate and self.surrogate._trained:
                ranked_seeds = self._surrogate_rank_seeds(valid_seeds, cycle)
                k = min(100, len(ranked_seeds))
                top_k = ranked_seeds[:k]
                print(
                    f"[Coordinator] Surrogate selected top-{k} candidates for RL refinement"
                )
            else:
                top_k = valid_seeds[:100]

            # STAGE 5: RL Refine - Micro-surgery on top-K only
            rl_ctx = {
                "candidates": top_k,
                "target_metrics": explore_ctx.get("target_metrics"),
            }
            refined = self.rl_worker.run(rl_ctx).get("candidates", [])
            print(f"[Coordinator] RL Agent refined {len(refined)} candidates")

            # STAGE 6: Optimize - Final gradient descent
            opt_ctx = {"initial_guesses": refined}
            res = self.opt_worker.run(opt_ctx)
            candidates = res.get("candidates", [])
            print(
                f"[Coordinator] Quad-Hybrid pipeline complete: {len(candidates)} final candidates"
            )

        # Check for periodic retraining
        if candidates:
            self._periodic_retrain(cycle, experiment_id, candidates)

        return candidates

    def _should_retrain(self, cycle: int, experiment_id: int) -> tuple[bool, str]:
        """Check if retraining should be triggered.

        Returns:
            (should_retrain, reason)
        """
        retraining_cfg = self.cfg.retraining

        if not retraining_cfg.enabled:
            return False, "disabled"

        # Trigger 1: Cycle cadence (every N cycles)
        if cycle > 0 and cycle % retraining_cfg.cycle_cadence == 0:
            return (
                True,
                f"cycle_cadence ({cycle} % {retraining_cfg.cycle_cadence} == 0)",
            )

        # Trigger 2: HV stagnation
        hv_delta = self.world_model.average_recent_hv_delta(
            experiment_id, lookback=retraining_cfg.hv_stagnation_lookback
        )
        if hv_delta is not None and hv_delta < retraining_cfg.hv_stagnation_threshold:
            return True, f"hv_stagnation (delta={hv_delta:.6f})"

        return False, "no_trigger"

    def _periodic_retrain(
        self,
        cycle: int,
        experiment_id: int,
        candidates: List[Dict[str, Any]],
    ) -> None:
        """Perform periodic retraining of generative model and surrogate.

        Args:
            cycle: Current cycle number.
            experiment_id: Experiment ID for world model queries.
            candidates: Candidates from the current cycle.
        """
        should_retrain, reason = self._should_retrain(cycle, experiment_id)

        if not should_retrain:
            return

        retraining_cfg = self.cfg.retraining
        print(
            f"[Coordinator] Periodic retraining triggered (cycle {cycle}, reason: {reason})"
        )

        # Collect elite candidates (top N by score)
        # Sort by any available score metric with fallback chain
        elites = []
        for cand in candidates:
            # Fallback chain: rl_score -> surrogate_score -> objective (negated)
            # -> geometric_energy (negated, lower is better)
            score = cand.get("rl_score")
            if score is None:
                score = cand.get("surrogate_score")
            if score is None:
                # For objective, lower is better so negate for sorting
                obj = cand.get("objective")
                if obj is not None:
                    score = -float(obj)
            if score is None:
                # For geometric_energy, lower is better so negate
                energy = cand.get("geometric_energy")
                if energy is not None:
                    score = -float(energy)
            if score is None:
                score = 0.0
            elites.append((score, cand))

        # Sort by score (higher is better for RL/surrogate scores)
        elites.sort(key=lambda x: x[0], reverse=True)
        top_elites = [cand for _, cand in elites[: retraining_cfg.min_elites]]

        if len(top_elites) < retraining_cfg.min_elites:
            print(
                f"[Coordinator] Skipping retraining: only {len(top_elites)} elites "
                f"(need {retraining_cfg.min_elites})"
            )
            return

        # 1. Fine-tune Generative Model (if available and supports fine-tuning)
        if self.generative_model is not None:
            try:
                # Use fine_tune_on_elites for incremental training (preserves PCA)
                if hasattr(self.generative_model, "fine_tune_on_elites"):
                    print(
                        f"[Coordinator] Fine-tuning generative model on {len(top_elites)} elites..."
                    )
                    self.generative_model.fine_tune_on_elites(top_elites)
                    print("[Coordinator] Generative model fine-tuning complete")
                elif hasattr(self.generative_model, "fit"):
                    # Fallback to fit() for VAE or other models
                    print(
                        f"[Coordinator] Retraining generative model on {len(top_elites)} elites..."
                    )
                    self.generative_model.fit(top_elites)
                    print("[Coordinator] Generative model retraining complete")
            except Exception as e:
                print(f"[Coordinator] Generative model retraining failed: {e}")

        # 2. Retrain Surrogate (if available)
        if self.surrogate is not None:
            try:
                # Prepare training data from elites
                metrics_list = []
                target_values = []

                # Determine training target and direction based on problem type
                problem = (self.cfg.problem or "p3").lower()
                if problem.startswith("p1"):
                    # P1: train on objective (elongation, minimize)
                    minimize_obj = True
                else:
                    # P2/P3: train on score (grad/aspect, maximize)
                    minimize_obj = False

                for cand in top_elites:
                    params = cand.get("params", cand.get("candidate_params"))
                    if params is None:
                        continue

                    # Extract actual metrics from evaluation (not the full candidate)
                    eval_data = cand.get("evaluation", {})
                    actual_metrics = eval_data.get("metrics", {})

                    # Skip if metrics are missing (don't train on fabricated targets)
                    if not actual_metrics:
                        continue

                    # Determine training target based on problem
                    if problem.startswith("p1"):
                        target = eval_data.get("objective", cand.get("objective"))
                    else:
                        # P2/P3: use score (higher = better)
                        target = eval_data.get("score", cand.get("score"))
                        if target is None:
                            # Fallback: compute score from metrics
                            grad = actual_metrics.get(
                                "minimum_normalized_magnetic_gradient_scale_length"
                            )
                            aspect = actual_metrics.get("aspect_ratio")
                            if grad is not None and aspect is not None:
                                target = float(grad) / max(1.0, float(aspect))
                            else:
                                continue  # Skip: can't compute target

                    if target is None:
                        continue

                    metrics_list.append(
                        {"candidate_params": params, "metrics": actual_metrics}
                    )
                    target_values.append(float(target))

                if metrics_list:
                    print(
                        f"[Coordinator] Retraining surrogate on {len(metrics_list)} samples "
                        f"(minimize_objective={minimize_obj})..."
                    )
                    self.surrogate.fit(
                        metrics_list, target_values, minimize_objective=minimize_obj
                    )
                    print("[Coordinator] Surrogate retraining complete")
            except Exception as e:
                print(f"[Coordinator] Surrogate retraining failed: {e}")

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

        # 1. Prepare seeds with Surrogate Ranking
        # Calculate pool size based on multiplier
        multiplier = config.proposal_mix.surrogate_pool_multiplier
        pool_size = int(max(10, 1 * multiplier))  # We only need 1 candidate for ASO

        # Generate larger pool
        raw_seeds = self._prepare_seeds(initial_seeds, cycle, pool_size)

        # Rank and select best
        ranked_seeds = self._surrogate_rank_seeds(raw_seeds, cycle)

        if not ranked_seeds:
            print("[Coordinator] No valid seeds, returning empty")
            return []

        best_seed = ranked_seeds[0]
        print(
            f"[Coordinator] Selected best seed from {len(raw_seeds)} candidates (Score: {best_seed.get('surrogate_score', 'N/A')})"
        )

        # 2. Run trajectory with real ALM
        traj = TrajectoryState(id=0, seed=best_seed)
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

        alm_context, alm_state = create_alm_context(
            boundary=boundary,
            problem=problem,
            settings=settings,
        )
        traj = traj.model_copy(
            update={"alm_context": alm_context, "alm_state": alm_state}
        )

        oracle_budget = alm.oracle_budget_initial

        while traj.budget_used < eval_budget and traj.status == "active":
            traj = traj.model_copy(update={"steps": traj.steps + 1})
            step_start = time.perf_counter()

            # 1. Execute ALM step
            result = step_alm(
                context=alm_context,
                state=traj.alm_state,
                budget=min(oracle_budget, eval_budget - traj.budget_used),
                num_workers=alm.oracle_num_workers,
            )

            new_history = traj.history + [result.state]
            traj = traj.model_copy(
                update={
                    "alm_state": result.state,
                    "budget_used": traj.budget_used + result.n_evals,
                    "history": new_history,
                }
            )

            # 2. Generate diagnostics from REAL ALM state
            diagnostics = self._generate_diagnostics(result.state, traj)

            # 3. Update trajectory tracking
            traj = self._update_trajectory_best(traj, diagnostics)

            # 4. Get directive (tiered supervision)
            llm_called = diagnostics.requires_llm_supervision(aso)
            directive = self.planner.supervise(diagnostics, cycle, aso)

            # 5. Log telemetry
            wall_time_ms = (time.perf_counter() - step_start) * 1000
            self._log_telemetry(
                experiment_id,
                cycle,
                traj,
                diagnostics,
                directive,
                wall_time_ms,
                llm_called,
            )

            # 6. Apply directive
            if directive.action == DirectiveAction.STOP:
                status = (
                    "converged"
                    if diagnostics.status == "FEASIBLE_FOUND"
                    else "stagnated"
                )
                traj = traj.model_copy(update={"status": status})
                print(f"[Coordinator] STOP: {directive.reasoning}")
                # Extract final candidate
                candidates.append(
                    {
                        "params": state_to_boundary_params(alm_context, traj.alm_state),
                        "objective": result.objective,
                        "max_violation": result.max_violation,
                        "source": "aso",
                        "seed": traj.seed.get("seed", 0),
                    }
                )
                break

            if directive.action == DirectiveAction.RESTART:
                # Try new seed
                new_seeds = self._prepare_seeds(None, cycle, 1)
                if new_seeds:
                    # Save current best before restart
                    candidates.append(
                        {
                            "params": state_to_boundary_params(
                                alm_context, traj.alm_state
                            ),
                            "objective": result.objective,
                            "max_violation": result.max_violation,
                            "source": "aso_pre_restart",
                            "seed": traj.seed.get("seed", 0),
                        }
                    )

                    boundary = self._seed_to_boundary(new_seeds[0])
                    new_context, new_state = create_alm_context(
                        boundary=boundary,
                        problem=problem,
                        settings=settings,
                    )
                    alm_context = new_context
                    traj = traj.model_copy(
                        update={
                            "seed": new_seeds[0],
                            "history": [],
                            "stagnation_count": 0,
                            "alm_context": new_context,
                            "alm_state": new_state,
                        }
                    )

                    oracle_budget = alm.oracle_budget_initial
                    print("[Coordinator] RESTART with new seed")
                else:
                    traj = traj.model_copy(update={"status": "abandoned"})
                    print("[Coordinator] RESTART failed, no seeds")
                    break
                continue

            if directive.action == DirectiveAction.ADJUST:
                # Apply ALM overrides directly to state
                if (
                    directive.alm_overrides
                    and "penalty_parameters" in directive.alm_overrides
                ):
                    new_penalties = jnp.array(
                        directive.alm_overrides["penalty_parameters"]
                    )
                    # Use model_copy for AugmentedLagrangianState as well
                    new_alm_state = traj.alm_state.model_copy(
                        update={"penalty_parameters": new_penalties}
                    )
                    traj = traj.model_copy(update={"alm_state": new_alm_state})
                    print(f"[Coordinator] ADJUST penalties: {directive.reasoning}")

            # Increase oracle budget for next iteration
            oracle_budget = min(
                alm.oracle_budget_max, oracle_budget + alm.oracle_budget_increment
            )

            # Auto-stop on excessive stagnation
            if traj.stagnation_count >= aso.max_stagnation_steps:
                traj = traj.model_copy(update={"status": "stagnated"})
                print("[Coordinator] Auto-STOP (stagnation limit)")
                candidates.append(
                    {
                        "params": state_to_boundary_params(alm_context, traj.alm_state),
                        "objective": result.objective,
                        "max_violation": result.max_violation,
                        "source": "aso_stagnation",
                        "seed": traj.seed.get("seed", 0),
                    }
                )
                break

        print(
            f"[Coordinator] Trajectory done: {traj.status}, {traj.steps} steps, "
            f"{traj.budget_used} evals, {len(candidates)} candidates"
        )

        return candidates

    def _generate_diagnostics(
        self,
        alm_state: AugmentedLagrangianState,
        traj: TrajectoryState,
    ) -> OptimizerDiagnostics:
        """Generate rich diagnostics from REAL ALM state.

        Args:
            alm_state: State containing:
                - constraints: Float[jnp.ndarray, "n_constraints"]
                - multipliers: Float[jnp.ndarray, "n_constraints"]
        """
        aso = self.cfg.aso
        assert traj.alm_context is not None, "ALM context required for diagnostics"
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

                if (
                    violation > prev_violation * (1 + aso.violation_increase_threshold)
                    and violation > 1e-4
                ):
                    trend = "increasing_violation"
                    diverging_count += 1
                elif violation < prev_violation * (
                    1 - aso.violation_decrease_threshold
                ):
                    trend = "decreasing_violation"

            constraint_diagnostics.append(
                ConstraintDiagnostic(
                    name=name,
                    violation=violation,
                    penalty=penalty,
                    multiplier=multiplier,
                    trend=trend,
                    delta=delta,
                )
            )

        # Status determination
        narrative = []
        if max_violation < aso.feasibility_threshold:
            status = "FEASIBLE_FOUND"
            narrative.append(
                f"Feasible region reached (max_violation={max_violation:.4f})"
            )
        elif diverging_count >= len(self.constraint_names) // 2:
            status = "DIVERGING"
            narrative.append(f"{diverging_count} constraints diverging")
        elif prev and abs(objective_delta) < aso.stagnation_objective_threshold:
            if max_violation > aso.stagnation_violation_threshold:
                status = "STAGNATION"
                narrative.append(
                    f"Stagnation: obj_delta={objective_delta:.6f}, violation={max_violation:.4f}"
                )
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

    def _update_trajectory_best(
        self, traj: TrajectoryState, diag: OptimizerDiagnostics
    ) -> TrajectoryState:
        """Update best values and stagnation counter."""
        updates = {}
        improved = False

        current_best_violation = traj.best_violation
        current_best_objective = traj.best_objective

        if diag.max_violation < current_best_violation:
            current_best_violation = diag.max_violation
            updates["best_violation"] = current_best_violation
            improved = True
        if (
            diag.objective < current_best_objective
            and diag.max_violation <= current_best_violation
        ):
            current_best_objective = diag.objective
            updates["best_objective"] = current_best_objective
            improved = True

        if improved:
            updates["stagnation_count"] = 0
        else:
            updates["stagnation_count"] = traj.stagnation_count + 1

        return traj.model_copy(update=updates)

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

        self.telemetry.append(
            {
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
                "surrogate_score": traj.seed.get("surrogate_score"),
                "surrogate_rank": traj.seed.get("surrogate_rank"),
                "surrogate_pool_size": traj.seed.get("surrogate_pool_size"),
            }
        )

    def _persist_telemetry(self, experiment_id: int):
        """Persist telemetry to JSONL file."""
        if not self.telemetry:
            return

        import json
        from pathlib import Path

        telemetry_dir = Path(self.cfg.reporting_dir) / "telemetry"
        telemetry_dir.mkdir(parents=True, exist_ok=True)
        telemetry_file = telemetry_dir / f"aso_exp{experiment_id}.jsonl"

        with open(telemetry_file, "a") as f:
            for event in self.telemetry:
                f.write(json.dumps(event) + "\n")

        self.telemetry = []

    def _surrogate_rank_seeds(
        self, seeds: List[Dict[str, Any]], cycle: int
    ) -> List[Dict[str, Any]]:
        """Rank seeds using the surrogate model."""
        if not seeds:
            return []

        if not self.surrogate or not self.surrogate._trained:
            print("[Coordinator] Surrogate not ready, using random selection")
            # Shuffle to avoid bias if generator is deterministic
            import random

            random.shuffle(seeds)
            return seeds

        print(
            f"[Coordinator] Ranking {len(seeds)} seeds with Neural Operator Surrogate..."
        )
        try:
            problem = (self.cfg.problem or "p3").lower()
            # Surrogate targets in this coordinator:
            # - P1: objective (max_elongation) → minimize
            # - P2/P3: score/HV proxies → maximize
            minimize_objective = problem.startswith("p1")
            # Rank candidates (higher score is better)
            ranked_preds = self.surrogate.rank_candidates(
                seeds,
                minimize_objective=minimize_objective,
                exploration_ratio=self.cfg.proposal_mix.exploration_ratio,
            )

            # Convert back to seed dicts with metadata
            sorted_seeds = []
            for i, pred in enumerate(ranked_preds):
                # Cast to Dict to allow mutation (pyright sees Mapping)
                # Cast to Dict to allow mutation (pyright sees Mapping)
                seed_data = dict(pred.metadata)  # type: ignore

                # Inject surrogate stats into seed for telemetry
                seed_data["surrogate_score"] = pred.expected_value
                seed_data["surrogate_rank"] = i
                seed_data["surrogate_pool_size"] = len(seeds)
                seed_data["surrogate_uncertainty"] = getattr(
                    pred, "uncertainty", 0.0
                )  # If available
                sorted_seeds.append(seed_data)

            return sorted_seeds

        except Exception as e:
            print(f"[Coordinator] Surrogate ranking failed: {e}")
            return seeds

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
            is_stellarator_symmetric=bool(
                params_map.get("is_stellarator_symmetric", True)
            ),
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
        from constellaration.forward_model import ConstellarationSettings
        from constellaration.optimization.augmented_lagrangian import (
            AugmentedLagrangianSettings,
        )
        from constellaration.optimization.settings import (
            AugmentedLagrangianMethodSettings,
            NevergradSettings,
            OptimizationSettings,
        )

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
            ),
        )
