"""Coordinator agent for Phase 5 (Hierarchical Autonomy).

The Coordinator manages the high-level strategy of the scientific process,
switching between Exploration (gathering new data/seeds) and Exploitation (optimizing candidates).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import jax.numpy as jnp
import numpy as np
import pydantic
from constellaration.optimization.augmented_lagrangian import AugmentedLagrangianState

from ai_scientist import config as ai_config
from ai_scientist import memory
from ai_scientist.forward_model import make_boundary_from_params
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
    DirectiveSource,
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

    NOTE: Uses the centralized constraint registry from ai_scientist.constraints.
    """
    from ai_scientist.constraints import get_constraint_names

    return get_constraint_names(problem, for_alm=True)


_VMEC_FAILURE_OBJECTIVE_SENTINEL = 9.0
_VMEC_FAILURE_VIOLATION_SENTINEL = 5.0
_VMEC_FAILURE_RESTART_STREAK = 2
_VMEC_FAILURE_ABORT_STREAK = 4


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

        else:
            candidates = self._run_standard_pipeline(
                cycle=cycle,
                n_candidates=n_candidates,
            )

        # Check for periodic retraining
        if candidates:
            self._periodic_retrain(cycle, experiment_id, candidates)

        return candidates

    def _run_standard_pipeline(
        self,
        *,
        cycle: int,
        n_candidates: int,
    ) -> List[Dict[str, Any]]:
        """Run the shared Quad-Hybrid stages used by EXPLOIT and HYBRID modes."""
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
        if self.cfg.proposal_mix.rl_refinement_enabled:
            rl_ctx = {
                "candidates": top_k,
                "target_metrics": explore_ctx.get("target_metrics"),
            }
            refined = self.rl_worker.run(rl_ctx).get("candidates", [])
            print(f"[Coordinator] RL Agent refined {len(refined)} candidates")
        else:
            refined = top_k
            print("[Coordinator] RL refinement disabled; skipping PPO-CMA stage")

        # STAGE 6: Optimize - Final gradient descent
        opt_ctx = {"initial_guesses": refined}
        res = self.opt_worker.run(opt_ctx)
        candidates = res.get("candidates", [])
        print(
            f"[Coordinator] Quad-Hybrid pipeline complete: {len(candidates)} final candidates"
        )
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

                    # Determine training target based on problem.
                    #
                    # IMPORTANT: We must keep `target_values` aligned with the
                    # `minimize_objective` flag used at ranking time:
                    # - If we train on a minimization objective (P1 max_elongation),
                    #   we must NOT negate it here, because `rank_candidates` will
                    #   handle the direction via `minimize_objective=True`.
                    # - For P2/P3 we train on a "higher is better" score proxy.
                    #
                    # See ai_scientist.objective_types for vocabulary.
                    from ai_scientist.objective_types import compute_ranking_score

                    if problem.startswith("p1"):
                        # P1: train on physics objective (max_elongation; lower is better)
                        target = eval_data.get("objective", cand.get("objective"))
                    else:
                        # P2/P3: use canonical ranking score (gradient / aspect)
                        target = eval_data.get("score", cand.get("score"))
                        if target is None:
                            # Recompute from metrics using canonical formula
                            target = compute_ranking_score(actual_metrics, problem)

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
        planner_intent: Mapping[str, Any] | None = None,
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
            planner_intent=planner_intent,
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
        planner_intent: Mapping[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        """Run trajectory with real ALM state and supervision."""
        aso = config.aso
        alm = config.alm
        candidates = []

        # Initialize ALM context and state
        boundary = self._seed_to_boundary(traj.seed)
        problem = self._get_problem(config)
        settings = self._build_optimization_settings(config)

        aspect_ratio_upper_bound = None
        if (config.problem or "").lower().startswith("p3"):
            aspect_ratio_upper_bound = config.alm.aspect_ratio_upper_bound

        alm_context, alm_state = create_alm_context(
            boundary=boundary,
            problem=problem,
            settings=settings,
            aspect_ratio_upper_bound=aspect_ratio_upper_bound,
        )
        traj = traj.model_copy(
            update={"alm_context": alm_context, "alm_state": alm_state}
        )

        oracle_budget = alm.oracle_budget_initial
        prev_diag: OptimizerDiagnostics | None = None
        consecutive_vmec_failures = 0

        while traj.budget_used < eval_budget and traj.status == "active":
            # Mathematical Invariant: ALM state is mandatory for the optimization loop.
            # While the TrajectoryState schema allows None (for initialization),
            # the ASO loop cannot physically proceed without a defined Lagrangian state.
            assert traj.alm_state is not None, "ASO loop requires initialized ALM state"
            current_alm_state = traj.alm_state

            traj = traj.model_copy(update={"steps": traj.steps + 1})
            step_start = time.perf_counter()

            # 1. Execute ALM step
            result = step_alm(
                context=alm_context,
                state=current_alm_state,
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
            vmec_step_failed = self._is_vmec_failure_step(result)
            if vmec_step_failed:
                consecutive_vmec_failures += 1
            else:
                consecutive_vmec_failures = 0

            # 3. Update trajectory tracking
            traj = self._update_trajectory_best(traj, diagnostics)

            # 4. Get directive (tiered supervision)
            llm_called = diagnostics.requires_llm_supervision(aso)
            directive = self.planner.supervise(
                diagnostics,
                cycle,
                aso,
                planner_intent=planner_intent,
            )
            intent_agreement, computed_override_reason = (
                self._assess_intent_agreement(
                    planner_intent=planner_intent,
                    diagnostics=diagnostics,
                    directive=directive,
                )
            )
            recovery_override_reason: str | None = None
            if (
                vmec_step_failed
                and consecutive_vmec_failures >= _VMEC_FAILURE_ABORT_STREAK
            ):
                recovery_override_reason = (
                    "forced STOP after sustained VMEC-failure sentinel streak"
                )
                directive = OptimizationDirective(
                    action=DirectiveAction.STOP,
                    reasoning=(
                        "Consecutive VMEC-failure sentinel steps reached abort threshold; "
                        "stopping trajectory."
                    ),
                    source=DirectiveSource.FALLBACK,
                )
            elif (
                vmec_step_failed
                and consecutive_vmec_failures >= _VMEC_FAILURE_RESTART_STREAK
                and directive.action != DirectiveAction.RESTART
            ):
                recovery_override_reason = (
                    "forced RESTART after repeated VMEC-failure sentinel steps"
                )
                directive = OptimizationDirective(
                    action=DirectiveAction.RESTART,
                    reasoning=(
                        "Consecutive VMEC-failure sentinel steps reached restart threshold; "
                        "restarting with a fresh seed."
                    ),
                    source=DirectiveSource.FALLBACK,
                )

            if recovery_override_reason is not None:
                intent_agreement = "overridden"
            override_reason = directive.override_reason or computed_override_reason
            if recovery_override_reason is not None:
                if override_reason:
                    override_reason = (
                        f"{override_reason} | {recovery_override_reason}"
                    )
                else:
                    override_reason = recovery_override_reason
            violation_delta = (
                diagnostics.max_violation - prev_diag.max_violation
                if prev_diag is not None
                else None
            )

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
                planner_intent=planner_intent,
                intent_agreement=intent_agreement,
                override_reason=override_reason,
                violation_delta=violation_delta,
                vmec_step_failed=vmec_step_failed,
                vmec_failure_streak=consecutive_vmec_failures,
            )
            self.world_model.log_scratchpad_event(
                experiment_id=experiment_id,
                cycle=cycle,
                step=traj.steps,
                planner_intent=planner_intent,
                aso_action=directive.action.value,
                intent_agreement=intent_agreement,
                override_reason=override_reason,
                diagnostics={
                    "status": diagnostics.status,
                    "objective": diagnostics.objective,
                    "max_violation": diagnostics.max_violation,
                    "bounds_norm": diagnostics.bounds_norm,
                    "penalty_parameters": diagnostics.penalty_parameters,
                    "constraint_diagnostics": [
                        entry.model_dump(mode="json")
                        for entry in diagnostics.constraint_diagnostics
                    ],
                },
                outcome={
                    "objective_delta": diagnostics.objective_delta,
                    "violation_delta": violation_delta,
                    "steps_since_improvement": diagnostics.steps_since_improvement,
                    "llm_called": llm_called,
                    "vmec_step_failed": vmec_step_failed,
                    "vmec_failure_streak": consecutive_vmec_failures,
                },
            )
            prev_diag = diagnostics

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
                assert traj.alm_state is not None
                self._append_validated_aso_candidate(
                    candidates=candidates,
                    params=state_to_boundary_params(alm_context, traj.alm_state),
                    objective=result.objective,
                    max_violation=result.max_violation,
                    source="aso",
                    seed=traj.seed.get("seed", 0),
                )
                break

            if directive.action == DirectiveAction.RESTART:
                # Try new seed
                new_seeds = self._prepare_seeds(None, cycle, 1)
                if new_seeds:
                    # Save current best before restart
                    assert traj.alm_state is not None
                    self._append_validated_aso_candidate(
                        candidates=candidates,
                        params=state_to_boundary_params(alm_context, traj.alm_state),
                        objective=result.objective,
                        max_violation=result.max_violation,
                        source="aso_pre_restart",
                        seed=traj.seed.get("seed", 0),
                    )

                    boundary = self._seed_to_boundary(new_seeds[0])
                    new_context, new_state = create_alm_context(
                        boundary=boundary,
                        problem=problem,
                        settings=settings,
                        aspect_ratio_upper_bound=aspect_ratio_upper_bound,
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
                assert traj.alm_state is not None
                self._append_validated_aso_candidate(
                    candidates=candidates,
                    params=state_to_boundary_params(alm_context, traj.alm_state),
                    objective=result.objective,
                    max_violation=result.max_violation,
                    source="aso_stagnation",
                    seed=traj.seed.get("seed", 0),
                )
                break

        if not candidates and traj.alm_state is not None:
            # Preserve the terminal ALM state as a candidate even when we never hit
            # STOP/RESTART branches. Without this, budget-exhausted runs emit zero
            # candidates despite spending evaluations.
            constraints = [float(c) for c in traj.alm_state.constraints]
            max_violation = max(0.0, max(constraints)) if constraints else 0.0
            self._append_validated_aso_candidate(
                candidates=candidates,
                params=state_to_boundary_params(alm_context, traj.alm_state),
                objective=float(traj.alm_state.objective),
                max_violation=max_violation,
                source="aso_terminal_state",
                seed=traj.seed.get("seed", 0),
            )

        print(
            f"[Coordinator] Trajectory done: {traj.status}, {traj.steps} steps, "
            f"{traj.budget_used} evals, {len(candidates)} candidates"
        )

        return candidates

    def _is_vmec_failure_step(self, result: Any) -> bool:
        """Detect ALM steps dominated by VMEC/runtime failures."""
        if getattr(result, "metrics", None) is not None:
            return False
        objective = float(getattr(result, "objective", 0.0))
        max_violation = float(getattr(result, "max_violation", 0.0))
        return (
            objective >= _VMEC_FAILURE_OBJECTIVE_SENTINEL
            or max_violation >= _VMEC_FAILURE_VIOLATION_SENTINEL
        )

    def _append_validated_aso_candidate(
        self,
        *,
        candidates: List[Dict[str, Any]],
        params: Mapping[str, Any],
        objective: float,
        max_violation: float,
        source: str,
        seed: int,
    ) -> None:
        try:
            make_boundary_from_params(params)
        except Exception as exc:
            print(
                f"[Coordinator] Dropping invalid ASO candidate (source={source}, seed={seed}): {exc}"
            )
            return
        candidates.append(
            {
                "params": dict(params),
                "objective": float(objective),
                "max_violation": float(max_violation),
                "source": source,
                "seed": int(seed),
            }
        )

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

    def _assess_intent_agreement(
        self,
        *,
        planner_intent: Mapping[str, Any] | None,
        diagnostics: OptimizerDiagnostics,
        directive: OptimizationDirective,
    ) -> tuple[str, str | None]:
        if not planner_intent:
            return "unavailable", None

        constraint_priority = planner_intent.get("primary_constraint_order")
        focus_indices = planner_intent.get("penalty_focus_indices")
        restart_policy = planner_intent.get("restart_policy")

        if directive.action == DirectiveAction.ADJUST:
            if (
                isinstance(focus_indices, list)
                and focus_indices
                and directive.alm_overrides
                and isinstance(directive.alm_overrides.get("penalty_parameters"), list)
            ):
                focus_set = {int(idx) for idx in focus_indices}
                penalties = diagnostics.penalty_parameters
                adjusted_indices = []
                for idx, value in enumerate(directive.alm_overrides["penalty_parameters"]):
                    if idx >= len(penalties):
                        continue
                    if float(value) != float(penalties[idx]):
                        adjusted_indices.append(idx)
                if any(idx in focus_set for idx in adjusted_indices):
                    return "aligned", None
                return (
                    "overridden",
                    "directive penalty edits did not follow planner penalty_focus_indices.",
                )
            return "unknown", None

        if directive.action == DirectiveAction.RESTART and restart_policy is not None:
            restart_policy_text = str(restart_policy).strip().lower()
            if restart_policy_text in {"on_stagnation", "aggressive"}:
                return "aligned", None
            return (
                "overridden",
                f"directive requested restart against planner restart_policy={restart_policy}.",
            )

        if (
            isinstance(constraint_priority, list)
            and constraint_priority
            and diagnostics.constraint_diagnostics
        ):
            top_constraint = max(
                diagnostics.constraint_diagnostics,
                key=lambda item: item.violation,
            ).name
            if str(constraint_priority[0]) == str(top_constraint):
                return "aligned", None

        return "unknown", None

    def _log_telemetry(
        self,
        experiment_id: int,
        cycle: int,
        traj: TrajectoryState,
        diag: OptimizerDiagnostics,
        directive: OptimizationDirective,
        wall_time_ms: float,
        llm_called: bool,
        planner_intent: Mapping[str, Any] | None,
        intent_agreement: str,
        override_reason: str | None,
        violation_delta: float | None,
        vmec_step_failed: bool,
        vmec_failure_streak: int,
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
                "planner_intent": planner_intent,
                "aso_action": directive.action.value,
                "intent_agreement": intent_agreement,
                "override_reason": override_reason,
                "objective_delta": diag.objective_delta,
                "violation_delta": violation_delta,
                "vmec_step_failed": vmec_step_failed,
                "vmec_failure_streak": vmec_failure_streak,
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
        """Rank seeds using the surrogate model.

        IMPORTANT: The surrogate predicts a *ranking target* whose direction
        depends on the problem:
        - P1: physics objective `max_elongation` (lower is better → minimize)
        - P2/P3: score proxy `gradient / aspect` (higher is better → maximize)

        This must stay consistent with `_periodic_retrain()` and the
        `minimize_objective` flag passed into `surrogate.rank_candidates()`.
        """
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
                problem=problem,
            )

            # Convert back to seed dicts with metadata
            sorted_seeds = []
            for i, pred in enumerate(ranked_preds):
                # Handle Optional metadata - create empty dict if None
                seed_data: Dict[str, Any] = (
                    dict(pred.metadata) if pred.metadata is not None else {}
                )

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
            normalized_initial = self._normalize_seed_batch(
                initial_seeds,
                cycle,
                limit=n_needed,
            )
            if normalized_initial:
                return normalized_initial
            fallback = self._stable_fallback_seed(cycle)
            if fallback is not None:
                return [fallback]
            return []

        # Generate using Explore worker
        # Generate a batch to ensure we have enough valid seeds
        explore_ctx = {"n_samples": max(n_needed, 10), "cycle": cycle}
        res = self.explore_worker.run(explore_ctx)
        candidates = res.get("candidates", [])
        raw_candidates = list(candidates)

        # Filter with Geometer
        if candidates:
            geo_ctx = {"candidates": candidates}
            geo_res = self.geo_worker.run(geo_ctx)
            candidates = geo_res.get("candidates", [])

        if not candidates:
            fallback = self._stable_fallback_seed(cycle)
            if fallback is not None:
                print(
                    "[Coordinator] Geometer rejected all seeds; using stable fallback seed."
                )
                return [fallback]
            if raw_candidates:
                print(
                    "[Coordinator] Geometer rejected all seeds; using unfiltered seeds as fallback."
                )
                return self._normalize_seed_batch(
                    raw_candidates,
                    cycle,
                    limit=n_needed,
                )
            return []

        return self._normalize_seed_batch(
            candidates,
            cycle,
            limit=n_needed,
        )

    def _normalize_seed_batch(
        self,
        seeds: list[Any],
        cycle: int,
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        normalized = [
            seed_entry
            for seed_entry in (
                self._normalize_seed_entry(seed, cycle) for seed in seeds
            )
            if seed_entry is not None
        ]
        if limit is None:
            return normalized
        return normalized[:limit]

    def _default_seed_for_cycle(self, cycle: int) -> int:
        return int(self.cfg.random_seed + cycle)

    def _normalize_seed_entry(self, seed: Any, cycle: int) -> dict[str, Any] | None:
        if not isinstance(seed, Mapping):
            return None

        default_seed = self._default_seed_for_cycle(cycle)
        if "params" in seed and isinstance(seed["params"], Mapping):
            normalized = dict(seed)
            normalized["params"] = dict(seed["params"])
            normalized["seed"] = int(seed.get("seed", default_seed))
            return normalized

        if "r_cos" not in seed or "z_sin" not in seed:
            return None
        return {
            "seed": int(seed.get("seed", default_seed)),
            "params": dict(seed),
            "source": str(seed.get("source", "seed_normalized")),
        }

    def _stable_fallback_seed(self, cycle: int) -> dict[str, Any] | None:
        problem_key = (self.cfg.problem or "p3").lower()
        if problem_key.startswith("p1"):
            seed_path = Path("configs/seeds/p1_seeds.json")
        elif problem_key.startswith("p2"):
            seed_path = Path("configs/seeds/p2_seeds.json")
        else:
            seed_path = Path("configs/seeds/rotating_ellipse_p3.json")
            if not seed_path.exists():
                seed_path = Path("configs/seeds/p3_seeds.json")
        if not seed_path.exists():
            return None
        payload = json.loads(seed_path.read_text(encoding="utf-8"))
        params: dict[str, Any] | None = None
        if isinstance(payload, Mapping):
            if "r_cos" in payload and "z_sin" in payload:
                params = dict(payload)
        elif isinstance(payload, list) and payload:
            first = payload[0]
            if isinstance(first, Mapping):
                if "json" in first and isinstance(first["json"], str):
                    parsed = json.loads(first["json"])
                    if isinstance(parsed, Mapping):
                        params = dict(parsed)
                elif "r_cos" in first and "z_sin" in first:
                    params = dict(first)
        if params is None:
            return None
        return {
            "seed": self._default_seed_for_cycle(cycle),
            "params": params,
            "source": "stable_seed_fallback",
        }

    def _seed_to_boundary(self, seed):
        """Convert seed dict to SurfaceRZFourier."""
        from constellaration.geometry import surface_rz_fourier

        # Handle both "seed" dict and direct params dict
        params_map = seed.get("params", seed)
        r_sin = params_map.get("r_sin")
        z_cos = params_map.get("z_cos")

        return surface_rz_fourier.SurfaceRZFourier(
            r_cos=np.array(params_map["r_cos"]),
            z_sin=np.array(params_map["z_sin"]),
            n_field_periods=int(params_map.get("n_field_periods", 1)),
            is_stellarator_symmetric=bool(
                params_map.get("is_stellarator_symmetric", True)
            ),
            r_sin=np.array(r_sin) if r_sin is not None else None,
            z_cos=np.array(z_cos) if z_cos is not None else None,
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
        from constellaration.mhd import vmec_settings as vmec_settings_module

        problem_key = (config.problem or "p3").lower()
        if problem_key.startswith("p3"):
            vmec_fidelity = "very_low_fidelity"
        else:
            vmec_fidelity = "low_fidelity"

        fm_settings = ConstellarationSettings(
            vmec_preset_settings=vmec_settings_module.VmecPresetSettings(
                fidelity=vmec_fidelity,
            )
        )
        if problem_key.startswith("p1"):
            fm_settings = fm_settings.model_copy(
                update={
                    "boozer_preset_settings": None,
                    "qi_settings": None,
                    "turbulent_settings": None,
                }
            )

        return OptimizationSettings(
            max_poloidal_mode=config.boundary_template.max_poloidal_mode,
            max_toroidal_mode=config.boundary_template.max_toroidal_mode,
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
