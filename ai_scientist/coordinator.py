"""Coordinator agent for Phase 5 (Hierarchical Autonomy).

The Coordinator manages the high-level strategy of the scientific process,
switching between Exploration (gathering new data/seeds) and Exploitation (optimizing candidates).
"""

from __future__ import annotations

import json
import hashlib
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

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
from ai_scientist.model_router_reward import select_ucb_arm
from ai_scientist.problems import get_problem
from ai_scientist.staged_governor import (
    StagedSeedPlan,
    build_delta_replay_seeds,
    build_staged_seed_plan_from_snapshots,
    expand_parent_group_staged_offspring,
    worst_constraint_from_violations,
)
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
        self._last_trajectory_budget_used: int = 0

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

    def _as_finite_float(self, value: Any) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        value_f = float(value)
        if not np.isfinite(value_f):
            return None
        return value_f

    def _seed_bank_hash(self, seeds: Sequence[Mapping[str, Any]]) -> str:
        canonical: list[str] = []
        for seed in seeds:
            params = seed.get("params")
            if isinstance(params, Mapping):
                canonical.append(self._seed_identity(params))
        payload = "|".join(sorted(canonical))
        if not payload:
            return ""
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _derive_focus_constraint_context(
        self,
        snapshots: Sequence[Mapping[str, Any]],
    ) -> tuple[str | None, float | None]:
        if not snapshots:
            return None, None
        finite = [
            entry
            for entry in snapshots
            if self._as_finite_float(entry.get("feasibility")) is not None
        ]
        if not finite:
            return None, None
        focus = min(
            finite,
            key=lambda entry: float(entry.get("feasibility", float("inf"))),
        )
        margins = focus.get("constraint_margins")
        if not isinstance(margins, Mapping):
            return None, None
        worst_name: str | None = None
        worst_margin = 0.0
        for key, raw in margins.items():
            margin = self._as_finite_float(raw)
            if margin is None or margin <= 0.0:
                continue
            if margin > worst_margin:
                worst_margin = margin
                worst_name = str(key)
        if worst_name is None:
            return None, None
        return worst_name, float(worst_margin)

    def _select_runtime_ucb_arm(
        self,
        *,
        experiment_id: int,
        problem: str,
        min_eligible_events: int,
    ) -> dict[str, Any] | None:
        if not hasattr(self.world_model, "model_router_reward_eligible_history"):
            return None
        history = self.world_model.model_router_reward_eligible_history(
            experiment_id=experiment_id,
            problem=problem,
            limit=200,
        )
        if not isinstance(history, list) or len(history) < min_eligible_events:
            return None

        arm_stats: dict[str, dict[str, float | int]] = {}
        arm_payload: dict[str, dict[str, str]] = {}
        for row in history:
            if not isinstance(row, Mapping):
                continue
            model_route_raw = row.get("model_route")
            model_route = (
                str(model_route_raw).strip() if model_route_raw is not None else ""
            )
            if not model_route:
                model_route = "unknown"
            reward_raw = row.get("reward")
            reward = self._as_finite_float(reward_raw)
            if reward is None:
                continue
            components = row.get("reward_components")
            operator_family = "unknown"
            if isinstance(components, Mapping):
                op_raw = components.get("operator_family")
                if isinstance(op_raw, str) and op_raw.strip():
                    operator_family = op_raw.strip()
            arm_key = f"{operator_family}|{model_route}"
            stats = arm_stats.get(arm_key)
            if stats is None:
                arm_stats[arm_key] = {"pulls": 1, "mean_reward": reward}
                arm_payload[arm_key] = {
                    "operator_family": operator_family,
                    "model_route": model_route,
                }
                continue
            pulls_raw = stats.get("pulls", 0)
            pulls = int(pulls_raw) if isinstance(pulls_raw, (int, float)) else 0
            mean_raw = stats.get("mean_reward", 0.0)
            mean = float(mean_raw) if isinstance(mean_raw, (int, float)) else 0.0
            next_pulls = pulls + 1
            next_mean = ((mean * pulls) + reward) / max(1, next_pulls)
            stats["pulls"] = next_pulls
            stats["mean_reward"] = next_mean

        if not arm_stats:
            return None
        arm_key = select_ucb_arm(arm_stats, exploration_c=1.0)
        if arm_key is None:
            return None
        selected = arm_payload.get(arm_key)
        if selected is None:
            return None
        payload = dict(selected)
        payload["arm_key"] = arm_key
        return payload

    def _stamp_runtime_ucb_arm(
        self,
        *,
        seeds: list[dict[str, Any]],
        arm: Mapping[str, Any] | None,
    ) -> None:
        if not arm:
            return
        selected_route = arm.get("model_route")
        selected_operator = arm.get("operator_family")
        route_text = (
            str(selected_route)
            if isinstance(selected_route, str) and selected_route
            else "unknown"
        )
        operator_text = (
            str(selected_operator)
            if isinstance(selected_operator, str) and selected_operator
            else "unknown"
        )
        for seed in seeds:
            seed.setdefault("model_route", route_text)
            seed.setdefault("operator_family", operator_text)
            seed["runtime_ucb_arm"] = dict(arm)

    def produce_candidates_aso(
        self,
        cycle: int,
        experiment_id: int,
        eval_budget: int,
        template: ai_config.BoundaryTemplateConfig,
        initial_seeds: Optional[List[Dict[str, Any]]] = None,
        initial_config: Optional[ai_config.ExperimentConfig] = None,
        planner_intent: Mapping[str, Any] | None = None,
        planner_intents: Sequence[Mapping[str, Any] | None] | None = None,
    ) -> List[Dict[str, Any]]:
        """
        ASO loop with real ALM state supervision.
        """
        config = initial_config or self.cfg
        allow_seed_fallbacks = config.aso.seed_fallback_policy != "forbid"
        tree_enabled = bool(config.aso.tree_evolution_enabled)
        multiplier = config.proposal_mix.surrogate_pool_multiplier
        pool_size = int(max(10, 1 * multiplier))
        queue_limit = max(1, int(config.aso.tree_queue_max))
        branch_budget = max(1, int(config.aso.tree_branch_budget))

        fairness_lockstep = bool(config.aso.fair_ab_lockstep_seed_bank)
        user_initial_seeds = list(initial_seeds or [])
        if planner_intents is not None and user_initial_seeds:
            enriched_seeds: list[dict[str, Any]] = []
            for idx, seed in enumerate(user_initial_seeds):
                seed_payload = dict(seed)
                intent_payload = (
                    planner_intents[idx] if idx < len(planner_intents) else None
                )
                if isinstance(intent_payload, Mapping):
                    seed_payload["planner_intent"] = dict(intent_payload)
                enriched_seeds.append(seed_payload)
            user_initial_seeds = enriched_seeds

        snapshots: list[Mapping[str, Any]] = []
        if hasattr(self.world_model, "recent_candidate_snapshots"):
            snapshot_rows = self.world_model.recent_candidate_snapshots(
                experiment_id=experiment_id,
                problem=config.problem,
                limit=config.aso.staged_recent_limit,
            )
            if isinstance(snapshot_rows, list):
                snapshots = snapshot_rows
        focus_worst_constraint, focus_constraint_margin = (
            self._derive_focus_constraint_context(snapshots)
        )

        fairness_seed_bank_hash = self._seed_bank_hash(user_initial_seeds)
        for seed in user_initial_seeds:
            if fairness_seed_bank_hash:
                seed["fairness_seed_bank_hash"] = fairness_seed_bank_hash

        base_seed_sources: list[dict[str, Any]] = []
        deferred_planner_seeds: list[dict[str, Any]] = []
        for seed in user_initial_seeds:
            source_raw = seed.get("source")
            source_name = str(source_raw) if isinstance(source_raw, str) else ""
            if fairness_lockstep and source_name == "planner_extra":
                deferred_planner_seeds.append(dict(seed))
            else:
                base_seed_sources.append(dict(seed))

        parent_group: list[Mapping[str, Any]] = []
        if hasattr(self.world_model, "select_parent_group_performance_novelty"):
            selected = self.world_model.select_parent_group_performance_novelty(
                experiment_id=experiment_id,
                problem=config.problem,
                group_size=max(1, int(config.aso.parent_group_size)),
                limit=config.aso.staged_recent_limit,
                near_feasibility_threshold=config.aso.staged_near_feasibility_threshold,
                worst_constraint=focus_worst_constraint,
                focus_constraint_margin=focus_constraint_margin,
                leverage_weight=config.aso.parent_selection_leverage_weight,
            )
            if isinstance(selected, list):
                parent_group = selected

        parent_by_hash: dict[str, Mapping[str, Any]] = {}
        for parent in parent_group:
            params_payload = parent.get("params")
            if not isinstance(params_payload, Mapping):
                continue
            design_hash = str(parent.get("design_hash", ""))
            if design_hash:
                parent_by_hash[design_hash] = parent
            base_seed_sources.append(
                {
                    "seed": int(
                        parent.get("seed", self._default_seed_for_cycle(cycle))
                    ),
                    "params": dict(params_payload),
                    "source": "parent_group",
                    "lineage_parent_hashes": [design_hash],
                    "operator_family": "parent_group",
                    "novelty_score": parent.get("novelty_score"),
                    "parent_feasibility": parent.get("feasibility"),
                    "parent_objective": parent.get("objective"),
                    "improvement_reason": "performance_novelty_leverage_selection",
                    "fairness_seed_bank_hash": fairness_seed_bank_hash,
                }
            )

        staged_plan: StagedSeedPlan | None = None
        staged_seeds: list[dict[str, Any]] = []
        if config.aso.staged_governor_enabled:
            if parent_group:
                focus_parent = parent_group[0]
                raw_violations = focus_parent.get("constraint_margins")
                violations = (
                    dict(raw_violations) if isinstance(raw_violations, Mapping) else {}
                )
                worst_constraint, _ = worst_constraint_from_violations(violations)
                staged_seeds = expand_parent_group_staged_offspring(
                    parent_group=parent_group,
                    worst_constraint=worst_constraint,
                    max_repair_candidates=config.aso.staged_repair_candidates,
                    bridge_blend_t=config.aso.staged_bridge_blend_t,
                    offspring_per_parent=config.aso.offspring_per_parent,
                )
            else:
                if isinstance(snapshots, list) and snapshots:
                    staged_plan = build_staged_seed_plan_from_snapshots(
                        snapshots=snapshots,
                        problem=config.problem,
                        near_feasibility_threshold=config.aso.staged_near_feasibility_threshold,
                        max_repair_candidates=config.aso.staged_repair_candidates,
                        bridge_blend_t=config.aso.staged_bridge_blend_t,
                    )
                if staged_plan is not None and staged_plan.seeds:
                    staged_seeds = staged_plan.seeds

        replay_seeds: list[dict[str, Any]] = []
        if (
            parent_group
            and hasattr(self.world_model, "nearest_case_deltas")
            and int(config.aso.delta_replay_top_k) > 0
        ):
            focus_parent = parent_group[0]
            focus_params_raw = focus_parent.get("params")
            if isinstance(focus_params_raw, Mapping):
                nearest = self.world_model.nearest_case_deltas(
                    experiment_id=experiment_id,
                    problem=config.problem,
                    seed_params=focus_params_raw,
                    limit=max(2, int(config.aso.delta_replay_top_k) * 2),
                    near_feasibility_threshold=config.aso.staged_near_feasibility_threshold,
                    include_recipes=True,
                )
                if isinstance(nearest, list):
                    focus_hash = str(focus_parent.get("design_hash", ""))
                    replay_seeds = build_delta_replay_seeds(
                        focus_params=focus_params_raw,
                        case_deltas=nearest,
                        top_k=int(config.aso.delta_replay_top_k),
                        focus_hash=focus_hash,
                        worst_constraint=focus_worst_constraint,
                    )

        seed_sources: list[dict[str, Any]] = list(base_seed_sources)
        if replay_seeds:
            seed_sources = replay_seeds + seed_sources
        if staged_seeds:
            seed_sources = staged_seeds + seed_sources

        for staged_seed in seed_sources:
            parent_hashes = staged_seed.get("lineage_parent_hashes")
            if not isinstance(parent_hashes, list) or not parent_hashes:
                continue
            focus_hash = str(parent_hashes[0])
            parent = parent_by_hash.get(focus_hash)
            if parent is None:
                continue
            if "parent_feasibility" not in staged_seed:
                staged_seed["parent_feasibility"] = parent.get("feasibility")
            if "parent_objective" not in staged_seed:
                staged_seed["parent_objective"] = parent.get("objective")
            if fairness_seed_bank_hash and "fairness_seed_bank_hash" not in staged_seed:
                staged_seed["fairness_seed_bank_hash"] = fairness_seed_bank_hash

        if staged_seeds:
            print(
                "[Coordinator] Staged governor prepared seeds "
                f"(count={len(staged_seeds)})."
            )
        elif staged_plan is not None and staged_plan.seeds:
            print(
                "[Coordinator] Staged governor prepared seeds "
                f"(focus={staged_plan.focus_hash[:12]}, "
                f"worst={staged_plan.worst_constraint}, "
                f"count={len(staged_plan.seeds)})."
            )
        if replay_seeds:
            print(
                f"[Coordinator] Applied case-delta replay seeds (count={len(replay_seeds)})."
            )

        runtime_ucb_arm: Mapping[str, Any] | None = None
        if config.aso.runtime_ucb_enabled:
            runtime_ucb_arm = self._select_runtime_ucb_arm(
                experiment_id=experiment_id,
                problem=config.problem,
                min_eligible_events=max(
                    1, int(config.aso.runtime_ucb_min_eligible_events)
                ),
            )
            if runtime_ucb_arm is not None:
                print(
                    "[Coordinator] Runtime UCB selected arm "
                    f"{runtime_ucb_arm.get('arm_key', 'unknown')}."
                )
                self._stamp_runtime_ucb_arm(
                    seeds=seed_sources,
                    arm=runtime_ucb_arm,
                )
                self._stamp_runtime_ucb_arm(
                    seeds=deferred_planner_seeds,
                    arm=runtime_ucb_arm,
                )

        ranked_seeds = self._prepare_gated_ranked_seeds(
            seeds=seed_sources or None,
            cycle=cycle,
            n_needed=max(pool_size, queue_limit),
            allow_fallbacks=allow_seed_fallbacks,
            experiment_id=experiment_id,
            problem=config.problem,
            gate_limit=queue_limit,
        )
        seed_queue: list[dict[str, Any]] = [
            dict(seed) for seed in ranked_seeds[:queue_limit]
        ]

        if not seed_queue and deferred_planner_seeds:
            ranked_deferred = self._prepare_gated_ranked_seeds(
                seeds=deferred_planner_seeds,
                cycle=cycle,
                n_needed=max(pool_size, queue_limit),
                allow_fallbacks=False,
                experiment_id=experiment_id,
                problem=config.problem,
                gate_limit=queue_limit,
            )
            seed_queue = [dict(seed) for seed in ranked_deferred[:queue_limit]]
            deferred_planner_seeds = []

        if not seed_queue:
            if not allow_seed_fallbacks:
                raise RuntimeError(
                    "ASO seed queue is empty after novelty/validity gate and aso.seed_fallback_policy='forbid' disables fallback seeding."
                )
            print("[Coordinator] No valid ASO seeds after gating; returning empty.")
            return []

        print(
            f"[Coordinator] ASO seed queue initialized with {len(seed_queue)} candidate(s)."
        )

        remaining_budget = int(eval_budget)
        trajectory_id = 0
        candidates: list[dict[str, Any]] = []
        while remaining_budget > 0 and seed_queue:
            seed_payload = seed_queue.pop(0)
            seed_specific_intent = seed_payload.get("planner_intent")
            effective_planner_intent = (
                dict(seed_specific_intent)
                if isinstance(seed_specific_intent, Mapping)
                else planner_intent
            )
            branch_eval_budget = (
                min(remaining_budget, branch_budget)
                if tree_enabled
                else remaining_budget
            )
            traj = TrajectoryState(id=trajectory_id, seed=seed_payload)
            branch_candidates = self._run_trajectory_aso(
                traj=traj,
                eval_budget=branch_eval_budget,
                cycle=cycle,
                experiment_id=experiment_id,
                config=config,
                planner_intent=effective_planner_intent,
                restart_seed_queue=seed_queue,
            )
            candidates.extend(branch_candidates)
            consumed = self._last_trajectory_budget_used
            if consumed <= 0:
                consumed = branch_eval_budget
            remaining_budget = max(0, remaining_budget - consumed)
            trajectory_id += 1

            if deferred_planner_seeds and trajectory_id >= 1:
                ranked_deferred = self._prepare_gated_ranked_seeds(
                    seeds=deferred_planner_seeds,
                    cycle=cycle,
                    n_needed=len(deferred_planner_seeds),
                    allow_fallbacks=False,
                    experiment_id=experiment_id,
                    problem=config.problem,
                    gate_limit=queue_limit,
                )
                self._enqueue_seed_queue(
                    queue=seed_queue,
                    new_items=ranked_deferred,
                    max_size=queue_limit,
                    tie_epsilon=config.aso.frontier_objective_tie_epsilon,
                )
                deferred_planner_seeds = []

            if not tree_enabled:
                break

            expanded = self._expand_tree_seeds_from_candidates(
                candidates=branch_candidates,
                parent_group=parent_group,
                config=config,
                experiment_id=experiment_id,
            )
            if expanded:
                ranked_expanded = self._prepare_gated_ranked_seeds(
                    seeds=expanded,
                    cycle=cycle,
                    n_needed=len(expanded),
                    allow_fallbacks=False,
                    experiment_id=experiment_id,
                    problem=config.problem,
                    gate_limit=queue_limit,
                )
                self._enqueue_seed_queue(
                    queue=seed_queue,
                    new_items=ranked_expanded,
                    max_size=queue_limit,
                    tie_epsilon=config.aso.frontier_objective_tie_epsilon,
                )

        if (
            remaining_budget > 0
            and not seed_queue
            and not allow_seed_fallbacks
            and not candidates
        ):
            raise RuntimeError(
                "ASO queue depleted before budget exhaustion and aso.seed_fallback_policy='forbid' prevents fresh fallback seeds."
            )

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
        restart_seed_queue: list[dict[str, Any]] | None = None,
    ) -> List[Dict[str, Any]]:
        """Run trajectory with real ALM state and supervision."""
        aso = config.aso
        alm = config.alm
        allow_seed_fallbacks = aso.seed_fallback_policy != "forbid"
        candidates = []
        self._last_trajectory_budget_used = 0

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
            intent_agreement, computed_override_reason = self._assess_intent_agreement(
                planner_intent=planner_intent,
                diagnostics=diagnostics,
                directive=directive,
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
                    override_reason = f"{override_reason} | {recovery_override_reason}"
                else:
                    override_reason = recovery_override_reason
            violation_delta = (
                diagnostics.max_violation - prev_diag.max_violation
                if prev_diag is not None
                else None
            )
            feasibility_delta = violation_delta
            worst_before_name, worst_before_value = self._worst_constraint_snapshot(
                prev_diag
            )
            worst_after_name, worst_after_value = self._worst_constraint_snapshot(
                diagnostics
            )
            chosen_operator = self._chosen_operator_label(
                seed_payload=traj.seed,
                directive=directive,
            )
            parent_hashes = self._parent_hashes_from_seed(traj.seed)
            bridge_flag = self._bridge_flag_from_seed(traj.seed)
            hv_delta_raw = self.world_model.average_recent_hv_delta(
                experiment_id, lookback=1
            )
            try:
                hv_delta = float(hv_delta_raw) if hv_delta_raw is not None else None
            except (TypeError, ValueError):
                hv_delta = None

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
                feasibility_delta=feasibility_delta,
                hv_delta=hv_delta,
                worst_constraint_before=worst_before_name,
                worst_constraint_before_value=worst_before_value,
                worst_constraint_after=worst_after_name,
                worst_constraint_after_value=worst_after_value,
                chosen_operator=chosen_operator,
                parent_hashes=parent_hashes,
                bridge_flag=bridge_flag,
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
                    "worst_constraint_before": worst_before_name,
                    "worst_constraint_before_value": worst_before_value,
                    "worst_constraint_after": worst_after_name,
                    "worst_constraint_after_value": worst_after_value,
                    "chosen_operator": chosen_operator,
                    "parent_hashes": parent_hashes,
                    "bridge_flag": bridge_flag,
                    "constraint_diagnostics": [
                        entry.model_dump(mode="json")
                        for entry in diagnostics.constraint_diagnostics
                    ],
                },
                outcome={
                    "objective_delta": diagnostics.objective_delta,
                    "feasibility_delta": feasibility_delta,
                    "hv_delta": hv_delta,
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
                    seed_payload=traj.seed,
                )
                break

            if directive.action == DirectiveAction.RESTART:
                # Strict restart contract: consume queued seeds first.
                new_seeds: list[dict[str, Any]] = []
                if restart_seed_queue:
                    next_seed = restart_seed_queue.pop(0)
                    if isinstance(next_seed, Mapping):
                        new_seeds = [dict(next_seed)]
                elif allow_seed_fallbacks:
                    new_seeds = self._prepare_seeds(
                        None,
                        cycle,
                        1,
                        allow_fallbacks=True,
                    )
                else:
                    raise RuntimeError(
                        "ASO restart requested but queue is empty and aso.seed_fallback_policy='forbid' prevents fresh seed generation."
                    )
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
                        seed_payload=traj.seed,
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
                    seed_payload=traj.seed,
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
                seed_payload=traj.seed,
            )

        print(
            f"[Coordinator] Trajectory done: {traj.status}, {traj.steps} steps, "
            f"{traj.budget_used} evals, {len(candidates)} candidates"
        )

        self._last_trajectory_budget_used = int(traj.budget_used)
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
        seed_payload: Mapping[str, Any] | None = None,
    ) -> None:
        try:
            make_boundary_from_params(params)
        except Exception as exc:
            print(
                f"[Coordinator] Dropping invalid ASO candidate (source={source}, seed={seed}): {exc}"
            )
            return
        seed_payload_mapping = seed_payload if isinstance(seed_payload, Mapping) else {}
        parent_hashes = self._parent_hashes_from_seed(seed_payload_mapping)
        parent_feasibility = self._as_finite_float(
            seed_payload_mapping.get("parent_feasibility")
        )
        parent_objective = self._as_finite_float(
            seed_payload_mapping.get("parent_objective")
        )
        violation_delta_vs_parent = None
        if parent_feasibility is not None:
            violation_delta_vs_parent = float(max_violation) - float(parent_feasibility)
        staged_raw = seed_payload_mapping.get("staged_governor")
        staged_meta = dict(staged_raw) if isinstance(staged_raw, Mapping) else {}
        candidates.append(
            {
                "params": dict(params),
                "objective": float(objective),
                "max_violation": float(max_violation),
                "source": source,
                "seed": int(seed),
                "lineage_parent_hashes": parent_hashes,
                "operator_family": self._chosen_operator_label(
                    seed_payload=seed_payload_mapping,
                    directive=None,
                ),
                "model_route": seed_payload_mapping.get("model_route"),
                "bridge_flag": self._bridge_flag_from_seed(seed_payload_mapping),
                "staged_governor": staged_meta,
                "verify_contract_phases": ["cheap", "strict", "vmec"],
                "parent_feasibility": parent_feasibility,
                "parent_objective": parent_objective,
                "violation_delta_vs_parent": violation_delta_vs_parent,
                "improvement_reason": seed_payload_mapping.get("improvement_reason"),
                "fairness_seed_bank_hash": seed_payload_mapping.get(
                    "fairness_seed_bank_hash"
                ),
            }
        )

    def _worst_constraint_snapshot(
        self, diagnostics: OptimizerDiagnostics | None
    ) -> tuple[str | None, float | None]:
        if diagnostics is None or not diagnostics.constraint_diagnostics:
            return None, None
        worst = max(
            diagnostics.constraint_diagnostics,
            key=lambda entry: float(entry.violation),
        )
        return str(worst.name), float(worst.violation)

    def _chosen_operator_label(
        self,
        *,
        seed_payload: Mapping[str, Any],
        directive: OptimizationDirective | None,
    ) -> str:
        staged = seed_payload.get("staged_governor")
        if isinstance(staged, Mapping):
            phase = staged.get("phase")
            if isinstance(phase, str) and phase:
                return f"staged:{phase}"
        operator = seed_payload.get("operator_family")
        if isinstance(operator, str) and operator:
            return operator
        if directive is not None:
            return f"aso:{directive.action.value.lower()}"
        source = seed_payload.get("source")
        if isinstance(source, str) and source:
            return source
        return "unknown"

    def _parent_hashes_from_seed(self, seed_payload: Mapping[str, Any]) -> list[str]:
        lineage = seed_payload.get("lineage_parent_hashes")
        if isinstance(lineage, list):
            return [str(item) for item in lineage if str(item)]

        staged = seed_payload.get("staged_governor")
        if isinstance(staged, Mapping):
            hashes: list[str] = []
            focus_hash = staged.get("focus_hash")
            if isinstance(focus_hash, str) and focus_hash:
                hashes.append(focus_hash)
            partner_hash = staged.get("partner_hash")
            if isinstance(partner_hash, str) and partner_hash:
                hashes.append(partner_hash)
            return hashes
        return []

    def _bridge_flag_from_seed(self, seed_payload: Mapping[str, Any]) -> bool:
        staged = seed_payload.get("staged_governor")
        if not isinstance(staged, Mapping):
            return False
        return str(staged.get("phase", "")).lower() == "bridge"

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
                for idx, value in enumerate(
                    directive.alm_overrides["penalty_parameters"]
                ):
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
        feasibility_delta: float | None,
        hv_delta: float | None,
        worst_constraint_before: str | None,
        worst_constraint_before_value: float | None,
        worst_constraint_after: str | None,
        worst_constraint_after_value: float | None,
        chosen_operator: str,
        parent_hashes: Sequence[str],
        bridge_flag: bool,
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
                "worst_constraint_before": worst_constraint_before,
                "worst_constraint_before_value": worst_constraint_before_value,
                "worst_constraint_after": worst_constraint_after,
                "worst_constraint_after_value": worst_constraint_after_value,
                "chosen_operator": chosen_operator,
                "parent_hashes": list(parent_hashes),
                "bridge_flag": bool(bridge_flag),
                "objective_delta": diag.objective_delta,
                "feasibility_delta": feasibility_delta,
                "hv_delta": hv_delta,
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
                "fairness_seed_bank_hash": traj.seed.get("fairness_seed_bank_hash"),
                "runtime_ucb_arm": traj.seed.get("runtime_ucb_arm"),
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
    def _prepare_gated_ranked_seeds(
        self,
        *,
        seeds: Sequence[Mapping[str, Any]] | None,
        cycle: int,
        n_needed: int,
        allow_fallbacks: bool,
        experiment_id: int,
        problem: str,
        gate_limit: int,
    ) -> list[dict[str, Any]]:
        prepared = self._prepare_seeds(
            seeds,
            cycle,
            n_needed,
            allow_fallbacks=allow_fallbacks,
        )
        gated = self._apply_seed_novelty_validity_gate(
            prepared,
            experiment_id=experiment_id,
            problem=problem,
            limit=gate_limit,
        )
        return self._surrogate_rank_seeds(gated, cycle)

    def _prepare_seeds(
        self,
        initial_seeds,
        cycle,
        n_needed,
        *,
        allow_fallbacks: bool = True,
    ):
        """Prepare and validate seeds using ExplorationWorker + GeometerWorker."""
        if initial_seeds:
            normalized_initial = self._normalize_seed_batch(
                initial_seeds,
                cycle,
                limit=n_needed,
            )
            if normalized_initial:
                return normalized_initial
            if not allow_fallbacks:
                print(
                    "[Coordinator] Initial seeds rejected and fallback seeding is disabled."
                )
                return []
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
            if not allow_fallbacks:
                print(
                    "[Coordinator] Geometer rejected all seeds and fallback seeding is disabled."
                )
                return []
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

    def _seed_identity(self, params: Mapping[str, Any]) -> str:
        return json.dumps(params, sort_keys=True, separators=(",", ":"))

    def _flatten_seed_params(self, params: Mapping[str, Any]) -> np.ndarray:
        values: list[float] = []
        for key in ("r_cos", "z_sin", "r_sin", "z_cos"):
            matrix = params.get(key)
            if not isinstance(matrix, list):
                continue
            for row in matrix:
                if not isinstance(row, list):
                    continue
                for value in row:
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        continue
                    value_f = float(value)
                    if np.isfinite(value_f):
                        values.append(value_f)
        if not values:
            return np.zeros(0, dtype=float)
        return np.asarray(values, dtype=float)

    def _seed_distance(self, left: np.ndarray, right: np.ndarray) -> float:
        if left.size == 0 or right.size == 0:
            return float("inf")
        shared = min(left.size, right.size)
        if shared <= 0:
            return float("inf")
        lhs = left[:shared]
        rhs = right[:shared]
        dist = float(np.linalg.norm(lhs - rhs))
        if left.size > shared:
            dist = float(np.sqrt((dist**2) + float(np.linalg.norm(left[shared:]) ** 2)))
        if right.size > shared:
            dist = float(
                np.sqrt((dist**2) + float(np.linalg.norm(right[shared:]) ** 2))
            )
        return dist

    def _apply_seed_novelty_validity_gate(
        self,
        seeds: Sequence[Mapping[str, Any]],
        *,
        experiment_id: int,
        problem: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Canonical novelty + validity gate applied to every seed source."""
        if not seeds:
            return []

        history = self.world_model.recent_candidate_snapshots(
            experiment_id=experiment_id,
            problem=problem,
            limit=max(64, limit * 8),
        )
        historical_hashes: set[str] = set()
        historical_vectors: list[np.ndarray] = []
        for snapshot in history:
            params = snapshot.get("params")
            if not isinstance(params, Mapping):
                continue
            historical_hashes.add(self._seed_identity(params))
            historical_vectors.append(self._flatten_seed_params(params))

        accepted: list[dict[str, Any]] = []
        accepted_hashes: set[str] = set()
        accepted_vectors: list[np.ndarray] = []
        novelty_floor = 1e-8
        for seed in seeds:
            params_payload = seed.get("params")
            if not isinstance(params_payload, Mapping):
                continue
            params = dict(params_payload)
            try:
                make_boundary_from_params(params)
            except Exception:
                continue
            identity = self._seed_identity(params)
            if identity in historical_hashes or identity in accepted_hashes:
                continue
            vector = self._flatten_seed_params(params)
            novelty_score = 1.0
            if historical_vectors:
                novelty_score = min(
                    self._seed_distance(vector, baseline)
                    for baseline in historical_vectors
                )
            if accepted_vectors:
                novelty_score = min(
                    novelty_score,
                    min(
                        self._seed_distance(vector, baseline)
                        for baseline in accepted_vectors
                    ),
                )
            if novelty_score <= novelty_floor:
                continue
            payload = dict(seed)
            payload["params"] = params
            payload["novelty_score"] = float(novelty_score)
            accepted.append(payload)
            accepted_hashes.add(identity)
            accepted_vectors.append(vector)
            if len(accepted) >= limit:
                break
        return accepted

    def _expand_tree_seeds_from_candidates(
        self,
        *,
        candidates: Sequence[Mapping[str, Any]],
        parent_group: Sequence[Mapping[str, Any]],
        config: ai_config.ExperimentConfig,
        experiment_id: int,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []

        finite_candidates = [
            candidate
            for candidate in candidates
            if np.isfinite(float(candidate.get("max_violation", float("inf"))))
            and isinstance(candidate.get("params"), Mapping)
        ]
        if not finite_candidates:
            return []
        focus_candidate = min(
            finite_candidates,
            key=lambda item: float(item.get("max_violation", float("inf"))),
        )
        params_payload = focus_candidate.get("params")
        if not isinstance(params_payload, Mapping):
            return []

        staged_meta = focus_candidate.get("staged_governor")
        worst_constraint = None
        if isinstance(staged_meta, Mapping):
            raw_worst = staged_meta.get("worst_constraint")
            if isinstance(raw_worst, str) and raw_worst:
                worst_constraint = raw_worst

        focus_hash = self._seed_identity(params_payload)[:16]
        focus_node: dict[str, Any] = {
            "design_hash": focus_hash,
            "params": dict(params_payload),
            "feasibility": float(focus_candidate.get("max_violation", float("inf"))),
            "is_feasible": bool(
                float(focus_candidate.get("max_violation", float("inf")))
                <= config.aso.feasibility_threshold
            ),
            "constraint_margins": {},
        }
        parent_nodes: list[Mapping[str, Any]] = [focus_node]
        if parent_group:
            parent_nodes.append(parent_group[0])
        expanded = expand_parent_group_staged_offspring(
            parent_group=parent_nodes,
            worst_constraint=worst_constraint,
            max_repair_candidates=config.aso.staged_repair_candidates,
            bridge_blend_t=config.aso.staged_bridge_blend_t,
            offspring_per_parent=config.aso.offspring_per_parent,
        )
        replay: list[dict[str, Any]] = []
        if (
            hasattr(self.world_model, "nearest_case_deltas")
            and int(config.aso.delta_replay_top_k) > 0
        ):
            nearest = self.world_model.nearest_case_deltas(
                experiment_id=experiment_id,
                problem=config.problem,
                seed_params=dict(params_payload),
                limit=max(2, int(config.aso.delta_replay_top_k) * 2),
                near_feasibility_threshold=config.aso.staged_near_feasibility_threshold,
                include_recipes=True,
            )
            if isinstance(nearest, list):
                replay = build_delta_replay_seeds(
                    focus_params=dict(params_payload),
                    case_deltas=nearest,
                    top_k=int(config.aso.delta_replay_top_k),
                    focus_hash=focus_hash,
                    worst_constraint=worst_constraint,
                )
        merged = replay + expanded
        parent_feasibility = self._as_finite_float(
            focus_candidate.get("max_violation", focus_candidate.get("feasibility"))
        )
        parent_objective = self._as_finite_float(focus_candidate.get("objective"))
        for item in merged:
            item.setdefault("parent_feasibility", parent_feasibility)
            item.setdefault("parent_objective", parent_objective)
            if "improvement_reason" not in item:
                item["improvement_reason"] = "tree_branch_expansion"
            staged = item.get("staged_governor")
            phase = (
                str(staged.get("phase", "")).lower()
                if isinstance(staged, Mapping)
                else ""
            )
            if parent_feasibility is not None:
                if phase == "repair":
                    item.setdefault("estimated_feasibility", parent_feasibility * 0.9)
                elif phase in {"bridge", "delta_replay"}:
                    item.setdefault("estimated_feasibility", parent_feasibility * 0.95)
                else:
                    item.setdefault("estimated_feasibility", parent_feasibility)
            if parent_objective is not None:
                if phase in {"repair", "delta_replay"}:
                    item.setdefault("estimated_objective", parent_objective * 0.99)
                else:
                    item.setdefault("estimated_objective", parent_objective)
        return merged

    def _enqueue_seed_queue(
        self,
        *,
        queue: list[dict[str, Any]],
        new_items: Sequence[Mapping[str, Any]],
        max_size: int,
        tie_epsilon: float,
    ) -> None:
        if max_size <= 0:
            return
        existing: set[str] = set()
        for seed in queue:
            params = seed.get("params")
            if isinstance(params, Mapping):
                existing.add(self._seed_identity(params))

        tie_eps = float(max(0.0, tie_epsilon))
        for item in new_items:
            if len(queue) >= max_size:
                break
            params = item.get("params")
            if not isinstance(params, Mapping):
                continue
            identity = self._seed_identity(params)
            if identity in existing:
                continue

            parent_hashes = item.get("lineage_parent_hashes")
            if not isinstance(parent_hashes, list) or not parent_hashes:
                print(
                    "[Coordinator] Queue rejection: missing lineage_parent_hashes for frontier admission."
                )
                continue
            reason_raw = item.get("improvement_reason")
            reason = str(reason_raw).strip() if reason_raw is not None else ""
            if not reason:
                print(
                    "[Coordinator] Queue rejection: missing improvement_reason for frontier admission."
                )
                continue

            parent_feasibility = self._as_finite_float(item.get("parent_feasibility"))
            estimated_feasibility = self._as_finite_float(
                item.get("estimated_feasibility")
            )
            parent_objective = self._as_finite_float(item.get("parent_objective"))
            estimated_objective = self._as_finite_float(item.get("estimated_objective"))
            if parent_feasibility is not None and estimated_feasibility is not None:
                improved_feasibility = float(estimated_feasibility) + tie_eps < float(
                    parent_feasibility
                )
                tied_feasibility = (
                    abs(float(estimated_feasibility) - float(parent_feasibility))
                    <= tie_eps
                )
                improved_objective = False
                if parent_objective is not None and estimated_objective is not None:
                    improved_objective = float(estimated_objective) + tie_eps < float(
                        parent_objective
                    )
                if not improved_feasibility and not (
                    tied_feasibility and improved_objective
                ):
                    print(
                        "[Coordinator] Queue rejection: frontier admission policy rejected non-improving seed."
                    )
                    continue

            queue.append(dict(item))
            existing.add(identity)

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
        params_payload = dict(seed)
        planner_intent = params_payload.pop("planner_intent", None)
        normalized_seed = {
            "seed": int(seed.get("seed", default_seed)),
            "params": params_payload,
            "source": str(seed.get("source", "seed_normalized")),
        }
        if isinstance(planner_intent, Mapping):
            normalized_seed["planner_intent"] = dict(planner_intent)
        return normalized_seed

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
