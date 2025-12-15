"""
Cycle execution logic extracted from runner.py.
Handles the core orchestration of a single governance cycle:
- Candidate generation (ASO, ALM, Samplers)
- Surrogate ranking and screening
- High-fidelity evaluation and promotion
- Reporting and persistence
"""

from __future__ import annotations

import json
import logging
import math
import os
import platform
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence, Set, Tuple, cast

import jax.numpy as jnp
import numpy as np
from constellaration.geometry import surface_rz_fourier as surface_module
from constellaration.initial_guess import generate_nae, generate_rotating_ellipse
from constellaration.optimization.augmented_lagrangian import (
    AugmentedLagrangianSettings,
    AugmentedLagrangianState,
    augmented_lagrangian_function,
    update_augmented_lagrangian_state,
)
from constellaration.utils import pytree

from ai_scientist import adapter, memory, reporting, tools
from ai_scientist import config as ai_config
from ai_scientist import planner as ai_planner
from ai_scientist.budget_manager import BudgetController, BudgetSnapshot
from ai_scientist.coordinator import Coordinator
from ai_scientist.fidelity_controller import FidelityController
from ai_scientist.optim.generative import DiffusionDesignModel, GenerativeDesignModel
from ai_scientist.optim.samplers import NearAxisSampler
from ai_scientist.optim.surrogate import BaseSurrogate, SurrogatePrediction
from ai_scientist.optim.surrogate_v2 import NeuralOperatorSurrogate
from constellaration import forward_model
from orchestration import adaptation as adaptation_helpers
from ai_scientist.prefilter import FeasibilityPrefilter
from ai_scientist.optim import geometry
from ai_scientist.objective_types import get_training_target

# Constants
FEASIBILITY_CUTOFF = getattr(tools, "_DEFAULT_RELATIVE_TOLERANCE", 1e-2)
P3_REFERENCE_POINT = getattr(tools, "_P3_REFERENCE_POINT", (1.0, 20.0))
_BOUNDARY_SEED_CACHE: dict[Path, dict[str, Any]] = {}
_LAST_SURROGATE_FIT_SEC = 0.0
NAN_TO_HIGH_VALUE = 10.0


@dataclass
class CycleResult:
    """Outcome of a single execution cycle."""

    cycle_index: int
    candidates_evaluated: int
    candidates_promoted: int
    best_objective: float | None
    hypervolume: float | None
    feasibility_rate: float
    report_path: Path | None
    best_eval: dict[str, Any] | None
    p3_summary: tools.P3Summary | None


class ProblemEvaluator(Protocol):
    def __call__(
        self,
        boundary_params: Mapping[str, Any],
        *,
        stage: str,
        use_cache: bool = True,
    ) -> dict[str, Any]: ...


class WorldModelLike(Protocol):
    def log_statement(
        self,
        experiment_id: int,
        cycle: int,
        stage: str,
        text: str,
        status: str,
        tool_name: str,
        tool_input: Mapping[str, Any],
        *,
        metrics_id: int | None = None,
        seed: int | None = None,
        git_sha: str,
        repro_cmd: str,
        created_at: str | None = None,
        commit: bool = True,
    ) -> int: ...


_PROBLEM_EVALUATORS: dict[str, tuple[str, ProblemEvaluator]] = {
    "p1": ("evaluate_p1", tools.evaluate_p1),
    "p2": ("evaluate_p2", tools.evaluate_p2),
    "p3": ("evaluate_p3", tools.evaluate_p3),
}


def _problem_evaluator(problem: str) -> ProblemEvaluator:
    try:
        return _PROBLEM_EVALUATORS[problem][1]
    except KeyError as exc:
        raise NotImplementedError(
            "Problem '%s' is not supported; choose one of %s."
            % (problem, ", ".join(sorted(_PROBLEM_EVALUATORS)))
        ) from exc


def _problem_tool_name(problem: str) -> str:
    try:
        return _PROBLEM_EVALUATORS[problem][0]
    except KeyError as exc:
        raise NotImplementedError(
            "Problem '%s' is not supported; choose one of %s."
            % (problem, ", ".join(sorted(_PROBLEM_EVALUATORS)))
        ) from exc


def serialize_experiment_config(
    cfg: ai_config.ExperimentConfig, constellaration_sha: str | None = None
) -> dict[str, Any]:
    boundary_template = asdict(cfg.boundary_template)
    seed_path = boundary_template.get("seed_path")
    if seed_path is not None:
        boundary_template["seed_path"] = str(seed_path)
    return {
        "problem": cfg.problem,
        "cycles": cfg.cycles,
        "random_seed": cfg.random_seed,
        "budgets": asdict(cfg.budgets),
        "adaptive_budgets": asdict(cfg.adaptive_budgets),
        "proposal_mix": asdict(cfg.proposal_mix),
        "fidelity_ladder": asdict(cfg.fidelity_ladder),
        "boundary_template": boundary_template,
        "stage_gates": asdict(cfg.stage_gates),
        "governance": asdict(cfg.governance),
        "source_config": str(cfg.source_config),
        "reporting_dir": str(cfg.reporting_dir),
        "memory_db": str(cfg.memory_db),
        "constellaration_sha": constellaration_sha or "unknown",
        "reporting": cfg.reporting,
        "planner": cfg.planner,
    }


class CycleExecutor:
    def __init__(
        self,
        config: ai_config.ExperimentConfig,
        world_model: memory.WorldModel,
        planner: Any,  # ai_planner.PlanningAgent or similar
        budget_controller: BudgetController,
        fidelity_controller: FidelityController,
    ):
        self.config = config
        self.world_model = world_model
        self.planner = planner
        self.budget_controller = budget_controller
        self.fidelity_controller = fidelity_controller
        self.prefilter = FeasibilityPrefilter()

    def run_cycle(
        self,
        cycle_index: int,
        experiment_id: int,
        governance_stage: str,
        git_sha: str,
        constellaration_sha: str,
        surrogate_model: BaseSurrogate,
        generative_model: GenerativeDesignModel | DiffusionDesignModel | None = None,
        prev_feasibility_rate: float | None = None,
        suggested_params: list[dict[str, Any]] | None = None,
        config_overrides: Mapping[str, Any] | None = None,
        # Runtime flags passed as kwargs to avoid coupling with RunnerCLIConfig
        verbose: bool = False,
        slow: bool = False,
        screen_only: bool = False,
        log_cache_stats: bool = False,
    ) -> CycleResult:
        tool_name = _problem_tool_name(self.config.problem)
        base_evaluate = _problem_evaluator(self.config.problem)
        verifier_evaluate_fn = adapter.with_peft(base_evaluate, tool_name=tool_name)
        # Keep the centralized batch evaluator as the default (forward_model_batch),
        # but allow tests to inject a lightweight evaluator by patching
        # `_problem_evaluator` to return a different callable.
        fidelity_evaluate_fn = (
            None
            if base_evaluate
            in (tools.evaluate_p1, tools.evaluate_p2, tools.evaluate_p3)
            else base_evaluate
        )
        cycle_start = time.perf_counter()
        cycle_number = cycle_index + 1
        global _LAST_SURROGATE_FIT_SEC
        _LAST_SURROGATE_FIT_SEC = 0.0
        sleep_per_eval = 0.05 if slow else 0.0

        # Apply config-defined run_overrides first (for testing/fixed behavior)
        current_overrides = config_overrides
        if self.config.run_overrides:
            try:
                if current_overrides:
                    temp_overrides = dict(current_overrides)
                    for key, value in self.config.run_overrides.items():
                        if (
                            isinstance(value, Mapping)
                            and key in temp_overrides
                            and isinstance(temp_overrides[key], Mapping)
                        ):
                            temp_overrides[key] = {**temp_overrides[key], **value}
                        else:
                            temp_overrides[key] = value
                    current_overrides = temp_overrides
                else:
                    current_overrides = self.config.run_overrides
                if verbose:
                    print(
                        f"[runner][cycle={cycle_number}] Applying config-defined run_overrides: {self.config.run_overrides}"
                    )
            except Exception as exc:
                print(
                    f"[runner][cycle={cycle_number}] Failed to apply run_overrides from config: {exc}"
                )

        # Apply agent-driven config overrides
        active_cfg = self.config
        optimizer_mode = "default"
        alm_settings_overrides: Mapping[str, Any] = {}

        if current_overrides:
            try:
                overrides_log = []
                if "proposal_mix" in current_overrides:
                    new_mix = replace(
                        active_cfg.proposal_mix, **current_overrides["proposal_mix"]
                    )
                    active_cfg = replace(active_cfg, proposal_mix=new_mix)
                    overrides_log.append(
                        f"proposal_mix={current_overrides['proposal_mix']}"
                    )
                if "budgets" in current_overrides:
                    new_budgets = replace(
                        active_cfg.budgets, **current_overrides["budgets"]
                    )
                    active_cfg = replace(active_cfg, budgets=new_budgets)
                    overrides_log.append(f"budgets={current_overrides['budgets']}")
                if "constraint_weights" in current_overrides:
                    new_weights = replace(
                        active_cfg.constraint_weights,
                        **current_overrides["constraint_weights"],
                    )
                    active_cfg = replace(active_cfg, constraint_weights=new_weights)
                    overrides_log.append(
                        f"constraint_weights={current_overrides['constraint_weights']}"
                    )

                if "optimizer" in current_overrides:
                    optimizer_mode = str(current_overrides["optimizer"]).lower()
                    overrides_log.append(f"optimizer={optimizer_mode}")
                if "alm_settings" in current_overrides:
                    alm_settings_overrides = current_overrides["alm_settings"]
                    overrides_log.append(f"alm_settings={alm_settings_overrides}")
                if "initialization_strategy" in current_overrides:
                    new_init_strategy = str(
                        current_overrides["initialization_strategy"]
                    )
                    active_cfg = replace(
                        active_cfg, initialization_strategy=new_init_strategy
                    )
                    overrides_log.append(f"initialization_strategy={new_init_strategy}")

                if overrides_log:
                    print(
                        f"[runner][cycle={cycle_number}] Applying agent config overrides: {', '.join(overrides_log)}"
                    )
            except Exception as exc:
                print(
                    f"[runner][cycle={cycle_number}] Failed to apply config overrides: {exc}"
                )

        # Create local fidelity controller with active config
        if self.fidelity_controller:
            fidelity_ctl = self.fidelity_controller
        else:
            fidelity_ctl = FidelityController(active_cfg)

        budget_snapshot = self.budget_controller.snapshot()
        active_budgets = replace(
            active_cfg.budgets,
            screen_evals_per_cycle=budget_snapshot.screen_evals_per_cycle,
            promote_top_k=budget_snapshot.promote_top_k,
            max_high_fidelity_evals_per_cycle=budget_snapshot.max_high_fidelity_evals_per_cycle,
        )

        # Re-apply agent overrides if they exist
        if current_overrides and "budgets" in current_overrides:
            active_budgets = replace(active_budgets, **current_overrides["budgets"])
            if verbose:
                print(
                    f"[runner] Agent forced budget overrides: {current_overrides['budgets']}"
                )

        if active_cfg.adaptive_budgets.enabled and verbose:
            print(
                f"[runner][budget] cycle={cycle_number} override "
                f"screen={active_budgets.screen_evals_per_cycle} "
                f"promote_top_k={active_budgets.promote_top_k} "
                f"max_high_fidelity={active_budgets.max_high_fidelity_evals_per_cycle}"
            )
        cache_hit_rate = 0.0
        pool_size = _surrogate_candidate_pool_size(
            active_budgets.screen_evals_per_cycle,
            active_cfg.proposal_mix.surrogate_pool_multiplier,
        )

        if generative_model:
            # P3 is multi-objective: surrogate predicts hypervolume contribution for
            # ranking, not the single physics objective (aspect_ratio). This is
            # intentional—see forward_model.compute_objective() docstring.
            # Use SSOT function to determine target (avoids semantic drift).
            gen_history = self.world_model.surrogate_training_data(
                target=get_training_target(self.config.problem).value,
                problem=self.config.problem,
            )
            if gen_history:
                generative_model.fit([item[0] for item in gen_history])

        # Train Feasibility Prefilter on historical data
        # Query all historical evaluations from the database
        try:
            rows = self.world_model._conn.execute(
                """
                SELECT m.raw_json, m.is_feasible
                FROM metrics m
                JOIN candidates c ON m.candidate_id = c.id
                WHERE c.experiment_id = ? AND c.problem = ?
                AND m.raw_json IS NOT NULL
                """,
                (experiment_id, self.config.problem),
            ).fetchall()

            if len(rows) >= 10:
                X_list = []
                y_list = []
                for row in rows:
                    try:
                        metrics = json.loads(row["raw_json"])
                        # Extract features that the prefilter expects
                        features = []
                        has_all_features = True
                        for key in self.prefilter.feature_keys:
                            if key in metrics:
                                features.append(metrics[key])
                            else:
                                has_all_features = False
                                break

                        if has_all_features:
                            X_list.append(features)
                            y_list.append(bool(row["is_feasible"]))
                    except (json.JSONDecodeError, KeyError):
                        continue

                if len(X_list) >= 10:
                    X = np.array(X_list)
                    y = np.array(y_list)
                    self.prefilter.train(X, y)
                    if verbose:
                        print(
                            f"[runner][cycle={cycle_number}] Trained prefilter on {len(X)} historical evaluations."
                        )
        except Exception as e:
            if verbose:
                print(
                    f"[runner][cycle={cycle_number}] Prefilter training failed: {e}. Continuing without prefilter."
                )

        candidate_pool: list[dict[str, Any]] = []

        # Initialize Coordinator (Phase 5)
        # Always rebuild with active_cfg to honor per-cycle overrides (P1 fix)
        coordinator = Coordinator(
            active_cfg,
            self.world_model,
            planner=ai_planner.PlanningAgent(),  # Default planner
            surrogate=surrogate_model
            if isinstance(surrogate_model, NeuralOperatorSurrogate)
            else None,
            generative_model=generative_model,
        )

        if active_cfg.aso.enabled:
            if verbose:
                print(f"[runner][cycle={cycle_number}] ASO mode with real ALM state.")

            initial_seeds = []
            if suggested_params:
                initial_seeds = suggested_params

            candidate_pool = cast(
                list[dict[str, Any]],
                coordinator.produce_candidates_aso(
                    cycle=cycle_number,
                    experiment_id=experiment_id,
                    eval_budget=active_budgets.screen_evals_per_cycle,
                    template=active_cfg.boundary_template,
                    initial_seeds=initial_seeds,
                    initial_config=active_cfg,
                ),
            )
            candidates = candidate_pool

        elif self.config.optimizer_backend == "gradient_descent":
            if verbose:
                print(
                    f"[runner][cycle={cycle_number}] V2 Gradient Descent Optimization active (Phase 5 Coordinator)."
                )

            candidate_pool = cast(
                list[dict[str, Any]],
                coordinator.produce_candidates(
                    cycle=cycle_number,
                    experiment_id=experiment_id,
                    n_candidates=pool_size,
                    template=active_cfg.boundary_template,
                ),
            )

            if not candidate_pool:
                if verbose:
                    print(
                        f"[runner][cycle={cycle_number}] Coordinator produced no candidates. Falling back to legacy sampler."
                    )

                candidate_pool, _, _, _ = _propose_p3_candidates_for_cycle(
                    cfg=active_cfg,
                    cycle_index=cycle_index,
                    world_model=self.world_model,
                    experiment_id=experiment_id,
                    screen_budget=active_budgets.screen_evals_per_cycle,
                    total_candidates=pool_size,
                    generative_model=generative_model,
                    suggested_params=suggested_params,
                )

            candidates = _surrogate_rank_screen_candidates(
                active_cfg,
                active_budgets.screen_evals_per_cycle,
                candidate_pool,
                self.world_model,
                surrogate_model,
                cycle=cycle_number,
                verbose=verbose,
            )

        elif (optimizer_mode == "alm" or optimizer_mode == "sa-alm") or (
            self.config.optimizer_backend in ["alm", "sa-alm"]
        ):
            # ALM Execution Branch (legacy)
            candidates = self._run_alm_optimization(
                active_cfg,
                cycle_index,
                experiment_id,
                surrogate_model,
                suggested_params,
                optimizer_mode,
                alm_settings_overrides,
                active_budgets,
                cycle_number,
                verbose,
                cycle_start,
            )
            candidate_pool = candidates

        elif pool_size <= 0:
            if verbose:
                print(
                    f"[runner][cycle={cycle_number}] screen budget zero; skipping candidate generation"
                )
            candidate_pool = []
            candidates = []
        else:
            (
                candidate_pool,
                sampler_count,
                random_count,
                vae_count,
            ) = cast(
                tuple[list[dict[str, Any]], int, int, int],
                _propose_p3_candidates_for_cycle(
                    active_cfg,
                    cycle_index,
                    self.world_model,
                    experiment_id,
                    screen_budget=active_budgets.screen_evals_per_cycle,
                    total_candidates=pool_size,
                    prev_feasibility_rate=prev_feasibility_rate,
                    suggested_params=suggested_params,
                    generative_model=generative_model,
                ),
            )
            if verbose:
                print(
                    f"[runner][cycle={cycle_number}] candidate mix (pool={len(candidate_pool)}): sampler={sampler_count} random={random_count} vae={vae_count} agent={len(suggested_params or [])}"
                )

            # Apply Feasibility Prefilter
            if self.prefilter.is_trained:
                # Compute geometric features for prefilter
                try:
                    import torch

                    r_cos_list = []
                    z_sin_list = []
                    nfp_list = []
                    indices_to_compute = []

                    for i, cand in enumerate(candidate_pool):
                        # Only compute if not already present
                        if "aspect_ratio" not in cand or "max_elongation" not in cand:
                            p = cand["params"]
                            if "r_cos" in p and "z_sin" in p:
                                r_cos_list.append(p["r_cos"])
                                z_sin_list.append(p["z_sin"])
                                nfp_list.append(p.get("n_field_periods", 1))
                                indices_to_compute.append(i)

                    if indices_to_compute:
                        # Convert to tensors for geometry module
                        r_cos_t = torch.tensor(r_cos_list, dtype=torch.float32)
                        z_sin_t = torch.tensor(z_sin_list, dtype=torch.float32)
                        nfp_t = torch.tensor(nfp_list, dtype=torch.float32)

                        ar_batch = geometry.aspect_ratio(r_cos_t, z_sin_t, nfp_t)
                        elo_batch = geometry.elongation_isoperimetric(
                            r_cos_t, z_sin_t, nfp_t
                        )

                        for k, idx in enumerate(indices_to_compute):
                            candidate_pool[idx]["aspect_ratio"] = float(
                                ar_batch[k].item()
                            )
                            candidate_pool[idx]["max_elongation"] = float(
                                elo_batch[k].item()
                            )
                except Exception as e:
                    if verbose:
                        print(f"[runner] Failed to compute prefilter features: {e}")

                original_count = len(candidate_pool)
                candidate_pool = self.prefilter.filter_candidates(candidate_pool)
                if verbose:
                    print(
                        f"[runner][cycle={cycle_number}] Prefilter: {original_count} -> {len(candidate_pool)} candidates"
                    )

            candidates = _surrogate_rank_screen_candidates(
                active_cfg,
                active_budgets.screen_evals_per_cycle,
                candidate_pool,
                self.world_model,
                surrogate_model,
                cycle=cycle_number,
                verbose=verbose,
            )

        screen_stage = active_cfg.fidelity_ladder.screen
        if candidates:
            screen_results = fidelity_ctl.evaluate_stage(
                candidates,
                stage=screen_stage,
                budgets=active_budgets,
                cycle_start=cycle_start,
                evaluate_fn=fidelity_evaluate_fn,
                sleep_per_eval=sleep_per_eval,
                tool_name=tool_name,
            )
        else:
            screen_results = []
        self.budget_controller.consume(len(screen_results))
        screen_cache_stats = tools.get_cache_stats(screen_stage)
        cache_hit_rate = self.budget_controller.capture_cache_hit_rate(
            screen_stage, stats=screen_cache_stats
        )
        if log_cache_stats:
            _maybe_log_cache_stats(
                self.config, cycle_index, screen_stage, screen_cache_stats
            )

        screen_design_map = _latest_evaluations_by_design(screen_results, screen_stage)
        screen_summary = tools.summarize_p3_candidates(
            list(screen_design_map.values()), reference_point=P3_REFERENCE_POINT
        )

        # Adaptive Promotion Logic
        sufficient_feasible = (
            screen_summary.feasible_count
            >= self.config.governance.min_feasible_for_promotion
        )

        promote_limit = 0
        to_promote: list[dict[str, Any]] = []

        if screen_design_map:
            promote_limit = min(
                active_budgets.promote_top_k,
                active_budgets.max_high_fidelity_evals_per_cycle,
                len(screen_design_map),
            )

            if sufficient_feasible:
                prioritized_screen = fidelity_ctl.get_promotion_candidates(
                    screen_design_map,
                    active_budgets.promote_top_k,
                    P3_REFERENCE_POINT,
                )
                to_promote = cast(
                    list[dict[str, Any]], prioritized_screen[:promote_limit]
                )
            else:
                if verbose:
                    print(
                        f"[runner][promotion] Insufficient feasible ({screen_summary.feasible_count} < {self.config.governance.min_feasible_for_promotion}); promoting by lowest max_violation."
                    )
                sorted_by_violation = sorted(
                    screen_design_map.values(),
                    key=lambda entry: float(
                        entry["evaluation"].get("max_violation", float("inf"))
                    ),
                )
                to_promote = cast(
                    list[dict[str, Any]], sorted_by_violation[:promote_limit]
                )
                for candidate in to_promote:
                    candidate["promotion_reason"] = "restoration"

        promote_stage = self.config.fidelity_ladder.promote
        promote_results: list[dict[str, Any]] = []
        if not screen_only and promote_limit > 0:
            promote_results = fidelity_ctl.evaluate_stage(
                to_promote,
                stage=promote_stage,
                budgets=active_budgets,
                cycle_start=cycle_start,
                evaluate_fn=fidelity_evaluate_fn,
                sleep_per_eval=sleep_per_eval,
                tool_name=tool_name,
            )
            promote_cache_stats = tools.get_cache_stats(promote_stage)
            if log_cache_stats:
                _maybe_log_cache_stats(
                    self.config, cycle_index, promote_stage, promote_cache_stats
                )
        elif screen_only and verbose:
            print("[runner] screen-only flag active; skipping promotion evaluations.")

        self.budget_controller.consume(len(promote_results))

        aggregated = screen_results + promote_results
        if not aggregated:
            self.world_model.record_stage_history(
                experiment_id=experiment_id,
                cycle=cycle_number,
                stage=governance_stage,
            )
            return CycleResult(cycle_index, 0, 0, None, None, 0.0, None, None, None)

        latest_by_design = _latest_evaluations_by_design(
            aggregated, self.config.fidelity_ladder.promote
        )
        if not latest_by_design:
            self.world_model.record_stage_history(
                experiment_id=experiment_id,
                cycle=cycle_number,
                stage=governance_stage,
            )
            return CycleResult(
                cycle_index,
                len(screen_results),
                len(promote_results),
                None,
                None,
                0.0,
                None,
                None,
                None,
            )

        p3_summary = tools.summarize_p3_candidates(
            list(latest_by_design.values()), reference_point=P3_REFERENCE_POINT
        )
        total_designs = len(latest_by_design)
        feasibility_rate = float(p3_summary.feasible_count) / float(
            max(1, total_designs)
        )
        vmec_failure_rate = _vmec_failure_rate(aggregated)
        hv_display = (
            f"{p3_summary.hv_score:.6f}" if p3_summary.feasible_count > 0 else "n/a"
        )
        print(
            f"[runner][cycle={cycle_number}] feasible={p3_summary.feasible_count}/{total_designs} hv={hv_display}"
        )
        _log_observability_metrics(
            self.config,
            cycle_index,
            hv=p3_summary.hv_score,
            feasible_count=p3_summary.feasible_count,
            vmec_failure_rate=vmec_failure_rate,
            retrain_time=_LAST_SURROGATE_FIT_SEC,
            cache_hit_rate=cache_hit_rate,
            budget_snapshot=budget_snapshot,
        )

        if self.config.reporting.get("prometheus_export_enabled", False):
            prom_path = Path(self.config.reporting_dir) / "metrics.prom"
            reporting.export_metrics_to_prometheus_textfile(
                {
                    "cycle_hv": p3_summary.hv_score or 0.0,
                    "feasible_count": p3_summary.feasible_count,
                    "vmec_failure_rate": vmec_failure_rate,
                    "surrogate_retrain_seconds": _LAST_SURROGATE_FIT_SEC,
                    "cache_hit_rate": cache_hit_rate,
                    "cycle": cycle_number,
                },
                prom_path,
            )

        p3_summary_path = adaptation_helpers.write_p3_summary(
            base_dir=self.config.reporting_dir,
            cycle=cycle_number,
            summary=p3_summary,
        )

        # M3 FIX: Prefer feasible candidates for best selection (scientific correctness)
        feasible_entries = [
            e
            for e in latest_by_design.values()
            if e.get("evaluation", {}).get("is_feasible", False)
        ]
        if feasible_entries:
            best_entry = min(feasible_entries, key=_oriented_objective)
        else:
            # Fallback: use lowest max_violation if no feasible candidates
            best_entry = min(
                latest_by_design.values(),
                key=lambda e: (
                    e.get("evaluation", {}).get("max_violation", float("inf")),
                    _oriented_objective(e),
                ),
            )
        best_eval: dict[str, Any] = dict(best_entry["evaluation"])
        metrics_payload = best_eval.setdefault("metrics", {})
        metrics_payload["cycle_hv"] = p3_summary.hv_score
        best_eval["cycle_hv"] = p3_summary.hv_score
        best_eval["design_hash"] = best_entry.get("design_hash", "")
        cycle_duration = time.perf_counter() - cycle_start
        best_seed = int(best_entry.get("seed", self.config.random_seed))
        previous_baseline = self.world_model.previous_best_hv(
            experiment_id, cycle_number
        )
        current_hv = float(p3_summary.hv_score)
        hv_delta = (
            current_hv - previous_baseline
            if previous_baseline is not None
            else current_hv
        )
        self.budget_controller.adjust_for_cycle(
            hv_delta=hv_delta,
            feasibility_rate=feasibility_rate,
            cache_hit_rate=cache_hit_rate,
        )
        best_metrics_id: int | None = None
        logged_hashes: Set[str] = set()
        config_snapshot = dict(
            serialize_experiment_config(
                self.config, constellaration_sha=constellaration_sha
            )
        )
        config_snapshot["cycle_seed"] = best_seed
        adapter_version = adapter.current_adapter_version(
            tool_name, self.config.fidelity_ladder.promote
        ) or adapter.current_adapter_version(
            tool_name, self.config.fidelity_ladder.screen
        )
        config_snapshot["adapter_version"] = adapter_version
        cycle_json = {
            "experiment_id": experiment_id,
            "git_sha": git_sha,
            "constellaration_sha": constellaration_sha,
            "cycle": cycle_number,
            "stage": governance_stage,
            "feasible_count": p3_summary.feasible_count,
            "hv": p3_summary.hv_score,
            "vmec_failure_rate": vmec_failure_rate,
            "surrogate_retrain_seconds": _LAST_SURROGATE_FIT_SEC,
            "reference_point": p3_summary.reference_point,
            "archive_size": p3_summary.archive_size,
            "screened": len(screen_results),
            "promoted": len(promote_results),
            "feasibility_rate": feasibility_rate,
            "adapter_version": adapter_version,
            "last_feedback": {
                "hv_delta": hv_delta,
                "feasibility_rate": feasibility_rate,
                "cache_hit_rate": cache_hit_rate,
            },
            "budget_controller": self.budget_controller.to_dict(),
        }
        cycle_json_path = Path(self.config.reporting_dir) / f"cycle_{cycle_number}.json"

        with self.world_model.transaction():
            self.world_model.record_cycle(
                experiment_id=experiment_id,
                cycle_number=cycle_number,
                screen_evals=len(screen_results),
                promoted_evals=len(promote_results),
                high_fidelity_evals=len(promote_results),
                wall_seconds=cycle_duration,
                best_params=best_entry["params"],
                best_evaluation=best_eval,
                seed=best_seed,
                log_best_candidate=False,
                problem=self.config.problem,
                commit=False,
            )
            self.world_model.record_cycle_summary(
                experiment_id=experiment_id,
                cycle_number=cycle_number,
                stage=governance_stage,
                feasible_count=p3_summary.feasible_count,
                hv_score=p3_summary.hv_score,
                commit=False,
            )
            self.world_model.record_cycle_hv(
                experiment_id=experiment_id,
                cycle_number=cycle_number,
                hv_score=p3_summary.hv_score,
                reference_point=p3_summary.reference_point,
                pareto_entries=[
                    entry.as_mapping() for entry in p3_summary.pareto_entries
                ],
                n_feasible=p3_summary.feasible_count,
                n_archive=p3_summary.archive_size,
                hv_lookback=self.config.governance.hv_lookback,
                commit=False,
            )
            logged_hashes, metrics_by_hash = _persist_pareto_archive(
                world_model=self.world_model,
                experiment_id=experiment_id,
                cycle_number=cycle_number,
                problem=self.config.problem,
                entries_by_design=latest_by_design,
                p3_summary=p3_summary,
                git_sha=git_sha,
                constellaration_sha=constellaration_sha,
            )
            if (
                best_entry.get("design_hash")
                and best_entry["design_hash"] not in logged_hashes
            ):
                _, metrics_id = self.world_model.log_candidate(
                    experiment_id=experiment_id,
                    problem=self.config.problem,
                    params=best_entry["params"],
                    seed=best_seed,
                    status=best_eval.get("stage", "unknown"),
                    evaluation=best_eval,
                    design_hash=best_entry["design_hash"],
                    commit=False,
                )
                best_metrics_id = metrics_id
            self.world_model.record_deterministic_snapshot(
                experiment_id=experiment_id,
                cycle_number=cycle_number,
                snapshot=config_snapshot,
                constellaration_sha=constellaration_sha,
                seed=best_seed,
                commit=False,
            )

        if best_metrics_id is None:
            best_metrics_id = metrics_by_hash.get(best_entry.get("design_hash", ""))

        cycle_json_path.parent.mkdir(parents=True, exist_ok=True)
        cycle_json_path.write_text(json.dumps(cycle_json, indent=2), encoding="utf-8")

        # Generate reproduction command (assuming args will be filled by user or stored context,
        # but here we construct a best effort command since we don't have full CLI args in this class)
        repro_command = (
            f"python -m ai_scientist.runner --config {self.config.source_config} --problem {self.config.problem} "
            f"--cycles {self.config.cycles} --eval-budget {active_budgets.screen_evals_per_cycle} "
            f"--workers {active_budgets.n_workers} --pool-type {active_budgets.pool_type}"
        )

        env_block = (
            f"- Python: {sys.version.splitlines()[0]}\n"
            f"- Platform: {platform.platform()}\n"
            f"- Executable: {sys.executable}\n"
            f"- Host: {platform.node()}\n"
            f"- CPU: {platform.processor() or 'unknown'} | cores: {os.cpu_count() or 'unknown'}\n"
        )
        pareto_entries = p3_summary.pareto_entries
        if pareto_entries:
            pareto_lines = "\n".join(
                f"- design {entry.design_hash[:8]} seed={entry.seed} stage={entry.stage}: "
                f"gradient={entry.gradient:.4f}, aspect={entry.aspect_ratio:.4f}, "
                f"feasibility={entry.feasibility:.4f}"
                for entry in pareto_entries
            )
        else:
            pareto_lines = "- none (pareto front empty)"

        if pareto_entries:
            replay_entry = pareto_entries[0]
            reproduction_snippet = (
                "```bash\n"
                "python - <<'PY'\n"
                "import json, sqlite3\n"
                "from ai_scientist import tools\n"
                f"conn = sqlite3.connect('{self.config.memory_db}')\n"
                "row = conn.execute(\n"
                '    "SELECT params_json FROM candidates WHERE design_hash = ? ORDER BY id DESC LIMIT 1",\n'
                f"    ('{replay_entry.design_hash}',),\n"
                ").fetchone()\n"
                "assert row, 'Design hash not found in world model'\n"
                "params = json.loads(row[0])\n"
                f"print(tools.{tool_name}(params, stage='{replay_entry.stage or self.config.fidelity_ladder.promote}'))\n"
                "PY\n"
                "```\n"
            )
        else:
            reproduction_snippet = (
                "No Pareto archive entries available to replay this cycle.\n"
            )

        stage_label = best_eval.get("stage") or self.config.fidelity_ladder.promote
        statement_status = _verify_best_claim(
            world_model=self.world_model,
            experiment_id=experiment_id,
            cycle_number=cycle_number,
            best_entry=best_entry,
            best_eval=best_eval,
            evaluation_fn=verifier_evaluate_fn,
            tool_name=tool_name,
            best_seed=best_seed,
            git_sha=git_sha,
            reproduction_command=repro_command,
            stage=stage_label,
            metrics_id=best_metrics_id,
        )

        tool_input = {"params": best_entry["params"], "stage": stage_label}
        tool_input_hash = memory.hash_payload(tool_input)
        preference_pairs_path = adaptation_helpers.append_preference_pair(
            base_dir=self.config.reporting_dir,
            cycle=cycle_number,
            pair={
                "stage": stage_label,
                "status": statement_status,
                "tool_name": tool_name,
                "tool_input_hash": tool_input_hash,
                "reproduction_command": repro_command,
                "metrics": best_eval.get("metrics", {}),
                "design_hash": best_entry.get("design_hash"),
                "problem": self.config.problem,
                "seed": best_seed,
            },
        )
        trajectory_path = adaptation_helpers.append_trajectory_entry(
            base_dir=self.config.reporting_dir,
            cycle=cycle_number,
            entry={
                "stage": stage_label,
                "seed": best_seed,
                "tool_name": tool_name,
                "tool_input_hash": tool_input_hash,
                "reproduction_steps": [
                    repro_command,
                    f"git checkout {git_sha}",
                    f"(cd constellaration && git checkout {constellaration_sha})",
                    f"tools.{tool_name}(params, stage='{stage_label}')",
                ],
                "reproduction_snippet": reproduction_snippet,
                "reproduction_command": repro_command,
                "params": best_entry["params"],
                "metrics": best_eval.get("metrics", {}),
                "problem": self.config.problem,
                "design_hash": best_entry.get("design_hash", ""),
            },
        )

        preference_pairs_anchor = _repo_relative(preference_pairs_path)
        p3_summary_anchor = _repo_relative(p3_summary_path)
        trajectory_anchor = _repo_relative(trajectory_path)
        baseline_display = (
            f"{previous_baseline:.6f}" if previous_baseline is not None else "n/a"
        )
        preference_pairs_display = preference_pairs_anchor or preference_pairs_path.name
        trajectory_display = trajectory_anchor or trajectory_path.name
        p3_summary_display = p3_summary_anchor or p3_summary_path.name
        hv_text = (
            f"Cycle {cycle_number} hypervolume {current_hv:.6f} vs baseline "
            f"{baseline_display} (delta {hv_delta:+.6f}) recorded in cycle_hv "
            f"and adaptation logs {preference_pairs_display} and {trajectory_display} "
            f"with summary {p3_summary_display} (docs/TASKS_CODEX_MINI.md:238; docs/MASTER_PLAN_AI_SCIENTIST.md:226-247)."
        )
        hv_tool_input = {
            "cycle": cycle_number,
            "stage": stage_label,
            "current_hv": current_hv,
            "baseline_hv": previous_baseline,
            "delta": hv_delta,
            "preference_pairs_anchor": preference_pairs_anchor,
            "p3_summary_anchor": p3_summary_anchor,
            "trajectory_anchor": trajectory_anchor,
        }
        self.world_model.log_statement(
            experiment_id=experiment_id,
            cycle=cycle_number,
            stage=stage_label,
            text=hv_text,
            status="PENDING",
            tool_name="hv_delta_comparison",
            tool_input=hv_tool_input,
            metrics_id=best_metrics_id,
            seed=best_seed,
            git_sha=git_sha,
            repro_cmd=repro_command,
        )
        statements = self.world_model.statements_for_cycle(experiment_id, cycle_number)
        figure_path = reporting.save_pareto_figure(
            p3_summary.pareto_entries,
            self.config.reporting_dir,
            title=self.config.problem,
            cycle_index=cycle_index,
        )
        figure_paths = [figure_path] if figure_path else []
        metrics_payload = best_eval.get("metrics", {})
        metrics_path = adaptation_helpers.write_metrics_snapshot(
            base_dir=self.config.reporting_dir,
            cycle=cycle_number,
            payload=metrics_payload,
        )
        artifact_entries: list[tuple[str, Path]] = [("metrics_snapshot", metrics_path)]
        self.world_model.log_artifact(
            experiment_id=experiment_id,
            path=metrics_path,
            kind="metrics_snapshot",
        )
        artifact_entries.append(("p3_summary", p3_summary_path))
        self.world_model.log_artifact(
            experiment_id=experiment_id,
            path=p3_summary_path,
            kind="p3_summary",
        )
        artifact_entries.append(("preference_pairs", preference_pairs_path))
        self.world_model.log_artifact(
            experiment_id=experiment_id,
            path=preference_pairs_path,
            kind="preference_pairs",
        )
        artifact_entries.append(("trajectory_entry", trajectory_path))
        self.world_model.log_artifact(
            experiment_id=experiment_id,
            path=trajectory_path,
            kind="trajectory_entry",
        )
        if figure_path:
            artifact_entries.append(("pareto_figure", figure_path))
            self.world_model.log_artifact(
                experiment_id=experiment_id,
                path=figure_path,
                kind="pareto_figure",
            )
        self.world_model.record_stage_history(
            experiment_id=experiment_id,
            cycle=cycle_number,
            stage=governance_stage,
        )
        stage_history_entries = self.world_model.stage_history(experiment_id)
        property_graph_summary = self.world_model.property_graph_summary(experiment_id)
        rag_citations = (
            property_graph_summary.get("citations") if property_graph_summary else None
        )
        adaptation_figures = reporting.collect_adaptation_figures(
            self.config.reporting_dir
        )
        anchor_candidates = (
            ("preference_pairs", preference_pairs_anchor),
            ("p3_summary", p3_summary_anchor),
            ("trajectory", trajectory_anchor),
        )
        positioning_artifacts = {
            name: anchor for name, anchor in anchor_candidates if anchor is not None
        }
        if not positioning_artifacts:
            positioning_artifacts = None
        references = [
            "docs/TASKS_CODEX_MINI.md:200-248",
            "docs/TASKS_CODEX_MINI.md:206-238",
            "docs/MASTER_PLAN_AI_SCIENTIST.md:247-368",
        ]
        for anchor in (preference_pairs_anchor, p3_summary_anchor, trajectory_anchor):
            if anchor:
                references.append(anchor)
        content = reporting.build_cycle_report(
            cycle_index=cycle_index,
            problem=self.config.problem,
            screened=len(screen_results),
            promoted=len(promote_results),
            governance_stage=governance_stage,
            best_metrics=best_eval["metrics"],
            config_snapshot=config_snapshot,
            reproduction_steps=[
                repro_command,
                f"git checkout {git_sha}",
                f"(cd constellaration && git checkout {constellaration_sha})",
                f"tools.{tool_name}(params, stage='{stage_label}')",
            ],
            reproduction_snippet=reproduction_snippet,
            environment_block=env_block,
            pareto_lines=pareto_lines,
            p3_summary={
                "hv_score": p3_summary.hv_score,
                "reference_point": p3_summary.reference_point,
                "feasible_count": p3_summary.feasible_count,
                "archive_size": p3_summary.archive_size,
            },
            statements=statements,
            references=references,
            positioning_artifacts=positioning_artifacts,
            stage_history=stage_history_entries,
            artifact_entries=artifact_entries,
            adaptation_figures=adaptation_figures,
            property_graph_summary=property_graph_summary,
            rag_citations=rag_citations,
            figure_paths=figure_paths,
            out_dir=self.config.reporting_dir,
        )

        title = f"{self.config.problem}_cycle_{cycle_index + 1}"
        report_path = reporting.write_report(
            title, content, out_dir=self.config.reporting_dir
        )

        # Log cache info at cycle end (AoT recommendation for observability)
        if log_cache_stats:
            from ai_scientist import forward_model as fm

            cache_info = fm.get_cache_info()
            print(
                f"[runner][cycle={cycle_number}] Cache stats: "
                f"size={cache_info.get('currsize', 0)}/{cache_info.get('maxsize', 0)} "
                f"hits={cache_info.get('hits', 0)} misses={cache_info.get('misses', 0)} "
                f"rate={100 * cache_info.get('hits', 0) / (cache_info.get('hits', 0) + cache_info.get('misses', 0) + 1e-9):.1f}%"
            )

        return CycleResult(
            cycle_index=cycle_index,
            candidates_evaluated=len(screen_results),
            candidates_promoted=len(promote_results),
            best_objective=best_eval.get("objective"),
            hypervolume=p3_summary.hv_score,
            feasibility_rate=feasibility_rate,
            report_path=report_path,
            best_eval=best_eval,
            p3_summary=p3_summary,
        )

    def _run_alm_optimization(
        self,
        active_cfg,
        cycle_index,
        experiment_id,
        surrogate_model,
        suggested_params,
        optimizer_mode,
        alm_settings_overrides,
        active_budgets,
        cycle_number,
        verbose,
        cycle_start,
    ) -> list[dict[str, Any]]:
        """Extracted ALM optimization logic."""
        if verbose:
            print(
                f"[runner][cycle={cycle_number}] {optimizer_mode if optimizer_mode != 'default' else self.config.optimizer_backend} optimizer mode active (LEGACY ALM path)."
            )

        initial_params_map: Mapping[str, Any]
        if suggested_params and suggested_params[0]:
            initial_params_map = suggested_params[0]
        else:
            if active_cfg.initialization_strategy == "nae":
                if verbose:
                    print("[runner] Using NAE for initial ALM design.")
                initial_params_map = _generate_nae_candidate_params(
                    active_cfg.boundary_template
                )
            else:
                if verbose:
                    print("[runner] Using template for initial ALM design.")
                initial_params_map = _build_template_params_for_alm(
                    active_cfg.boundary_template
                )

        boundary_obj = tools.make_boundary_from_params(initial_params_map)

        max_poloidal = active_cfg.boundary_template.n_poloidal_modes - 1
        max_toroidal = (active_cfg.boundary_template.n_toroidal_modes - 1) // 2

        boundary_obj = surface_module.set_max_mode_numbers(
            surface=boundary_obj,
            max_poloidal_mode=max_poloidal,
            max_toroidal_mode=max_toroidal,
        )

        mask = surface_module.build_mask(
            boundary_obj,
            max_poloidal_mode=max_poloidal,
            max_toroidal_mode=max_toroidal,
        )

        initial_guess, unravel_and_unmask_fn = pytree.mask_and_ravel(
            pytree=boundary_obj,
            mask=mask,
        )

        scale = surface_module.compute_infinity_norm_spectrum_scaling_fun(
            poloidal_modes=boundary_obj.poloidal_modes.flatten(),
            toroidal_modes=boundary_obj.toroidal_modes.flatten(),
            alpha=0.5,
        ).reshape(boundary_obj.poloidal_modes.shape)
        scale = jnp.array(np.concatenate([scale[mask.r_cos], scale[mask.z_sin]]))

        x0 = jnp.array(initial_guess) / scale

        fm_settings = tools._settings_for_stage(active_cfg.fidelity_ladder.promote)

        alm_settings_obj = AugmentedLagrangianSettings(**alm_settings_overrides)

        sa_alm_predictor: (
            Callable[[Mapping[str, Any]], Tuple[float, Sequence[float]]] | None
        ) = None
        if optimizer_mode == "sa-alm" or self.config.optimizer_backend == "sa-alm":
            problem = (self.config.problem or "").lower()
            # P3 is multi-objective: surrogate predicts hypervolume contribution for
            # ranking, not the single physics objective (aspect_ratio). This is
            # intentional—see forward_model.compute_objective() docstring.
            # Use SSOT function to determine target (avoids semantic drift).
            target_column = get_training_target(self.config.problem).value
            history = self.world_model.surrogate_training_data(
                target=target_column,
                problem=self.config.problem,
                experiment_id=experiment_id,
            )
            metrics_list: tuple[Mapping[str, Any], ...] = ()
            target_values: tuple[float, ...] = ()
            if history:
                metrics_list, target_values = zip(*history)
                if surrogate_model.should_retrain(len(history), cycle=cycle_number):
                    surrogate_model.fit(
                        metrics_list,
                        target_values,
                        minimize_objective=(problem == "p1"),
                        cycle=cycle_number,
                    )
                    if isinstance(surrogate_model, NeuralOperatorSurrogate):
                        self.world_model.register_surrogate(
                            experiment_id=experiment_id,
                            cycle=cycle_number,
                            backend_type="neural_operator_ensemble",
                            training_samples=len(history),
                            model_hash=f"ensemble_n{surrogate_model._n_ensembles}_c{cycle_number}",
                            weights_path="memory_only",
                            commit=True,
                        )

            def surrogate_predictor(
                params: Mapping[str, Any],
            ) -> Tuple[float, Sequence[float]]:
                dummy_candidate = {"candidate_params": params}
                p_key = (self.config.problem or "").lower()
                predicted_list = surrogate_model.rank_candidates(
                    [dummy_candidate],
                    minimize_objective=p_key.startswith("p1"),
                    problem=p_key or "p3",
                )
                predicted = predicted_list[0]

                mhd = (
                    predicted.predicted_mhd
                    if predicted.predicted_mhd is not None
                    else 0.0
                )
                qi = (
                    predicted.predicted_qi
                    if predicted.predicted_qi is not None
                    else 1.0
                )

                if predicted.predicted_elongation is not None:
                    elongation = predicted.predicted_elongation
                else:
                    try:
                        import torch

                        from ai_scientist.optim import geometry

                        r_cos_list = params.get("r_cos")
                        z_sin_list = params.get("z_sin")
                        nfp = params.get("n_field_periods", 1)

                        if r_cos_list is not None and z_sin_list is not None:
                            r_cos_t = torch.tensor(
                                r_cos_list, dtype=torch.float32
                            ).unsqueeze(0)
                            z_sin_t = torch.tensor(
                                z_sin_list, dtype=torch.float32
                            ).unsqueeze(0)
                            elongation = float(
                                geometry.elongation_isoperimetric(
                                    r_cos_t, z_sin_t, nfp
                                ).item()
                            )
                        else:
                            elongation = 1.0
                    except Exception:
                        elongation = 1.0

                if p_key.startswith("p1"):
                    predicted_alm_constraints = [0.0, 0.0, 0.0]
                elif p_key.startswith("p2"):
                    c_elo = max(0.0, elongation - 5.0)
                    qi_log = math.log10(qi) if qi > 0 else 10.0
                    c_qi = max(0.0, qi_log - (-4.0))
                    predicted_alm_constraints = [0.0, 0.0, 0.0, c_elo, c_qi]
                else:
                    c_mhd = max(0.0, -mhd)
                    qi_log = math.log10(qi) if qi > 0 else 10.0
                    c_qi = max(0.0, qi_log - (-3.5))
                    predicted_alm_constraints = [0.0, 0.0, c_mhd, 0.0, c_qi]

                return float(predicted.predicted_objective), predicted_alm_constraints

            sa_alm_predictor = surrogate_predictor

        (initial_objective, initial_constraints), _ = _objective_constraints(
            x0,
            scale,
            unravel_and_unmask_fn,
            fm_settings,
            self.config.problem,
            predictor=sa_alm_predictor,
        )

        state = AugmentedLagrangianState(
            x=jnp.copy(x0),
            multipliers=jnp.zeros_like(initial_constraints),
            penalty_parameters=jnp.ones_like(initial_constraints) * 1.0,
            objective=initial_objective,
            constraints=initial_constraints,
            bounds=jnp.ones_like(x0) * 0.1,
        )

        alm_candidates: list[dict[str, Any]] = []
        budget_per_step = 8
        num_alm_steps = max(1, active_budgets.screen_evals_per_cycle // budget_per_step)

        p_key = (self.config.problem or "").lower()
        # Use centralized constraint registry for consistent naming
        from ai_scientist.constraints import get_constraint_names

        constraint_names = get_constraint_names(p_key)

        import nevergrad

        for k in range(num_alm_steps):
            if (
                self.config.surrogate.backend == "neural_operator"
                and isinstance(surrogate_model, NeuralOperatorSurrogate)
                and surrogate_model._trained
            ):
                from ai_scientist.optim import differentiable

                alm_state_dict = {
                    "multipliers": np.array(state.multipliers),
                    "penalty_parameters": np.array(state.penalty_parameters),
                }
                x_new_np = differentiable.optimize_alm_inner_loop(
                    x_initial=np.array(state.x),
                    scale=np.array(scale),
                    surrogate=surrogate_model,
                    alm_state=alm_state_dict,
                    n_field_periods_val=initial_params_map.get("n_field_periods", 1),
                    problem=self.config.problem,
                    steps=budget_per_step,
                    target=get_training_target(self.config.problem or "p3"),
                )
                x_new = jnp.array(x_new_np)

                (obj_new, constr_new), metrics = _objective_constraints(
                    x_new,
                    scale,
                    unravel_and_unmask_fn,
                    fm_settings,
                    self.config.problem,
                    predictor=sa_alm_predictor,
                )
                cand_boundary = unravel_and_unmask_fn(jnp.asarray(x_new * scale))
                cand_params = {
                    "r_cos": np.array(cand_boundary.r_cos).tolist(),
                    "z_sin": np.array(cand_boundary.z_sin).tolist(),
                    "n_field_periods": cand_boundary.n_field_periods,
                    "is_stellarator_symmetric": cand_boundary.is_stellarator_symmetric,
                }
                if metrics:
                    p3_margins = tools.compute_constraint_margins(
                        metrics, self.config.problem, stage=fm_settings.stage
                    )
                    max_viol = tools._max_violation(p3_margins)
                else:
                    max_viol = float(jnp.max(constr_new))

                alm_candidates.append(
                    {
                        "seed": self.config.random_seed
                        + cycle_index * 10000
                        + len(alm_candidates),
                        "params": cand_params,
                        "design_hash": tools.design_hash(cand_params),
                        "constraint_distance": max_viol,
                        "source": "sa-alm_diff",
                        "evaluation": {
                            "metrics": metrics.model_dump() if metrics else {},
                            "feasibility": max_viol,
                            "stage": active_cfg.fidelity_ladder.promote,
                        },
                    }
                )

            else:
                parametrization = nevergrad.p.Array(
                    init=np.array(state.x),
                    lower=np.array(state.x - state.bounds),
                    upper=np.array(state.x + state.bounds),
                )

                oracle = nevergrad.optimizers.NGOpt(
                    parametrization=parametrization,
                    budget=budget_per_step,
                    num_workers=1,
                )
                oracle.suggest(np.array(state.x))

                for _ in range(budget_per_step):
                    candidate = oracle.ask()

                    (obj, constr), metrics = _objective_constraints(
                        jnp.array(candidate.value),
                        scale,
                        unravel_and_unmask_fn,
                        fm_settings,
                        self.config.problem,
                        predictor=sa_alm_predictor,
                    )

                    cand_boundary = unravel_and_unmask_fn(
                        jnp.asarray(candidate.value * scale)
                    )
                    cand_params = {
                        "r_cos": np.array(cand_boundary.r_cos).tolist(),
                        "z_sin": np.array(cand_boundary.z_sin).tolist(),
                        "n_field_periods": cand_boundary.n_field_periods,
                        "is_stellarator_symmetric": cand_boundary.is_stellarator_symmetric,
                    }

                    if metrics:
                        p3_margins = tools.compute_constraint_margins(
                            metrics, self.config.problem, stage=fm_settings.stage
                        )
                        max_viol = tools._max_violation(p3_margins)
                    else:
                        max_viol = float(jnp.max(constr))

                    alm_candidates.append(
                        {
                            "seed": self.config.random_seed
                            + cycle_index * 10000
                            + len(alm_candidates),
                            "params": cand_params,
                            "design_hash": tools.design_hash(cand_params),
                            "constraint_distance": max_viol,
                            "source": optimizer_mode,
                            "evaluation": {
                                "metrics": metrics.model_dump() if metrics else {},
                                "feasibility": max_viol,
                                "stage": active_cfg.fidelity_ladder.promote,
                            },
                        }
                    )

                    loss = augmented_lagrangian_function(obj, constr, state).item()
                    oracle.tell(candidate, loss)

                recommendation = oracle.provide_recommendation()
                x_new = jnp.array(recommendation.value)

            if (
                optimizer_mode == "sa-alm" or self.config.optimizer_backend == "sa-alm"
            ) and k % 2 == 0:
                if verbose:
                    print(
                        f"[runner] SA-ALM: Verifying best candidate with true forward model (step {k})."
                    )
                (obj_new, constr_new), true_metrics = _objective_constraints(
                    x_new,
                    scale,
                    unravel_and_unmask_fn,
                    fm_settings,
                    self.config.problem,
                    predictor=None,
                )
                verified_params_obj = unravel_and_unmask_fn(x_new * scale)
                verified_params = {
                    "r_cos": np.asarray(verified_params_obj.r_cos).tolist(),
                    "z_sin": np.asarray(verified_params_obj.z_sin).tolist(),
                    "n_field_periods": initial_params_map.get("n_field_periods", 1),
                    "is_stellarator_symmetric": initial_params_map.get(
                        "is_stellarator_symmetric", True
                    ),
                }

                if true_metrics:
                    self.world_model.log_candidate(
                        experiment_id=experiment_id,
                        problem=self.config.problem,
                        params=verified_params,
                        seed=self.config.random_seed + cycle_index * 100000 + k,
                        status=active_cfg.fidelity_ladder.promote,
                        evaluation={
                            "objective": obj_new.item(),
                            "feasibility": float(jnp.max(constr_new)),
                            "stage": active_cfg.fidelity_ladder.promote,
                            "metrics": true_metrics.model_dump(),
                        },
                        design_hash=tools.design_hash(verified_params),
                        commit=True,
                    )
                state = update_augmented_lagrangian_state(
                    x=x_new,
                    objective=obj_new,
                    constraints=constr_new,
                    state=state,
                    settings=alm_settings_obj,
                )
            else:
                (obj_new, constr_new), _ = _objective_constraints(
                    x_new,
                    scale,
                    unravel_and_unmask_fn,
                    fm_settings,
                    self.config.problem,
                    predictor=sa_alm_predictor,
                )
                state = update_augmented_lagrangian_state(
                    x=x_new,
                    objective=obj_new,
                    constraints=constr_new,
                    state=state,
                    settings=alm_settings_obj,
                )

            for i, c_name in enumerate(constraint_names):
                self.world_model.log_alm_state(
                    experiment_id=experiment_id,
                    cycle=cycle_number,
                    step_index=k,
                    constraint_name=c_name,
                    multiplier_value=float(state.multipliers[i]),
                    penalty_parameter=float(state.penalty_parameters[i]),
                    violation_magnitude=float(state.constraints[i]),
                    commit=True,
                )

        return list(alm_candidates)


# --- Helpers moved from runner.py ---


def _expand_matrix_to_mode(
    matrix: np.ndarray,
    max_poloidal_mode: int,
    max_toroidal_mode: int,
) -> np.ndarray:
    """Expand or validate Fourier coefficient matrix to target mode numbers.

    Canonicalization Fix: Zero-pads if smaller, logs warning if larger.
    The center column (n=0) is preserved during expansion.
    """
    target_rows = max_poloidal_mode + 1
    target_cols = 2 * max_toroidal_mode + 1

    if matrix.shape == (target_rows, target_cols):
        return matrix

    logging.info(
        "[canonicalization] Expanding seed matrix from (%d, %d) to (%d, %d)",
        matrix.shape[0],
        matrix.shape[1],
        target_rows,
        target_cols,
    )

    expanded = np.zeros((target_rows, target_cols), dtype=float)

    # Copy existing values, centering the toroidal modes
    seed_ntor = (matrix.shape[1] - 1) // 2
    target_ntor = max_toroidal_mode
    col_offset = target_ntor - seed_ntor

    rows_to_copy = min(matrix.shape[0], target_rows)
    cols_to_copy = min(matrix.shape[1], target_cols)

    for m in range(rows_to_copy):
        for n_idx in range(cols_to_copy):
            target_col = n_idx + col_offset
            if 0 <= target_col < target_cols:
                expanded[m, target_col] = matrix[m, n_idx]

    return expanded


def _load_seed_boundary(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    cached = _BOUNDARY_SEED_CACHE.get(resolved)
    if cached is None:
        raw = json.loads(resolved.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            if not raw:
                raise ValueError(f"Seed file {path} is an empty list.")
            raw = raw[0]

        payload: dict[str, Any] = {
            "r_cos": np.asarray(raw["r_cos"], dtype=float),
            "z_sin": np.asarray(raw["z_sin"], dtype=float),
            "r_sin": np.asarray(raw["r_sin"], dtype=float)
            if raw.get("r_sin") is not None
            else None,
            "z_cos": np.asarray(raw["z_cos"], dtype=float)
            if raw.get("z_cos") is not None
            else None,
            "n_field_periods": int(raw.get("n_field_periods") or raw.get("nfp") or 1),
            "is_stellarator_symmetric": bool(raw.get("is_stellarator_symmetric", True)),
        }
        _BOUNDARY_SEED_CACHE[resolved] = payload
        cached = payload
    return {
        key: (np.array(value, copy=True) if isinstance(value, np.ndarray) else value)
        for key, value in cached.items()
    }


def _build_template_params_for_alm(
    template: ai_config.BoundaryTemplateConfig,
) -> Mapping[str, Any]:
    n_poloidal = template.n_poloidal_coefficients  # mpol + 1
    n_toroidal = template.n_toroidal_coefficients  # 2*ntor + 1 (odd)
    center_idx = template.max_toroidal_mode  # ntor
    r_cos = []
    z_sin = []
    for pol in range(n_poloidal):
        r_row = []
        z_row = []
        for tor in range(n_toroidal):
            r_val = (
                template.base_major_radius if pol == 0 and tor == center_idx else 0.0
            )
            z_val = (
                template.base_minor_radius
                if pol == 1 and tor == center_idx and n_poloidal > 1
                else 0.0
            )
            r_row.append(r_val)
            z_row.append(z_val)
        r_cos.append(r_row)
        z_sin.append(z_row)
    return {
        "r_cos": r_cos,
        "z_sin": z_sin,
        "n_field_periods": template.n_field_periods,
        "is_stellarator_symmetric": True,
    }


def _generate_nae_candidate_params(
    template: ai_config.BoundaryTemplateConfig,
    aspect_ratio: float = 8.0,
    max_elongation: float = 1.5,
    rotational_transform: float = 0.4,
    mirror_ratio: float = 0.5,
) -> Mapping[str, Any]:
    nae_boundary = generate_nae(
        aspect_ratio=aspect_ratio,
        max_elongation=max_elongation,
        rotational_transform=rotational_transform,
        mirror_ratio=mirror_ratio,
        n_field_periods=template.n_field_periods,
        max_poloidal_mode=template.max_poloidal_mode,
        max_toroidal_mode=template.max_toroidal_mode,
    )
    return {
        "r_cos": np.asarray(nae_boundary.r_cos).tolist(),
        "z_sin": np.asarray(nae_boundary.z_sin).tolist(),
        "n_field_periods": nae_boundary.n_field_periods,
        "is_stellarator_symmetric": nae_boundary.is_stellarator_symmetric,
    }


def _objective_constraints(
    x: jnp.ndarray,
    scale: jnp.ndarray,
    unravel_and_unmask_fn: Any,
    settings: forward_model.ConstellarationSettings,
    problem_type: str,
    predictor: Callable[[Mapping[str, Any]], Tuple[float, Sequence[float]]]
    | None = None,
) -> tuple[
    tuple[jnp.ndarray, jnp.ndarray], forward_model.ConstellarationMetrics | None
]:
    boundary_obj = unravel_and_unmask_fn(jnp.asarray(x * scale))

    boundary_params = {
        "r_cos": np.asarray(boundary_obj.r_cos).tolist(),
        "z_sin": np.asarray(boundary_obj.z_sin).tolist(),
        "n_field_periods": boundary_obj.n_field_periods,
        "is_stellarator_symmetric": boundary_obj.is_stellarator_symmetric,
    }

    if predictor:
        predicted_objective, predicted_constraints = predictor(boundary_params)
        return (
            (
                jnp.asarray(predicted_objective, dtype=jnp.float32),
                jnp.asarray(predicted_constraints, dtype=jnp.float32),
            ),
            None,
        )

    with tempfile.TemporaryDirectory() as _:
        metrics = None
        try:
            metrics, _ = forward_model.forward_model(
                boundary=boundary_obj,
                settings=settings,
            )
        except Exception as _:
            pass

        if metrics is None:
            objective = jnp.array(NAN_TO_HIGH_VALUE)
            p_key = (problem_type or "").lower()
            if p_key.startswith("p1"):
                c_size = 3
            else:
                c_size = 5
            constraints = jnp.ones(c_size) * NAN_TO_HIGH_VALUE
        else:
            if problem_type.lower().startswith("p1"):
                objective = jnp.array(metrics.max_elongation)
            elif problem_type.lower().startswith("p2"):
                # P2: maximize gradient scale length
                objective = jnp.array(
                    metrics.minimum_normalized_magnetic_gradient_scale_length
                )
            else:
                # P3: minimize aspect ratio
                objective = jnp.array(metrics.aspect_ratio)

            margins = tools.compute_constraint_margins(
                metrics, problem_type, stage="high"
            )
            constraints = jnp.array(list(margins.values()))

        return ((objective, constraints), metrics)


def _generate_candidate_params(
    template: ai_config.BoundaryTemplateConfig, seed: int
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    seed_data: dict[str, Any] | None = None
    seed_path = template.seed_path
    if seed_path is None:
        default_seed = Path("configs/seeds/rotating_ellipse_p3.json")
        seed_path = default_seed if default_seed.exists() else None

    if seed_path is not None:
        seed_data = _load_seed_boundary(seed_path)
        # Canonicalization Fix: Expand seed matrices to template's max modes
        r_cos = _expand_matrix_to_mode(
            seed_data["r_cos"],
            template.max_poloidal_mode,
            template.max_toroidal_mode,
        )
        z_sin = _expand_matrix_to_mode(
            seed_data["z_sin"],
            template.max_poloidal_mode,
            template.max_toroidal_mode,
        )
        r_sin = seed_data["r_sin"]
        if r_sin is not None:
            r_sin = _expand_matrix_to_mode(
                r_sin,
                template.max_poloidal_mode,
                template.max_toroidal_mode,
            )
        z_cos = seed_data["z_cos"]
        if z_cos is not None:
            z_cos = _expand_matrix_to_mode(
                z_cos,
                template.max_poloidal_mode,
                template.max_toroidal_mode,
            )
        n_field_periods = int(seed_data["n_field_periods"])
        is_stellarator_symmetric = bool(seed_data["is_stellarator_symmetric"])

    else:
        base_surface = generate_rotating_ellipse(
            aspect_ratio=4.0,
            elongation=1.5,
            rotational_transform=1.2,
            n_field_periods=template.n_field_periods,
        )
        max_poloidal = max(1, template.max_poloidal_mode)
        max_toroidal = max(1, template.max_toroidal_mode)
        expanded = surface_module.set_max_mode_numbers(
            base_surface,
            max_poloidal_mode=max_poloidal,
            max_toroidal_mode=max_toroidal,
        )
        r_cos = np.asarray(expanded.r_cos, dtype=float)
        z_sin = np.asarray(expanded.z_sin, dtype=float)
        center_idx = r_cos.shape[1] // 2
        r_cos[0, center_idx] = template.base_major_radius
        if r_cos.shape[0] > 1:
            z_sin[1, center_idx] = template.base_minor_radius

        r_sin = None
        z_cos = None
        n_field_periods = template.n_field_periods
        is_stellarator_symmetric = True

    r_cos += rng.normal(scale=template.perturbation_scale, size=r_cos.shape)
    z_sin += rng.normal(scale=template.perturbation_scale / 2, size=z_sin.shape)
    if seed_data is not None:
        if r_sin is not None:
            r_sin += rng.normal(scale=template.perturbation_scale / 2, size=r_sin.shape)
        if z_cos is not None:
            z_cos += rng.normal(scale=template.perturbation_scale / 2, size=z_cos.shape)

    if is_stellarator_symmetric:
        n_cols = r_cos.shape[1]
        center_idx = n_cols // 2
        if center_idx > 0:
            r_cos[0, :center_idx] = 0.0
        z_sin[0, :] = 0.0

    params: dict[str, Any] = {
        "r_cos": r_cos.tolist(),
        "z_sin": z_sin.tolist(),
        "n_field_periods": template.n_field_periods or n_field_periods,
        "is_stellarator_symmetric": is_stellarator_symmetric,
    }

    if r_sin is not None:
        params["r_sin"] = r_sin.tolist()

    if z_cos is not None:
        params["z_cos"] = z_cos.tolist()

    return {
        "seed": seed,
        "params": params,
        "design_hash": tools.design_hash(params),
    }


def _propose_p3_candidates_for_cycle(
    cfg: ai_config.ExperimentConfig,
    cycle_index: int,
    world_model: memory.WorldModel,
    experiment_id: int,
    *,
    screen_budget: int,
    total_candidates: int | None = None,
    prev_feasibility_rate: float | None = None,
    suggested_params: list[dict[str, Any]] | None = None,
    generative_model: GenerativeDesignModel | DiffusionDesignModel | None = None,
) -> tuple[list[dict[str, Any]], int, int, int]:
    pool_size = total_candidates if total_candidates is not None else screen_budget
    total_candidates = int(pool_size)
    if total_candidates <= 0:
        return [], 0, 0, 0

    agent_results: list[dict[str, Any]] = []
    if suggested_params:
        for idx, params in enumerate(suggested_params):
            agent_results.append(
                {
                    "seed": cfg.random_seed + cycle_index * 1000 + idx,
                    "params": params,
                    "design_hash": tools.design_hash(params),
                    "constraint_distance": 0.0,
                    "source": "agent_suggestion",
                }
            )

    remaining_candidates = max(0, total_candidates - len(agent_results))
    if remaining_candidates == 0:
        return agent_results, 0, 0, 0

    gen_results: list[dict[str, Any]] = []
    if generative_model and generative_model._trained:
        gen_target = int(remaining_candidates * 0.4)
        if gen_target > 0:
            seed_base = cfg.random_seed + cycle_index * 20000
            if isinstance(generative_model, DiffusionDesignModel):
                target_metrics = {
                    "aspect_ratio": 8.0,
                    "minimum_normalized_magnetic_gradient_scale_length": 20.0,
                    "max_elongation": 1.5,
                    "edge_rotational_transform_over_n_field_periods": 0.4,
                }
                gen_results = cast(
                    list[dict[str, Any]],
                    generative_model.sample(
                        gen_target, target_metrics=target_metrics, seed=seed_base
                    ),
                )
            else:
                gen_results = cast(
                    list[dict[str, Any]],
                    generative_model.sample(gen_target, seed=seed_base),
                )

    remaining_candidates = max(0, remaining_candidates - len(gen_results))

    mix = cfg.proposal_mix
    exploration_weight = mix.exploration_ratio
    if prev_feasibility_rate is not None:
        exploration_weight *= 1.0 - min(1.0, max(0.0, prev_feasibility_rate))

    ratio_sum = mix.constraint_ratio + exploration_weight
    if ratio_sum <= 0.0:
        sampler_target = 0
    else:
        sampler_target = int(
            round(remaining_candidates * (mix.constraint_ratio / ratio_sum))
        )
    sampler_target = min(sampler_target, remaining_candidates)

    stage_limit = max(remaining_candidates * 4, 16)
    stage_records = world_model.recent_stage_candidates(
        experiment_id=experiment_id,
        problem=cfg.problem,
        stage=cfg.fidelity_ladder.promote,
        limit=int(stage_limit),
    )

    if stage_records and sampler_target > 0:
        base_designs = [record[0] for record in stage_records]
        feasibilities = [max(0.0, float(record[1])) for record in stage_records]
        max_feas = max(feasibilities, default=0.0)
        normalized_distances = [
            0.0 if max_feas <= 0.0 else min(1.0, value / max_feas)
            for value in feasibilities
        ]
        rng_seed = cfg.random_seed + cycle_index + total_candidates
        sampler_params = tools.normalized_constraint_distance_sampler(
            base_designs,
            normalized_distances=normalized_distances,
            proposal_count=sampler_target,
            jitter_scale=mix.jitter_scale,
            rng=np.random.default_rng(rng_seed),
            include_distances=True,
        )
    else:
        sampler_params = []

    candidate_seeds = [
        cfg.random_seed + cycle_index * total_candidates + i
        for i in range(remaining_candidates)
    ]
    seed_iter = iter(candidate_seeds)
    sampler_results: list[dict[str, Any]] = []
    for params in sampler_params:
        try:
            seed = next(seed_iter)
        except StopIteration:
            break
        if isinstance(params, Mapping) and "params" in params:
            payload = params.get("params", {})
            constraint_distance = float(
                cast(Any, params.get("normalized_constraint_distance", 0.0))
            )
        else:
            payload = params
            constraint_distance = 0.0
        sampler_results.append(
            {
                "seed": seed,
                "params": payload,
                "design_hash": tools.design_hash(cast(dict[str, Any], payload)),
                "constraint_distance": constraint_distance,
                "source": "constraint_sampler",
            }
        )

    remaining = remaining_candidates - len(sampler_results)
    remaining_seeds = []
    for _ in range(remaining):
        try:
            remaining_seeds.append(next(seed_iter))
        except StopIteration:
            break

    random_results: list[dict[str, Any]] = []

    if cfg.proposal_mix.sampler_type == "near_axis":
        try:
            sampler = NearAxisSampler(cfg.boundary_template)
            random_results = cast(
                list[dict[str, Any]], sampler.generate(remaining_seeds)
            )
        except Exception as exc:
            print(
                f"[runner] NearAxisSampler failed: {exc}; falling back to standard random"
            )
            random_results = []

    used_seeds = {c["seed"] for c in random_results}
    fallback_seeds = [s for s in remaining_seeds if s not in used_seeds]

    for seed in fallback_seeds:
        random_results.append(
            {
                **_generate_candidate_params(cfg.boundary_template, seed),
                "constraint_distance": 1.0,
                "source": "random",
            }
        )

    candidates = agent_results + gen_results + sampler_results + random_results
    return (
        cast(list[dict[str, Any]], candidates),
        len(sampler_results),
        len(random_results),
        len(gen_results),
    )


def _surrogate_candidate_pool_size(
    screen_budget: int,
    surrogate_pool_multiplier: float,
) -> int:
    if screen_budget <= 0:
        return 0
    multiplier = max(1.0, surrogate_pool_multiplier)
    proposed = int(math.ceil(screen_budget * multiplier))
    return max(proposed, screen_budget)


def _surrogate_rank_screen_candidates(
    cfg: ai_config.ExperimentConfig,
    screen_budget: int,
    candidates: list[dict[str, Any]],
    world_model: memory.WorldModel,
    surrogate_model: BaseSurrogate,
    *,
    cycle: int = 0,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    if not candidates or screen_budget <= 0:
        return candidates

    problem = (cfg.problem or "").lower()
    # P3 is multi-objective: surrogate predicts hypervolume contribution for
    # ranking, not the single physics objective (aspect_ratio). This is
    # intentional—see forward_model.compute_objective() docstring.
    # Use SSOT function to determine target (avoids semantic drift).
    target_column = get_training_target(cfg.problem).value
    minimize_objective = problem == "p1"

    history = world_model.surrogate_training_data(
        target=target_column, problem=cfg.problem
    )
    global _LAST_SURROGATE_FIT_SEC
    _LAST_SURROGATE_FIT_SEC = 0.0

    recent_feasibility = 0.0
    if history:
        recent_window = history[-50:]
        valid = 0
        for m, _ in recent_window:
            f_val = float(m.get("feasibility", float("inf")))
            if f_val <= FEASIBILITY_CUTOFF:
                valid += 1
        recent_feasibility = valid / len(recent_window)

    metrics_list: tuple[Mapping[str, Any], ...] = ()
    target_values: tuple[float, ...] = ()
    if history:
        metrics_list, target_values = zip(*history)
        if surrogate_model.should_retrain(len(history), cycle=cycle):
            start = time.perf_counter()
            surrogate_model.fit(
                metrics_list,
                target_values,
                minimize_objective=minimize_objective,
                cycle=cycle,
            )
            _LAST_SURROGATE_FIT_SEC = time.perf_counter() - start
    else:
        metrics_list = ()
        target_values = ()

    pool_entries: list[dict[str, Any]] = []
    for idx, candidate in enumerate(candidates):
        pool_entries.append(
            {
                "candidate_params": candidate["params"],
                "constraint_distance": candidate.get("constraint_distance", 0.0),
                "source": candidate.get("source"),
                "__surrogate_candidate_index": idx,
            }
        )

    # Compute exploration ratio with UCB decay schedule (Issue #11)
    from ai_scientist.exploration import compute_exploration_ratio_from_config

    ucb_exploration_ratio = compute_exploration_ratio_from_config(
        cycle,
        ucb_exploration_initial=cfg.proposal_mix.ucb_exploration_initial,
        ucb_exploration_final=cfg.proposal_mix.ucb_exploration_final,
        ucb_decay_cycles=cfg.proposal_mix.ucb_decay_cycles,
    )

    ranked_predictions = surrogate_model.rank_candidates(
        pool_entries,
        minimize_objective=minimize_objective,
        exploration_ratio=ucb_exploration_ratio,
        problem=problem,
    )

    restoration_active = len(history) > 10 and recent_feasibility < 0.05

    if restoration_active:
        if verbose:
            print(
                f"[runner][restoration] Feasibility rate {recent_feasibility:.1%}; switching to restoration ranking."
            )

        weights = cfg.constraint_weights

        def _predicted_violation(pred: SurrogatePrediction) -> float:
            mhd_val = pred.predicted_mhd if pred.predicted_mhd is not None else 0.0
            mhd_viol = max(0.0, -mhd_val) * weights.mhd

            qi_val = pred.predicted_qi if pred.predicted_qi is not None else 1.0
            qi_viol = max(0.0, qi_val) * weights.qi

            if pred.predicted_elongation is not None:
                elo_val = pred.predicted_elongation
            else:
                idx = int(pred.metadata.get("__surrogate_candidate_index", -1))
                elo_val = 1.0
                if idx >= 0:
                    try:
                        import torch

                        from ai_scientist.optim import geometry

                        cand_params = candidates[idx]["params"]
                        r_cos_list = cand_params.get("r_cos")
                        z_sin_list = cand_params.get("z_sin")
                        nfp = cand_params.get("n_field_periods", 1)

                        if r_cos_list is not None and z_sin_list is not None:
                            r_cos_t = torch.tensor(
                                r_cos_list, dtype=torch.float32
                            ).unsqueeze(0)
                            z_sin_t = torch.tensor(
                                z_sin_list, dtype=torch.float32
                            ).unsqueeze(0)
                            elo_val = float(
                                geometry.elongation_isoperimetric(
                                    r_cos_t, z_sin_t, nfp
                                ).item()
                            )
                    except Exception:
                        pass

            elo_viol = max(0.0, elo_val) * weights.elongation
            return mhd_viol + qi_viol + elo_viol

        ranked_predictions.sort(key=_predicted_violation)

    elif cfg.problem.lower() == "p2":
        prob_threshold = max(0.0, min(1.0, cfg.adaptive_budgets.feasibility_target))
        filtered_predictions = [
            prediction
            for prediction in ranked_predictions
            if prediction.prob_feasible >= prob_threshold
        ]
        if filtered_predictions:
            ranked_predictions = filtered_predictions

    selected: list[dict[str, Any]] = []
    needed = screen_budget
    for prediction in ranked_predictions:
        idx = int(prediction.metadata.get("__surrogate_candidate_index", -1))
        if idx < 0:
            continue
        selected.append(candidates[idx])
        if len(selected) >= needed:
            break

    if not selected:
        return candidates[:needed]

    if verbose:
        dropped = len(candidates) - len(selected)
        print(
            f"[runner][surrogate] trained on {len(history)} rows, "
            f"selected {len(selected)}/{len(candidates)} candidates (dropped {dropped})"
        )

    return selected


def _stage_rank(stage: str | None, promote_stage: str) -> int:
    if stage == promote_stage:
        return 2
    if stage:
        return 1
    return 0


def _feasibility_value(entry: Mapping[str, Any]) -> float:
    return float(entry["evaluation"].get("feasibility", float("inf")))


def _oriented_objective(entry: Mapping[str, Any]) -> float:
    evaluation = entry["evaluation"]
    objective = evaluation.get("objective")
    if objective is None:
        return float("inf")
    minimize = evaluation.get("minimize_objective", True)
    value = float(objective)
    return value if minimize else -value


def _prefer_entry(
    current: Mapping[str, Any],
    candidate: Mapping[str, Any],
    promote_stage: str,
) -> Mapping[str, Any]:
    current_rank = _stage_rank(current["evaluation"].get("stage"), promote_stage)
    candidate_rank = _stage_rank(candidate["evaluation"].get("stage"), promote_stage)
    if candidate_rank > current_rank:
        return candidate
    if candidate_rank < current_rank:
        return current

    current_feas = _feasibility_value(current)
    candidate_feas = _feasibility_value(candidate)
    if candidate_feas < current_feas:
        return candidate
    if candidate_feas > current_feas:
        return current

    current_obj = _oriented_objective(current)
    candidate_obj = _oriented_objective(candidate)
    if candidate_obj < current_obj:
        return candidate
    return current


def _close_metric(value_a: float | None, value_b: float | None) -> bool:
    if value_a is None or value_b is None:
        return value_a is None and value_b is None
    return math.isclose(value_a, value_b, rel_tol=1e-3, abs_tol=1e-3)


def _verify_best_claim(
    world_model: WorldModelLike,
    experiment_id: int,
    cycle_number: int,
    best_entry: Mapping[str, Any],
    best_eval: Mapping[str, Any],
    evaluation_fn: ProblemEvaluator,
    tool_name: str,
    best_seed: int,
    git_sha: str,
    reproduction_command: str,
    *,
    stage: str,
    metrics_id: int | None,
) -> str:
    tool_input = {"params": best_entry["params"], "stage": stage}
    replay_eval = evaluation_fn(
        best_entry["params"],
        stage=stage,
        use_cache=False,
    )
    differences: list[str] = []
    for metric in ("objective", "feasibility", "hv"):
        if not _close_metric(best_eval.get(metric), replay_eval.get(metric)):
            differences.append(metric)
    status = "SUPPORTED" if not differences else "REFUTED"
    statement_text = f"Replayed {tool_name} evaluation for design {best_entry.get('design_hash', '')[:8]} at stage {stage}."
    world_model.log_statement(
        experiment_id=experiment_id,
        cycle=cycle_number,
        stage=stage,
        text=statement_text,
        status=status,
        tool_name=tool_name,
        tool_input=tool_input,
        metrics_id=metrics_id,
        seed=best_seed,
        git_sha=git_sha,
        repro_cmd=reproduction_command,
    )
    print(
        f"[runner][verifier] statement status={status} differences={differences} for cycle {cycle_number}"
    )
    return status


def _latest_evaluations_by_design(
    aggregated: Sequence[Mapping[str, Any]], promote_stage: str
) -> dict[str, Mapping[str, Any]]:
    latest: dict[str, Mapping[str, Any]] = {}
    for entry in aggregated:
        design_hash = entry.get("design_hash")
        if not design_hash:
            continue
        existing = latest.get(design_hash)
        if existing is None:
            latest[design_hash] = entry
            continue
        latest[design_hash] = _prefer_entry(
            existing,
            entry,
            promote_stage,
        )
    return latest


def _persist_pareto_archive(
    *,
    world_model: memory.WorldModel,
    experiment_id: int,
    cycle_number: int,
    problem: str,
    entries_by_design: Mapping[str, Mapping[str, Any]],
    p3_summary: tools.P3Summary,
    git_sha: str,
    constellaration_sha: str,
) -> tuple[Set[str], dict[str, int]]:
    logged_hashes: Set[str] = set()
    archive_rows: list[Mapping[str, Any]] = []
    metrics_by_hash: dict[str, int] = {}
    for entry in p3_summary.pareto_entries:
        design_hash = entry.design_hash
        latest = entries_by_design.get(design_hash)
        if latest is None:
            continue
        evaluation = latest["evaluation"]
        candidate_id, metrics_id = world_model.log_candidate(
            experiment_id=experiment_id,
            problem=problem,
            params=latest["params"],
            seed=int(latest.get("seed", -1)),
            status=evaluation.get("stage", "unknown"),
            evaluation=evaluation,
            design_hash=design_hash,
            commit=False,
        )
        logged_hashes.add(design_hash)
        metrics_by_hash[design_hash] = metrics_id
        settings_json = json.dumps(
            evaluation.get("settings", {}), separators=(",", ":")
        )
        archive_rows.append(
            {
                "design_hash": design_hash,
                "fidelity": evaluation.get("stage", "unknown"),
                "gradient": entry.gradient,
                "aspect": entry.aspect_ratio,
                "metrics_id": metrics_id,
                "git_sha": git_sha,
                "constellaration_sha": constellaration_sha,
                "settings_json": settings_json,
                "seed": int(latest.get("seed", -1)),
            }
        )
        world_model.upsert_pareto(experiment_id, candidate_id)
    if archive_rows:
        world_model.record_pareto_archive(
            experiment_id,
            cycle_number,
            archive_rows,
            commit=False,
        )
    return logged_hashes, metrics_by_hash


def _cache_stats_log_path(report_dir: Path | str) -> Path:
    return Path(report_dir) / "cache_stats.jsonl"


def _observability_log_path(report_dir: Path | str) -> Path:
    return Path(report_dir) / "observability.jsonl"


def _maybe_log_cache_stats(
    cfg: ai_config.ExperimentConfig,
    cycle_index: int,
    stage: str,
    stats: Mapping[str, int],
) -> None:
    entry = {
        "cycle": cycle_index + 1,
        "stage": stage,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stats": stats,
    }
    log_path = _cache_stats_log_path(cfg.reporting_dir)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, separators=(",", ":")) + "\n")


def _vmec_failure_rate(results: Sequence[Mapping[str, Any]]) -> float:
    if not results:
        return 0.0
    total = len(results)
    failures = 0
    for entry in results:
        status = str(entry.get("evaluation", {}).get("vmec_status", "success")).lower()
        if status and status not in {"ok", "success"}:
            failures += 1
    return float(failures) / float(total)


def _log_observability_metrics(
    cfg: ai_config.ExperimentConfig,
    cycle_index: int,
    *,
    hv: float | None,
    feasible_count: int,
    vmec_failure_rate: float,
    retrain_time: float,
    cache_hit_rate: float,
    budget_snapshot: BudgetSnapshot,
) -> None:
    entry = {
        "cycle": cycle_index + 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hv": hv,
        "feasible_count": feasible_count,
        "vmec_failure_rate": vmec_failure_rate,
        "surrogate_retrain_seconds": retrain_time,
        "cache_hit_rate": cache_hit_rate,
        "budget_overrides": {
            "screen_evals_per_cycle": budget_snapshot.screen_evals_per_cycle,
            "promote_top_k": budget_snapshot.promote_top_k,
            "max_high_fidelity": budget_snapshot.max_high_fidelity_evals_per_cycle,
        },
    }
    path = _observability_log_path(cfg.reporting_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, separators=(",", ":")) + "\n")


def _repo_relative(path: Path) -> str | None:
    allowed_prefixes = (
        "docs/",
        "constellaration/",
        "Jr.AI-Scientist/",
        "reports/",
        "tests/",
    )
    try:
        rel = path.resolve().relative_to(Path.cwd()).as_posix()
    except ValueError:
        return None
    for prefix in allowed_prefixes:
        if rel.startswith(prefix):
            return rel
    return None
