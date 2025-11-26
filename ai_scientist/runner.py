"Runner that wires budgets, fidelity decisions, and minimal reporting (Tasks 4.1 + B.*)."

from __future__ import annotations

import argparse # Import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence, Set, Tuple, Callable
import random
import multiprocessing # Import multiprocessing

import numpy as np
import jax.numpy as jnp
import yaml
import nevergrad
import tempfile
from concurrent import futures

from ai_scientist import adapter
from ai_scientist import config as ai_config
from ai_scientist import memory
from ai_scientist import planner as ai_planner
from ai_scientist import rag
from ai_scientist import reporting
from ai_scientist import tools
from ai_scientist.coordinator import Coordinator
from ai_scientist.optim.samplers import NearAxisSampler
from ai_scientist.optim.surrogate import SurrogateBundle, SurrogatePrediction
from ai_scientist.optim.surrogate_v2 import NeuralOperatorSurrogate
from ai_scientist.optim.generative import GenerativeDesignModel
from constellaration.geometry import surface_rz_fourier as surface_module
from constellaration.initial_guess import generate_nae, generate_rotating_ellipse
from constellaration.optimization.augmented_lagrangian import (
    AugmentedLagrangianState,
    AugmentedLagrangianSettings,
    augmented_lagrangian_function,
    update_augmented_lagrangian_state,
)
from constellaration.optimization import augmented_lagrangian_runner # To register PyTree
from constellaration.utils import pytree
from constellaration import problems
from constellaration import forward_model
from orchestration import adaptation as adaptation_helpers

FEASIBILITY_CUTOFF = getattr(tools, "_DEFAULT_RELATIVE_TOLERANCE", 1e-2)
P3_REFERENCE_POINT = getattr(tools, "_P3_REFERENCE_POINT", (1.0, 20.0))
_BOUNDARY_SEED_CACHE: dict[Path, dict[str, Any]] = {}
_LAST_SURROGATE_FIT_SEC = 0.0
NAN_TO_HIGH_VALUE = 10.0


def _create_surrogate(cfg: ai_config.ExperimentConfig) -> SurrogateBundle | NeuralOperatorSurrogate:
    """Factory to create the appropriate surrogate model based on config."""
    if cfg.surrogate_backend == "neural_operator":
        print("[runner] V2 Active: Initializing NeuralOperatorSurrogate (Deep Learning Backend).")
        return NeuralOperatorSurrogate()
    return SurrogateBundle()


def _create_generative_model(cfg: ai_config.ExperimentConfig) -> GenerativeDesignModel | None:
    """Factory to create the generative model if enabled."""
    if cfg.generative.enabled:
        print("[runner] Generative Model Enabled (VAE).")
        return GenerativeDesignModel(
            latent_dim=cfg.generative.latent_dim,
            learning_rate=cfg.generative.learning_rate,
            epochs=cfg.generative.epochs,
            kl_weight=cfg.generative.kl_weight,
        )
    return None


def _load_seed_boundary(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    cached = _BOUNDARY_SEED_CACHE.get(resolved)
    if cached is None:
        raw = json.loads(resolved.read_text(encoding="utf-8"))
        payload: dict[str, Any] = {
            "r_cos": np.asarray(raw["r_cos"], dtype=float),
            "z_sin": np.asarray(raw["z_sin"], dtype=float),
            "r_sin": np.asarray(raw["r_sin"], dtype=float)
            if raw.get("r_sin") is not None
            else None,
            "z_cos": np.asarray(raw["z_cos"], dtype=float)
            if raw.get("z_cos") is not None
            else None,
            "n_field_periods": int(
                raw.get("n_field_periods") or raw.get("nfp") or 1
            ),
            "is_stellarator_symmetric": bool(
                raw.get("is_stellarator_symmetric", True)
            ),
        }
        _BOUNDARY_SEED_CACHE[resolved] = payload
        cached = payload
    return {
        key: (np.array(value, copy=True) if isinstance(value, np.ndarray) else value)
        for key, value in cached.items()
    }


def _build_template_params_for_alm(
    template: ai_config.BoundaryTemplateConfig
) -> Mapping[str, Any]:
    n_poloidal = template.n_poloidal_modes
    n_toroidal = template.n_toroidal_modes
    center_idx = n_toroidal // 2
    r_cos = []
    z_sin = []
    for pol in range(n_poloidal):
        r_row = []
        z_row = []
        for tor in range(n_toroidal):
            r_val = (
                template.base_major_radius
                if pol == 0 and tor == center_idx
                else 0.0
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
    """Generates initial parameters using Near-Axis Expansion (NAE)."""
    nae_boundary = generate_nae(
        aspect_ratio=aspect_ratio,
        max_elongation=max_elongation,
        rotational_transform=rotational_transform,
        mirror_ratio=mirror_ratio,
        n_field_periods=template.n_field_periods,
        max_poloidal_mode=template.n_poloidal_modes -1, # convert to 0-indexed max
        max_toroidal_mode=template.n_toroidal_modes -1, # convert to 0-indexed max
    )
    # Convert SurfaceRZFourier to dictionary format
    return {
        "r_cos": np.asarray(nae_boundary.r_cos).tolist(),
        "z_sin": np.asarray(nae_boundary.z_sin).tolist(),
        "n_field_periods": nae_boundary.n_field_periods,
        "is_stellarator_symmetric": nae_boundary.is_stellarator_symmetric,
    }


# ALM Helpers adapted from constellaration.optimization.augmented_lagrangian_runner

def _objective_constraints(
    x: jnp.ndarray,
    scale: jnp.ndarray,
    unravel_and_unmask_fn: Any, # Callable[[jnp.ndarray], surface_module.SurfaceRZFourier],
    settings: forward_model.ConstellarationSettings,
    problem_type: str,
    predictor: Callable[[Mapping[str, Any]], Tuple[float, Sequence[float]]] | None = None,
) -> tuple[
    tuple[jnp.ndarray, jnp.ndarray], forward_model.ConstellarationMetrics | None
]:
    # Reconstruct boundary_obj from optimized parameters x using unravel_and_unmask_fn
    boundary_obj = unravel_and_unmask_fn(jnp.asarray(x * scale))
    
    # Convert SurfaceRZFourier to dictionary format for predictor
    boundary_params = {
        "r_cos": np.asarray(boundary_obj.r_cos).tolist(),
        "z_sin": np.asarray(boundary_obj.z_sin).tolist(),
        "n_field_periods": boundary_obj.n_field_periods,
        "is_stellarator_symmetric": boundary_obj.is_stellarator_symmetric,
    }
    
    # If a predictor is provided, use it for faster evaluation
    if predictor:
        predicted_objective, predicted_constraints = predictor(boundary_params)
        return ((jnp.asarray(predicted_objective, dtype=jnp.float32), jnp.asarray(predicted_constraints, dtype=jnp.float32)), None)
        
    # Otherwise, use the expensive forward model
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
            constraints = jnp.ones(6) * NAN_TO_HIGH_VALUE
        else:
            objective = jnp.array(metrics.aspect_ratio)
            margins = tools.compute_constraint_margins(metrics, "p3")
            constraints = jnp.array(list(margins.values()))

        return ((objective, constraints), metrics)
    


@dataclass(frozen=True)
class BudgetSnapshot:
    screen_evals_per_cycle: int
    promote_top_k: int
    max_high_fidelity_evals_per_cycle: int


@dataclass(frozen=True)
class CycleBudgetFeedback:
    hv_delta: float | None
    feasibility_rate: float | None
    cache_hit_rate: float | None


class BudgetController:
    STATE_VERSION = 1
    _STATE_FIELDS = {"_last_feedback", "_cache_stats"}

    def __init__(
        self,
        base_budgets: ai_config.BudgetConfig,
        adaptive_cfg: ai_config.AdaptiveBudgetConfig,
    ) -> None:
        self._base = base_budgets
        self._adaptive_cfg = adaptive_cfg
        self._last_feedback: CycleBudgetFeedback | None = None
        self._cache_stats: dict[str, dict[str, int]] = {}
        # NOTE: If new adaptive state is added, include it in to_dict/restore and checkpoints for deterministic resume.

    def to_dict(self) -> dict[str, Any]:
        unknown_state = {
            key
            for key in self.__dict__
            if key.startswith("_")
            and key not in {"_base", "_adaptive_cfg"}
            and key not in self._STATE_FIELDS
        }
        if unknown_state:
            raise ValueError(
                f"BudgetController state fields {unknown_state} are not serialized; "
                "update _STATE_FIELDS/to_dict/restore for deterministic resume."
            )
        feedback = (
            {
                "hv_delta": self._last_feedback.hv_delta,
                "feasibility_rate": self._last_feedback.feasibility_rate,
                "cache_hit_rate": self._last_feedback.cache_hit_rate,
            }
            if self._last_feedback
            else None
        )
        return {
            "state_version": self.STATE_VERSION,
            "base_budgets": asdict(self._base),
            "adaptive_cfg": asdict(self._adaptive_cfg),
            "adaptive_enabled": self._adaptive_cfg.enabled,
            "last_feedback": feedback,
            "cache_stats": self._cache_stats,
        }

    def restore(self, payload: Mapping[str, Any]) -> None:
        version = payload.get("state_version", 0)
        if version != self.STATE_VERSION:
            print(
                f"[budget] warning: checkpoint state_version={version} "
                f"!= expected {self.STATE_VERSION}; attempting best-effort restore."
            )
        feedback = payload.get("last_feedback")
        if feedback:
            self._last_feedback = CycleBudgetFeedback(
                hv_delta=feedback.get("hv_delta"),
                feasibility_rate=feedback.get("feasibility_rate"),
                cache_hit_rate=feedback.get("cache_hit_rate"),
            )
        cache_stats = payload.get("cache_stats")
        if isinstance(cache_stats, dict):
            self._cache_stats = cache_stats

    def snapshot(self) -> BudgetSnapshot:
        if not self._adaptive_cfg.enabled or self._last_feedback is None:
            return BudgetSnapshot(
                screen_evals_per_cycle=self._base.screen_evals_per_cycle,
                promote_top_k=self._base.promote_top_k,
                max_high_fidelity_evals_per_cycle=self._base.max_high_fidelity_evals_per_cycle,
            )
        return BudgetSnapshot(
            screen_evals_per_cycle=self._blend_budget(
                self._base.screen_evals_per_cycle,
                self._adaptive_cfg.screen_bounds,
                self._screen_score(self._last_feedback),
            ),
            promote_top_k=self._blend_budget(
                self._base.promote_top_k,
                self._adaptive_cfg.promote_top_k_bounds,
                self._promote_score(self._last_feedback),
            ),
            max_high_fidelity_evals_per_cycle=self._blend_budget(
                self._base.max_high_fidelity_evals_per_cycle,
                self._adaptive_cfg.high_fidelity_bounds,
                self._high_fidelity_score(self._last_feedback),
            ),
        )

    def capture_cache_hit_rate(
        self,
        stage: str,
        stats: Mapping[str, int] | None = None,
    ) -> float:
        stats = stats or tools.get_cache_stats(stage)
        previous = self._cache_stats.get(stage)
        delta_hits = stats.get("hits", 0)
        delta_misses = stats.get("misses", 0)
        if previous:
            delta_hits -= previous.get("hits", 0)
            delta_misses -= previous.get("misses", 0)
        self._cache_stats[stage] = {
            "hits": stats.get("hits", 0),
            "misses": stats.get("misses", 0),
        }
        total = max(0, delta_hits + delta_misses)
        if total <= 0:
            return 0.0
        return float(max(0.0, min(1.0, delta_hits / total)))

    def record_feedback(self, feedback: CycleBudgetFeedback) -> None:
        self._last_feedback = feedback

    def _blend_budget(
        self,
        base_value: int,
        bounds: ai_config.BudgetRangeConfig,
        score: float,
    ) -> int:
        constrained_base = max(bounds.min, min(bounds.max, base_value))
        if bounds.min >= bounds.max:
            return constrained_base
        clamped = max(0.0, min(1.0, score))
        mid = 0.5
        if clamped == mid:
            return constrained_base
        if clamped > mid:
            ratio = (clamped - mid) * 2.0
            increment = bounds.max - constrained_base
            return min(bounds.max, constrained_base + int(round(increment * ratio)))
        ratio = (mid - clamped) * 2.0
        decrement = constrained_base - bounds.min
        return max(bounds.min, constrained_base - int(round(decrement * ratio)))

    def _normalize(self, value: float | None, target: float) -> float:
        if value is None or target <= 0.0:
            return 0.0
        ratio = float(value) / float(target)
        return max(0.0, min(1.0, ratio))

    def _screen_score(self, feedback: CycleBudgetFeedback) -> float:
        progress = max(0.0, feedback.hv_delta or 0.0)
        normalized = self._normalize(progress, self._adaptive_cfg.hv_slope_reference)
        feasibility = self._normalize(
            feedback.feasibility_rate, self._adaptive_cfg.feasibility_target
        )
        return 1.0 - (0.6 * normalized + 0.4 * feasibility)

    def _promote_score(self, feedback: CycleBudgetFeedback) -> float:
        progress = max(0.0, feedback.hv_delta or 0.0)
        normalized = self._normalize(progress, self._adaptive_cfg.hv_slope_reference)
        feasibility = self._normalize(
            feedback.feasibility_rate, self._adaptive_cfg.feasibility_target
        )
        return 0.6 * normalized + 0.4 * feasibility

    def _high_fidelity_score(self, feedback: CycleBudgetFeedback) -> float:
        progress = max(0.0, feedback.hv_delta or 0.0)
        normalized = self._normalize(progress, self._adaptive_cfg.hv_slope_reference)
        cache = self._normalize(
            feedback.cache_hit_rate, self._adaptive_cfg.cache_hit_target
        )
        return 0.5 * normalized + 0.5 * cache


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


@dataclass
class RunnerCLIConfig:
    config_path: Path
    problem: str | None
    cycles: int | None
    memory_db: Path | None
    eval_budget: int | None
    workers: int | None
    pool_type: str | None
    screen_only: bool
    promote_only: bool
    slow: bool
    verbose: bool
    log_cache_stats: bool
    run_preset: str | None
    planner: str
    resume_from: Path | None = None


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "AI Scientist runner (per docs/TASKS_CODEX_MINI.md:191-195). "
            "Set AI_SCIENTIST_PEFT=1 to load adapter bundles from reports/adapters."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ai_config.DEFAULT_EXPERIMENT_CONFIG_PATH,
        help="Path to the experiment configuration YAML (defaults to configs/experiment.yaml).",
    )
    parser.add_argument(
        "--problem",
        choices=["p1", "p2", "p3"],
        help=(
            "Problem identifier that overrides the config (p1=GeometricalProblem, "
            "p2=SimpleToBuildQIStellarator, p3=MHDStableQIStellarator)."
        ),
    )
    parser.add_argument(
        "--cycles",
        type=int,
        help="Number of governance cycles to run (overrides config; each cycle includes screening → reporting).",
    )
    parser.add_argument(
        "--memory-db",
        type=Path,
        help="Path to the shared SQLite world model (overrides config).",
    )
    parser.add_argument(
        "--eval-budget",
        type=int,
        help="Override the per-cycle screening budget (screen_evals_per_cycle).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Override n_workers from the config (also powers multiprocessing pools).",
    )
    parser.add_argument(
        "--pool-type",
        choices=["thread", "process"],
        help="Choose the executor pool type used when n_workers > 1.",
    )
    parser.add_argument(
        "--screen",
        action="store_true",
        help=(
            "Run only the screening stage (governance S1) and skip promotions. "
            "Cannot be combined with --promote or presets that advance directly to S2."
        ),
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help=(
            "Start governance in promote/refine mode (S2+) and report promotions. "
            "Cannot be combined with --screen or presets that force S1-only behavior."
        ),
    )
    parser.add_argument(
        "--slow",
        action="store_true",
        help="Throttle loop iterations for deterministic, long-wall-clock logging and traceability.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Emit additional runner diagnostics (candidate mixes, gating decisions).",
    )
    parser.add_argument(
        "--log-cache-stats",
        action="store_true",
        help="Write per-stage cache stats to reports/cache_stats.jsonl for Phase 5 observability.",
    )
    parser.add_argument(
        "--run-preset",
        type=str,
        help="Name of a preset from configs/run_presets.yaml that toggles --screen/--promote/--slow.",
    )
    parser.add_argument(
        "--planner",
        choices=["deterministic", "agent"],
        default="deterministic",
        help="Choose the planning driver (deterministic loop or Phase 3 agent).",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Path to a cycle checkpoint JSON to resume from (skips completed cycles).",
    )
    return parser


def parse_args(args: Sequence[str] | None = None) -> RunnerCLIConfig:
    parser = _build_argument_parser()
    namespace = parser.parse_args(args)
    if namespace.screen and namespace.promote:
        parser.error("--screen cannot be combined with --promote.")
    return RunnerCLIConfig(
        config_path=namespace.config,
        problem=namespace.problem,
        cycles=namespace.cycles,
        memory_db=namespace.memory_db,
        eval_budget=namespace.eval_budget,
        workers=namespace.workers,
        pool_type=namespace.pool_type,
        screen_only=bool(namespace.screen),
        promote_only=bool(namespace.promote),
        slow=bool(namespace.slow),
        verbose=bool(namespace.verbose),
        log_cache_stats=bool(namespace.log_cache_stats),
        run_preset=namespace.run_preset,
        planner=namespace.planner,
        resume_from=namespace.resume_from,
    )


def _validate_runtime_flags(runtime: RunnerCLIConfig) -> None:
    if runtime.screen_only and runtime.promote_only:
        raise ValueError(
            "--screen (S1-only) cannot be combined with promote-only mode (S2+) "
            f"(presets: {runtime.run_preset or '<none>'}). Remove one flag/preset."
        )


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
        r_cos = seed_data["r_cos"]
        z_sin = seed_data["z_sin"]
        r_sin = seed_data["r_sin"]
        z_cos = seed_data["z_cos"]
        n_field_periods = int(seed_data["n_field_periods"])
        is_stellarator_symmetric = bool(seed_data["is_stellarator_symmetric"])
    else:
        base_surface = generate_rotating_ellipse(
            aspect_ratio=4.0,
            elongation=1.5,
            rotational_transform=1.2,
            n_field_periods=template.n_field_periods,
        )
        max_poloidal = max(1, template.n_poloidal_modes - 1)
        max_toroidal = max(1, (template.n_toroidal_modes - 1) // 2)
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

    params = {
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
    suggested_params: list[Mapping[str, Any]] | None = None,
    generative_model: GenerativeDesignModel | None = None,
) -> tuple[list[Mapping[str, Any]], int, int, int]:
    """Blend constraint-aware sampling, VAE generation, and random noise per roadmap."""

    pool_size = total_candidates if total_candidates is not None else screen_budget
    total_candidates = int(pool_size)
    if total_candidates <= 0:
        return [], 0, 0, 0

    agent_results: list[Mapping[str, Any]] = []
    if suggested_params:
        for idx, params in enumerate(suggested_params):
            agent_results.append(
                {
                    "seed": cfg.random_seed + cycle_index * 1000 + idx,  # distinct seed space
                    "params": params,
                    "design_hash": tools.design_hash(params),
                    "constraint_distance": 0.0,  # Assume agent suggestions are good
                    "source": "agent_suggestion",
                }
            )
    
    # Adjust remaining quota
    remaining_candidates = max(0, total_candidates - len(agent_results))
    if remaining_candidates == 0:
         return agent_results, 0, 0, 0

    # VAE Generation
    vae_results: list[Mapping[str, Any]] = []
    if generative_model and generative_model._trained:
        # Allocate up to 40% to VAE if trained, to allow mixing
        vae_target = int(remaining_candidates * 0.4)
        if vae_target > 0:
            seed_base = cfg.random_seed + cycle_index * 20000
            vae_results = generative_model.sample(vae_target, seed=seed_base)
    
    remaining_candidates = max(0, remaining_candidates - len(vae_results))

    mix = cfg.proposal_mix
    exploration_weight = mix.exploration_ratio
    if prev_feasibility_rate is not None:
        # Decay exploration as feasibility improves (Workstream 1)
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
    sampler_results: list[Mapping[str, Any]] = []
    for params in sampler_params:
        try:
            seed = next(seed_iter)
        except StopIteration:
            break
        if isinstance(params, Mapping) and "params" in params:
            payload = params.get("params", {})
            constraint_distance = float(params.get("normalized_constraint_distance", 0.0))
        else:
            payload = params
            constraint_distance = 0.0
        sampler_results.append(
            {
                "seed": seed,
                "params": payload,
                "design_hash": tools.design_hash(payload),
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

    random_results: list[Mapping[str, Any]] = []
    
    if cfg.proposal_mix.sampler_type == "near_axis":
        try:
            sampler = NearAxisSampler(cfg.boundary_template)
            random_results = sampler.generate(remaining_seeds)
        except Exception as exc:
            print(f"[runner] NearAxisSampler failed: {exc}; falling back to standard random")
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

    candidates = agent_results + vae_results + sampler_results + random_results
    return candidates, len(sampler_results), len(random_results), len(vae_results)


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
    candidates: list[Mapping[str, Any]],
    world_model: memory.WorldModel,
    surrogate_model: SurrogateBundle,
    *,
    cycle: int = 0,
    verbose: bool = False,
) -> list[Mapping[str, Any]]:
    """Train SurrogateBundle on cached history and trim the pool."""

    if not candidates or screen_budget <= 0:
        return candidates

    problem = (cfg.problem or "").lower()
    target_column = "hv" if problem == "p3" else "objective"
    minimize_objective = problem == "p1"

    history = world_model.surrogate_training_data(
        target=target_column, problem=cfg.problem
    )
    global _LAST_SURROGATE_FIT_SEC
    _LAST_SURROGATE_FIT_SEC = 0.0
    
    # Check feasibility rate in recent history to trigger Restoration Mode
    # TODO: make window configurable? using last 100 or all history
    recent_feasibility = 0.0
    if history:
        # Look at last 50 entries
        recent_window = history[-50:]
        valid = 0
        for m, _ in recent_window:
            # Check "is_feasible" or "feasibility" metric
            # world_model returns raw metrics in m["metrics"] usually?
            # verify world_model.surrogate_training_data structure.
            # In memory.py it returns (metrics_json, target_val).
            # metrics_json is the dict stored in 'evaluation'.
            # We need to check 'feasibility' (float) <= tolerance.
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

    pool_entries: list[Mapping[str, Any]] = []
    for idx, candidate in enumerate(candidates):
        pool_entries.append(
            {
                "candidate_params": candidate["params"],
                "constraint_distance": candidate.get("constraint_distance", 0.0),
                "source": candidate.get("source"),
                "__surrogate_candidate_index": idx,
            }
        )

    ranked_predictions = surrogate_model.rank_candidates(
        pool_entries,
        minimize_objective=minimize_objective,
        exploration_ratio=cfg.proposal_mix.exploration_ratio,
    )

    # Restoration Mode Logic
    # If we have history but feasibility is very low, we override the ranking.
    # Threshold: e.g. < 5% feasible in recent window.
    restoration_active = (len(history) > 10 and recent_feasibility < 0.05)
    
    if restoration_active:
        if verbose:
            print(f"[runner][restoration] Feasibility rate {recent_feasibility:.1%}; switching to restoration ranking.")
        
        weights = cfg.constraint_weights
        
        def _predicted_violation(pred: SurrogatePrediction) -> float:
            # Estimate violations based on predictions
            # MHD: needs to be >= 0. Violation is max(0, -pred_mhd) * weight
            mhd_val = pred.predicted_mhd if pred.predicted_mhd is not None else 0.0
            mhd_viol = max(0.0, -mhd_val) * weights.mhd
            
            # QI: lower is better. How to define violation?
            # Maybe just weighted value if we want to drive it down?
            # Or relative to a threshold? Let's just drive it down.
            qi_val = pred.predicted_qi if pred.predicted_qi is not None else 1.0
            qi_viol = max(0.0, qi_val) * weights.qi # Assume QI always > 0
            
            # Elongation: lower is better.
            elo_val = pred.predicted_elongation if pred.predicted_elongation is not None else 1.0
            elo_viol = max(0.0, elo_val) * weights.elongation
            
            # Composite violation (lower is better)
            return mhd_viol + qi_viol + elo_viol

        # Re-sort by predicted violation
        ranked_predictions.sort(key=_predicted_violation)
    
    elif cfg.problem.lower() == "p2":
        # Phase 5 HV/Objective (docs/AI_SCIENTIST_UNIFIED_ROADMAP.md §5):
        # gate P2 promotions on feasibility probability before objective.
        prob_threshold = max(
            0.0, min(1.0, cfg.adaptive_budgets.feasibility_target)
        )
        filtered_predictions = [
            prediction
            for prediction in ranked_predictions
            if prediction.prob_feasible >= prob_threshold
        ]
        if filtered_predictions:
            ranked_predictions = filtered_predictions

    selected: list[Mapping[str, Any]] = []
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


def _time_exceeded(start: float, limit_minutes: float) -> bool:
    elapsed = time.perf_counter() - start
    return elapsed >= limit_minutes * 60


@dataclass
class CycleSummary:
    cycle: int
    objective: float | None
    feasibility: float | None
    hv: float | None
    stage: str


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


def _extract_objectives(entry: Mapping[str, Any]) -> tuple[float, float]:
    metrics = entry["evaluation"]["metrics"]
    gradient = float(metrics["minimum_normalized_magnetic_gradient_scale_length"])
    aspect = float(metrics["aspect_ratio"])
    return gradient, aspect


def _objective_proxy(entry: Mapping[str, Any]) -> float:
    gradient, aspect = _extract_objectives(entry)
    return gradient - aspect


def _crowding_distance(
    entries_by_design: Mapping[str, Mapping[str, Any]],
) -> dict[str, float]:
    if not entries_by_design:
        return {}
    values: list[tuple[str, float, float]] = []
    for design_hash, entry in entries_by_design.items():
        gradient, aspect = _extract_objectives(entry)
        values.append((design_hash, float(gradient), float(aspect)))
    distances = {design_hash: 0.0 for design_hash in entries_by_design}
    if len(values) <= 2:
        for design_hash in distances:
            distances[design_hash] = float("inf")
        return distances
    sorted_grad = sorted(values, key=lambda item: item[1], reverse=True)
    grad_values = [item[1] for item in sorted_grad]
    grad_span = max(max(grad_values) - min(grad_values), 1e-9)
    distances[sorted_grad[0][0]] = float("inf")
    distances[sorted_grad[-1][0]] = float("inf")
    for pos in range(1, len(sorted_grad) - 1):
        prev_val = sorted_grad[pos - 1][1]
        next_val = sorted_grad[pos + 1][1]
        distances[sorted_grad[pos][0]] += abs(next_val - prev_val) / grad_span

    sorted_aspect = sorted(values, key=lambda item: item[2], reverse=False)
    aspect_values = [item[2] for item in sorted_aspect]
    aspect_span = max(max(aspect_values) - min(aspect_values), 1e-9)
    distances[sorted_aspect[0][0]] = float("inf")
    distances[sorted_aspect[-1][0]] = float("inf")
    for pos in range(1, len(sorted_aspect) - 1):
        prev_val = sorted_aspect[pos - 1][2]
        next_val = sorted_aspect[pos + 1][2]
        distances[sorted_aspect[pos][0]] += abs(next_val - prev_val) / aspect_span
    return distances


def _rank_candidates_for_promotion(
    entries_by_design: Mapping[str, Mapping[str, Any]],
    promote_limit: int,
    reference_point: Tuple[float, float],
) -> list[Mapping[str, Any]]:
    if not entries_by_design:
        return []
    summary = tools.summarize_p3_candidates(
        list(entries_by_design.values()), reference_point=reference_point
    )
    # Phase 5 HV/Objective (docs/AI_SCIENTIST_UNIFIED_ROADMAP.md §5):
    # rank promotions by feasibility first, then objective proxy.
    ordered_hashes = [entry.design_hash for entry in summary.pareto_entries]
    ranked: list[Mapping[str, Any]] = sorted(
        (
            entries_by_design[h]
            for h in ordered_hashes
            if h in entries_by_design
        ),
        key=lambda entry: (
            0.0 if _feasibility_value(entry) <= FEASIBILITY_CUTOFF else 1.0,
            _feasibility_value(entry),
            -_objective_proxy(entry),
        ),
    )
    if len(ranked) >= promote_limit:
        return ranked[:promote_limit]
    remaining = {
        design_hash: entry
        for design_hash, entry in entries_by_design.items()
        if design_hash not in {entry.design_hash for entry in summary.pareto_entries}
    }
    crowding = _crowding_distance(remaining)

    def _sort_key(design_hash: str) -> tuple[float, float, float]:
        entry = remaining[design_hash]
        feas = _feasibility_value(entry)
        feasible_flag = 0.0 if feas <= FEASIBILITY_CUTOFF else 1.0
        return (
            feasible_flag,
            -crowding.get(design_hash, 0.0),
            feas,
            -_objective_proxy(entry),
        )

    for design_hash in sorted(remaining, key=_sort_key):
        ranked.append(remaining[design_hash])
        if len(ranked) >= promote_limit:
            break
    return ranked


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
    """Persist Phase 6 Pareto deliverables and return the logged design hashes."""

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
            evaluation.get("settings", {}), separators=( ",", ":")
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


def _relative_objective_improvement(
    history: list[CycleSummary], lookback: int
) -> float:
    if len(history) <= lookback:
        return 0.0
    earlier = history[-lookback - 1].objective
    latest = history[-1].objective
    if earlier is None or latest is None:
        return 0.0
    diff = earlier - latest
    denom = abs(earlier) if abs(earlier) > 1e-6 else 1.0
    return float(diff / denom)


def _should_transition_s1_to_s2(
    history: list[CycleSummary], gate_cfg: ai_config.StageGateConfig
) -> bool:
    if not history:
        return False
    last = history[-1]
    if last.feasibility is not None:
        triggered = last.feasibility <= gate_cfg.s1_to_s2_feasibility_margin
        print(
            f"[runner][stage-gate] S1→S2 feasibility check: margin={last.feasibility:.5f} "
            f"<= {gate_cfg.s1_to_s2_feasibility_margin:.5f} -> {triggered}"
        )
        if triggered:
            return True
    improvement = _relative_objective_improvement(
        history, gate_cfg.s1_to_s2_lookback_cycles
    )
    triggered_improvement = improvement >= gate_cfg.s1_to_s2_objective_improvement
    print(
        f"[runner][stage-gate] S1→S2 objective improvement check: "
        f"{improvement:.4f} >= {gate_cfg.s1_to_s2_objective_improvement:.4f} -> {triggered_improvement}"
    )
    return triggered_improvement


def _should_transition_s2_to_s3(
    history: list[CycleSummary],
    gate_cfg: ai_config.StageGateConfig,
    governance_cfg: ai_config.GovernanceConfig,
    world_model: memory.WorldModel,
    experiment_id: int,
    current_cycle: int,
    total_cycles: int,
) -> bool:
    avg_delta = world_model.average_recent_hv_delta(
        experiment_id, governance_cfg.hv_lookback
    )
    if avg_delta is not None:
        triggered_delta = avg_delta <= gate_cfg.s2_to_s3_hv_delta
        print(
            f"[runner][stage-gate] S2→S3 average HV delta over "
            f"{governance_cfg.hv_lookback} cycles: {avg_delta:.4f} <= {gate_cfg.s2_to_s3_hv_delta:.4f} -> {triggered_delta}"
        )
        if triggered_delta:
            return True
    else:
        print(
            f"[runner][stage-gate] insufficient HV delta history ({len(history)} cycles) "
            f"to evaluate lookback={governance_cfg.hv_lookback}; deferring promotion"
        )
    exhausted = current_cycle >= total_cycles
    print(
        f"[runner][stage-gate] S2→S3 budget check: cycle={current_cycle} >= total={total_cycles} -> {exhausted}"
    )
    return exhausted


def _process_worker_initializer() -> None:
    """Limit OpenMP threads inside process workers (Phase 5 observability safeguard)."""
    os.environ["OMP_NUM_THREADS"] = "1"


def _evaluate_stage(
    candidates: Iterable[Mapping[str, Any]],
    stage: str,
    budgets: ai_config.BudgetConfig,
    cycle_start: float,
    evaluate_fn: ProblemEvaluator,
    *,
    sleep_per_eval: float = 0.0,
) -> list[dict[str, Any]]:
    """Evaluate candidates at a given fidelity, respecting wall-clock budget."""

    results: list[dict[str, Any]] = []
    wall_limit = budgets.wall_clock_minutes

    if budgets.n_workers <= 1:
        for candidate in candidates:
            if _time_exceeded(cycle_start, wall_limit):
                break
            params = candidate["params"]
            design_id = candidate.get("design_hash") or tools.design_hash(params)
            try:
                evaluation = evaluate_fn(params, stage=stage)
                evaluation.setdefault("vmec_status", "ok")
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[runner][stage-eval] Failed evaluation for design {design_id} "
                    f"(seed={candidate['seed']} stage={stage}): {exc}"
                )
                evaluation = {
                    "stage": stage,
                    "feasibility": float("inf"),
                    "max_violation": float("inf"),
                    "objective": float("inf"),
                    "is_feasible": False,
                    "vmec_status": "exception",
                    "metrics": {
                        "minimum_normalized_magnetic_gradient_scale_length": 0.0,
                        "aspect_ratio": float("inf"),
                        "max_elongation": float("inf"),
                        "constraint_margins": {},
                        "max_violation": float("inf"),
                    },
                }
            results.append(
                {
                    "params": params,
                    "evaluation": evaluation,
                    "seed": int(candidate["seed"]),
                    "design_hash": design_id,
                }
            )
            if sleep_per_eval > 0:
                time.sleep(sleep_per_eval)
        return results

    future_payloads = {}
    executor_cls = (
        ThreadPoolExecutor if budgets.pool_type == "thread" else ProcessPoolExecutor
    )
    executor_kwargs: dict[str, Any] = {"max_workers": budgets.n_workers}
    if executor_cls is ProcessPoolExecutor:
        executor_kwargs["initializer"] = _process_worker_initializer
    with executor_cls(**executor_kwargs) as executor:
        for candidate in candidates:
            if _time_exceeded(cycle_start, wall_limit):
                break
            design_id = candidate.get("design_hash")
            future = executor.submit(evaluate_fn, candidate["params"], stage=stage)
            future_payloads[future] = (candidate, design_id)

        for future in as_completed(future_payloads):
            candidate, design_id = future_payloads[future]
            exc = future.exception()
            if exc is not None:
                design_hash = design_id or tools.design_hash(candidate["params"])
                print(
                    f"[runner][stage-eval] Failed evaluation for design {design_hash} "
                    f"(seed={candidate['seed']} stage={stage}): {exc}"
                )
                failure_eval = {
                    "stage": stage,
                    "feasibility": float("inf"),
                    "max_violation": float("inf"),
                    "objective": float("inf"),
                    "is_feasible": False,
                    "vmec_status": "exception",
                    "metrics": {
                        "minimum_normalized_magnetic_gradient_scale_length": 0.0,
                        "aspect_ratio": float("inf"),
                        "max_elongation": float("inf"),
                        "constraint_margins": {},
                        "max_violation": float("inf"),
                    },
                }
                results.append(
                    {
                        "params": candidate["params"],
                        "evaluation": failure_eval,
                        "seed": int(candidate["seed"]),
                        "design_hash": design_hash,
                    }
                )
                continue

            results.append(
                {
                    "params": candidate["params"],
                    "evaluation": {
                        **future.result(),
                        "vmec_status": future.result().get("vmec_status", "ok"),
                    },
                    "seed": int(candidate["seed"]),
                    "design_hash": design_id or tools.design_hash(candidate["params"]),
                }
            )
    return results


def _cache_stats_log_path(report_dir: Path | str) -> Path:
    return Path(report_dir) / "cache_stats.jsonl"


def _observability_log_path(report_dir: Path | str) -> Path:
    return Path(report_dir) / "observability.jsonl"


def _maybe_log_cache_stats(
    runtime: RunnerCLIConfig | None,
    cfg: ai_config.ExperimentConfig,
    cycle_index: int,
    stage: str,
    stats: Mapping[str, int],
) -> None:
    """Emit per-cycle cache stats for Phase 5 observability (see ai_scientist/roadmap.md & ai_scientist/improvement-plan.md)."""
    if not (runtime and runtime.log_cache_stats):
        return
    entry = {
        "cycle": cycle_index + 1,
        "stage": stage,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stats": stats,
    }
    log_path = _cache_stats_log_path(cfg.reporting_dir)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, separators=( ",", ":")) + "\n")


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
        handle.write(json.dumps(entry, separators=( ",", ":")) + "\n")


def _build_template_params_for_alm(
    template: ai_config.BoundaryTemplateConfig
) -> Mapping[str, Any]:
    n_poloidal = template.n_poloidal_modes
    n_toroidal = template.n_toroidal_modes
    center_idx = n_toroidal // 2
    r_cos = []
    z_sin = []
    for pol in range(n_poloidal):
        r_row = []
        z_row = []
        for tor in range(n_toroidal):
            r_val = (
                template.base_major_radius
                if pol == 0 and tor == center_idx
                else 0.0
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


def _run_cycle(
    cfg: ai_config.ExperimentConfig,
    cycle_index: int,
    world_model: memory.WorldModel,
    experiment_id: int,
    governance_stage: str,
    git_sha: str,
    constellaration_sha: str,
    surrogate_model: SurrogateBundle,
    *,
    runtime: RunnerCLIConfig | None = None,
    budget_controller: BudgetController,
    prev_feasibility_rate: float | None = None,
    suggested_params: list[Mapping[str, Any]] | None = None,
    config_overrides: Mapping[str, Any] | None = None,
    generative_model: GenerativeDesignModel | None = None,
) -> tuple[Path | None, dict[str, Any] | None, tools.P3Summary | None]:
    tool_name = _problem_tool_name(cfg.problem)
    base_evaluate = _problem_evaluator(cfg.problem)
    evaluate_fn = adapter.with_peft(base_evaluate, tool_name=tool_name)
    cycle_start = time.perf_counter()
    cycle_number = cycle_index + 1
    global _LAST_SURROGATE_FIT_SEC
    _LAST_SURROGATE_FIT_SEC = 0.0
    sleep_per_eval = 0.05 if runtime and runtime.slow else 0.0
    screen_only = bool(runtime and runtime.screen_only)

    # Apply config-defined run_overrides first (for testing/fixed behavior)
    if cfg.run_overrides:
        try:
            # Merge with existing config_overrides, prioritizing cfg.run_overrides
            if config_overrides:
                temp_overrides = dict(config_overrides)
                for key, value in cfg.run_overrides.items():
                    if isinstance(value, Mapping) and key in temp_overrides and isinstance(temp_overrides[key], Mapping):
                        temp_overrides[key] = {**temp_overrides[key], **value}
                    else:
                        temp_overrides[key] = value
                config_overrides = temp_overrides
            else:
                config_overrides = cfg.run_overrides
            if runtime and runtime.verbose:
                print(f"[runner][cycle={cycle_number}] Applying config-defined run_overrides: {cfg.run_overrides}")
        except Exception as exc:
            print(f"[runner][cycle={cycle_number}] Failed to apply run_overrides from config: {exc}")

    # Apply agent-driven config overrides
    active_cfg = cfg
    optimizer_mode = "default"  # Default optimizer mode
    alm_settings_overrides: Mapping[str, Any] = {}

    if config_overrides:
        try:
            overrides_log = []
            if "proposal_mix" in config_overrides:
                new_mix = replace(active_cfg.proposal_mix, **config_overrides["proposal_mix"])
                active_cfg = replace(active_cfg, proposal_mix=new_mix)
                overrides_log.append(f"proposal_mix={config_overrides['proposal_mix']}")
            if "budgets" in config_overrides:
                new_budgets = replace(active_cfg.budgets, **config_overrides["budgets"])
                active_cfg = replace(active_cfg, budgets=new_budgets)
                overrides_log.append(f"budgets={config_overrides['budgets']}")
            if "constraint_weights" in config_overrides:
                new_weights = replace(active_cfg.constraint_weights, **config_overrides["constraint_weights"])
                active_cfg = replace(active_cfg, constraint_weights=new_weights)
                overrides_log.append(f"constraint_weights={config_overrides['constraint_weights']}")
            
            # New: Handle optimizer mode and ALM settings
            if "optimizer" in config_overrides:
                optimizer_mode = str(config_overrides["optimizer"]).lower()
                overrides_log.append(f"optimizer={optimizer_mode}")
            if "alm_settings" in config_overrides:
                alm_settings_overrides = config_overrides["alm_settings"]
                overrides_log.append(f"alm_settings={alm_settings_overrides}")
            if "initialization_strategy" in config_overrides:
                new_init_strategy = str(config_overrides["initialization_strategy"])
                active_cfg = replace(active_cfg, initialization_strategy=new_init_strategy)
                overrides_log.append(f"initialization_strategy={new_init_strategy}")

            if overrides_log:
                print(f"[runner][cycle={cycle_number}] Applying agent config overrides: {', '.join(overrides_log)}")
        except Exception as exc:
            print(f"[runner][cycle={cycle_number}] Failed to apply config overrides: {exc}")

    budget_snapshot = budget_controller.snapshot()
    active_budgets = replace(
        active_cfg.budgets,
        screen_evals_per_cycle=budget_snapshot.screen_evals_per_cycle,
        promote_top_k=budget_snapshot.promote_top_k,
        max_high_fidelity_evals_per_cycle=budget_snapshot.max_high_fidelity_evals_per_cycle,
    )
    
    # Re-apply agent overrides if they exist, as snapshot() clobbered them.
    if config_overrides and "budgets" in config_overrides:
        active_budgets = replace(active_budgets, **config_overrides["budgets"])
        if runtime and runtime.verbose:
            print(f"[runner] Agent forced budget overrides: {config_overrides['budgets']}")

    if active_cfg.adaptive_budgets.enabled and runtime and runtime.verbose:
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
        gen_history = world_model.surrogate_training_data(
            target="hv" if cfg.problem == "p3" else "objective", 
            problem=cfg.problem
        )
        if gen_history:
            generative_model.fit([item[0] for item in gen_history])

    sampler_count = 0
    random_count = 0
    
    candidate_pool: list[Mapping[str, Any]] = []
    
    # Placeholder for ALM/SA-ALM block
    # Will be replaced with full SA-ALM integration
    
    candidate_pool: list[Mapping[str, Any]] = []
    
    # START ALM_BLOCK_PLACEHOLDER
    if cfg.optimizer_backend == "gradient_descent":
        if runtime and runtime.verbose:
            print(f"[runner][cycle={cycle_number}] V2 Gradient Descent Optimization active (Phase 5 Coordinator).")
        
        # Initialize Coordinator (Phase 5)
        coordinator = Coordinator(
            active_cfg, 
            world_model, 
            planner=None, # We don't pass the legacy planner instance here to avoid confusion, or we could. 
            surrogate=surrogate_model if isinstance(surrogate_model, NeuralOperatorSurrogate) else None, 
            generative_model=generative_model
        )
        
        candidate_pool = coordinator.produce_candidates(
            cycle=cycle_number,
            experiment_id=experiment_id,
            n_candidates=pool_size,
            template=active_cfg.boundary_template
        )
        
        # Note: Coordinator handles strategy (Explore/Exploit). 
        # If Exploit, candidates are already optimized. 
        
        candidates = _surrogate_rank_screen_candidates(
            active_cfg,
            active_budgets.screen_evals_per_cycle,
            candidate_pool,
            world_model,
            surrogate_model,
            cycle=cycle_number,
            verbose=bool(runtime and runtime.verbose),
        )

    elif optimizer_mode == "alm" or optimizer_mode == "sa-alm":
        # ALM Execution Branch
        if runtime and runtime.verbose:
            print(f"[runner][cycle={cycle_number}] {optimizer_mode} optimizer mode active.")
        
        initial_params_map: Mapping[str, Any]
        if suggested_params and suggested_params[0]:
            initial_params_map = suggested_params[0]
        else:
            if active_cfg.initialization_strategy == "nae":
                print("[runner] Using NAE for initial ALM design.")
                initial_params_map = _generate_nae_candidate_params(active_cfg.boundary_template)
            else: # Default to "template"
                print("[runner] Using template for initial ALM design.")
                initial_params_map = _build_template_params_for_alm(active_cfg.boundary_template)
            
        # Prepare boundary for optimization using constellaration utils
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
            alpha=0.5, # Default alpha
        ).reshape(boundary_obj.poloidal_modes.shape)
        scale = jnp.array(np.concatenate([scale[mask.r_cos], scale[mask.z_sin]]))
        
        x0 = jnp.array(initial_guess) / scale

        fm_settings = tools._settings_for_stage(active_cfg.fidelity_ladder.promote)
        
        # ALM settings
        alm_settings_obj = AugmentedLagrangianSettings(**alm_settings_overrides)

        # For SA-ALM, we need a predictor function for the inner loop
        sa_alm_predictor: Callable[[Mapping[str, Any]], Tuple[float, Sequence[float]]] | None = None
        if optimizer_mode == "sa-alm":
            # Retrain surrogate if needed
            problem = (cfg.problem or "").lower()
            target_column = "hv" if problem == "p3" else "objective"
            history = world_model.surrogate_training_data(
                target=target_column, problem=cfg.problem
            )
            metrics_list: tuple[Mapping[str, Any], ...] = ()
            target_values: tuple[float, ...] = ()
            if history:
                metrics_list, target_values = zip(*history)
                if surrogate_model.should_retrain(len(history), cycle=cycle_number):
                    surrogate_model.fit(
                        metrics_list,
                        target_values,
                        minimize_objective=(problem == "p1"), # assuming P1 objective is minimize
                        cycle=cycle_number,
                    )
            
            def surrogate_predictor(params: Mapping[str, Any]) -> Tuple[float, Sequence[float]]:
                # Predict objective and constraints using the surrogate
                dummy_candidate = {"candidate_params": params}
                predicted_list = surrogate_model.rank_candidates([dummy_candidate], minimize_objective=False) # min_obj doesn't matter much for individual predictions
                predicted = predicted_list[0]
                
                # Default values for constraints if not predicted
                mhd = predicted.predicted_mhd if predicted.predicted_mhd is not None else 0.0
                qi = predicted.predicted_qi if predicted.predicted_qi is not None else 1.0 # Assume always positive
                elongation = predicted.predicted_elongation if predicted.predicted_elongation is not None else 1.0
                
                # Simplified prediction of constraint values (positive for violation)
                predicted_alm_constraints = [
                    max(0.0, -mhd), # MHD violation: if vacuum_well < 0
                    max(0.0, qi - FEASIBILITY_CUTOFF), # QI violation: if qi > FEASIBILITY_CUTOFF (needs to be low)
                    max(0.0, elongation - 5.0), # Elongation violation: if > 5.0 (arbitrary threshold for now)
                ]
                
                return float(predicted.predicted_objective), predicted_alm_constraints

            sa_alm_predictor = surrogate_predictor
            
        (initial_objective, initial_constraints), _ = _objective_constraints(
            x0, scale, unravel_and_unmask_fn, fm_settings, cfg.problem, predictor=sa_alm_predictor
        )

        state = AugmentedLagrangianState(
            x=jnp.copy(x0),
            multipliers=jnp.zeros_like(initial_constraints),
            penalty_parameters=jnp.ones_like(initial_constraints) * 1.0,
            objective=initial_objective,
            constraints=initial_constraints,
            bounds=jnp.ones_like(x0) * 0.1, # Initial bounds
        )
        
        alm_candidates: list[Mapping[str, Any]] = []
        budget_per_step = 8 # Inner evaluations per ALM step
        num_alm_steps = max(1, active_budgets.screen_evals_per_cycle // budget_per_step)
        
        mp_context = multiprocessing.get_context("spawn") # Safer default for JAX/CUDA
        
        for k in range(num_alm_steps):
            # V2: Differentiable Optimization
            if cfg.surrogate_backend == "neural_operator" and isinstance(surrogate_model, NeuralOperatorSurrogate) and surrogate_model._trained:
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
                    steps=budget_per_step,
                )
                x_new = jnp.array(x_new_np)
                
                # Log the final result
                (obj_new, constr_new), metrics = _objective_constraints(
                    x_new, scale, unravel_and_unmask_fn, fm_settings, cfg.problem, predictor=sa_alm_predictor
                )
                cand_boundary = unravel_and_unmask_fn(jnp.asarray(x_new * scale))
                cand_params = {
                    "r_cos": np.array(cand_boundary.r_cos).tolist(),
                    "z_sin": np.array(cand_boundary.z_sin).tolist(),
                    "n_field_periods": cand_boundary.n_field_periods,
                    "is_stellarator_symmetric": cand_boundary.is_stellarator_symmetric,
                }
                if metrics:
                    p3_margins = tools.compute_constraint_margins(metrics, "p3")
                    max_viol = tools._max_violation(p3_margins)
                else:
                    max_viol = float(jnp.max(constr_new))
                
                alm_candidates.append({
                    "seed": cfg.random_seed + cycle_index * 10000 + len(alm_candidates),
                    "params": cand_params,
                    "design_hash": tools.design_hash(cand_params),
                    "constraint_distance": max_viol,
                    "source": "sa-alm_diff",
                    "evaluation": {
                        "metrics": metrics.model_dump() if metrics else {},
                        "feasibility": max_viol,
                        "stage": active_cfg.fidelity_ladder.promote
                    }
                })

            else:
                # V1: Nevergrad
                parametrization = nevergrad.p.Array(
                    init=np.array(state.x),
                    lower=np.array(state.x - state.bounds),
                    upper=np.array(state.x + state.bounds),
                )
                
                # Inner Optimizer (NGOpt)
                oracle = nevergrad.optimizers.NGOpt(
                    parametrization=parametrization,
                    budget=budget_per_step,
                    num_workers=1, # Keep it simple for now, avoid nested multiprocessing issues
                )
                oracle.suggest(np.array(state.x))
                
                # Inner loop execution
                for _ in range(budget_per_step):
                    candidate = oracle.ask()
                    
                    # Eval using predictor if SA-ALM, else use full model
                    (obj, constr), metrics = _objective_constraints(
                        jnp.array(candidate.value),
                        scale,
                        unravel_and_unmask_fn,
                        fm_settings,
                        cfg.problem,
                        predictor=sa_alm_predictor
                    )
                    
                    # Capture candidate details
                    cand_boundary = unravel_and_unmask_fn(jnp.asarray(candidate.value * scale))
                    cand_params = {
                        "r_cos": np.array(cand_boundary.r_cos).tolist(),
                        "z_sin": np.array(cand_boundary.z_sin).tolist(),
                        "n_field_periods": cand_boundary.n_field_periods,
                        "is_stellarator_symmetric": cand_boundary.is_stellarator_symmetric,
                    }
                    
                    # For SA-ALM, we need a functional metric check
                    if metrics:
                        p3_margins = tools.compute_constraint_margins(metrics, "p3")
                        max_viol = tools._max_violation(p3_margins)
                    else: 
                        max_viol = float(jnp.max(constr))
                    
                    alm_candidates.append({
                        "seed": cfg.random_seed + cycle_index * 10000 + len(alm_candidates),
                        "params": cand_params,
                        "design_hash": tools.design_hash(cand_params),
                        "constraint_distance": max_viol,
                        "source": optimizer_mode, 
                        "evaluation": { 
                            "metrics": metrics.model_dump() if metrics else {},
                            "feasibility": max_viol, 
                            "stage": active_cfg.fidelity_ladder.promote
                        } 
                    })

                    loss = augmented_lagrangian_function(obj, constr, state).item()
                    oracle.tell(candidate, loss)
                
                # Update ALM state
                recommendation = oracle.provide_recommendation()
                x_new = jnp.array(recommendation.value)
            
            # Periodically verify with true forward model if SA-ALM
            if optimizer_mode == "sa-alm" and k % 2 == 0: # Verify every N steps
                if runtime and runtime.verbose:
                    print(f"[runner] SA-ALM: Verifying best candidate with true forward model (step {k}).")
                (obj_new, constr_new), true_metrics = _objective_constraints( # Capture true_metrics
                    x_new, scale, unravel_and_unmask_fn, fm_settings, cfg.problem, predictor=None # Force true eval
                )
                # Add this true evaluation to world model for surrogate retraining
                # The unravel_and_unmask_fn already converts x to SurfaceRZFourier,
                # then we need to convert to dict params for log_candidate.
                # However, this doesn't directly give `FlattenSchema` for `structured_unflatten`.
                # We need to preserve the initial_params_map's metadata.
                verified_params_obj = unravel_and_unmask_fn(x_new * scale)
                verified_params = {
                    "r_cos": np.asarray(verified_params_obj.r_cos).tolist(),
                    "z_sin": np.asarray(verified_params_obj.z_sin).tolist(),
                    "n_field_periods": initial_params_map.get("n_field_periods", 1),
                    "is_stellarator_symmetric": initial_params_map.get("is_stellarator_symmetric", True),
                }

                # Use true_metrics for logging to world model
                if true_metrics:
                    world_model.log_candidate(
                        experiment_id=experiment_id,
                        problem=cfg.problem,
                        params=verified_params,
                        seed=cfg.random_seed + cycle_index * 100000 + k, # Unique seed
                        status=active_cfg.fidelity_ladder.promote,
                        evaluation={
                            "objective": obj_new.item(),
                            "feasibility": float(jnp.max(constr_new)), # Max constraint violation
                            "stage": active_cfg.fidelity_ladder.promote,
                            "metrics": true_metrics.model_dump(), # Use actual metrics
                        },
                        design_hash=tools.design_hash(verified_params),
                        commit=False,
                    )
                # Now use true obj/constr for ALM state update
                state = update_augmented_lagrangian_state(
                    x=x_new,
                    objective=obj_new,
                    constraints=constr_new,
                    state=state,
                    settings=alm_settings_obj,
                )
            else: # Use surrogate prediction for ALM state update
                (obj_new, constr_new), _ = _objective_constraints(
                    x_new, scale, unravel_and_unmask_fn, fm_settings, cfg.problem, predictor=sa_alm_predictor
                )
                state = update_augmented_lagrangian_state(
                    x=x_new,
                    objective=obj_new,
                    constraints=constr_new,
                    state=state,
                    settings=alm_settings_obj,
                )

        candidate_pool = alm_candidates
        candidates = list(alm_candidates)
    # END ALM_BLOCK_PLACEHOLDER
    
    elif pool_size <= 0:
        if runtime and runtime.verbose:
            print(
                f"[runner][cycle={cycle_number}] screen budget zero; skipping candidate generation"
            )
        candidate_pool: list[Mapping[str, Any]] = []
        candidates: list[Mapping[str, Any]] = []
    else:
        candidate_pool, sampler_count, random_count, vae_count = _propose_p3_candidates_for_cycle(
            active_cfg,
            cycle_index,
            world_model,
            experiment_id,
            screen_budget=active_budgets.screen_evals_per_cycle,
            total_candidates=pool_size,
            prev_feasibility_rate=prev_feasibility_rate,
            suggested_params=suggested_params,
            generative_model=generative_model,
        )
        if runtime and runtime.verbose:
            print(
                f"[runner][cycle={cycle_number}] candidate mix (pool={len(candidate_pool)}): sampler={sampler_count} random={random_count} vae={vae_count} agent={len(suggested_params or [])}"
            )
        candidates = _surrogate_rank_screen_candidates(
            active_cfg,
            active_budgets.screen_evals_per_cycle,
            candidate_pool,
            world_model,
            surrogate_model,
            cycle=cycle_number,
            verbose=bool(runtime and runtime.verbose),
        )

    screen_stage = active_cfg.fidelity_ladder.screen
    if candidates:
        screen_results = _evaluate_stage(
            candidates, 
            stage=screen_stage,
            budgets=active_budgets,
            cycle_start=cycle_start,
            evaluate_fn=evaluate_fn,
            sleep_per_eval=sleep_per_eval,
        )
    else:
        screen_results = []
    screen_cache_stats = tools.get_cache_stats(screen_stage)
    cache_hit_rate = budget_controller.capture_cache_hit_rate(
        screen_stage, stats=screen_cache_stats
    )
    _maybe_log_cache_stats(runtime, cfg, cycle_index, screen_stage, screen_cache_stats)

    screen_design_map = _latest_evaluations_by_design(screen_results, screen_stage)
    screen_summary = tools.summarize_p3_candidates(
        list(screen_design_map.values()), reference_point=P3_REFERENCE_POINT
    )
    
    # Adaptive Promotion Logic (Phase 1.2)
    sufficient_feasible = (
        screen_summary.feasible_count >= cfg.governance.min_feasible_for_promotion
    )
    
    promote_limit = 0
    to_promote: list[Mapping[str, Any]] = []
    
    if screen_design_map:
        promote_limit = min(
            active_budgets.promote_top_k,
            active_budgets.max_high_fidelity_evals_per_cycle,
            len(screen_design_map),
        )
        
        if sufficient_feasible:
            # Standard Pareto/Feasibility ranking
            prioritized_screen = _rank_candidates_for_promotion(
                screen_design_map, active_budgets.promote_top_k, P3_REFERENCE_POINT
            )
            to_promote = prioritized_screen[:promote_limit]
        else:
            # Feasibility Restoration ranking (lowest violation)
            print(
                f"[runner][promotion] Insufficient feasible ({screen_summary.feasible_count} < {cfg.governance.min_feasible_for_promotion}); promoting by lowest max_violation."
            )
            # Sort by max_violation (ascending)
            sorted_by_violation = sorted(
                screen_design_map.values(),
                key=lambda entry: float(entry["evaluation"].get("max_violation", float("inf")))
            )
            to_promote = sorted_by_violation[:promote_limit]
            
            # Tag them for analysis
            for candidate in to_promote:
                candidate["promotion_reason"] = "restoration"
            
    promote_stage = cfg.fidelity_ladder.promote
    promote_results: list[dict[str, Any]] = []
    if not screen_only and promote_limit > 0:
        promote_results = _evaluate_stage(
            to_promote,
            stage=promote_stage,
            budgets=active_budgets,
            cycle_start=cycle_start,
            evaluate_fn=evaluate_fn,
            sleep_per_eval=sleep_per_eval,
        )
        promote_cache_stats = tools.get_cache_stats(promote_stage)
        _maybe_log_cache_stats(
            runtime, cfg, cycle_index, promote_stage, promote_cache_stats
        )
    elif screen_only:
        print("[runner] screen-only flag active; skipping promotion evaluations.")

    aggregated = screen_results + promote_results
    if not aggregated:
        world_model.record_stage_history(
            experiment_id=experiment_id,
            cycle=cycle_number,
            stage=governance_stage,
        )
        return None, None, None

    latest_by_design = _latest_evaluations_by_design(
        aggregated, cfg.fidelity_ladder.promote
    )
    if not latest_by_design:
        world_model.record_stage_history(
            experiment_id=experiment_id,
            cycle=cycle_number,
            stage=governance_stage,
        )
        return None, None, None

    p3_summary = tools.summarize_p3_candidates(
        list(latest_by_design.values()), reference_point=P3_REFERENCE_POINT
    )
    total_designs = len(latest_by_design)
    feasibility_rate = float(p3_summary.feasible_count) / float(max(1, total_designs))
    vmec_failure_rate = _vmec_failure_rate(aggregated)
    hv_display = (
        f"{p3_summary.hv_score:.6f}" if p3_summary.feasible_count > 0 else "n/a"
    )
    print(
        f"[runner][cycle={cycle_number}] feasible={p3_summary.feasible_count}/{total_designs} hv={hv_display}"
    )
    _log_observability_metrics(
        cfg,
        cycle_index,
        hv=p3_summary.hv_score,
        feasible_count=p3_summary.feasible_count,
        vmec_failure_rate=vmec_failure_rate,
        retrain_time=_LAST_SURROGATE_FIT_SEC,
        cache_hit_rate=cache_hit_rate,
        budget_snapshot=active_budgets,
    )
    
    if cfg.reporting.get("prometheus_export_enabled", False):
        prom_path = Path(cfg.reporting_dir) / "metrics.prom"
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
        base_dir=cfg.reporting_dir,
        cycle=cycle_number,
        summary=p3_summary,
    )

    best_entry = min(latest_by_design.values(), key=_oriented_objective)
    best_eval = dict(best_entry["evaluation"])
    metrics_payload = best_eval.setdefault("metrics", {})
    metrics_payload["cycle_hv"] = p3_summary.hv_score
    best_eval["cycle_hv"] = p3_summary.hv_score
    best_eval["design_hash"] = best_entry.get("design_hash", "")
    cycle_duration = time.perf_counter() - cycle_start
    best_seed = int(best_entry.get("seed", cfg.random_seed))
    previous_baseline = world_model.previous_best_hv(experiment_id, cycle_number)
    current_hv = float(p3_summary.hv_score)
    hv_delta = (
        current_hv - previous_baseline if previous_baseline is not None else current_hv
    )
    budget_controller.record_feedback(
        CycleBudgetFeedback(
            hv_delta=hv_delta,
            feasibility_rate=feasibility_rate,
            cache_hit_rate=cache_hit_rate,
        )
    )
    best_metrics_id: int | None = None
    logged_hashes: Set[str] = set()
    config_snapshot = dict(
        _serialize_experiment_config(cfg, constellaration_sha=constellaration_sha)
    )
    config_snapshot["cycle_seed"] = best_seed
    adapter_version = adapter.current_adapter_version(
        tool_name, cfg.fidelity_ladder.promote
    ) or adapter.current_adapter_version(tool_name, cfg.fidelity_ladder.screen)
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
        "budget_controller": budget_controller.to_dict(),
    }
    cycle_json_path = Path(cfg.reporting_dir) / f"cycle_{cycle_number}.json"

    with world_model.transaction():
        world_model.record_cycle(
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
            problem=cfg.problem,
            commit=False,
        )
        world_model.record_cycle_summary(
            experiment_id=experiment_id,
            cycle_number=cycle_number,
            stage=governance_stage,
            feasible_count=p3_summary.feasible_count,
            hv_score=p3_summary.hv_score,
            commit=False,
        )
        world_model.record_cycle_hv(
            experiment_id=experiment_id,
            cycle_number=cycle_number,
            hv_score=p3_summary.hv_score,
            reference_point=p3_summary.reference_point,
            pareto_entries=[entry.as_mapping() for entry in p3_summary.pareto_entries],
            n_feasible=p3_summary.feasible_count,
            n_archive=p3_summary.archive_size,
            hv_lookback=cfg.governance.hv_lookback,
            commit=False,
        )
        logged_hashes, metrics_by_hash = _persist_pareto_archive(
            world_model=world_model,
            experiment_id=experiment_id,
            cycle_number=cycle_number,
            problem=cfg.problem,
            entries_by_design=latest_by_design,
            p3_summary=p3_summary,
            git_sha=git_sha,
            constellaration_sha=constellaration_sha,
        )
        if (
            best_entry.get("design_hash")
            and best_entry["design_hash"] not in logged_hashes
        ):
            _, metrics_id = world_model.log_candidate(
                experiment_id=experiment_id,
                problem=cfg.problem,
                params=best_entry["params"],
                seed=best_seed,
                status=best_eval.get("stage", "unknown"),
                evaluation=best_eval,
                design_hash=best_entry["design_hash"],
                commit=False,
            )
            best_metrics_id = metrics_id
        world_model.record_deterministic_snapshot(
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
    repro_command = (
        f"python -m ai_scientist.runner --config {cfg.source_config} --problem {cfg.problem} "
        f"--cycles {cfg.cycles} --eval-budget {active_budgets.screen_evals_per_cycle} "
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
            f"conn = sqlite3.connect('{cfg.memory_db}')\n"
            "row = conn.execute(\"\n    'SELECT params_json FROM candidates WHERE design_hash = ? ORDER BY id DESC LIMIT 1',\n    ('{replay_entry.design_hash}',),\n").fetchone()\n"
            "assert row, 'Design hash not found in world model'\n"
            "params = json.loads(row[0])\n"
            f"print(tools.{tool_name}(params, stage='{replay_entry.stage or cfg.fidelity_ladder.promote}'))\n"
            "PY\n"
            "```\n"
        )
    else:
        reproduction_snippet = (
            "No Pareto archive entries available to replay this cycle.\n"
        )
    stage_label = best_eval.get("stage") or cfg.fidelity_ladder.promote
    statement_status = _verify_best_claim(
        world_model=world_model,
        experiment_id=experiment_id,
        cycle_number=cycle_number,
        best_entry=best_entry,
        best_eval=best_eval,
        evaluation_fn=evaluate_fn,
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
        base_dir=cfg.reporting_dir,
        cycle=cycle_number,
        pair={
            "stage": stage_label,
            "status": statement_status,
            "tool_name": tool_name,
            "tool_input_hash": tool_input_hash,
            "reproduction_command": repro_command,
            "metrics": best_eval.get("metrics", {}),
            "design_hash": best_entry.get("design_hash"),
            "problem": cfg.problem,
            "seed": best_seed,
        },
    )
    trajectory_path = adaptation_helpers.append_trajectory_entry(
        base_dir=cfg.reporting_dir,
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
            "problem": cfg.problem,
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
    world_model.log_statement(
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
    statements = world_model.statements_for_cycle(experiment_id, cycle_number)
    figure_path = reporting.save_pareto_figure(
        p3_summary.pareto_entries,
        cfg.reporting_dir,
        title=cfg.problem,
        cycle_index=cycle_index,
    )
    figure_paths = [figure_path] if figure_path else []
    metrics_payload = best_eval.get("metrics", {})
    metrics_path = adaptation_helpers.write_metrics_snapshot(
        base_dir=cfg.reporting_dir,
        cycle=cycle_number,
        payload=metrics_payload,
    )
    artifact_entries: list[tuple[str, Path]] = [("metrics_snapshot", metrics_path)]
    world_model.log_artifact(
        experiment_id=experiment_id,
        path=metrics_path,
        kind="metrics_snapshot",
    )
    artifact_entries.append(("p3_summary", p3_summary_path))
    world_model.log_artifact(
        experiment_id=experiment_id,
        path=p3_summary_path,
        kind="p3_summary",
    )
    artifact_entries.append(("preference_pairs", preference_pairs_path))
    world_model.log_artifact(
        experiment_id=experiment_id,
        path=preference_pairs_path,
        kind="preference_pairs",
    )
    artifact_entries.append(("trajectory_entry", trajectory_path))
    world_model.log_artifact(
        experiment_id=experiment_id,
        path=trajectory_path,
        kind="trajectory_entry",
    )
    if figure_path:
        artifact_entries.append(("pareto_figure", figure_path))
        world_model.log_artifact(
            experiment_id=experiment_id,
            path=figure_path,
            kind="pareto_figure",
        )
    world_model.record_stage_history(
        experiment_id=experiment_id,
        cycle=cycle_number,
        stage=governance_stage,
    )
    stage_history_entries = world_model.stage_history(experiment_id)
    property_graph_summary = world_model.property_graph_summary(experiment_id)
    rag_citations = (
        property_graph_summary.get("citations") if property_graph_summary else None
    )
    adaptation_figures = reporting.collect_adaptation_figures(cfg.reporting_dir)
    anchor_candidates = (
        ("preference_pairs", preference_pairs_anchor),
        ("p3_summary", p3_summary_anchor),
        ("trajectory", trajectory_anchor),
    )
    positioning_artifacts = {name: anchor for name, anchor in anchor_candidates if anchor is not None}
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
        problem=cfg.problem,
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
        out_dir=cfg.reporting_dir,
    )

    title = f"{cfg.problem}_cycle_{cycle_index + 1}"
    report_path = reporting.write_report(title, content, out_dir=cfg.reporting_dir)
    return report_path, best_eval, p3_summary


def run(
    cfg: ai_config.ExperimentConfig, runtime: RunnerCLIConfig | None = None
) -> None:
    index_status = rag.ensure_index()
    runtime_label = (
        f"screen_only={runtime.screen_only} promote_only={runtime.promote_only} "
        f"log_cache_stats={runtime.log_cache_stats} slow={runtime.slow} "
        f"planner={runtime.planner} preset={runtime.run_preset or 'none'}"
        if runtime
        else "default"
    )
    print(
        f"[runner] RAG index ready: {index_status.chunks_indexed} chunks ({index_status.index_path}); runtime={runtime_label}"
    )
    tools.clear_evaluation_cache()
    planner_mode = (
        runtime.planner.lower() if runtime and runtime.planner else "deterministic"
    )
    budget_controller = BudgetController(cfg.budgets, cfg.adaptive_budgets)
    last_p3_summary: tools.P3Summary | None = None
    surrogate_model = _create_surrogate(cfg)
    generative_model = _create_generative_model(cfg)

    with memory.WorldModel(cfg.memory_db) as world_model:
        git_sha = _resolve_git_sha()
        constellaration_sha = _resolve_git_sha("constellaration")
        
        experiment_id: int
        start_cycle_index = 0
        stage_history: list[CycleSummary] = []
        governance_stage = "s1"

        if runtime and runtime.resume_from:
            resume_data = json.loads(runtime.resume_from.read_text(encoding="utf-8"))
            resume_cycle = int(resume_data["cycle"])
            # We expect experiment_id in the checkpoint (deterministic resume)
            # If missing (legacy checkpoint), we fail fast per feedback requirements
            if "experiment_id" not in resume_data:
                raise ValueError(f"Checkpoint {runtime.resume_from} missing experiment_id; cannot resume deterministically.")
            
            experiment_id = int(resume_data["experiment_id"])
            print(f"[runner] resuming experiment_id={experiment_id} from cycle {resume_cycle}")
            
            # Verify DB consistency
            if world_model.cycles_completed(experiment_id) < resume_cycle:
                 # This might happen if DB write failed but JSON wrote, or DB was deleted.
                 print(f"[runner] warning: DB has fewer cycles than checkpoint for exp {experiment_id}")
            
            start_cycle_index = resume_cycle
            
            # Restore history for governance transitions
            restored = world_model.cycle_summaries(experiment_id)
            for row in restored:
                stage_history.append(
                    CycleSummary(
                        cycle=row["cycle"],
                        objective=row["objective"],
                        feasibility=row["feasibility"],
                        hv=row["hv"],
                        stage=row["stage"],
                    )
                )
            
            # Determine governance stage for the NEXT cycle (start_cycle_index + 1)
            # We run transition checks on the full history up to resume_cycle
            last_stage = stage_history[-1].stage if stage_history else "s1"
            governance_stage = last_stage
            
            # Check transitions based on restored history
            if last_stage == "s1" and _should_transition_s1_to_s2(stage_history, cfg.stage_gates):
                governance_stage = "s2"
            elif last_stage == "s2" and _should_transition_s2_to_s3(
                stage_history, cfg.stage_gates, cfg.governance, world_model, experiment_id, resume_cycle, cfg.cycles
            ):
                governance_stage = "s3"
                
            print(f"[runner] resumed state: start_index={start_cycle_index} next_stage={governance_stage}")
            bc_state = resume_data.get("budget_controller")
            if bc_state:
                budget_controller.restore(bc_state)

        else:
            experiment_id = world_model.start_experiment(
                _serialize_experiment_config(cfg, constellaration_sha=constellaration_sha),
                git_sha,
                constellaration_sha=constellaration_sha,
            )
            if runtime and runtime.promote_only:
                governance_stage = "s2"
                print("[runner] promote-only flag engaged; starting governance in S2.")

        # Reset deterministic RNG baselines after resume/start so subsequent cycles match a fresh run.
        np.random.seed(cfg.random_seed + start_cycle_index)
        random.seed(cfg.random_seed + start_cycle_index)

        planning_agent = (
            ai_planner.PlanningAgent(world_model=world_model)
            if planner_mode == "agent"
            else None
        )
        last_best_objective: float | None = None
        if stage_history:
            # Seed last_best_objective so reward diffs remain monotonic after resume
            last_best_objective = next(
                (entry.objective for entry in reversed(stage_history) if entry.objective is not None),
                None,
            )
        
        last_feasibility_rate: float | None = None

        for idx in range(cfg.cycles):
            cycle_number = idx + 1
            if idx < start_cycle_index:
                continue

            print(
                f"[runner] starting cycle {cycle_number} stage={governance_stage.upper()} "
                f"screen_budget={cfg.budgets.screen_evals_per_cycle}"
            )
            suggested_params: list[Mapping[str, Any]] | None = None
            config_overrides: Mapping[str, Any] | None = None
            if planning_agent:
                # We pass the *reconstructed* stage_history
                stage_payload = [
                    {
                        "cycle": entry.cycle,
                        "stage": entry.stage,
                        "selected_at": datetime.now(timezone.utc).isoformat(), # Approximation if not tracked precisely in summary
                    }
                    for entry in stage_history
                ]
                # Note: 'selected_at' in stage_history table exists, but CycleSummary doesn't store it. 
                # For planner context, accurate timestamp is less critical than the sequence.
                
                plan_outcome = planning_agent.plan_cycle(
                    cfg=cfg,
                    cycle_index=idx,
                    stage_history=stage_payload,
                    last_summary=last_p3_summary,
                    experiment_id=experiment_id,
                )
                context_snapshot = json.dumps(plan_outcome.context, indent=2)
                print(f"[planner][cycle={idx + 1}] context:\n{context_snapshot}")
                
                if plan_outcome.suggested_params:
                    suggested_params = [plan_outcome.suggested_params]
                if plan_outcome.config_overrides:
                    config_overrides = plan_outcome.config_overrides

            report_path, best_eval, p3_summary = _run_cycle(
                cfg, 
                idx, 
                world_model, 
                experiment_id, 
                governance_stage, 
                git_sha, 
                constellaration_sha, 
                surrogate_model, 
                runtime=runtime, 
                budget_controller=budget_controller, 
                prev_feasibility_rate=last_feasibility_rate, 
                suggested_params=suggested_params, 
                config_overrides=config_overrides, 
                generative_model=generative_model,
            )
            last_p3_summary = p3_summary
            if p3_summary:
                total = max(1, p3_summary.feasible_count + (p3_summary.archive_size or 0)) # approximate total if not tracked directly
                # Actually p3_summary doesn't store total candidates, but _run_cycle logs feasibility_rate to JSON.
                # We can infer it from the feasible_count if we knew the total. 
                # Better: _run_cycle *returns* p3_summary, but feasibility_rate was local. 
                # Let's check _run_cycle output. It returns (Path, dict, P3Summary).
                # The P3Summary has feasible_count. 
                # We can track feasibility rate if we modify _run_cycle to return it or just assume 0.5 for now?
                # No, better to capture it. 
                # However, modifying return signature again is annoying. 
                # Let's assume feasibility_rate ~ p3_summary.feasible_count / (screened + promoted) roughly?
                # Actually, best_eval has "feasibility_rate" in the JSON payload logic but not in the returned dict structure clearly. 
                # Let's look at how `last_feedback` was constructed in _run_cycle: 
                # `feasibility_rate = float(p3_summary.feasible_count) / float(max(1, total_designs))`
                # `total_designs` came from `len(latest_by_design)`.
                # We can approximately reconstruct it or just let the decay be 0 for the first pass if not critical. 
                # BUT, I can calculate it here if I assume total_designs is roughly comparable to what p3_summary holds. 
                # `p3_summary` holds `pareto_entries` (tuple). 
                # The simplest way is to trust that if feasible_count > 0, rate > 0. 
                # Let's check `p3_summary` fields. 
                pass
            
            # To properly implement decay, I need the rate. 
            # I will check budget_controller._last_feedback since it WAS recorded in _run_cycle.
            if budget_controller._last_feedback and budget_controller._last_feedback.feasibility_rate is not None:
                last_feasibility_rate = budget_controller._last_feedback.feasibility_rate
            if report_path:
                print(f"[runner] cycle {idx + 1} report saved to {report_path}")
            else:
                print(f"[runner] cycle {idx + 1} aborted (wall-clock or budget).")
            summary = CycleSummary(
                cycle=idx + 1,
                objective=best_eval.get("objective") if best_eval else None,
                feasibility=best_eval.get("feasibility") if best_eval else None,
                hv=best_eval.get("cycle_hv") if best_eval else None,
                stage=governance_stage,
            )
            stage_history.append(summary)
            if best_eval:
                current_objective = best_eval.get("objective")
                reward_diff = 0.0
                if current_objective is not None and last_best_objective is not None:
                    reward_diff = float(current_objective) - float(last_best_objective)
                adaptation_helpers.append_preference_record(
                    base_dir=cfg.reporting_dir,
                    record={
                        "cycle": idx + 1,
                        "stage": governance_stage,
                        "candidate_hash": best_eval.get("design_hash", "") or "",
                        "reward_diff": reward_diff,
                    },
                )
                if current_objective is not None:
                    last_best_objective = float(current_objective)
            
            next_stage = governance_stage
            if governance_stage == "s1":
                if _should_transition_s1_to_s2(stage_history, cfg.stage_gates):
                    next_stage = "s2"
                    print(
                        f"[runner][stage-gate] governance stage advanced to S2 after cycle {idx + 1}"
                    )
            elif governance_stage == "s2":
                if _should_transition_s2_to_s3(
                    stage_history,
                    cfg.stage_gates,
                    cfg.governance,
                    world_model,
                    experiment_id,
                    idx + 1,
                    cfg.cycles,
                ):
                    next_stage = "s3"
                    print(
                        f"[runner][stage-gate] governance stage advanced to S3 after cycle {idx + 1}"
                    )
            governance_stage = next_stage
        batch_summary_path = _export_batch_reports(cfg.reporting_dir, stage_history)
        world_model.log_artifact(
            experiment_id=experiment_id,
            path=batch_summary_path,
            kind="batch_summary",
        )
        usage = world_model.budget_usage(experiment_id)
        print(
            f"[runner] logged {usage.screen_evals} screen + {usage.promoted_evals} promote evaluations ("
            f"{usage.high_fidelity_evals} high-fidelity) into {cfg.memory_db}",
        )


def _serialize_experiment_config(
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
    }


_RUN_PRESETS_PATH = Path("configs/run_presets.yaml")


def _load_run_presets(path: Path | str | None = None) -> dict[str, dict[str, bool]]:
    target = Path(path or _RUN_PRESETS_PATH)
    if not target.exists():
        return {}
    raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    presets: dict[str, dict[str, bool]] = {}
    for key, values in raw.items():
        if not isinstance(values, dict):
            continue
        presets[key] = {
            "screen_only": bool(values.get("screen_only", False)),
            "promote_only": bool(values.get("promote_only", False)),
            "slow": bool(values.get("slow", False)),
        }
    return presets


def _apply_run_preset(cli: RunnerCLIConfig) -> RunnerCLIConfig:
    preset_name = cli.run_preset or os.getenv("AI_SCIENTIST_RUN_PRESET")
    if not preset_name:
        return cli
    presets = _load_run_presets()
    preset = presets.get(preset_name)
    if preset is None:
        raise ValueError(
            "Unknown run preset '%s'; available presets are %s."
            % (preset_name, ", ".join(sorted(presets or ["<none>"])))
        )
    return replace(
        cli,
        screen_only=cli.screen_only or preset["screen_only"],
        promote_only=cli.promote_only or preset["promote_only"],
        slow=cli.slow or preset["slow"],
    )


def _export_batch_reports(
    report_dir: Path | str, history: Sequence[CycleSummary]
) -> Path:
    base_path = Path(report_dir)
    figures_dir = base_path / "figures"
    stage_dir = figures_dir / "batch_stage_summaries"
    figures_dir.mkdir(parents=True, exist_ok=True)
    stage_dir.mkdir(parents=True, exist_ok=True)
    stage_entries: dict[str, list[CycleSummary]] = {}
    for cycle_summary in history:
        stage_entries.setdefault(cycle_summary.stage, []).append(cycle_summary)
    stage_refs: dict[str, dict[str, Any]] = {}
    for stage, entries in stage_entries.items():
        objectives = [
            entry.objective for entry in entries if entry.objective is not None
        ]
        feasibilities = [
            entry.feasibility for entry in entries if entry.feasibility is not None
        ]
        hv_values = [entry.hv for entry in entries if entry.hv is not None]
        stage_payload = {
            "stage": stage,
            "cycles": len(entries),
            "best_objective": max(objectives) if objectives else None,
            "best_feasibility": min(feasibilities) if feasibilities else None,
            "max_hv": max(hv_values) if hv_values else None,
            "entries": [
                {
                    "cycle": entry.cycle,
                    "objective": entry.objective,
                    "feasibility": entry.feasibility,
                    "hv": entry.hv,
                }
                for entry in entries
            ],
        }
        stage_path = stage_dir / f"{stage}_summary.json"
        stage_path.write_text(json.dumps(stage_payload, indent=2), encoding="utf-8")
        stage_refs[stage] = {
            "cycles": len(entries),
            "path": str(stage_path.resolve()),
        }
    summary_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_cycles": len(history),
        "stage_files": stage_refs,
    }
    summary_path = figures_dir / "batch_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    return summary_path


def _resolve_git_sha(repo_path: str | None = None) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", repo_path, "rev-parse", "HEAD"]
            if repo_path
            else ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


def main() -> None:
    try:
        cli = _apply_run_preset(parse_args())
        _validate_runtime_flags(cli)
    except ValueError as exc:
        print(f"[runner] invalid CLI flags: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    experiment = ai_config.load_experiment_config(cli.config_path)
    if cli.problem:
        experiment = replace(experiment, problem=cli.problem)
    if cli.cycles:
        experiment = replace(experiment, cycles=cli.cycles)
    if cli.memory_db:
        experiment = replace(experiment, memory_db=cli.memory_db)
    if cli.eval_budget is not None:
        experiment = replace(
            experiment,
            budgets=replace(experiment.budgets, screen_evals_per_cycle=cli.eval_budget),
        )
    if cli.workers is not None:
        experiment = replace(
            experiment,
            budgets=replace(experiment.budgets, n_workers=cli.workers),
        )
    if cli.pool_type is not None:
        experiment = replace(
            experiment,
            budgets=replace(experiment.budgets, pool_type=cli.pool_type),
        )
    if cli.slow:
        experiment = replace(
            experiment,
            budgets=replace(
                experiment.budgets,
                wall_clock_minutes=experiment.budgets.wall_clock_minutes * 1.5,
            ),
        )
    preset_label = cli.run_preset or os.getenv("AI_SCIENTIST_RUN_PRESET") or "none"
    print(
        f"[runner] starting problem={experiment.problem} cycles={experiment.cycles} "
        f"screen_budget={experiment.budgets.screen_evals_per_cycle} "
        f"screen_only={cli.screen_only} promote_only={cli.promote_only} "
        f"log_cache_stats={cli.log_cache_stats} slow={cli.slow} preset={preset_label}"
    )
    run(experiment, runtime=cli)


if __name__ == "__main__":
    main()