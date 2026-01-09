"""
Configuration and setup helpers for the AI Scientist runner.
"""

import argparse
import os
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import yaml

from ai_scientist import config as ai_config
from ai_scientist.optim.generative import DiffusionDesignModel, GenerativeDesignModel
from ai_scientist.optim.surrogate import BaseSurrogate, SurrogateBundle
from ai_scientist.optim.surrogate_v2 import NeuralOperatorSurrogate


def create_surrogate(cfg: ai_config.ExperimentConfig) -> BaseSurrogate:
    """Factory to create the appropriate surrogate model based on config."""
    if cfg.surrogate.backend == "neural_operator":
        print(
            f"[runner] V2 Active: Initializing NeuralOperatorSurrogate (Deep Learning Backend, Ensembles={cfg.surrogate.n_ensembles})."
        )
        surrogate = NeuralOperatorSurrogate(
            min_samples=cfg.surrogate.min_samples,
            points_cadence=cfg.surrogate.points_cadence,
            cycle_cadence=cfg.surrogate.cycle_cadence,
            device=cfg.surrogate.device,
            learning_rate=cfg.surrogate.learning_rate,
            epochs=cfg.surrogate.epochs,
            batch_size=cfg.surrogate.batch_size,
            n_ensembles=cfg.surrogate.n_ensembles,
            hidden_dim=cfg.surrogate.hidden_dim,
        )

        if cfg.surrogate.use_offline_dataset:
            checkpoint_path = Path("checkpoints/surrogate_physics_v2.pt")
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Surrogate checkpoint not found at {checkpoint_path}. "
                    "Please run 'python scripts/train_surrogate.py' or set "
                    "surrogate.use_offline_dataset=false in your config."
                )
            surrogate.load_checkpoint(checkpoint_path)

        return surrogate
    return SurrogateBundle()


def create_generative_model(
    cfg: ai_config.ExperimentConfig,
) -> GenerativeDesignModel | DiffusionDesignModel | None:
    """Factory to create the generative model if enabled."""
    if not cfg.generative.enabled:
        return None

    if cfg.generative.backend == "diffusion":
        print("[runner] Generative Model Enabled (Diffusion).")
        model = DiffusionDesignModel(
            learning_rate=cfg.generative.learning_rate,
            epochs=cfg.generative.epochs,
            # StellarForge Upgrades
            hidden_dim=cfg.generative.hidden_dim,
            n_layers=cfg.generative.n_layers,
            pca_components=cfg.generative.pca_components,
            batch_size=cfg.generative.batch_size,
            diffusion_timesteps=cfg.generative.diffusion_timesteps,
            device=cfg.generative.device,
        )

        if cfg.generative.checkpoint_path:
            if cfg.generative.checkpoint_path.exists():
                print(
                    f"[runner] Loading Generative Model checkpoint: {cfg.generative.checkpoint_path}"
                )
                model.load_checkpoint(cfg.generative.checkpoint_path)
            else:
                print(
                    f"[runner] Warning: Checkpoint not found at {cfg.generative.checkpoint_path}"
                )

        return model

    print("[runner] Generative Model Enabled (VAE).")
    return GenerativeDesignModel(
        latent_dim=cfg.generative.latent_dim,
        learning_rate=cfg.generative.learning_rate,
        epochs=cfg.generative.epochs,
        kl_weight=cfg.generative.kl_weight,
    )


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
    disable_rl: bool = False
    resume_from: Path | None = None
    aso: bool = False
    preset: str | None = None


def build_argument_parser() -> argparse.ArgumentParser:
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
        "--preset",
        choices=["p3-high-fidelity", "p3-quick", "p3-aso"],
        help="Use a predefined configuration preset. Overrides --config.",
    )
    parser.add_argument(
        "--planner",
        choices=["deterministic", "agent"],
        default=None,
        help="Choose the planning driver (deterministic loop or Phase 3 agent).",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Path to a cycle checkpoint JSON to resume from (skips completed cycles).",
    )
    parser.add_argument(
        "--aso",
        action="store_true",
        help="Enable Agent-Supervised Optimization with real ALM state",
    )
    parser.add_argument(
        "--no-rl",
        action="store_true",
        help="Disable RL refinement (PPO-CMA) in the Phase 5 Coordinator pipeline.",
    )
    return parser


def parse_args(args: Sequence[str] | None = None) -> RunnerCLIConfig:
    parser = build_argument_parser()
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
        disable_rl=bool(namespace.no_rl),
        resume_from=namespace.resume_from,
        aso=bool(namespace.aso),
        preset=namespace.preset,
    )


def validate_runtime_flags(runtime: RunnerCLIConfig) -> None:
    if runtime.screen_only and runtime.promote_only:
        raise ValueError(
            "--screen (S1-only) cannot be combined with promote-only mode (S2+) "
            f"(presets: {runtime.run_preset or '<none>'}). Remove one flag/preset."
        )


def resolve_git_sha(repo_path: str | None = None) -> str:
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


_RUN_PRESETS_PATH = Path("configs/run_presets.yaml")


def load_run_presets(path: Path | str | None = None) -> dict[str, dict[str, bool]]:
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


def apply_run_preset(cli: RunnerCLIConfig) -> RunnerCLIConfig:
    preset_name = cli.run_preset or os.getenv("AI_SCIENTIST_RUN_PRESET")
    if not preset_name:
        return cli
    presets = load_run_presets()
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
