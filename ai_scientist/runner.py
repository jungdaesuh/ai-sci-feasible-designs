"Runner that wires budgets, fidelity decisions, and minimal reporting (Tasks 4.1 + B.*)."

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import random
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from ai_scientist import config as ai_config
from ai_scientist import memory
from ai_scientist import planner as ai_planner
from ai_scientist import rag
from ai_scientist import tools
from ai_scientist.budget_manager import BudgetController
from ai_scientist.fidelity_controller import CycleSummary, FidelityController
from ai_scientist.optim.surrogate import BaseSurrogate, SurrogateBundle
from ai_scientist.optim.surrogate_v2 import NeuralOperatorSurrogate
from ai_scientist.optim.generative import GenerativeDesignModel, DiffusionDesignModel
from orchestration import adaptation as adaptation_helpers

from ai_scientist.cycle_executor import CycleExecutor, serialize_experiment_config


def _create_surrogate(cfg: ai_config.ExperimentConfig) -> BaseSurrogate:
    """Factory to create the appropriate surrogate model based on config."""
    if cfg.surrogate.backend == "neural_operator":
        print(f"[runner] V2 Active: Initializing NeuralOperatorSurrogate (Deep Learning Backend, Ensembles={cfg.surrogate.n_ensembles}).")
        surrogate = NeuralOperatorSurrogate(
            learning_rate=cfg.surrogate.learning_rate,
            epochs=cfg.surrogate.epochs,
            n_ensembles=cfg.surrogate.n_ensembles,
            hidden_dim=cfg.surrogate.hidden_dim,
        )
        
        if cfg.surrogate.use_offline_dataset:
            ckpt_path = Path("checkpoints/surrogate_physics_v2.pt")
            if ckpt_path.exists():
                print(f"[runner] Loading offline surrogate checkpoint: {ckpt_path}")
                surrogate.load_checkpoint(ckpt_path)
            else:
                print(f"[runner] Warning: use_offline_dataset=True but {ckpt_path} not found. Starting cold.")
        
        return surrogate
    return SurrogateBundle()


def _create_generative_model(cfg: ai_config.ExperimentConfig) -> GenerativeDesignModel | DiffusionDesignModel | None:
    """Factory to create the generative model if enabled."""
    if not cfg.generative.enabled:
        return None
        
    if cfg.generative.backend == "diffusion":
        print("[runner] Generative Model Enabled (Diffusion).")
        return DiffusionDesignModel(
            learning_rate=cfg.generative.learning_rate,
            epochs=cfg.generative.epochs,
            min_samples=32,
        )

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
    resume_from: Path | None = None
    aso: bool = False

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
        aso=bool(namespace.aso),
    )


def _validate_runtime_flags(runtime: RunnerCLIConfig) -> None:
    if runtime.screen_only and runtime.promote_only:
        raise ValueError(
            "--screen (S1-only) cannot be combined with promote-only mode (S2+) "
            f"(presets: {runtime.run_preset or '<none>'}). Remove one flag/preset."
        )

def run(
    cfg: ai_config.ExperimentConfig,
    runtime: RunnerCLIConfig | None = None,
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
        runtime.planner.lower()
        if runtime and runtime.planner
        else cfg.planner.lower()
    )
    budget_controller = BudgetController(cfg)
    fidelity_ctl = FidelityController(cfg)
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
            if "experiment_id" not in resume_data:
                raise ValueError(f"Checkpoint {runtime.resume_from} missing experiment_id; cannot resume deterministically.")
            
            experiment_id = int(resume_data["experiment_id"])
            print(f"[runner] resuming experiment_id={experiment_id} from cycle {resume_cycle}")
            
            if world_model.cycles_completed(experiment_id) < resume_cycle:
                 print(f"[runner] warning: DB has fewer cycles than checkpoint for exp {experiment_id}")
            
            start_cycle_index = resume_cycle
            
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
            
            last_stage = stage_history[-1].stage if stage_history else "s1"
            governance_stage = last_stage
            
            if last_stage == "s1" and fidelity_ctl.should_transition_s1_to_s2(stage_history):
                governance_stage = "s2"
            elif last_stage == "s2" and fidelity_ctl.should_transition_s2_to_s3(
                stage_history,
                world_model,
                experiment_id,
                resume_cycle,
            ):
                governance_stage = "s3"
                
            print(f"[runner] resumed state: start_index={start_cycle_index} next_stage={governance_stage}")
            bc_state = resume_data.get("budget_controller")
            if bc_state:
                budget_controller.restore(bc_state)

        else:
            experiment_id = world_model.start_experiment(
                serialize_experiment_config(cfg, constellaration_sha=constellaration_sha),
                git_sha,
                constellaration_sha=constellaration_sha,
            )
            if runtime and runtime.promote_only:
                governance_stage = "s2"
                print("[runner] promote-only flag engaged; starting governance in S2.")

        np.random.seed(cfg.random_seed + start_cycle_index)
        random.seed(cfg.random_seed + start_cycle_index)

        planning_agent = (
            ai_planner.PlanningAgent(world_model=world_model)
            if planner_mode == "agent"
            else None
        )
        
        cycle_executor = CycleExecutor(
            config=cfg,
            world_model=world_model,
            planner=planning_agent,
            coordinator=None, 
            budget_controller=budget_controller,
            fidelity_controller=fidelity_ctl
        )

        last_best_objective: float | None = None
        if stage_history:
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
                stage_payload = [
                    {
                        "cycle": entry.cycle,
                        "stage": entry.stage,
                        "selected_at": datetime.now(timezone.utc).isoformat(),
                    }
                    for entry in stage_history
                ]
                
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

            result = cycle_executor.run_cycle(
                cycle_index=idx,
                experiment_id=experiment_id,
                governance_stage=governance_stage,
                git_sha=git_sha,
                constellaration_sha=constellaration_sha,
                surrogate_model=surrogate_model,
                generative_model=generative_model,
                prev_feasibility_rate=last_feasibility_rate,
                suggested_params=suggested_params,
                config_overrides=config_overrides,
                verbose=bool(runtime and runtime.verbose),
                slow=bool(runtime and runtime.slow),
                screen_only=bool(runtime and runtime.screen_only),
                log_cache_stats=bool(runtime and runtime.log_cache_stats),
            )
            
            last_p3_summary = result.p3_summary
            last_feasibility_rate = result.feasibility_rate

            if result.report_path:
                print(f"[runner] cycle {idx + 1} report saved to {result.report_path}")
            else:
                print(f"[runner] cycle {idx + 1} aborted (wall-clock or budget).")
            
            summary = CycleSummary(
                cycle=idx + 1,
                objective=result.best_eval.get("objective") if result.best_eval else None,
                feasibility=result.best_eval.get("feasibility") if result.best_eval else None,
                hv=result.hypervolume,
                stage=governance_stage,
            )
            stage_history.append(summary)
            if result.best_eval:
                current_objective = result.best_eval.get("objective")
                reward_diff = 0.0
                if current_objective is not None and last_best_objective is not None:
                    reward_diff = float(current_objective) - float(last_best_objective)
                adaptation_helpers.append_preference_record(
                    base_dir=cfg.reporting_dir,
                    record={
                        "cycle": idx + 1,
                        "stage": governance_stage,
                        "candidate_hash": result.best_eval.get("design_hash", "") or "",
                        "reward_diff": reward_diff,
                    },
                )
                if current_objective is not None:
                    last_best_objective = float(current_objective)
            
            next_stage = governance_stage
            if governance_stage == "s1":
                if fidelity_ctl.should_transition_s1_to_s2(stage_history):
                    next_stage = "s2"
                    print(
                        f"[runner][stage-gate] governance stage advanced to S2 after cycle {idx + 1}"
                    )
            elif governance_stage == "s2":
                if fidelity_ctl.should_transition_s2_to_s3(
                    stage_history,
                    world_model,
                    experiment_id,
                    idx + 1,
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
    report_dir: Path | str,
    history: Sequence[CycleSummary],
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
    if cli.aso:
        experiment = replace(experiment, aso=replace(experiment.aso, enabled=True))
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