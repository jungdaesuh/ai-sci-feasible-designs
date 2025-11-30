"""
Experiment runner module.
Orchestrates the experiment execution by composing BudgetController, FidelityController, CycleExecutor, and Coordinator.
"""
from __future__ import annotations

import json
import sys
import random
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence
from pathlib import Path

import numpy as np
from dataclasses import replace

from ai_scientist import config as ai_config
from ai_scientist import memory
from ai_scientist import planner as ai_planner
from ai_scientist import rag
from ai_scientist import tools
from ai_scientist.budget_manager import BudgetController
from ai_scientist.fidelity_controller import CycleSummary, FidelityController
from ai_scientist.coordinator import Coordinator
from ai_scientist.cycle_executor import CycleExecutor, serialize_experiment_config
from ai_scientist.optim.surrogate_v2 import NeuralOperatorSurrogate
from ai_scientist.experiment_setup import (
    RunnerCLIConfig,
    parse_args,
    apply_run_preset,
    validate_runtime_flags,
    create_surrogate,
    create_generative_model,
    resolve_git_sha,
)
from orchestration import adaptation as adaptation_helpers


def run_experiment(
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
    surrogate_model = create_surrogate(cfg)
    generative_model = create_generative_model(cfg)

    with memory.WorldModel(cfg.memory_db) as world_model:
        git_sha = resolve_git_sha()
        constellaration_sha = resolve_git_sha("constellaration")
        
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

        coordinator = Coordinator(
            cfg, 
            world_model, 
            planner=planning_agent or ai_planner.PlanningAgent(world_model=world_model),
            surrogate=surrogate_model if isinstance(surrogate_model, NeuralOperatorSurrogate) else None, 
            generative_model=generative_model
        )
        
        cycle_executor = CycleExecutor(
            config=cfg,
            world_model=world_model,
            planner=planning_agent,
            coordinator=coordinator, 
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


def main() -> None:
    try:
        cli = apply_run_preset(parse_args())
        validate_runtime_flags(cli)
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
    run_experiment(experiment, runtime=cli)


if __name__ == "__main__":
    main()
