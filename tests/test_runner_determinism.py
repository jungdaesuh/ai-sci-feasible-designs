from __future__ import annotations

import json
import shutil
import sqlite3
from dataclasses import replace
from typing import Mapping
from unittest.mock import patch

from ai_scientist import config as ai_config
from ai_scientist import cycle_executor, memory, runner, tools
from ai_scientist.budget_manager import BudgetController, CycleBudgetFeedback


class _DummyWorldModel:
    def __init__(self) -> None:
        self.logged: list[Mapping[str, object]] = []

    def log_statement(
        self,
        experiment_id: int,
        cycle: int,
        stage: str,
        text: str,
        status: str,
        tool_name: str,
        tool_input: Mapping[str, object],
        *,
        metrics_id: int | None = None,
        seed: int | None = None,
        git_sha: str,
        repro_cmd: str,
        created_at: str | None = None,
        commit: bool = True,
    ) -> int:
        self.logged.append(
            {
                "experiment_id": experiment_id,
                "cycle": cycle,
                "stage": stage,
                "status": status,
                "tool_name": tool_name,
                "seed": seed,
            }
        )
        return len(self.logged)


def _stub_runner_evaluator():
    def evaluate(boundary_params, *, stage, use_cache=True):
        r_cos = boundary_params.get("r_cos", [])
        z_sin = boundary_params.get("z_sin", [])
        r_total = sum(sum(row) for row in r_cos)
        z_total = sum(sum(row) for row in z_sin)
        gradient = float(abs(r_total - z_total)) + 1.0
        aspect = float((r_total + z_total) % 5.0 + 1.0)
        return {
            "stage": stage,
            "objective": gradient,
            "minimize_objective": True,
            "feasibility": 0.0,
            "hv": gradient,
            "metrics": {
                "minimum_normalized_magnetic_gradient_scale_length": gradient,
                "aspect_ratio": aspect,
            },
        }

    return evaluate


def test_verify_best_claim_supported_when_seed_replay_matches():
    world_model = _DummyWorldModel()
    best_seed = 7
    best_entry = {"params": {"seed": best_seed}, "design_hash": "seed-hash"}
    best_eval = {
        "stage": "screen",
        "objective": best_seed * 0.01,
        "feasibility": 0.0,
        "hv": best_seed * 0.1,
    }

    def evaluation_fn(
        boundary_params: Mapping[str, float], *, stage: str, use_cache: bool = True
    ):
        seed_value = float(boundary_params.get("seed", 0))
        return {
            "stage": stage,
            "objective": seed_value * 0.01,
            "feasibility": 0.0,
            "hv": seed_value * 0.1,
        }

    status = cycle_executor._verify_best_claim(
        world_model=world_model,
        experiment_id=1,
        cycle_number=1,
        best_entry=best_entry,
        best_eval=best_eval,
        evaluation_fn=evaluation_fn,
        tool_name="evaluate_p3",
        best_seed=best_seed,
        git_sha="deadbeef",
        reproduction_command="echo replay",
        stage="screen",
        metrics_id=42,
    )

    assert status == "SUPPORTED"
    assert world_model.logged[-1]["status"] == "SUPPORTED"


def test_verify_best_claim_refuted_when_metric_differs_beyond_tolerance():
    world_model = _DummyWorldModel()
    best_seed = 13
    best_entry = {"params": {"seed": best_seed}, "design_hash": "seed-hash"}
    best_eval = {
        "stage": "promote",
        "objective": best_seed * 0.01,
        "feasibility": 0.0,
        "hv": best_seed * 0.1,
    }

    def evaluation_fn(
        boundary_params: Mapping[str, float], *, stage: str, use_cache: bool = True
    ):
        seed_value = float(boundary_params.get("seed", 0))
        # Introduce a shift larger than the allowed tolerance
        return {
            "stage": stage,
            "objective": seed_value * 0.01 + 0.02,
            "feasibility": 0.0,
            "hv": seed_value * 0.1,
        }

    status = cycle_executor._verify_best_claim(
        world_model=world_model,
        experiment_id=2,
        cycle_number=1,
        best_entry=best_entry,
        best_eval=best_eval,
        evaluation_fn=evaluation_fn,
        tool_name="evaluate_p3",
        best_seed=best_seed,
        git_sha="feedbeef",
        reproduction_command="echo replay",
        stage="promote",
        metrics_id=7,
    )

    assert status == "REFUTED"
    assert world_model.logged[-1]["status"] == "REFUTED"


def test_run_cycle_deterministic_snapshot(tmp_path):
    base_config = ai_config.load_experiment_config(
        ai_config.DEFAULT_EXPERIMENT_CONFIG_PATH
    )
    report_dir = tmp_path / "reports"
    db_path = tmp_path / "world.db"
    cfg = replace(
        base_config,
        cycles=1,
        reporting_dir=report_dir,
        memory_db=db_path,
        budgets=replace(
            base_config.budgets,
            screen_evals_per_cycle=2,
            promote_top_k=1,
            max_high_fidelity_evals_per_cycle=1,
            n_workers=1,
        ),
    )
    git_sha = "runner-det"
    constellaration_sha = "const-det"

    def capture_snapshot():
        if db_path.exists():
            db_path.unlink()
        if report_dir.exists():
            shutil.rmtree(report_dir)
        tools.clear_evaluation_cache()
        with memory.WorldModel(db_path) as world_model:
            experiment_id = world_model.start_experiment(
                runner.serialize_experiment_config(
                    cfg, constellaration_sha=constellaration_sha
                ),
                git_sha,
                constellaration_sha=constellaration_sha,
            )
            with (
                patch(
                    "ai_scientist.cycle_executor._problem_evaluator",
                    return_value=_stub_runner_evaluator(),
                ),
                patch(
                    "ai_scientist.cycle_executor._problem_tool_name",
                    return_value="stub_tool",
                ),
            ):
                budget_controller = runner.BudgetController(cfg)
                fidelity_controller = runner.FidelityController(cfg)
                executor = cycle_executor.CycleExecutor(
                    config=cfg,
                    world_model=world_model,
                    planner=None,
                    coordinator=None,
                    budget_controller=budget_controller,
                    fidelity_controller=fidelity_controller,
                )
                executor.run_cycle(
                    cycle_index=0,
                    experiment_id=experiment_id,
                    governance_stage="s1",
                    git_sha=git_sha,
                    constellaration_sha=constellaration_sha,
                    surrogate_model=runner.SurrogateBundle(),
                )
        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT snapshot_json, seed FROM deterministic_snapshots WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone()
        conn.close()
        assert row is not None
        return json.loads(row[0]), row[1]

    snapshot_a, seed_a = capture_snapshot()
    snapshot_b, seed_b = capture_snapshot()
    assert seed_a == seed_b
    assert snapshot_a == snapshot_b


def test_budget_controller_serialization_roundtrip():
    base = ai_config.BudgetConfig(
        screen_evals_per_cycle=4,
        promote_top_k=2,
        max_high_fidelity_evals_per_cycle=1,
        wall_clock_minutes=1.0,
        n_workers=1,
        pool_type="thread",
    )
    adaptive = ai_config.AdaptiveBudgetConfig(
        enabled=True,
        hv_slope_reference=0.1,
        feasibility_target=0.5,
        cache_hit_target=0.5,
        screen_bounds=ai_config.BudgetRangeConfig(min=1, max=8),
        promote_top_k_bounds=ai_config.BudgetRangeConfig(min=1, max=4),
        high_fidelity_bounds=ai_config.BudgetRangeConfig(min=1, max=2),
    )
    dummy_cfg = replace(
        ai_config.load_experiment_config(),
        budgets=base,
        adaptive_budgets=adaptive,
    )
    bc = BudgetController(dummy_cfg)
    bc.capture_cache_hit_rate("screen", stats={"hits": 5, "misses": 3})
    bc.record_feedback(
        CycleBudgetFeedback(hv_delta=0.2, feasibility_rate=0.7, cache_hit_rate=0.6)
    )
    data = bc.to_dict()
    assert data["state_version"] == BudgetController.STATE_VERSION
    assert data["adaptive_cfg"]["enabled"] is True
    clone = BudgetController(dummy_cfg)
    clone.restore(data)
    assert clone._last_feedback.hv_delta == 0.2
    assert clone._last_feedback.feasibility_rate == 0.7
    assert clone._cache_stats == bc._cache_stats
