"""Tests for the runner resume functionality."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ai_scientist import config as ai_config
from ai_scientist import memory
from ai_scientist import runner


@pytest.fixture
def temp_workspace(tmp_path):
    """Return a minimal ExperimentConfig plus paths for DB/reporting."""
    db_path = tmp_path / "test_memory.db"
    report_dir = tmp_path / "reports"
    report_dir.mkdir()

    config = ai_config.ExperimentConfig(
        problem="p1",
        cycles=2,
        random_seed=42,
        budgets=ai_config.BudgetConfig(
            screen_evals_per_cycle=1,
            promote_top_k=1,
            max_high_fidelity_evals_per_cycle=1,
            wall_clock_minutes=1.0,
            n_workers=1,
            pool_type="thread",
        ),
        adaptive_budgets=ai_config.AdaptiveBudgetConfig(
            enabled=False,
            hv_slope_reference=0.1,
            feasibility_target=0.8,
            cache_hit_target=0.5,
            screen_bounds=ai_config.BudgetRangeConfig(min=10, max=100),
            promote_top_k_bounds=ai_config.BudgetRangeConfig(min=1, max=10),
            high_fidelity_bounds=ai_config.BudgetRangeConfig(min=1, max=5),
        ),
        proposal_mix=ai_config.ProposalMixConfig(
            constraint_ratio=0.5,
            exploration_ratio=0.5,
            jitter_scale=0.01,
            surrogate_pool_multiplier=2.0,
        ),
        fidelity_ladder=ai_config.FidelityLadder(screen="screen", promote="promote"),
        boundary_template=ai_config.BoundaryTemplateConfig(
            n_field_periods=3,
            n_poloidal_modes=4,
            n_toroidal_modes=4,
            base_major_radius=1.0,
            base_minor_radius=0.1,
            perturbation_scale=0.01,
        ),
        stage_gates=ai_config.StageGateConfig(
            s1_to_s2_feasibility_margin=0.1,
            s1_to_s2_lookback_cycles=3,
            s1_to_s2_objective_improvement=0.01,
            s2_to_s3_hv_delta=0.001,
            s2_to_s3_lookback_cycles=3,
        ),
        governance=ai_config.GovernanceConfig(
            hv_lookback=5, min_feasible_for_promotion=1
        ),
        constraint_weights=ai_config.ConstraintWeightsConfig(
            mhd=1.0,
            qi=1.0,
            elongation=1.0,
        ),
        generative=ai_config.GenerativeConfig(
            enabled=False,
            backend="vae",
            latent_dim=16,
            learning_rate=1e-3,
            epochs=1,
            kl_weight=1e-3,
        ),
        source_config=tmp_path / "experiment.yaml",
        reporting_dir=report_dir,
        memory_db=db_path,
    )
    return config, db_path, report_dir


def test_runner_resume_flow(temp_workspace):
    """Verify that resume skips completed cycles and preserves experiment_id."""
    config, db_path, report_dir = temp_workspace

    # Mock _run_cycle to create a checkpoint and minimal DB rows for cycle 1.
    with patch("ai_scientist.runner._run_cycle") as mock_run_cycle:
        mock_summary = MagicMock()
        mock_summary.hv_score = 0.5
        mock_summary.feasible_count = 1
        mock_summary.archive_size = 1
        mock_summary.reference_point = (1.0, 1.0)
        mock_summary.pareto_entries = []

        mock_eval = {
            "objective": 0.5,
            "feasibility": 0.0,
            "metrics": {
                "aspect_ratio": 5.0,
                "minimum_normalized_magnetic_gradient_scale_length": 2.0,
                "max_elongation": 1.0,
            },
            "design_hash": "hash1",
            "stage": "screen",
        }

        def side_effect(cfg, cycle_index, world_model, experiment_id, *args, **kwargs):
            cp_path = cfg.reporting_dir / f"cycle_{cycle_index + 1}.json"
            payload = {
                "experiment_id": experiment_id,
                "cycle": cycle_index + 1,
                "git_sha": "test",
                "constellaration_sha": "test",
                "stage": "screen",
                "feasible_count": 1,
                "hv": 0.5,
                "reference_point": [1.0, 1.0],
                "archive_size": 1,
                "screened": 1,
                "promoted": 0,
                "feasibility_rate": 1.0,
            }
            cp_path.write_text(json.dumps(payload), encoding="utf-8")
            # Minimal DB rows to satisfy resume validation.
            with world_model.transaction():
                world_model._conn.execute(
                    """
                    INSERT OR REPLACE INTO cycles
                    (experiment_id, cycle, stage, feasible_count, hv_score, hv_exists, created_at)
                    VALUES (?, ?, 'screen', 1, 0.5, 1, 'now')
                    """,
                    (experiment_id, cycle_index + 1),
                )
                world_model._conn.execute(
                    """
                    INSERT OR REPLACE INTO budgets
                    (experiment_id, cycle, screen_evals, promoted_evals, high_fidelity_evals,
                     wall_seconds, best_objective, best_feasibility, best_score, best_stage)
                    VALUES (?, ?, 1, 0, 0, 1.0, 0.5, 0.0, 0.5, 'screen')
                    """,
                    (experiment_id, cycle_index + 1),
                )
                world_model._conn.execute(
                    """
                    INSERT OR REPLACE INTO stage_history
                    (experiment_id, cycle, stage, selected_at)
                    VALUES (?, ?, 'screen', 'now')
                    """,
                    (experiment_id, cycle_index + 1),
                )
            return (Path("report.md"), mock_eval, mock_summary)

        mock_run_cycle.side_effect = side_effect

        config_c1 = replace(config, cycles=1)
        runner.run(config_c1)

        checkpoint_path = report_dir / "cycle_1.json"
        assert checkpoint_path.exists()
        data = json.loads(checkpoint_path.read_text())
        exp_id = data["experiment_id"]
        assert data["cycle"] == 1

    # Resume to cycle 2; _run_cycle should be called only once for index 1.
    with patch("ai_scientist.runner._run_cycle") as mock_run_cycle_2:
        mock_run_cycle_2.return_value = (Path("report2.md"), mock_eval, mock_summary)

        resume_cli = runner.RunnerCLIConfig(
            config_path=Path("dummy"),
            problem=None,
            cycles=None,
            memory_db=None,
            eval_budget=None,
            workers=None,
            pool_type=None,
            screen_only=False,
            promote_only=False,
            slow=False,
            verbose=True,
            log_cache_stats=False,
            run_preset=None,
            planner="deterministic",
            resume_from=checkpoint_path,
        )

        config_c2 = replace(config, cycles=2)
        runner.run(config_c2, runtime=resume_cli)

        assert mock_run_cycle_2.call_count == 1
        args, _ = mock_run_cycle_2.call_args
        assert args[1] == 1  # cycle index

        with memory.WorldModel(db_path) as wm:
            rows = wm._conn.execute(
                "SELECT id FROM experiments"
            ).fetchall()
            assert len(rows) == 1
            assert rows[0][0] == exp_id


def test_runner_resume_integration(temp_workspace):
    """Integration-style test: run cycle1 then resume for cycle2 using real DB writes."""
    config, db_path, report_dir = temp_workspace

    mock_eval_func = MagicMock()
    mock_eval_func.return_value = {
        "stage": "screen",
        "objective": 1.0,
        "feasibility": 0.0,
        "metrics": {
            "max_elongation": 1.0,
            "minimum_normalized_magnetic_gradient_scale_length": 2.0,
            "aspect_ratio": 10.0,
        },
        "settings": {},
        "design_hash": "hash_test",
    }

    with patch("ai_scientist.runner._problem_evaluator", return_value=mock_eval_func):
        config_c1 = replace(config, cycles=1)
        runner.run(config_c1)

        cp_path = report_dir / "cycle_1.json"
        assert cp_path.exists()
        data = json.loads(cp_path.read_text())
        exp_id = data["experiment_id"]

        resume_cli = runner.RunnerCLIConfig(
            config_path=Path("dummy"),
            problem=None,
            cycles=None,
            memory_db=None,
            eval_budget=None,
            workers=None,
            pool_type=None,
            screen_only=False,
            promote_only=False,
            slow=False,
            verbose=True,
            log_cache_stats=False,
            run_preset=None,
            planner="deterministic",
            resume_from=cp_path,
        )

        config_c2 = replace(config, cycles=2)
        runner.run(config_c2, runtime=resume_cli)

        with memory.WorldModel(db_path) as wm:
            rows = wm._conn.execute(
                "SELECT cycle FROM budgets WHERE experiment_id = ?", (exp_id,)
            ).fetchall()
            cycles = sorted(row[0] for row in rows)
            assert cycles == [1, 2]
