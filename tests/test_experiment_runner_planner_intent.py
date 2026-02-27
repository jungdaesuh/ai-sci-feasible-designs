from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from ai_scientist import config as ai_config
from ai_scientist.cycle_executor import CycleResult
from ai_scientist.experiment_runner import run_experiment


def test_run_experiment_threads_planner_intent_to_cycle_executor(
    monkeypatch, tmp_path: Path
) -> None:
    cfg = replace(
        ai_config.ExperimentConfig.p3_quick_validation(),
        cycles=1,
        planner="agent",
        reporting_dir=tmp_path / "reports",
        memory_db=tmp_path / "world.db",
    )
    planner_intent = {
        "primary_constraint_order": ["aspect_ratio", "qi"],
        "penalty_focus_indices": [0, 2],
        "confidence": 0.7,
    }
    planner_intent_alt = {
        "primary_constraint_order": ["qi", "aspect_ratio"],
        "penalty_focus_indices": [2, 0],
        "confidence": 0.5,
    }
    seed_payload = {
        "r_cos": [[1.0, 0.0], [0.0, 0.2]],
        "z_sin": [[0.0, 0.0], [0.0, 0.1]],
        "n_field_periods": cfg.boundary_template.n_field_periods,
        "is_stellarator_symmetric": True,
    }
    captured: dict[str, object] = {}

    class _FakeWorldModel:
        db_path = tmp_path / "world.db"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def start_experiment(self, *_args, **_kwargs) -> int:
            return 11

        def log_artifact(self, **_kwargs) -> None:
            return None

        def budget_usage(self, _experiment_id: int):
            return SimpleNamespace(
                screen_evals=0,
                promoted_evals=0,
                high_fidelity_evals=0,
            )

    class _FakePlanningAgent:
        def plan_cycle(self, **_kwargs):
            return SimpleNamespace(
                context={},
                suggested_params_list=[seed_payload],
                suggested_params=None,
                config_overrides=None,
                planner_intent=planner_intent,
                planner_intent_list=[planner_intent, planner_intent_alt],
            )

    class _FakeCycleExecutor:
        def __init__(self, **_kwargs):
            return None

        def run_cycle(self, **kwargs):
            captured["planner_intent"] = kwargs.get("planner_intent")
            captured["planner_intents"] = kwargs.get("planner_intents")
            return CycleResult(
                cycle_index=0,
                candidates_evaluated=0,
                candidates_promoted=0,
                best_objective=None,
                hypervolume=None,
                feasibility_rate=0.0,
                report_path=None,
                best_eval=None,
                p3_summary=None,
            )

    monkeypatch.setattr(
        "ai_scientist.experiment_runner.rag.ensure_index",
        lambda: SimpleNamespace(chunks_indexed=0, index_path="rag_index.db"),
    )
    monkeypatch.setattr(
        "ai_scientist.experiment_runner.tools.clear_evaluation_cache", lambda: None
    )
    monkeypatch.setattr(
        "ai_scientist.experiment_runner.memory.WorldModel",
        lambda _path: _FakeWorldModel(),
    )
    monkeypatch.setattr(
        "ai_scientist.experiment_runner.resolve_git_sha", lambda *_args: "deadbeef"
    )
    monkeypatch.setattr(
        "ai_scientist.experiment_runner.create_surrogate", lambda _cfg: MagicMock()
    )
    monkeypatch.setattr(
        "ai_scientist.experiment_runner.create_generative_model", lambda _cfg: None
    )
    monkeypatch.setattr(
        "ai_scientist.experiment_runner.ai_planner.PlanningAgent",
        lambda **_kwargs: _FakePlanningAgent(),
    )
    monkeypatch.setattr(
        "ai_scientist.experiment_runner.CycleExecutor", _FakeCycleExecutor
    )

    run_experiment(cfg)

    assert captured["planner_intent"] == planner_intent
    assert captured["planner_intents"] == [planner_intent, planner_intent_alt]
