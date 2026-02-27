"""Coordinator pipeline dedupe regression tests for PR-2."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from ai_scientist.config import load_experiment_config

if TYPE_CHECKING:
    from ai_scientist.coordinator import Coordinator


def _install_coordinator_import_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    if "jax" not in sys.modules:
        jax_module = ModuleType("jax")
        jax_numpy_module = ModuleType("jax.numpy")
        jax_module.numpy = jax_numpy_module
        monkeypatch.setitem(sys.modules, "jax", jax_module)
        monkeypatch.setitem(sys.modules, "jax.numpy", jax_numpy_module)

    constellaration_module = ModuleType("constellaration")
    optimization_module = ModuleType("constellaration.optimization")
    augmented_module = ModuleType("constellaration.optimization.augmented_lagrangian")

    class _AugmentedLagrangianState:
        pass

    augmented_module.AugmentedLagrangianState = _AugmentedLagrangianState
    optimization_module.augmented_lagrangian = augmented_module
    constellaration_module.optimization = optimization_module
    monkeypatch.setitem(sys.modules, "constellaration", constellaration_module)
    monkeypatch.setitem(
        sys.modules, "constellaration.optimization", optimization_module
    )
    monkeypatch.setitem(
        sys.modules,
        "constellaration.optimization.augmented_lagrangian",
        augmented_module,
    )

    alm_bridge_module = ModuleType("ai_scientist.optim.alm_bridge")

    class _ALMContext:
        pass

    alm_bridge_module.ALMContext = _ALMContext
    alm_bridge_module.create_alm_context = lambda *_args, **_kwargs: (None, None)
    alm_bridge_module.state_to_boundary_params = lambda *_args, **_kwargs: {}
    alm_bridge_module.step_alm = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "ai_scientist.optim.alm_bridge", alm_bridge_module)

    generative_module = ModuleType("ai_scientist.optim.generative")

    class _GenerativeDesignModel:
        pass

    class _DiffusionDesignModel:
        pass

    generative_module.GenerativeDesignModel = _GenerativeDesignModel
    generative_module.DiffusionDesignModel = _DiffusionDesignModel
    monkeypatch.setitem(sys.modules, "ai_scientist.optim.generative", generative_module)

    surrogate_module = ModuleType("ai_scientist.optim.surrogate_v2")

    class _NeuralOperatorSurrogate:
        pass

    surrogate_module.NeuralOperatorSurrogate = _NeuralOperatorSurrogate
    monkeypatch.setitem(
        sys.modules,
        "ai_scientist.optim.surrogate_v2",
        surrogate_module,
    )

    workers_module = ModuleType("ai_scientist.workers")

    class _Worker:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def run(self, *_args: object, **_kwargs: object) -> dict[str, object]:
            return {"candidates": []}

    workers_module.ExplorationWorker = _Worker
    workers_module.GeometerWorker = _Worker
    workers_module.OptimizationWorker = _Worker
    workers_module.PreRelaxWorker = _Worker
    workers_module.RLRefinementWorker = _Worker
    monkeypatch.setitem(sys.modules, "ai_scientist.workers", workers_module)


@pytest.fixture
def coordinator_instance(monkeypatch: pytest.MonkeyPatch) -> "Coordinator":
    _install_coordinator_import_stubs(monkeypatch)
    sys.modules.pop("ai_scientist.coordinator", None)

    from ai_scientist.coordinator import Coordinator

    cfg = load_experiment_config("configs/experiment.example.yaml")
    world_model = MagicMock()
    world_model.average_recent_hv_delta.return_value = None
    planner = MagicMock()

    with (
        patch("ai_scientist.coordinator.OptimizationWorker"),
        patch("ai_scientist.coordinator.ExplorationWorker"),
        patch("ai_scientist.coordinator.GeometerWorker"),
        patch("ai_scientist.coordinator.PreRelaxWorker"),
        patch("ai_scientist.coordinator.RLRefinementWorker"),
    ):
        coordinator = Coordinator(cfg=cfg, world_model=world_model, planner=planner)

    coordinator.explore_worker = MagicMock()
    coordinator.prerelax_worker = MagicMock()
    coordinator.geo_worker = MagicMock()
    coordinator.rl_worker = MagicMock()
    coordinator.opt_worker = MagicMock()
    return coordinator


def test_produce_candidates_exploit_routes_to_standard_pipeline(
    coordinator_instance: "Coordinator",
) -> None:
    coordinator_instance.decide_strategy = MagicMock(return_value="EXPLOIT")
    with patch.object(
        coordinator_instance.__class__,
        "_run_standard_pipeline",
        return_value=[{"id": "exploit"}],
    ) as mock_pipeline:
        out = coordinator_instance.produce_candidates(
            cycle=7,
            experiment_id=1,
            n_candidates=3,
            template=coordinator_instance.cfg.boundary_template,
        )

    assert out == [{"id": "exploit"}]
    mock_pipeline.assert_called_once_with(cycle=7, n_candidates=3)


def test_produce_candidates_hybrid_routes_to_standard_pipeline(
    coordinator_instance: "Coordinator",
) -> None:
    coordinator_instance.decide_strategy = MagicMock(return_value="HYBRID")
    with patch.object(
        coordinator_instance.__class__,
        "_run_standard_pipeline",
        return_value=[{"id": "hybrid"}],
    ) as mock_pipeline:
        out = coordinator_instance.produce_candidates(
            cycle=6,
            experiment_id=1,
            n_candidates=4,
            template=coordinator_instance.cfg.boundary_template,
        )

    assert out == [{"id": "hybrid"}]
    mock_pipeline.assert_called_once_with(cycle=6, n_candidates=4)


def test_produce_candidates_explore_keeps_lightweight_branch(
    coordinator_instance: "Coordinator",
) -> None:
    coordinator_instance.decide_strategy = MagicMock(return_value="EXPLORE")
    seeds = [{"id": 1}, {"id": 2}]
    coordinator_instance.explore_worker.run.return_value = {"candidates": seeds}
    coordinator_instance.prerelax_worker.run.return_value = {"candidates": seeds}
    coordinator_instance.geo_worker.run.return_value = {"candidates": [seeds[0]]}
    coordinator_instance.opt_worker.run.return_value = {
        "candidates": [{"id": "unused"}]
    }

    with patch.object(
        coordinator_instance.__class__,
        "_run_standard_pipeline",
        return_value=[{"id": "unexpected"}],
    ) as mock_pipeline:
        out = coordinator_instance.produce_candidates(
            cycle=10,
            experiment_id=1,
            n_candidates=2,
            template=coordinator_instance.cfg.boundary_template,
        )

    assert out == [seeds[0]]
    mock_pipeline.assert_not_called()
    coordinator_instance.opt_worker.run.assert_not_called()


def test_standard_pipeline_preserves_stage_order_and_output_contract(
    coordinator_instance: "Coordinator",
) -> None:
    coordinator_instance.surrogate = MagicMock()
    coordinator_instance.surrogate._trained = True
    coordinator_instance.surrogate._schema = {"version": 1}

    call_order: list[str] = []
    seeds = [
        {"id": "s1", "params": {"r_cos": [[1.0]], "z_sin": [[0.0]]}},
        {"id": "s2", "params": {"r_cos": [[1.1]], "z_sin": [[0.1]]}},
    ]
    ranked = [seeds[1], seeds[0]]
    refined = [{"id": "r2"}, {"id": "r1"}]
    final_candidates = [
        {"id": "c2", "params": {"r_cos": [[1.2]], "z_sin": [[0.2]]}},
        {"id": "c1", "params": {"r_cos": [[1.3]], "z_sin": [[0.3]]}},
    ]

    def _stage(name: str, payload: dict[str, object]) -> dict[str, object]:
        call_order.append(name)
        if name == "explore":
            assert payload["n_samples"] == 2
            return {"candidates": seeds}
        if name == "prerelax":
            assert payload["candidates"] == seeds
            return {"candidates": seeds}
        if name == "geo":
            assert payload["candidates"] == seeds
            return {"candidates": seeds}
        if name == "rl":
            assert payload["candidates"] == ranked
            return {"candidates": refined}
        if name == "opt":
            assert payload["initial_guesses"] == refined
            return {"candidates": final_candidates}
        raise AssertionError(f"unexpected stage {name}")

    coordinator_instance.explore_worker.run.side_effect = lambda ctx: _stage(
        "explore", ctx
    )
    coordinator_instance.prerelax_worker.run.side_effect = lambda ctx: _stage(
        "prerelax", ctx
    )
    coordinator_instance.geo_worker.run.side_effect = lambda ctx: _stage("geo", ctx)
    coordinator_instance.rl_worker.run.side_effect = lambda ctx: _stage("rl", ctx)
    coordinator_instance.opt_worker.run.side_effect = lambda ctx: _stage("opt", ctx)

    def _rank(
        valid_seeds: list[dict[str, object]], cycle: int
    ) -> list[dict[str, object]]:
        assert valid_seeds == seeds
        assert cycle == 11
        call_order.append("surrogate_rank")
        return ranked

    coordinator_instance._surrogate_rank_seeds = MagicMock(side_effect=_rank)

    out = coordinator_instance._run_standard_pipeline(cycle=11, n_candidates=2)

    assert call_order == ["explore", "prerelax", "geo", "surrogate_rank", "rl", "opt"]
    assert out == final_candidates
    assert len(out) == 2
    assert [candidate["id"] for candidate in out] == ["c2", "c1"]
    assert all("params" in candidate for candidate in out)
