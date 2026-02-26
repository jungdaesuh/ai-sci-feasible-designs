from __future__ import annotations

import dataclasses
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

if "pymoo.indicators.hv" not in sys.modules:
    pymoo_module = types.ModuleType("pymoo")
    indicators_module = types.ModuleType("pymoo.indicators")
    hv_module = types.ModuleType("pymoo.indicators.hv")

    class _HV:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __call__(self, *args, **kwargs) -> float:
            return 0.0

    hv_module.HV = _HV
    indicators_module.hv = hv_module
    pymoo_module.indicators = indicators_module
    sys.modules["pymoo"] = pymoo_module
    sys.modules["pymoo.indicators"] = indicators_module
    sys.modules["pymoo.indicators.hv"] = hv_module

from ai_scientist import planner
from ai_scientist import config as ai_config


class _Gate:
    def __init__(self) -> None:
        self.model_alias = "test-gate"
        self.provider_model = "test-gate"
        self.allowed_tools = (
            "retrieve_rag",
            "evaluate_p1",
            "evaluate_p2",
            "evaluate_p3",
            "propose_boundary",
            "recombine_designs",
            "make_boundary",
        )

    def allows(self, tool_name: str) -> bool:
        return tool_name in self.allowed_tools


def _make_agent(monkeypatch: pytest.MonkeyPatch) -> planner.PlanningAgent:
    gate = _Gate()
    monkeypatch.setattr(
        planner.agent_module,
        "provision_model_tier",
        lambda role, config: gate,
    )
    cfg = SimpleNamespace(agent_gates=[])
    return planner.PlanningAgent(config=cfg, rag_index=Path("rag_index.db"))


def _set_tools_evaluator_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        planner.tools, "evaluate_p1", lambda *_args, **_kwargs: {"name": "p1"}
    )
    monkeypatch.setattr(
        planner.tools, "evaluate_p2", lambda *_args, **_kwargs: {"name": "p2"}
    )
    monkeypatch.setattr(
        planner.tools, "evaluate_p3", lambda *_args, **_kwargs: {"name": "p3"}
    )


@pytest.mark.parametrize(
    ("problem", "expected_method"),
    (("p1", "p1"), ("p2", "p2"), ("p3", "p3")),
)
def test_plan_cycle_routes_seed_evaluation_by_problem(
    monkeypatch: pytest.MonkeyPatch, problem: str, expected_method: str
) -> None:
    agent = _make_agent(monkeypatch)
    cfg = dataclasses.replace(
        ai_config.load_experiment_config(), problem=problem, cycles=1
    )
    calls = {"p1": 0, "p2": 0, "p3": 0}

    monkeypatch.setattr(agent, "retrieve_rag", lambda query, k=3: [])
    monkeypatch.setattr(
        agent,
        "make_boundary",
        lambda params: SimpleNamespace(
            n_poloidal_modes=2,
            n_toroidal_modes=3,
            n_field_periods=params.get("n_field_periods", 3),
            is_stellarator_symmetric=True,
        ),
    )

    def _eval_factory(name: str):
        def _evaluate(params, stage=None):
            calls[name] += 1
            return {
                "stage": stage or name,
                "objective": 1.0,
                "feasibility": 0.0,
                "gradient_proxy": 0.1,
            }

        return _evaluate

    monkeypatch.setattr(agent, "evaluate_p1", _eval_factory("p1"))
    monkeypatch.setattr(agent, "evaluate_p2", _eval_factory("p2"))
    monkeypatch.setattr(agent, "evaluate_p3", _eval_factory("p3"))

    outcome = agent.plan_cycle(
        cfg=cfg,
        cycle_index=0,
        stage_history=[],
        last_summary=None,
    )

    assert calls[expected_method] == 1
    for name in ("p1", "p2", "p3"):
        if name != expected_method:
            assert calls[name] == 0
    assert outcome.evaluation_summary["stage"] == cfg.fidelity_ladder.screen


def test_execute_planning_tool_supports_all_problem_evaluators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _make_agent(monkeypatch)
    params = {"r_cos": [[1.0]], "z_sin": [[0.0]], "n_field_periods": 3}
    monkeypatch.setattr(agent, "evaluate_p1", lambda **_: {"name": "p1"})
    monkeypatch.setattr(agent, "evaluate_p2", lambda **_: {"name": "p2"})
    monkeypatch.setattr(agent, "evaluate_p3", lambda **_: {"name": "p3"})

    assert agent._execute_planning_tool("evaluate_p1", {"params": params}) == {
        "name": "p1"
    }
    assert agent._execute_planning_tool("evaluate_p2", {"params": params}) == {
        "name": "p2"
    }
    assert agent._execute_planning_tool("evaluate_p3", {"params": params}) == {
        "name": "p3"
    }


def test_execute_planning_tool_accepts_problem_kwarg_for_evaluators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _make_agent(monkeypatch)
    params = {"r_cos": [[1.0]], "z_sin": [[0.0]], "n_field_periods": 3}
    _set_tools_evaluator_stubs(monkeypatch)

    assert agent._execute_planning_tool(
        "evaluate_p1", {"params": params, "problem": "p1"}
    ) == {"name": "p1"}
    assert agent._execute_planning_tool(
        "evaluate_p2", {"params": params, "problem": "p2"}
    ) == {"name": "p2"}
    assert agent._execute_planning_tool(
        "evaluate_p3", {"params": params, "problem": "p3"}
    ) == {"name": "p3"}
