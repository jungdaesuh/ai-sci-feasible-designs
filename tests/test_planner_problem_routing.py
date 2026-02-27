from __future__ import annotations

import dataclasses
import json
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
from ai_scientist import memory


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


def _make_agent(
    monkeypatch: pytest.MonkeyPatch, *, use_agent_gates: bool = False
) -> planner.PlanningAgent:
    gate = _Gate()
    monkeypatch.setattr(
        planner.agent_module,
        "provision_model_tier",
        lambda role, config: gate,
    )
    cfg = ai_config.load_model_config()
    if not use_agent_gates:
        cfg = dataclasses.replace(cfg, agent_gates=())
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


def test_plan_cycle_rejects_config_only_finalize_until_seed_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _make_agent(monkeypatch, use_agent_gates=True)
    cfg = dataclasses.replace(
        ai_config.load_experiment_config(), problem="p1", cycles=1
    )
    seed_payload = {
        "r_cos": [[1.0, 0.0, 0.0], [0.0, 0.3, 0.0]],
        "z_sin": [[0.0, 0.0, 0.0], [0.0, 0.2, 0.0]],
        "n_field_periods": cfg.boundary_template.n_field_periods,
        "is_stellarator_symmetric": True,
    }
    responses = iter(
        [
            json.dumps({"config_overrides": {"proposal_mix": {"jitter_scale": 0.02}}}),
            json.dumps(
                {
                    "suggested_params": seed_payload,
                    "config_overrides": {"proposal_mix": {"jitter_scale": 0.01}},
                }
            ),
        ]
    )
    call_count = {"n": 0}

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
    monkeypatch.setattr(
        agent,
        "evaluate_p1",
        lambda params, stage=None: {
            "stage": stage or "screen",
            "objective": 1.0,
            "feasibility": 0.0,
            "gradient_proxy": 0.1,
        },
    )

    def _fake_invoke_chat_completion(*_args, **_kwargs):
        call_count["n"] += 1
        content = next(responses)
        return SimpleNamespace(
            status_code=200,
            body={"choices": [{"message": {"content": content}}]},
        )

    from ai_scientist import model_provider

    monkeypatch.setattr(
        model_provider,
        "invoke_chat_completion",
        _fake_invoke_chat_completion,
    )

    outcome = agent.plan_cycle(
        cfg=cfg,
        cycle_index=0,
        stage_history=[],
        last_summary=None,
    )

    assert call_count["n"] == 2
    assert outcome.suggested_params == seed_payload
    assert outcome.config_overrides == {"proposal_mix": {"jitter_scale": 0.01}}


def test_plan_cycle_accepts_explicit_template_seed_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _make_agent(monkeypatch, use_agent_gates=True)
    cfg = dataclasses.replace(
        ai_config.load_experiment_config(), problem="p2", cycles=1
    )
    response_payload = json.dumps(
        {
            "seed_fallback": "template",
            "config_overrides": {"constraint_weights": {"mhd": 2.0}},
        }
    )

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
    monkeypatch.setattr(
        agent,
        "evaluate_p2",
        lambda params, stage=None: {
            "stage": stage or "screen",
            "objective": 1.0,
            "feasibility": 0.0,
            "gradient_proxy": 0.1,
        },
    )

    def _fake_invoke_chat_completion(*_args, **_kwargs):
        return SimpleNamespace(
            status_code=200,
            body={"choices": [{"message": {"content": response_payload}}]},
        )

    from ai_scientist import model_provider

    monkeypatch.setattr(
        model_provider,
        "invoke_chat_completion",
        _fake_invoke_chat_completion,
    )

    outcome = agent.plan_cycle(
        cfg=cfg,
        cycle_index=0,
        stage_history=[],
        last_summary=None,
    )

    assert outcome.suggested_params is not None
    assert "r_cos" in outcome.suggested_params
    assert "z_sin" in outcome.suggested_params
    assert outcome.config_overrides == {"constraint_weights": {"mhd": 2.0}}


def test_plan_cycle_applies_template_fallback_when_no_valid_finalize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _make_agent(monkeypatch, use_agent_gates=True)
    cfg = dataclasses.replace(
        ai_config.load_experiment_config(), problem="p1", cycles=1
    )
    call_count = {"n": 0}

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
    monkeypatch.setattr(
        agent,
        "evaluate_p1",
        lambda params, stage=None: {
            "stage": stage or "screen",
            "objective": 1.0,
            "feasibility": 0.0,
            "gradient_proxy": 0.1,
        },
    )

    def _fake_invoke_chat_completion(*_args, **_kwargs):
        call_count["n"] += 1
        return SimpleNamespace(
            status_code=200,
            body={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "config_overrides": {
                                        "proposal_mix": {"jitter_scale": 0.02}
                                    }
                                }
                            )
                        }
                    }
                ]
            },
        )

    from ai_scientist import model_provider

    monkeypatch.setattr(
        model_provider,
        "invoke_chat_completion",
        _fake_invoke_chat_completion,
    )

    outcome = agent.plan_cycle(
        cfg=cfg,
        cycle_index=0,
        stage_history=[],
        last_summary=None,
    )

    assert call_count["n"] == 5
    assert outcome.suggested_params is not None
    assert "r_cos" in outcome.suggested_params
    assert "z_sin" in outcome.suggested_params
    assert outcome.config_overrides is None


def test_plan_cycle_uses_template_fallback_when_suggested_params_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _make_agent(monkeypatch, use_agent_gates=True)
    cfg = dataclasses.replace(
        ai_config.load_experiment_config(), problem="p2", cycles=1
    )
    call_count = {"n": 0}
    response_payload = json.dumps(
        {
            "suggested_params": {"invalid": True},
            "seed_fallback": "template",
            "config_overrides": {"constraint_weights": {"mhd": 3.0}},
        }
    )

    monkeypatch.setattr(agent, "retrieve_rag", lambda query, k=3: [])

    def _make_boundary(params):
        if "r_cos" not in params or "z_sin" not in params:
            raise ValueError("missing required Fourier coefficient arrays")
        return SimpleNamespace(
            n_poloidal_modes=2,
            n_toroidal_modes=3,
            n_field_periods=params.get("n_field_periods", 3),
            is_stellarator_symmetric=True,
        )

    monkeypatch.setattr(agent, "make_boundary", _make_boundary)
    monkeypatch.setattr(
        agent,
        "evaluate_p2",
        lambda params, stage=None: {
            "stage": stage or "screen",
            "objective": 1.0,
            "feasibility": 0.0,
            "gradient_proxy": 0.1,
        },
    )

    def _fake_invoke_chat_completion(*_args, **_kwargs):
        call_count["n"] += 1
        return SimpleNamespace(
            status_code=200,
            body={"choices": [{"message": {"content": response_payload}}]},
        )

    from ai_scientist import model_provider

    monkeypatch.setattr(
        model_provider,
        "invoke_chat_completion",
        _fake_invoke_chat_completion,
    )

    outcome = agent.plan_cycle(
        cfg=cfg,
        cycle_index=0,
        stage_history=[],
        last_summary=None,
    )

    assert call_count["n"] == 1
    assert outcome.suggested_params is not None
    assert "r_cos" in outcome.suggested_params
    assert "z_sin" in outcome.suggested_params
    assert outcome.config_overrides == {"constraint_weights": {"mhd": 3.0}}


def test_plan_cycle_accepts_multi_seed_finalize_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _make_agent(monkeypatch, use_agent_gates=True)
    cfg = dataclasses.replace(
        ai_config.load_experiment_config(), problem="p1", cycles=1
    )
    seed_a = {
        "r_cos": [[1.0, 0.0, 0.0], [0.0, 0.3, 0.0]],
        "z_sin": [[0.0, 0.0, 0.0], [0.0, 0.2, 0.0]],
        "n_field_periods": cfg.boundary_template.n_field_periods,
        "is_stellarator_symmetric": True,
    }
    seed_b = {
        "r_cos": [[1.0, 0.0, 0.0], [0.0, 0.25, 0.0]],
        "z_sin": [[0.0, 0.0, 0.0], [0.0, 0.15, 0.0]],
        "n_field_periods": cfg.boundary_template.n_field_periods,
        "is_stellarator_symmetric": True,
    }
    response_payload = json.dumps({"suggested_params_list": [seed_a, seed_b]})

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
    monkeypatch.setattr(
        agent,
        "evaluate_p1",
        lambda params, stage=None: {
            "stage": stage or "screen",
            "objective": 1.0,
            "feasibility": 0.0,
            "gradient_proxy": 0.1,
        },
    )

    def _fake_invoke_chat_completion(*_args, **_kwargs):
        return SimpleNamespace(
            status_code=200,
            body={"choices": [{"message": {"content": response_payload}}]},
        )

    from ai_scientist import model_provider

    monkeypatch.setattr(
        model_provider,
        "invoke_chat_completion",
        _fake_invoke_chat_completion,
    )

    outcome = agent.plan_cycle(
        cfg=cfg,
        cycle_index=0,
        stage_history=[],
        last_summary=None,
    )

    assert outcome.suggested_params == seed_a
    assert outcome.suggested_params_list == [seed_a, seed_b]


def test_plan_cycle_context_includes_balanced_experience_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _make_agent(monkeypatch)
    cfg = dataclasses.replace(
        ai_config.load_experiment_config(), problem="p3", cycles=1
    )

    graph = memory.PropertyGraph()
    world_model = SimpleNamespace(
        to_networkx=lambda _exp_id: graph,
        recent_experience_pack=lambda **_kwargs: {
            "recent_successes": [{"design_hash": "succ", "worst_constraint": None}],
            "recent_near_successes": [
                {"design_hash": "near", "worst_constraint": "qi"}
            ],
            "recent_failures": [{"design_hash": "fail", "worst_constraint": "mhd"}],
            "feedback_adapter": {
                "worst_constraint_trend": {"sequence": ["mhd"], "counts": {"mhd": 1}},
                "recent_effective_deltas": [
                    {
                        "from_design_hash": "fail",
                        "to_design_hash": "near",
                        "feasibility_delta": -0.1,
                        "objective_delta": None,
                        "worst_constraint_delta": -0.2,
                    }
                ],
            },
        },
    )
    agent.world_model = world_model
    monkeypatch.setattr(agent, "_should_inject_experience", lambda **_kwargs: True)
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
    monkeypatch.setattr(
        agent,
        "evaluate_p3",
        lambda params, stage=None: {
            "stage": stage or "screen",
            "objective": 1.0,
            "feasibility": 0.1,
            "gradient_proxy": 0.1,
        },
    )

    outcome = agent.plan_cycle(
        cfg=cfg,
        cycle_index=0,
        stage_history=[],
        last_summary=None,
        experiment_id=99,
    )

    assert len(outcome.context["recent_successes"]) == 1
    assert len(outcome.context["recent_near_successes"]) == 1
    assert len(outcome.context["recent_failures"]) == 1
    assert "feedback_adapter" in outcome.context
    assert outcome.context["experience_memo"] != ""


def test_plan_cycle_composes_gate_prompt_with_repo_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _make_agent(monkeypatch, use_agent_gates=True)
    setattr(agent.planning_gate, "system_prompt", "GATE_PROMPT_LAYER")
    cfg = dataclasses.replace(
        ai_config.load_experiment_config(), problem="p2", cycles=1
    )
    captured = {"system_prompt": ""}
    response_payload = json.dumps({"seed_fallback": "template"})

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
    monkeypatch.setattr(
        agent,
        "evaluate_p2",
        lambda params, stage=None: {
            "stage": stage or "screen",
            "objective": 1.0,
            "feasibility": 0.0,
            "gradient_proxy": 0.1,
        },
    )

    def _fake_invoke_chat_completion(*_args, **kwargs):
        messages = kwargs.get("messages") or []
        if messages:
            captured["system_prompt"] = messages[0].get("content", "")
        return SimpleNamespace(
            status_code=200,
            body={"choices": [{"message": {"content": response_payload}}]},
        )

    from ai_scientist import model_provider

    monkeypatch.setattr(
        model_provider,
        "invoke_chat_completion",
        _fake_invoke_chat_completion,
    )

    agent.plan_cycle(
        cfg=cfg,
        cycle_index=0,
        stage_history=[],
        last_summary=None,
    )

    assert "GATE_PROMPT_LAYER" in captured["system_prompt"]
    assert "Repository contract" in captured["system_prompt"]


def test_plan_cycle_accepts_planner_intent_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _make_agent(monkeypatch, use_agent_gates=True)
    cfg = dataclasses.replace(
        ai_config.load_experiment_config(), problem="p3", cycles=1
    )
    seed_payload = {
        "r_cos": [[1.0, 0.0, 0.0], [0.0, 0.2, 0.0]],
        "z_sin": [[0.0, 0.0, 0.0], [0.0, 0.15, 0.0]],
        "n_field_periods": cfg.boundary_template.n_field_periods,
        "is_stellarator_symmetric": True,
    }
    planner_intent = {
        "primary_constraint_order": ["aspect_ratio", "qi"],
        "target_move_family": "scale_groups",
        "forbidden_moves": ["random_reset"],
        "penalty_focus_indices": [0, 2],
        "restart_policy": "on_stagnation",
        "confidence": 0.72,
    }
    response_payload = json.dumps(
        {"suggested_params": seed_payload, "planner_intent": planner_intent}
    )

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
    monkeypatch.setattr(
        agent,
        "evaluate_p3",
        lambda params, stage=None: {
            "stage": stage or "screen",
            "objective": 1.0,
            "feasibility": 0.0,
            "gradient_proxy": 0.1,
        },
    )

    def _fake_invoke_chat_completion(*_args, **_kwargs):
        return SimpleNamespace(
            status_code=200,
            body={"choices": [{"message": {"content": response_payload}}]},
        )

    from ai_scientist import model_provider

    monkeypatch.setattr(
        model_provider,
        "invoke_chat_completion",
        _fake_invoke_chat_completion,
    )

    outcome = agent.plan_cycle(
        cfg=cfg,
        cycle_index=0,
        stage_history=[],
        last_summary=None,
    )

    assert outcome.suggested_params == seed_payload
    assert outcome.planner_intent == planner_intent


def test_plan_cycle_retries_when_planner_intent_invalid_then_accepts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _make_agent(monkeypatch, use_agent_gates=True)
    cfg = dataclasses.replace(
        ai_config.load_experiment_config(), problem="p3", cycles=1
    )
    seed_payload = {
        "r_cos": [[1.0, 0.0, 0.0], [0.0, 0.2, 0.0]],
        "z_sin": [[0.0, 0.0, 0.0], [0.0, 0.15, 0.0]],
        "n_field_periods": cfg.boundary_template.n_field_periods,
        "is_stellarator_symmetric": True,
    }
    valid_planner_intent = {
        "primary_constraint_order": ["aspect_ratio"],
        "target_move_family": "blend",
        "forbidden_moves": [],
        "penalty_focus_indices": [0],
        "restart_policy": "on_stagnation",
        "confidence": 0.6,
    }
    responses = iter(
        [
            json.dumps(
                {
                    "suggested_params": seed_payload,
                    "planner_intent": {
                        **valid_planner_intent,
                        "penalty_focus_indices": [999],
                    },
                }
            ),
            json.dumps(
                {
                    "suggested_params": seed_payload,
                    "planner_intent": valid_planner_intent,
                }
            ),
        ]
    )
    call_count = {"n": 0}

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
    monkeypatch.setattr(
        agent,
        "evaluate_p3",
        lambda params, stage=None: {
            "stage": stage or "screen",
            "objective": 1.0,
            "feasibility": 0.0,
            "gradient_proxy": 0.1,
        },
    )

    def _fake_invoke_chat_completion(*_args, **_kwargs):
        call_count["n"] += 1
        return SimpleNamespace(
            status_code=200,
            body={"choices": [{"message": {"content": next(responses)}}]},
        )

    from ai_scientist import model_provider

    monkeypatch.setattr(
        model_provider,
        "invoke_chat_completion",
        _fake_invoke_chat_completion,
    )

    outcome = agent.plan_cycle(
        cfg=cfg,
        cycle_index=0,
        stage_history=[],
        last_summary=None,
    )

    assert call_count["n"] == 2
    assert outcome.planner_intent == valid_planner_intent


def test_plan_cycle_context_includes_scratchpad_summary_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _make_agent(monkeypatch)
    cfg = dataclasses.replace(
        ai_config.load_experiment_config(), problem="p3", cycles=2
    )

    graph = memory.PropertyGraph()
    world_model = SimpleNamespace(
        to_networkx=lambda _exp_id: graph,
        recent_experience_pack=lambda **_kwargs: {
            "recent_successes": [],
            "recent_near_successes": [],
            "recent_failures": [],
            "feedback_adapter": {},
        },
        scratchpad_cycle_summary=lambda **_kwargs: {
            "cycle": 1,
            "event_count": 2,
            "action_counts": {"ADJUST": 1, "CONTINUE": 1},
            "events": [],
        },
    )
    agent.world_model = world_model
    monkeypatch.setattr(agent, "_should_inject_experience", lambda **_kwargs: True)
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
    monkeypatch.setattr(
        agent,
        "evaluate_p3",
        lambda params, stage=None: {
            "stage": stage or "screen",
            "objective": 1.0,
            "feasibility": 0.1,
            "gradient_proxy": 0.1,
        },
    )

    outcome = agent.plan_cycle(
        cfg=cfg,
        cycle_index=1,
        stage_history=[],
        last_summary=None,
        experiment_id=99,
    )

    assert outcome.context["scratchpad_summary"]["event_count"] == 2


def test_supervision_prompt_includes_planner_intent_prior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _make_agent(monkeypatch)
    diagnostics = planner.OptimizerDiagnostics(
        step=1,
        trajectory_id=0,
        objective=1.0,
        objective_delta=-0.1,
        max_violation=0.2,
        constraints_raw=[0.2, 0.1],
        multipliers=[1.0, 0.5],
        penalty_parameters=[10.0, 5.0],
        bounds_norm=0.5,
        status="STAGNATION",
        constraint_diagnostics=[
            planner.ConstraintDiagnostic(
                name="aspect_ratio",
                violation=0.2,
                penalty=10.0,
                multiplier=1.0,
                trend="increasing_violation",
                delta=0.05,
            ),
            planner.ConstraintDiagnostic(
                name="qi",
                violation=0.1,
                penalty=5.0,
                multiplier=0.5,
                trend="stable",
                delta=0.0,
            ),
        ],
        narrative=["stagnating"],
        steps_since_improvement=2,
    )
    prompt = agent._build_supervision_prompt(
        cycle=2,
        rag_context=[],
        diagnostics=diagnostics,
        planner_intent={
            "primary_constraint_order": ["aspect_ratio", "qi"],
            "penalty_focus_indices": [0],
            "confidence": 0.7,
        },
    )

    assert "Planner intent prior (SOFT PRIOR)" in prompt
    assert '"penalty_focus_indices": [' in prompt
    assert "0" in prompt
