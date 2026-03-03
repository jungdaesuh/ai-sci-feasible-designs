from __future__ import annotations

import pytest
from typing import Mapping

from ai_scientist.llm_controller import (
    Decision,
    build_observation,
    decide,
    evaluate_phase_transition,
    parse_decision_payload,
    select_restart_plan,
    validate_decision,
)
from ai_scientist.problem_profiles import get_problem_profile


def test_parse_and_validate_decision_for_p3() -> None:
    profile = get_problem_profile("p3")
    decision = parse_decision_payload(
        {
            "action": "repair",
            "target_constraint": "log10_qi",
            "mutations": [
                {"parameter_group": "axisym_z", "normalized_delta": 0.05},
                {"parameter_group": "m_ge_3", "normalized_delta": -0.1},
            ],
            "expected_effect": "reduce_log10_qi_violation",
            "restart_plan": None,
        }
    )
    validated = validate_decision(profile=profile, decision=decision)
    assert validated.selected_action == "repair"
    assert validated.selected_constraint == "log10_qi"


def test_validate_decision_rejects_action_not_allowed() -> None:
    profile = get_problem_profile("p3")
    decision = parse_decision_payload(
        {
            "action": "jump",
            "target_constraint": "log10_qi",
            "mutations": [],
            "expected_effect": "test",
            "restart_plan": None,
        }
    )
    with pytest.raises(ValueError):
        validate_decision(profile=profile, decision=decision)


def test_validate_decision_rejects_mutation_cap_overflow() -> None:
    profile = get_problem_profile("p3")
    decision = parse_decision_payload(
        {
            "action": "repair",
            "target_constraint": "log10_qi",
            "mutations": [{"parameter_group": "axisym_z", "normalized_delta": 0.99}],
            "expected_effect": "test",
            "restart_plan": None,
        }
    )
    with pytest.raises(ValueError):
        validate_decision(profile=profile, decision=decision)


def test_validate_decision_rejects_global_restart_with_mutations() -> None:
    profile = get_problem_profile("p3")
    decision = parse_decision_payload(
        {
            "action": "global_restart",
            "target_constraint": "log10_qi",
            "mutations": [{"parameter_group": "axisym_z", "normalized_delta": 0.01}],
            "expected_effect": "restart",
            "restart_plan": "global_restart",
        }
    )
    with pytest.raises(ValueError, match="global_restart does not accept mutations"):
        validate_decision(profile=profile, decision=decision)


def test_parse_decision_rejects_non_finite_delta() -> None:
    with pytest.raises(ValueError):
        parse_decision_payload(
            {
                "action": "repair",
                "target_constraint": "log10_qi",
                "mutations": [
                    {"parameter_group": "axisym_z", "normalized_delta": float("nan")}
                ],
                "expected_effect": "test",
                "restart_plan": None,
            }
        )


def test_parse_decision_rejects_invalid_restart_plan_enum() -> None:
    with pytest.raises(ValueError, match="decision.restart_plan must be one of"):
        parse_decision_payload(
            {
                "action": "repair",
                "target_constraint": "log10_qi",
                "mutations": [],
                "expected_effect": "test",
                "restart_plan": "not_a_real_plan",
            }
        )


def test_validate_decision_rejects_invalid_restart_plan_in_decision_object() -> None:
    profile = get_problem_profile("p3")
    decision = Decision(
        action="repair",
        target_constraint="log10_qi",
        mutations=tuple(),
        expected_effect="test",
        restart_plan="not_a_real_plan",
    )
    with pytest.raises(ValueError, match="restart_plan is invalid"):
        validate_decision(profile=profile, decision=decision)


def test_validate_decision_rejects_case_duplicate_groups() -> None:
    profile = get_problem_profile("p3")
    decision = parse_decision_payload(
        {
            "action": "repair",
            "target_constraint": "log10_qi",
            "mutations": [
                {"parameter_group": "axisym_z", "normalized_delta": 0.05},
                {"parameter_group": "Axisym_Z", "normalized_delta": -0.02},
            ],
            "expected_effect": "test",
            "restart_plan": None,
        }
    )
    with pytest.raises(ValueError):
        validate_decision(profile=profile, decision=decision)


def test_phase_transition_rules() -> None:
    profile = get_problem_profile("p3")
    improved = evaluate_phase_transition(
        profile=profile,
        current_phase="feasibility_recovery",
        accepted_feasible_last20=3,
        dominant_violation_rate_last20=0.2,
        accepted_feasible_last10=1,
    )
    assert improved == "frontier_improvement"

    reverted = evaluate_phase_transition(
        profile=profile,
        current_phase="frontier_improvement",
        accepted_feasible_last20=5,
        dominant_violation_rate_last20=0.1,
        accepted_feasible_last10=0,
    )
    assert reverted == "feasibility_recovery"


def test_restart_plan_thresholds() -> None:
    profile = get_problem_profile("p3")
    assert (
        select_restart_plan(
            profile=profile,
            consecutive_transient_failures=0,
            queue_desync_events_last20=0,
            stagnation_cycles=0,
            budget_remaining=1.0,
            invalid_llm_outputs_last20=3,
            schema_compatible=True,
            frontier_integrity_ok=True,
        )
        == "circuit_break"
    )
    assert (
        select_restart_plan(
            profile=profile,
            consecutive_transient_failures=3,
            queue_desync_events_last20=0,
            stagnation_cycles=0,
            budget_remaining=1.0,
            invalid_llm_outputs_last20=0,
            schema_compatible=True,
            frontier_integrity_ok=True,
        )
        == "degraded_restart"
    )
    assert (
        select_restart_plan(
            profile=profile,
            consecutive_transient_failures=0,
            queue_desync_events_last20=0,
            stagnation_cycles=9,
            budget_remaining=1.0,
            invalid_llm_outputs_last20=0,
            schema_compatible=True,
            frontier_integrity_ok=True,
        )
        == "global_restart"
    )
    assert (
        select_restart_plan(
            profile=profile,
            consecutive_transient_failures=1,
            queue_desync_events_last20=0,
            stagnation_cycles=0,
            budget_remaining=1.0,
            invalid_llm_outputs_last20=0,
            schema_compatible=True,
            frontier_integrity_ok=True,
        )
        == "soft_retry"
    )
    assert (
        select_restart_plan(
            profile=profile,
            consecutive_transient_failures=0,
            queue_desync_events_last20=0,
            stagnation_cycles=0,
            budget_remaining=1.0,
            invalid_llm_outputs_last20=0,
            schema_compatible=True,
            frontier_integrity_ok=False,
        )
        == "global_restart"
    )


def test_build_observation_includes_prompt_contract_blocks() -> None:
    profile = get_problem_profile("p3")
    observation = build_observation(
        profile=profile,
        rows=[{"is_feasible": False}, {"is_feasible": True}],
        context={
            "phase": "feasibility_recovery",
            "dominant_violation": "log10_qi",
            "hv_at_decision": 10.0,
            "record_hv": 12.0,
            "lesson_summary": {"operator_stats": [{"operator_family": "scale_groups"}]},
        },
    )
    assert observation["challenge"]["name"] == "constellaration"
    assert observation["problem"]["problem"] == "p3"
    assert observation["phase"]["dominant_violation"] == "log10_qi"
    assert observation["frontier"]["gap_to_record"] == 2.0
    assert observation["output_contract"]["json_only"] is True


def test_decide_returns_validated_decision_and_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = get_problem_profile("p3")

    def _none_file(path):
        return None

    def _payload_command(*, command, observation, model, session_id):
        assert command == "codex run --json"
        assert model == "codex"
        assert session_id == "sess-1"
        assert observation["phase"]["name"] == "feasibility_recovery"
        return {
            "action": "repair",
            "target_constraint": "log10_qi",
            "mutations": [],
            "expected_effect": "reduce violation",
            "restart_plan": None,
        }

    monkeypatch.setattr(
        "ai_scientist.llm_controller.load_optional_decision_file", _none_file
    )
    monkeypatch.setattr(
        "ai_scientist.llm_controller.load_optional_decision_command",
        _payload_command,
    )

    observation = build_observation(
        profile=profile,
        rows=[],
        context={
            "phase": "feasibility_recovery",
            "dominant_violation": "log10_qi",
            "hv_at_decision": 0.0,
            "record_hv": 1.0,
        },
    )
    result = decide(
        profile=profile,
        observation=observation,
        model="codex",
        session_id="sess-1",
        decision_file=None,
        decision_command="codex run --json",
    )
    assert result.input_source == "codex_command"
    assert result.validated_decision.selected_action == "repair"
    assert result.validated_decision.used_fallback is False
    assert result.validated_decision.fallback_reason is None


def test_decide_repairs_invalid_output_with_feedback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = get_problem_profile("p3")
    calls: list[Mapping[str, object]] = []

    def _none_file(path):
        return None

    def _payload_command(*, command, observation, model, session_id):
        calls.append(dict(observation))
        if len(calls) == 1:
            return {
                "action": "jump",
                "target_constraint": "log10_qi",
                "mutations": [],
                "expected_effect": "bad action for p3",
                "restart_plan": None,
            }
        feedback = observation["state"]["context"]["llm_repair_feedback"]
        assert int(feedback["attempt_index"]) == 1
        return {
            "action": "repair",
            "target_constraint": "log10_qi",
            "mutations": [],
            "expected_effect": "repaired output",
            "restart_plan": None,
        }

    monkeypatch.setattr(
        "ai_scientist.llm_controller.load_optional_decision_file", _none_file
    )
    monkeypatch.setattr(
        "ai_scientist.llm_controller.load_optional_decision_command",
        _payload_command,
    )
    observation = build_observation(
        profile=profile,
        rows=[],
        context={
            "phase": "feasibility_recovery",
            "dominant_violation": "log10_qi",
            "hv_at_decision": 0.0,
            "record_hv": 1.0,
        },
    )
    result = decide(
        profile=profile,
        observation=observation,
        model="codex",
        session_id="sess-2",
        decision_file=None,
        decision_command="codex run --json",
        repair_attempts=2,
    )
    assert len(calls) == 2
    assert result.validated_decision.selected_action == "repair"
    assert result.validated_decision.used_fallback is True
    assert result.validated_decision.fallback_reason is not None
    assert str(result.validated_decision.fallback_reason).startswith(
        "in_session_repair:"
    )


def test_decide_raises_after_repair_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = get_problem_profile("p3")

    def _none_file(path):
        return None

    def _invalid_command(*, command, observation, model, session_id):
        return {
            "action": "jump",
            "target_constraint": "log10_qi",
            "mutations": [],
            "expected_effect": "always invalid for p3",
            "restart_plan": None,
        }

    monkeypatch.setattr(
        "ai_scientist.llm_controller.load_optional_decision_file", _none_file
    )
    monkeypatch.setattr(
        "ai_scientist.llm_controller.load_optional_decision_command",
        _invalid_command,
    )
    observation = build_observation(
        profile=profile,
        rows=[],
        context={
            "phase": "feasibility_recovery",
            "dominant_violation": "log10_qi",
            "hv_at_decision": 0.0,
            "record_hv": 1.0,
        },
    )
    with pytest.raises(ValueError, match="after 2 attempts"):
        decide(
            profile=profile,
            observation=observation,
            model="codex",
            session_id="sess-3",
            decision_file=None,
            decision_command="codex run --json",
            repair_attempts=1,
        )


def test_decide_rejects_empty_mutations_under_stagnation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = get_problem_profile("p3")

    def _none_file(path):
        return None

    def _payload_command(*, command, observation, model, session_id):
        return {
            "action": "repair",
            "target_constraint": "log10_qi",
            "mutations": [],
            "expected_effect": "noop",
            "restart_plan": None,
        }

    monkeypatch.setattr(
        "ai_scientist.llm_controller.load_optional_decision_file", _none_file
    )
    monkeypatch.setattr(
        "ai_scientist.llm_controller.load_optional_decision_command",
        _payload_command,
    )
    observation = build_observation(
        profile=profile,
        rows=[],
        context={
            "phase": "feasibility_recovery",
            "dominant_violation": "log10_qi",
            "hv_at_decision": 0.0,
            "record_hv": 1.0,
            "stagnation_cycles": 2,
        },
    )
    with pytest.raises(ValueError, match="novelty contract"):
        decide(
            profile=profile,
            observation=observation,
            model="codex",
            session_id="sess-4",
            decision_file=None,
            decision_command="codex run --json",
            repair_attempts=0,
        )


def test_decide_rejects_small_delta_mutations_under_stagnation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = get_problem_profile("p3")

    def _none_file(path):
        return None

    def _payload_command(*, command, observation, model, session_id):
        return {
            "action": "repair",
            "target_constraint": "log10_qi",
            "mutations": [
                {"parameter_group": "axisym_z", "normalized_delta": 0.001},
                {"parameter_group": "m_ge_3", "normalized_delta": -0.005},
            ],
            "expected_effect": "tiny",
            "restart_plan": None,
        }

    monkeypatch.setattr(
        "ai_scientist.llm_controller.load_optional_decision_file", _none_file
    )
    monkeypatch.setattr(
        "ai_scientist.llm_controller.load_optional_decision_command",
        _payload_command,
    )
    observation = build_observation(
        profile=profile,
        rows=[],
        context={
            "phase": "feasibility_recovery",
            "dominant_violation": "log10_qi",
            "hv_at_decision": 0.0,
            "record_hv": 1.0,
            "stagnation_cycles": 3,
        },
    )
    with pytest.raises(ValueError, match="minimum"):
        decide(
            profile=profile,
            observation=observation,
            model="codex",
            session_id="sess-5",
            decision_file=None,
            decision_command="codex run --json",
            repair_attempts=0,
        )


def test_decide_accepts_nonzero_mutation_under_stagnation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = get_problem_profile("p3")

    def _none_file(path):
        return None

    def _payload_command(*, command, observation, model, session_id):
        return {
            "action": "repair",
            "target_constraint": "log10_qi",
            "mutations": [{"parameter_group": "axisym_z", "normalized_delta": 0.02}],
            "expected_effect": "repair",
            "restart_plan": None,
        }

    monkeypatch.setattr(
        "ai_scientist.llm_controller.load_optional_decision_file", _none_file
    )
    monkeypatch.setattr(
        "ai_scientist.llm_controller.load_optional_decision_command",
        _payload_command,
    )
    observation = build_observation(
        profile=profile,
        rows=[],
        context={
            "phase": "feasibility_recovery",
            "dominant_violation": "log10_qi",
            "hv_at_decision": 0.0,
            "record_hv": 1.0,
            "stagnation_cycles": 2,
        },
    )
    result = decide(
        profile=profile,
        observation=observation,
        model="codex",
        session_id="sess-6",
        decision_file=None,
        decision_command="codex run --json",
        repair_attempts=0,
    )
    assert result.validated_decision.selected_action == "repair"
