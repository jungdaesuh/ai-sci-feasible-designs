"""LLM decision contract helpers for autonomous governor loops.

This module contains only planning/schema logic. It does not execute physics
or enqueue/evaluate candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import shlex
import subprocess
from typing import Mapping, Sequence

from ai_scientist.problem_profiles import ProblemProfile, profile_prompt_block

_ALLOWED_RESTART_PLANS = frozenset(
    {"soft_retry", "degraded_restart", "global_restart", "circuit_break"}
)


@dataclass(frozen=True)
class MutationEdit:
    parameter_group: str
    normalized_delta: float


@dataclass(frozen=True)
class Decision:
    action: str
    target_constraint: str
    mutations: tuple[MutationEdit, ...]
    expected_effect: str
    restart_plan: str | None


@dataclass(frozen=True)
class ValidatedDecision:
    decision: Decision
    selected_action: str
    selected_constraint: str
    used_fallback: bool
    fallback_reason: str | None


@dataclass(frozen=True)
class DecideResult:
    validated_decision: ValidatedDecision
    payload: Mapping[str, object]
    input_source: str


def _coerce_float(value: object, *, default: float) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def build_observation(
    *,
    profile: ProblemProfile,
    rows: Sequence[Mapping[str, object]],
    context: Mapping[str, object],
) -> dict:
    feasible_count = 0
    for row in rows:
        is_feasible_raw = row.get("is_feasible")
        if isinstance(is_feasible_raw, bool) and is_feasible_raw:
            feasible_count += 1

    recent_size = len(rows)
    feasible_rate = (float(feasible_count) / float(recent_size)) if recent_size else 0.0

    dominant_violation = str(context.get("dominant_violation", "unknown"))
    phase = str(context.get("phase", "feasibility_recovery"))
    frontier_hv = _coerce_float(context.get("hv_at_decision"), default=0.0)
    frontier_record = _coerce_float(context.get("record_hv"), default=0.0)
    lessons = context.get("lesson_summary", {})
    if not isinstance(lessons, Mapping):
        lessons = {}

    return {
        "challenge": {
            "name": "constellaration",
            "deterministic_validity_required": True,
            "purpose": "Recover feasibility first, then improve objective frontier under hard physics constraints.",
        },
        "problem": profile_prompt_block(profile),
        "phase": {
            "name": phase,
            "target": (
                "feasibility_recovery"
                if phase == "feasibility_recovery"
                else "frontier_improvement"
            ),
            "dominant_violation": dominant_violation,
        },
        "frontier": {
            "hv_at_decision": frontier_hv,
            "record_hv": frontier_record,
            "gap_to_record": max(0.0, frontier_record - frontier_hv),
        },
        "lessons": {str(key): value for key, value in lessons.items()},
        "output_contract": {
            "json_only": True,
            "required_fields": [
                "action",
                "target_constraint",
                "mutations",
                "expected_effect",
            ],
        },
        "state": {
            "recent_sample_size": recent_size,
            "recent_feasible_rate": feasible_rate,
            "context": {str(key): value for key, value in context.items()},
        },
    }


def parse_decision_payload(payload: Mapping[str, object]) -> Decision:
    action_raw = payload.get("action")
    target_raw = payload.get("target_constraint")
    expected_effect_raw = payload.get("expected_effect")
    mutations_raw = payload.get("mutations", [])
    restart_plan_raw = payload.get("restart_plan")

    if not isinstance(action_raw, str) or not action_raw.strip():
        raise ValueError("decision.action must be a non-empty string")
    if not isinstance(target_raw, str) or not target_raw.strip():
        raise ValueError("decision.target_constraint must be a non-empty string")
    if not isinstance(expected_effect_raw, str) or not expected_effect_raw.strip():
        raise ValueError("decision.expected_effect must be a non-empty string")
    if not isinstance(mutations_raw, list):
        raise ValueError("decision.mutations must be a list")

    parsed_mutations: list[MutationEdit] = []
    for item in mutations_raw:
        if not isinstance(item, Mapping):
            raise ValueError("each mutation must be an object")
        parameter_group_raw = item.get("parameter_group")
        normalized_delta_raw = item.get("normalized_delta")
        if not isinstance(parameter_group_raw, str) or not parameter_group_raw.strip():
            raise ValueError("mutation.parameter_group must be a non-empty string")
        if isinstance(normalized_delta_raw, bool) or not isinstance(
            normalized_delta_raw, (int, float)
        ):
            raise ValueError("mutation.normalized_delta must be numeric")
        normalized_delta = float(normalized_delta_raw)
        if not math.isfinite(normalized_delta):
            raise ValueError("mutation.normalized_delta must be finite")
        parsed_mutations.append(
            MutationEdit(
                parameter_group=str(parameter_group_raw),
                normalized_delta=normalized_delta,
            )
        )

    restart_plan: str | None = None
    if restart_plan_raw is not None:
        if not isinstance(restart_plan_raw, str) or not restart_plan_raw.strip():
            raise ValueError("decision.restart_plan must be null or non-empty string")
        normalized_restart_plan = str(restart_plan_raw).strip().lower()
        if normalized_restart_plan not in _ALLOWED_RESTART_PLANS:
            allowed = ", ".join(sorted(_ALLOWED_RESTART_PLANS))
            raise ValueError(f"decision.restart_plan must be one of: {allowed}")
        restart_plan = normalized_restart_plan

    return Decision(
        action=str(action_raw),
        target_constraint=str(target_raw),
        mutations=tuple(parsed_mutations),
        expected_effect=str(expected_effect_raw),
        restart_plan=restart_plan,
    )


def load_optional_decision_file(path: Path | None) -> Mapping[str, object] | None:
    if path is None:
        return None
    try:
        payload = json.loads(path.read_text())
    except OSError as exc:
        raise ValueError(f"Unable to read LLM decision file {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid LLM decision JSON at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("LLM decision payload must be a JSON object.")
    return payload


def _parse_command_json_output(stdout: str) -> Mapping[str, object]:
    text = str(stdout).strip()
    if not text:
        raise ValueError("LLM decision command produced empty stdout.")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            raise ValueError("LLM decision command produced empty stdout.")
        try:
            payload = json.loads(lines[-1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON from LLM decision command: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("LLM decision command output must be a JSON object.")
    return payload


def load_optional_decision_command(
    *,
    command: str | None,
    observation: Mapping[str, object],
    model: str,
    session_id: str | None,
) -> Mapping[str, object] | None:
    if command is None:
        return None
    command_text = str(command).strip()
    if not command_text:
        raise ValueError("LLM decision command must not be empty.")
    request_payload = {
        "model": str(model),
        "session_id": session_id,
        "observation": dict(observation),
    }
    result = subprocess.run(
        shlex.split(command_text),
        input=json.dumps(request_payload, separators=(",", ":")),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = str(result.stderr or "").strip()
        raise ValueError(
            f"LLM decision command failed with exit={result.returncode}: {stderr}"
        )
    return _parse_command_json_output(result.stdout)


def resolve_decision_payload(
    *,
    decision_file: Path | None,
    decision_command: str | None,
    observation: Mapping[str, object],
    model: str,
    session_id: str | None,
) -> tuple[Mapping[str, object], str]:
    file_error: ValueError | None = None
    try:
        from_file = load_optional_decision_file(decision_file)
    except ValueError as exc:
        if decision_command is None:
            raise
        file_error = exc
        from_file = None
    if from_file is not None:
        return dict(from_file), "file"
    from_command = load_optional_decision_command(
        command=decision_command,
        observation=observation,
        model=model,
        session_id=session_id,
    )
    if from_command is not None:
        return dict(from_command), "codex_command"
    if file_error is not None:
        raise file_error
    raise ValueError(
        "No LLM decision source resolved; provide codex command transport or explicit decision file."
    )


def _repair_observation(
    *,
    observation: Mapping[str, object],
    attempt_index: int,
    previous_error: str,
) -> dict:
    out = dict(observation)
    state_raw = out.get("state")
    state = dict(state_raw) if isinstance(state_raw, Mapping) else {}
    context_raw = state.get("context")
    context = dict(context_raw) if isinstance(context_raw, Mapping) else {}
    context["llm_repair_feedback"] = {
        "attempt_index": int(attempt_index),
        "previous_error": str(previous_error),
        "instruction": (
            "Return JSON only. Obey profile allowed_actions/constraints, mutation caps, "
            "and required output fields."
        ),
    }
    state["context"] = context
    out["state"] = state
    return out


def _stagnation_cycles(observation: Mapping[str, object]) -> int:
    state_raw = observation.get("state")
    if not isinstance(state_raw, Mapping):
        return 0
    context_raw = state_raw.get("context")
    if not isinstance(context_raw, Mapping):
        return 0
    value = context_raw.get("stagnation_cycles")
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(0, int(value))
    if isinstance(value, float):
        if not math.isfinite(float(value)):
            return 0
        return max(0, int(value))
    return 0


def _policy_restart_plan(observation: Mapping[str, object]) -> str:
    state_raw = observation.get("state")
    if not isinstance(state_raw, Mapping):
        return ""
    context_raw = state_raw.get("context")
    if not isinstance(context_raw, Mapping):
        return ""
    value = context_raw.get("policy_restart_plan")
    if not isinstance(value, str):
        return ""
    return str(value).strip().lower()


def _enforce_novelty_contract(
    *,
    profile: ProblemProfile,
    decision: Decision,
    observation: Mapping[str, object],
) -> None:
    action = decision.action.strip().lower()
    if action not in {"repair", "bridge", "jump"}:
        return
    if _policy_restart_plan(observation) in {"global_restart", "circuit_break"}:
        return
    stagnation_cycles = _stagnation_cycles(observation)
    min_stagnation = max(1, int(profile.autonomy_policy.anti_repeat_no_progress_cycles))
    if stagnation_cycles < min_stagnation:
        return
    if not decision.mutations:
        raise ValueError(
            f"stagnation novelty contract: action {action} requires non-empty mutations"
        )
    min_delta = float(profile.autonomy_policy.stagnation_min_mutation_delta)
    has_nonzero = any(
        abs(float(mutation.normalized_delta)) >= min_delta
        for mutation in decision.mutations
    )
    if not has_nonzero:
        raise ValueError(
            "stagnation novelty contract: all mutation deltas are below "
            f"minimum {min_delta:.6f}"
        )


def decide(
    *,
    profile: ProblemProfile,
    observation: Mapping[str, object],
    model: str,
    session_id: str | None,
    decision_file: Path | None,
    decision_command: str | None,
    repair_attempts: int = 2,
) -> DecideResult:
    max_attempts = max(1, int(repair_attempts) + 1)
    last_error: ValueError | None = None
    for attempt in range(max_attempts):
        attempt_observation: Mapping[str, object]
        if attempt == 0:
            attempt_observation = observation
        else:
            if last_error is None:
                break
            if decision_command is None:
                # Decision-file-only mode cannot self-repair in-process.
                break
            attempt_observation = _repair_observation(
                observation=observation,
                attempt_index=attempt,
                previous_error=str(last_error),
            )
        try:
            payload, input_source = resolve_decision_payload(
                decision_file=decision_file,
                decision_command=decision_command,
                observation=attempt_observation,
                model=model,
                session_id=session_id,
            )
            decision = parse_decision_payload(payload)
            validated_decision = validate_decision(
                profile=profile,
                decision=decision,
            )
            _enforce_novelty_contract(
                profile=profile,
                decision=decision,
                observation=attempt_observation,
            )
            if attempt > 0 and last_error is not None:
                validated_decision = ValidatedDecision(
                    decision=validated_decision.decision,
                    selected_action=validated_decision.selected_action,
                    selected_constraint=validated_decision.selected_constraint,
                    used_fallback=True,
                    fallback_reason=f"in_session_repair:{last_error}",
                )
            return DecideResult(
                validated_decision=validated_decision,
                payload=payload,
                input_source=input_source,
            )
        except ValueError as exc:
            last_error = exc
            continue
    if last_error is None:
        raise ValueError("LLM decision unresolved with no captured error.")
    raise ValueError(
        f"LLM decision unresolved after {max_attempts} attempts: {last_error}"
    )


def validate_decision(
    *,
    profile: ProblemProfile,
    decision: Decision,
) -> ValidatedDecision:
    action = decision.action.strip().lower()
    if not profile.allows_action(action):
        raise ValueError(
            f"Action {decision.action!r} is not allowed for problem {profile.problem}"
        )

    allowed_constraints = set(profile.allowed_constraint_names())
    target_constraint = decision.target_constraint.strip().lower()
    if target_constraint not in allowed_constraints:
        raise ValueError(
            f"Constraint {decision.target_constraint!r} is not allowed for problem {profile.problem}"
        )
    if decision.restart_plan is not None:
        restart_plan = decision.restart_plan.strip().lower()
        if restart_plan not in _ALLOWED_RESTART_PLANS:
            allowed = ", ".join(sorted(_ALLOWED_RESTART_PLANS))
            raise ValueError(f"restart_plan is invalid; expected one of: {allowed}")

    if action == "global_restart" and decision.mutations:
        raise ValueError("global_restart does not accept mutations")

    max_mutations = int(profile.mutation_budget.max_mutations_per_candidate)
    if len(decision.mutations) > max_mutations:
        raise ValueError(
            f"Too many mutations: {len(decision.mutations)} > {max_mutations}"
        )

    normalized_groups = [
        mutation.parameter_group.strip().lower() for mutation in decision.mutations
    ]
    if len(set(normalized_groups)) != len(normalized_groups):
        raise ValueError("duplicate mutation.parameter_group entries are not allowed")
    unique_groups = set(normalized_groups)
    if len(unique_groups) > int(
        profile.mutation_budget.max_mutation_groups_per_candidate
    ):
        raise ValueError(
            "Too many mutation groups: "
            f"{len(unique_groups)} > {profile.mutation_budget.max_mutation_groups_per_candidate}"
        )

    delta_cap = float(profile.mutation_budget.action_delta_cap(action))
    for mutation in decision.mutations:
        if abs(float(mutation.normalized_delta)) > delta_cap:
            raise ValueError(
                "Mutation delta exceeds cap for action "
                f"{action}: {mutation.normalized_delta} > {delta_cap}"
            )

    return ValidatedDecision(
        decision=decision,
        selected_action=action,
        selected_constraint=target_constraint,
        used_fallback=False,
        fallback_reason=None,
    )


def evaluate_phase_transition(
    *,
    profile: ProblemProfile,
    current_phase: str,
    accepted_feasible_last20: int,
    dominant_violation_rate_last20: float,
    accepted_feasible_last10: int,
) -> str:
    phase = str(current_phase)
    thresholds = profile.phase_switch_thresholds
    if phase == "feasibility_recovery":
        if int(accepted_feasible_last20) >= int(
            thresholds.improve_min_accepted_feasible_last20
        ) and float(dominant_violation_rate_last20) <= float(
            thresholds.improve_max_dominant_violation_rate_last20
        ):
            return "frontier_improvement"
        return "feasibility_recovery"

    if int(accepted_feasible_last10) <= int(
        thresholds.revert_if_accepted_feasible_last10_eq
    ):
        return "feasibility_recovery"
    return "frontier_improvement"


def select_restart_plan(
    *,
    profile: ProblemProfile,
    consecutive_transient_failures: int,
    queue_desync_events_last20: int,
    stagnation_cycles: int,
    budget_remaining: float,
    invalid_llm_outputs_last20: int,
    schema_compatible: bool,
    frontier_integrity_ok: bool,
) -> str:
    thresholds = profile.restart_thresholds
    transient_failures = int(consecutive_transient_failures)
    if not schema_compatible:
        return "circuit_break"
    if float(budget_remaining) <= 0.0:
        return "circuit_break"
    if int(invalid_llm_outputs_last20) >= int(
        thresholds.circuit_break_min_invalid_outputs_last20
    ):
        return "circuit_break"
    if not frontier_integrity_ok:
        return "global_restart"
    if int(stagnation_cycles) >= int(thresholds.global_restart_min_stagnation_cycles):
        return "global_restart"
    if transient_failures >= int(thresholds.degraded_restart_min_consecutive_failures):
        return "degraded_restart"
    if int(queue_desync_events_last20) >= int(
        thresholds.degraded_restart_min_queue_desync_events_last20
    ):
        return "degraded_restart"
    if transient_failures > 0 and transient_failures <= int(
        thresholds.soft_retry_max_consecutive_failures
    ):
        return "soft_retry"
    return "continue"
