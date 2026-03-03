#!/usr/bin/env python3
"""Stateful Codex transport for governor LLM decision calls.

Reads a request JSON on stdin:
{
  "model": "...",
  "session_id": "...",
  "observation": {...}
}

Returns a strict JSON decision object on stdout.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STATE_DIR = Path(".codex") / "stateful_llm_sessions"
HISTORY_LIMIT = 24
PROMPT_HISTORY_LIMIT = 8

OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "action",
        "target_constraint",
        "mutations",
        "expected_effect",
        "restart_plan",
    ],
    "properties": {
        "action": {"type": "string"},
        "target_constraint": {"type": "string"},
        "mutations": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["parameter_group", "normalized_delta"],
                "properties": {
                    "parameter_group": {"type": "string"},
                    "normalized_delta": {"type": "number"},
                },
                "additionalProperties": False,
            },
        },
        "expected_effect": {"type": "string"},
        "restart_plan": {"type": ["string", "null"]},
    },
    "additionalProperties": False,
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_session_key(raw: str | None) -> str:
    if raw is None:
        return "default"
    text = str(raw).strip()
    if not text:
        return "default"
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "_", text)
    return cleaned[:120] if len(cleaned) > 120 else cleaned


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"history": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"history": []}
    if not isinstance(payload, dict):
        return {"history": []}
    history = payload.get("history", [])
    if not isinstance(history, list):
        history = []
    return {"history": history}


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    history = state.get("history", [])
    if not isinstance(history, list):
        history = []
    state["history"] = history[-HISTORY_LIMIT:]
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _fallback_decision(observation: dict[str, Any]) -> dict[str, Any]:
    problem = observation.get("problem", {})
    constraints = problem.get("constraints", []) if isinstance(problem, dict) else []
    allowed_actions = (
        problem.get("allowed_actions", []) if isinstance(problem, dict) else []
    )
    context = (
        observation.get("state", {}).get("context", {})
        if isinstance(observation.get("state"), dict)
        else {}
    )
    dominant = context.get("dominant_violation", "unknown")
    allowed_names: list[str] = []
    for item in constraints:
        if isinstance(item, dict):
            name = item.get("name")
            if isinstance(name, str) and name.strip():
                allowed_names.append(name.strip())
    target = (
        dominant
        if isinstance(dominant, str) and dominant in set(allowed_names)
        else (allowed_names[0] if allowed_names else "log10_qi")
    )

    action = "repair"
    if not isinstance(allowed_actions, list) or "repair" not in allowed_actions:
        if isinstance(allowed_actions, list) and allowed_actions:
            first = allowed_actions[0]
            if isinstance(first, str) and first.strip():
                action = first.strip()

    mutations: list[dict[str, Any]] = []
    if action in {"repair", "bridge", "jump"}:
        mutations = [{"parameter_group": "axisym_z", "normalized_delta": 0.02}]

    restart_plan: str | None = None
    if action == "global_restart":
        restart_plan = "global_restart"

    return {
        "action": action,
        "target_constraint": target,
        "mutations": mutations,
        "expected_effect": "fallback_transport_decision",
        "restart_plan": restart_plan,
    }


def _build_prompt(
    *,
    request: dict[str, Any],
    history: list[dict[str, Any]],
) -> str:
    observation = request.get("observation", {})
    if not isinstance(observation, dict):
        observation = {}
    problem = observation.get("problem", {})
    constraints = problem.get("constraints", []) if isinstance(problem, dict) else []
    allowed_actions = (
        problem.get("allowed_actions", []) if isinstance(problem, dict) else []
    )
    context = (
        observation.get("state", {}).get("context", {})
        if isinstance(observation.get("state"), dict)
        else {}
    )
    recent = history[-PROMPT_HISTORY_LIMIT:]
    return (
        "You are a strict JSON decision engine for fusion-governor control.\n"
        "Return JSON only. No markdown and no explanation.\n"
        "Follow the output schema exactly.\n"
        "Use only allowed actions and constraints.\n"
        "If policy_restart_plan is global_restart or circuit_break, obey hard policy.\n"
        "If stagnation_cycles is high, avoid no-op mutations.\n\n"
        f"Allowed actions: {json.dumps(allowed_actions)}\n"
        f"Constraints: {json.dumps(constraints)}\n"
        f"Context: {json.dumps(context)}\n"
        f"Recent decision history: {json.dumps(recent)}\n\n"
        f"Current request JSON:\n{json.dumps(request)}\n"
    )


def _run_codex(
    *, model: str, prompt: str, schema_path: Path, output_path: Path
) -> bool:
    cmd = [
        os.environ.get("REAL_CODEX_BIN", "codex"),
        "exec",
        "--skip-git-repo-check",
        "--output-schema",
        str(schema_path),
        "-o",
        str(output_path),
        "-",
    ]
    if model.strip():
        cmd.extend(["-m", model.strip()])
    result = subprocess.run(
        cmd,
        input=prompt,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def main() -> None:
    raw = sys.stdin.read()
    request: dict[str, Any] = {}
    try:
        loaded = json.loads(raw) if raw.strip() else {}
        if isinstance(loaded, dict):
            request = loaded
    except json.JSONDecodeError:
        request = {}

    model = str(request.get("model", "gpt-5"))
    session_key = _safe_session_key(request.get("session_id"))
    observation = request.get("observation", {})
    if not isinstance(observation, dict):
        observation = {}

    state_path = STATE_DIR / f"{session_key}.json"
    state = _load_state(state_path)
    history_raw = state.get("history", [])
    history: list[dict[str, Any]] = (
        [item for item in history_raw if isinstance(item, dict)]
        if isinstance(history_raw, list)
        else []
    )

    decision = _fallback_decision(observation)
    with tempfile.TemporaryDirectory(prefix="codex_stateful_") as tmp_dir_text:
        tmp_dir = Path(tmp_dir_text)
        schema_path = tmp_dir / "schema.json"
        output_path = tmp_dir / "out.json"
        schema_path.write_text(json.dumps(OUTPUT_SCHEMA), encoding="utf-8")
        prompt = _build_prompt(request=request, history=history)
        ok = _run_codex(
            model=model,
            prompt=prompt,
            schema_path=schema_path,
            output_path=output_path,
        )
        if ok and output_path.exists():
            try:
                parsed = json.loads(output_path.read_text(encoding="utf-8"))
                if isinstance(parsed, dict):
                    decision = parsed
            except json.JSONDecodeError:
                pass

    history.append(
        {
            "timestamp": _now_iso(),
            "decision": decision,
            "phase": observation.get("phase", {}),
            "frontier": observation.get("frontier", {}),
        }
    )
    state["history"] = history
    _save_state(state_path, state)
    sys.stdout.write(json.dumps(decision))


if __name__ == "__main__":
    main()
