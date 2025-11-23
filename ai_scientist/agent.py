"""Client gating helpers so K2-Instruct/K2-Thinking emit valid tool calls (docs/TASKS_CODEX_MINI.md:157-190)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

from ai_scientist.config import ModelConfig, load_model_config
from ai_scientist.tools_api import TOOL_SCHEMA_BY_NAME

_LOGGER = logging.getLogger(__name__)

_ROLE_ALIAS_MAP = {
    "screen": lambda cfg: cfg.instruct_model,
    "short_loop": lambda cfg: cfg.instruct_model,
    "prompt": lambda cfg: cfg.instruct_model,
    "planning": lambda cfg: cfg.thinking_model,
    "report": lambda cfg: cfg.thinking_model,
    "verification": lambda cfg: cfg.thinking_model,
    "literature": lambda cfg: cfg.role_map.get("literature", cfg.thinking_model),
    "analysis": lambda cfg: cfg.role_map.get("analysis", cfg.thinking_model),
}


@dataclass(frozen=True)
class AgentGate:
    model_alias: str
    allowed_tools: Tuple[str, ...]
    system_prompt: str
    provider_model: str

    def allows(self, tool_name: str) -> bool:
        return tool_name in self.allowed_tools


def gates_from_config(config: ModelConfig) -> tuple[AgentGate, ...]:
    return tuple(
        AgentGate(
            model_alias=gate.model_alias,
            allowed_tools=tuple(gate.allowed_tools),
            system_prompt=gate.system_prompt or "",
            provider_model=gate.provider_model or gate.model_alias,
        )
        for gate in config.agent_gates
    )


def gate_for_model(config: ModelConfig, model_alias: str) -> AgentGate | None:
    for gate in gates_from_config(config):
        if gate.model_alias == model_alias:
            return gate
    return None


def _resolve_alias_for_role(role: str | None, config: ModelConfig) -> str:
    normalized = (role or "short_loop").lower()
    resolver = _ROLE_ALIAS_MAP.get(normalized)
    if resolver:
        return resolver(config)
    return config.instruct_model


def provision_model_tier(
    role: str | None = None, *, config: ModelConfig | None = None
) -> AgentGate:
    """Return the AgentGate backing the K2 tier that best fits the requested role."""

    resolved = config or load_model_config()
    alias = _resolve_alias_for_role(role, resolved)
    gate = gate_for_model(resolved, alias)
    if gate is None:
        raise ValueError(
            f"Configured model '{alias}' is not declared in configs/model.yaml"
        )
    _LOGGER.info(
        "Provisioned %s tier for role=%s (base_url=%s, tools=%s)",
        alias,
        (role or "short_loop").lower(),
        resolved.base_url,
        gate.allowed_tools,
    )
    return gate


def validate_tool_call(config: ModelConfig, model_alias: str, tool_name: str) -> None:
    gate = gate_for_model(config, model_alias)
    if gate is None:
        raise ValueError(f"Unknown agent model '{model_alias}'")
    if tool_name not in gate.allowed_tools:
        raise ValueError(
            f"Tool '{tool_name}' is not permitted for {model_alias}; allowed {gate.allowed_tools}"
        )
    if tool_name not in TOOL_SCHEMA_BY_NAME:
        raise ValueError(f"No schema registered for tool '{tool_name}'")
