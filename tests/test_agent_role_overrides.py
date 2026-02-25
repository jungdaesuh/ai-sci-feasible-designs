from __future__ import annotations

from ai_scientist import agent
from ai_scientist.config import load_model_config


def test_role_map_env_overrides_planning_gate(monkeypatch) -> None:
    monkeypatch.setenv("AI_SCIENTIST_ROLE_PLANNING_MODEL", "codex-native-full")
    config = load_model_config()
    gate = agent.provision_model_tier(role="planning", config=config)
    assert gate.model_alias == "codex-native-full"


def test_role_map_env_overrides_literature_gate(monkeypatch) -> None:
    monkeypatch.setenv("AI_SCIENTIST_ROLE_LITERATURE_MODEL", "codex-native-full")
    config = load_model_config()
    gate = agent.provision_model_tier(role="literature", config=config)
    assert gate.model_alias == "codex-native-full"


def test_role_map_env_overrides_analysis_gate(monkeypatch) -> None:
    monkeypatch.setenv("AI_SCIENTIST_ROLE_ANALYSIS_MODEL", "codex-native-full")
    config = load_model_config()
    gate = agent.provision_model_tier(role="analysis", config=config)
    assert gate.model_alias == "codex-native-full"
