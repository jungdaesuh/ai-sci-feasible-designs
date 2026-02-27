from __future__ import annotations

from pathlib import Path

from ai_scientist.config import load_model_config


def test_codex_native_canary_defaults_are_pinned() -> None:
    config = load_model_config(Path("configs/model.codex_native_canary.yaml"))
    assert config.default_provider == "codex_native"
    assert config.instruct_model == "codex-native-short-loop"
    assert config.thinking_model == "codex-native-full"
    assert config.role_map["planning"] == "codex-native-full"
    assert config.role_map["literature"] == "codex-native-full"
    assert config.role_map["analysis"] == "codex-native-full"
