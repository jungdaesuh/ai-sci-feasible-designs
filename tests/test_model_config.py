import pytest

from ai_scientist.config import ModelConfig, ProviderConfig, load_model_config


def _base_model_config() -> ModelConfig:
    provider = ProviderConfig(
        name="openrouter",
        base_url="https://dummy",
        chat_path="/chat",
        auth_env="OPENROUTER_KEY",
        default_model="dummy-model",
        extra_headers=(),
    )
    return ModelConfig(
        base_url="https://dummy",
        instruct_model="kimi-k2-instruct",
        thinking_model="kimi-k2-thinking",
        request_timeout_seconds=60,
        rate_limit_per_minute=600,
        context_length=8192,
        dtype="bf16",
        tensor_parallel=1,
        default_provider="openrouter",
        providers=(provider,),
        agent_gates=(),
    )


def test_get_provider_missing_fails() -> None:
    config = _base_model_config()
    try:
        config.get_provider("unknown-alias")
        raise AssertionError("expected ValueError for unknown provider alias")
    except ValueError as exc:
        assert "unknown-alias" in str(exc)


def test_env_overrides_instruct_and_thinking(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    config_path = tmp_path / "model.yaml"
    config_path.write_text(
        """
model:
  provider: openrouter
  instruct_model: kimi-default
  thinking_model: kimi-think
  providers:
    openrouter:
      base_url: https://dummy
      chat_path: /chat
      auth_env: OPENROUTER_KEY
      default_model: kimi-default
""".strip()
    )
    monkeypatch.setenv("AI_SCIENTIST_INSTRUCT_MODEL", "env-instruct")
    monkeypatch.setenv("AI_SCIENTIST_THINKING_MODEL", "env-thinking")
    config = load_model_config(config_path)
    assert config.instruct_model == "env-instruct"
    assert config.thinking_model == "env-thinking"


def test_env_overrides_default_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    config_path = tmp_path / "model.yaml"
    config_path.write_text(
        """
model:
  provider: openrouter
  providers:
    openrouter:
      base_url: https://dummy
      chat_path: /chat
      auth_env: OPENROUTER_KEY
      default_model: kimi-default
    moonshot:
      base_url: https://moon
      chat_path: /chat
      auth_env: MOON_KEY
      default_model: moon-model
""".strip()
    )
    monkeypatch.setenv("MODEL_PROVIDER", "moonshot")
    config = load_model_config(config_path)
    assert config.default_provider == "moonshot"


def test_env_overrides_model_config_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    model_default = tmp_path / "model.default.yaml"
    model_default.write_text(
        """
model:
  provider: openrouter
  instruct_model: default-instruct
  thinking_model: default-thinking
  providers:
    openrouter:
      base_url: https://dummy
      chat_path: /chat
      auth_env: OPENROUTER_KEY
      default_model: default-model
""".strip()
    )
    model_canary = tmp_path / "model.canary.yaml"
    model_canary.write_text(
        """
model:
  provider: codex_native
  instruct_model: codex-native-short-loop
  thinking_model: codex-native-full
  role_map:
    planning: codex-native-full
    literature: codex-native-full
    analysis: codex-native-full
  providers:
    codex_native:
      base_url: https://chatgpt.com/backend-api
      chat_path: /codex/responses
      auth_env: CODEX_NATIVE_BEARER_TOKEN
      default_model: gpt-5.3-codex
""".strip()
    )
    monkeypatch.setenv("AI_SCIENTIST_MODEL_CONFIG_PATH", str(model_canary))
    config = load_model_config(model_default)
    assert config.default_provider == "openrouter"

    config_from_env = load_model_config()
    assert config_from_env.default_provider == "codex_native"
    assert config_from_env.instruct_model == "codex-native-short-loop"
    assert config_from_env.thinking_model == "codex-native-full"
    assert config_from_env.role_map["planning"] == "codex-native-full"

    # Empty path should behave like omitted path and still honor env override.
    config_from_empty = load_model_config("")
    assert config_from_empty.default_provider == "codex_native"
