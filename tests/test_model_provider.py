"""Smoke tests for the per-provider request builder."""

from __future__ import annotations

import json
import os
from email.message import Message
from io import BytesIO
from typing import Any
from urllib.error import HTTPError

import pytest

from ai_scientist.auth_profile import list_auth_profiles, upsert_auth_profile
from ai_scientist.config import load_model_config
from ai_scientist.model_provider import build_chat_request, invoke_chat_completion


@pytest.fixture(autouse=True)
def _clear_env() -> None:
    for key in (
        "OPENROUTER_API_KEY",
        "MOONSHOT_API_KEY",
        "WQ_API_KEY",
        "OPENCLAW_GATEWAY_TOKEN",
        "CODEX_NATIVE_BEARER_TOKEN",
        "AI_SCIENTIST_AUTH_PROFILE_STORE_PATH",
        "AI_SCIENTIST_AUTH_SECRET_STORE_PATH",
        "AI_SCIENTIST_AUTH_MANAGED_PROVIDERS",
    ):
        os.environ.pop(key, None)


def test_openrouter_chat_request_includes_extra_headers() -> None:
    os.environ["OPENROUTER_API_KEY"] = "open-token"
    config = load_model_config()
    provider = config.get_provider("openrouter")
    request = build_chat_request(
        provider,
        tool_call={"name": "make_boundary", "arguments": {}},
        model=config.instruct_model,
    )
    assert request.path == provider.chat_path
    assert request.headers["Authorization"] == "Bearer open-token"
    assert request.headers["X-AI-Scientist-Tool-Name"] == "make_boundary"
    assert request.body["tool_call"]["name"] == "make_boundary"
    assert request.headers.get("HTTP-Referer") == "https://openrouter.ai/docs"
    assert request.headers.get("X-Title") == "ConStellaration Grok"


def test_streamlake_falls_back_to_placeholder_token() -> None:
    config = load_model_config()
    provider = config.get_provider("streamlake")
    request = build_chat_request(provider, tool_call={"name": "make_boundary"})
    assert request.headers["Authorization"].startswith("Bearer LOCAL-")
    assert request.headers["Content-Type"] == "application/json"
    assert request.path == provider.chat_path


def test_moonshot_requests_can_override_messages() -> None:
    os.environ["MOONSHOT_API_KEY"] = "moon-token"
    config = load_model_config()
    provider = config.get_provider("moonshot")
    custom_messages = [
        {"role": "user", "content": "custom prompt"},
        {"role": "assistant", "content": "ok"},
    ]
    request = build_chat_request(
        provider,
        tool_call={"name": "evaluate_p1", "arguments": {}},
        messages=custom_messages,
    )
    assert request.headers["Authorization"] == "Bearer moon-token"
    assert request.headers["X-AI-Scientist-Tool-Name"] == "evaluate_p1"
    assert request.body["messages"] == custom_messages


def test_openclaw_uses_gateway_token_auth() -> None:
    os.environ["OPENCLAW_GATEWAY_TOKEN"] = "gateway-token"
    config = load_model_config()
    provider = config.get_provider("openclaw")
    request = build_chat_request(
        provider, tool_call={"name": "evaluate_p3", "arguments": {}}
    )
    assert request.headers["Authorization"] == "Bearer gateway-token"
    assert request.headers["X-AI-Scientist-Tool-Name"] == "evaluate_p3"
    assert request.path == "/chat/completions"


def test_codex_native_uses_bearer_token_auth(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv(
        "AI_SCIENTIST_AUTH_PROFILE_STORE_PATH", str(tmp_path / "auth_profiles.json")
    )
    monkeypatch.setenv(
        "AI_SCIENTIST_AUTH_SECRET_STORE_PATH", str(tmp_path / "auth_secrets.json")
    )
    os.environ["CODEX_NATIVE_BEARER_TOKEN"] = "codex-token"
    config = load_model_config()
    provider = config.get_provider("codex_native")
    request = build_chat_request(
        provider, tool_call={"name": "evaluate_p3", "arguments": {}}
    )
    assert request.headers["Authorization"] == "Bearer codex-token"
    assert request.headers["X-AI-Scientist-Tool-Name"] == "evaluate_p3"
    assert "tool_call" not in request.body
    assert request.path == "/codex/responses"
    assert request.body["model"] == "gpt-5.3-codex"
    assert request.body["stream"] is True
    assert request.body["store"] is False


def test_codex_native_normalizes_prefixed_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv(
        "AI_SCIENTIST_AUTH_PROFILE_STORE_PATH", str(tmp_path / "auth_profiles.json")
    )
    monkeypatch.setenv(
        "AI_SCIENTIST_AUTH_SECRET_STORE_PATH", str(tmp_path / "auth_secrets.json")
    )
    os.environ["CODEX_NATIVE_BEARER_TOKEN"] = "codex-token"
    config = load_model_config()
    provider = config.get_provider("codex_native")
    request = build_chat_request(
        provider,
        tool_call={"name": "evaluate_p3", "arguments": {}},
        model="openai-codex/gpt-5.3-codex",
    )
    assert request.body["model"] == "gpt-5.3-codex"


def test_codex_native_uses_auth_profile_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv(
        "AI_SCIENTIST_AUTH_PROFILE_STORE_PATH", str(tmp_path / "auth_profiles.json")
    )
    monkeypatch.setenv(
        "AI_SCIENTIST_AUTH_SECRET_STORE_PATH", str(tmp_path / "auth_secrets.json")
    )
    monkeypatch.setenv("AI_SCIENTIST_AUTH_MANAGED_PROVIDERS", "codex_native")
    upsert_auth_profile(
        profile_id="codex-main",
        provider="codex_native",
        mode="api_key",
        account_label="main",
        api_key="profile-token",
    )
    config = load_model_config()
    provider = config.get_provider("codex_native")
    request = build_chat_request(
        provider, tool_call={"name": "evaluate_p3", "arguments": {}}
    )
    assert request.headers["Authorization"] == "Bearer profile-token"


def test_invoke_chat_completion_failsover_to_next_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv(
        "AI_SCIENTIST_AUTH_PROFILE_STORE_PATH", str(tmp_path / "auth_profiles.json")
    )
    monkeypatch.setenv(
        "AI_SCIENTIST_AUTH_SECRET_STORE_PATH", str(tmp_path / "auth_secrets.json")
    )
    monkeypatch.setenv("AI_SCIENTIST_AUTH_MANAGED_PROVIDERS", "codex_native")
    upsert_auth_profile(
        profile_id="codex-a",
        provider="codex_native",
        mode="api_key",
        account_label="a",
        priority=20,
        api_key="token-a",
    )
    upsert_auth_profile(
        profile_id="codex-b",
        provider="codex_native",
        mode="api_key",
        account_label="b",
        priority=10,
        api_key="token-b",
    )

    config = load_model_config()
    provider = config.get_provider("codex_native")
    tool_call = {"name": "make_boundary", "arguments": {}}
    seen_headers: list[str] = []

    class _FakeHTTPResponse:
        def __init__(self, payload: bytes, *, status: int = 200) -> None:
            self._payload = payload
            self.status = status

        def read(self) -> bytes:
            return self._payload

        def getcode(self) -> int:
            return self.status

        def __enter__(self) -> "_FakeHTTPResponse":
            return self

        def __exit__(self, *_: object) -> bool:
            return False

    def _fake_urlopen(request: Any, **_: object) -> _FakeHTTPResponse:
        authorization = request.headers.get("Authorization", "")
        seen_headers.append(authorization)
        if authorization == "Bearer token-a":
            headers = Message()
            raise HTTPError(
                url="http://127.0.0.1:18790/codex/responses",
                code=401,
                msg="Unauthorized",
                hdrs=headers,
                fp=BytesIO(b'{"error":"bad token"}'),
            )
        sse_payload = (
            "event: response.completed\n"
            'data: {"type":"response.completed","response":{"id":"resp-test","status":"completed","model":"gpt-5.3-codex","created_at":1700000000,"output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}]}}\n\n'
            "data: [DONE]\n"
        )
        return _FakeHTTPResponse(sse_payload.encode("utf-8"))

    monkeypatch.setattr("ai_scientist.model_provider.urlopen", _fake_urlopen)
    response = invoke_chat_completion(
        provider,
        tool_call,
        base_url_override="http://127.0.0.1:18790/v1",
        messages=[{"role": "user", "content": "ping"}],
    )
    assert response.status_code == 200
    content = (response.body.get("choices") or [{}])[0].get("message", {})
    assert content.get("content") == "ok"
    assert seen_headers == ["Bearer token-a", "Bearer token-b"]
    profiles = {profile.profile_id: profile for profile in list_auth_profiles()}
    assert profiles["codex-a"].consecutive_failures == 1
    assert profiles["codex-b"].consecutive_failures == 0


def test_invoke_chat_completion_does_not_cooldown_on_non_retryable_http_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv(
        "AI_SCIENTIST_AUTH_PROFILE_STORE_PATH", str(tmp_path / "auth_profiles.json")
    )
    monkeypatch.setenv(
        "AI_SCIENTIST_AUTH_SECRET_STORE_PATH", str(tmp_path / "auth_secrets.json")
    )
    monkeypatch.setenv("AI_SCIENTIST_AUTH_MANAGED_PROVIDERS", "codex_native")
    upsert_auth_profile(
        profile_id="codex-a",
        provider="codex_native",
        mode="api_key",
        account_label="a",
        api_key="token-a",
    )

    config = load_model_config()
    provider = config.get_provider("codex_native")

    def _fake_urlopen(*_: object, **__: object) -> object:
        headers = Message()
        raise HTTPError(
            url="http://127.0.0.1:18790/codex/responses",
            code=400,
            msg="Bad Request",
            hdrs=headers,
            fp=BytesIO(b'{"error":"invalid payload"}'),
        )

    monkeypatch.setattr("ai_scientist.model_provider.urlopen", _fake_urlopen)
    with pytest.raises(RuntimeError):
        invoke_chat_completion(
            provider,
            {"name": "make_boundary", "arguments": {}},
            base_url_override="http://127.0.0.1:18790/v1",
            messages=[{"role": "user", "content": "ping"}],
        )
    profile = {item.profile_id: item for item in list_auth_profiles()}["codex-a"]
    assert profile.consecutive_failures == 0
    assert profile.cooldown_until == 0.0


def test_invoke_chat_completion_accepts_non_stream_codex_response_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv(
        "AI_SCIENTIST_AUTH_PROFILE_STORE_PATH", str(tmp_path / "auth_profiles.json")
    )
    monkeypatch.setenv(
        "AI_SCIENTIST_AUTH_SECRET_STORE_PATH", str(tmp_path / "auth_secrets.json")
    )
    monkeypatch.setenv("CODEX_NATIVE_BEARER_TOKEN", "codex-token")
    config = load_model_config()
    provider = config.get_provider("codex_native")

    class _FakeHTTPResponse:
        def __init__(self, payload: dict[str, Any], *, status: int = 200) -> None:
            self._payload = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self.status = status

        def read(self) -> bytes:
            return self._payload

        def getcode(self) -> int:
            return self.status

        def __enter__(self) -> "_FakeHTTPResponse":
            return self

        def __exit__(self, *_: object) -> bool:
            return False

    payload = {
        "id": "resp-json",
        "status": "completed",
        "created_at": 1700000000,
        "model": "gpt-5.3-codex",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "ok-json"}],
            }
        ],
    }

    monkeypatch.setattr(
        "ai_scientist.model_provider.urlopen",
        lambda *_args, **_kwargs: _FakeHTTPResponse(payload),
    )

    response = invoke_chat_completion(
        provider,
        {"name": "make_boundary", "arguments": {}},
        base_url_override="https://chatgpt.com/backend-api",
        messages=[{"role": "user", "content": "ping"}],
    )

    assert response.status_code == 200
    choice = (response.body.get("choices") or [{}])[0]
    message = choice.get("message", {})
    assert message.get("content") == "ok-json"
    assert choice.get("finish_reason") == "stop"


def test_invoke_chat_completion_hits_mock_endpoint() -> None:
    from unittest.mock import patch

    config = load_model_config()
    provider = config.get_provider("openrouter")
    tool_call = {"name": "make_boundary", "arguments": {}}

    expected_body: dict[str, Any] = {
        "choices": [
            {
                "message": {
                    "content": {"type": "tool_call", "tool": tool_call["name"]},
                }
            }
        ]
    }

    class _FakeHTTPResponse:
        def __init__(self, payload: dict[str, Any], *, status: int = 200) -> None:
            self._payload = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self.status = status

        def read(self) -> bytes:
            return self._payload

        def getcode(self) -> int:
            return self.status

        def __enter__(self) -> "_FakeHTTPResponse":
            return self

        def __exit__(self, *_: object) -> bool:
            return False

    def _fake_urlopen(*_: object, **__: object) -> _FakeHTTPResponse:
        return _FakeHTTPResponse(expected_body, status=200)

    with patch("ai_scientist.model_provider.urlopen", new=_fake_urlopen):
        response = invoke_chat_completion(
            provider,
            tool_call,
            base_url_override="http://127.0.0.1:0",
            messages=[{"role": "user", "content": "ping"}],
        )

    assert response.status_code == 200
    content = (response.body.get("choices") or [{}])[0].get("message", {})
    payload = content.get("content", {})
    assert payload.get("tool") == "make_boundary"
