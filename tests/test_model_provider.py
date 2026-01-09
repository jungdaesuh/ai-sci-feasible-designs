"""Smoke tests for the per-provider request builder."""

from __future__ import annotations

import json
import os
from typing import Any

import pytest

from ai_scientist.config import load_model_config
from ai_scientist.model_provider import build_chat_request, invoke_chat_completion


@pytest.fixture(autouse=True)
def _clear_env() -> None:
    for key in ("OPENROUTER_API_KEY", "MOONSHOT_API_KEY", "WQ_API_KEY"):
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
    assert request.body["messages"] == custom_messages
    assert request.body["tool_call"]["name"] == "evaluate_p1"


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
