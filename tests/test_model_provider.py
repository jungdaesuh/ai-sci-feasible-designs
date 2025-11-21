"""Smoke tests for the per-provider request builder."""

from __future__ import annotations

import os

import pytest

from ai_scientist import model_endpoint
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
    assert request.headers.get("X-Title") == "ConStellaration K2"


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
    config = load_model_config()
    provider = config.get_provider("openrouter")
    tool_call = {"name": "make_boundary", "arguments": {}}
    with model_endpoint.run_model_endpoint(
        config=config, provider_name=provider.name
    ) as server:
        response = invoke_chat_completion(
            provider,
            tool_call,
            base_url_override=server.url,
            messages=[{"role": "user", "content": "ping"}],
        )
    assert response.status_code == 200
    content = (response.body.get("choices") or [{}])[0].get("message", {})
    payload = content.get("content", {})
    assert payload.get("tool") == "make_boundary"
