from __future__ import annotations

import json
import socket
from urllib.request import Request, urlopen

import pytest

from ai_scientist.config import load_model_config
from ai_scientist.local_gateway import run_local_gateway
from ai_scientist.model_provider import ChatResponse


def test_local_gateway_forwards_chat_completion(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_invoke_chat_completion(*args, **kwargs) -> ChatResponse:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return ChatResponse(
            status_code=200,
            body={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    monkeypatch.setattr(
        "ai_scientist.local_gateway.invoke_chat_completion",
        _fake_invoke_chat_completion,
    )

    config = load_model_config()
    with run_local_gateway(
        config=config,
        provider_name="codex_native",
        host="127.0.0.1",
        port=0,
        upstream_base_url="https://api.openai.com/v1",
    ) as gateway:
        payload = {
            "model": "openai-codex/gpt-5.3-codex",
            "messages": [{"role": "user", "content": "ping"}],
        }
        request = Request(
            f"{gateway.url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-AI-Scientist-Tool-Name": "gateway_chat",
            },
            method="POST",
        )
        with urlopen(request, timeout=5.0) as response:
            body = json.loads(response.read().decode("utf-8"))

    assert body["id"] == "chatcmpl-test"
    forwarded_args_obj = captured["args"]
    assert isinstance(forwarded_args_obj, tuple)
    forwarded_tool_call = forwarded_args_obj[1]
    assert isinstance(forwarded_tool_call, dict)
    assert forwarded_tool_call["name"] == "gateway_chat"
    forwarded_kwargs = captured["kwargs"]
    assert isinstance(forwarded_kwargs, dict)
    assert forwarded_kwargs["base_url_override"] == "https://api.openai.com/v1"
    assert forwarded_kwargs["model"] == "openai-codex/gpt-5.3-codex"


def _find_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_local_gateway_rejects_alias_equivalent_self_origin() -> None:
    config = load_model_config()
    port = _find_open_port()
    with pytest.raises(ValueError):
        with run_local_gateway(
            config=config,
            provider_name="codex_native",
            host="127.0.0.1",
            port=port,
            upstream_base_url=f"http://localhost:{port}/v1",
        ):
            raise AssertionError("gateway should reject alias-equivalent self-origin")
