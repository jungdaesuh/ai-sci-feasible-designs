"""Helpers for building OpenAI-style calls that match OpenRouter, Moonshot, and StreamLake APIs."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ai_scientist.config import ProviderConfig

_LOGGER = logging.getLogger(__name__)
_DEFAULT_TIMEOUT_SECONDS = 60.0


@dataclass(frozen=True)
class ChatRequest:
    path: str
    headers: Mapping[str, str]
    body: Mapping[str, Any]


@dataclass(frozen=True)
class ChatResponse:
    status_code: int
    body: Mapping[str, Any]


def _resolve_auth_header(provider: ProviderConfig) -> str:
    token = os.getenv(provider.auth_env)
    if not token:
        placeholder = provider.auth_env or provider.name.upper()
        token = f"LOCAL-{placeholder}"
    return f"Bearer {token}"


def build_chat_request(
    provider: ProviderConfig,
    tool_call: Mapping[str, Any],
    *,
    messages: Sequence[Mapping[str, str]] | None = None,
    model: str | None = None,
) -> ChatRequest:
    """Return the per-provider path, headers, and body for a chat completion call."""

    headers: dict[str, str] = {
        "Authorization": _resolve_auth_header(provider),
        "Content-Type": "application/json",
    }
    for key, value in provider.extra_headers:
        headers[key] = value
    payload = {
        "model": model or provider.default_model,
        "messages": list(messages)
        if messages
        else [{"role": "user", "content": "tool request"}],
        "tool_call": tool_call,
    }
    _LOGGER.info(
        "Built chat request provider=%s path=%s model=%s tool=%s",
        provider.name,
        provider.chat_path,
        payload["model"],
        tool_call.get("name"),
    )
    return ChatRequest(path=provider.chat_path, headers=headers, body=payload)


def invoke_chat_completion(
    provider: ProviderConfig,
    tool_call: Mapping[str, Any],
    *,
    messages: Sequence[Mapping[str, str]] | None = None,
    model: str | None = None,
    base_url_override: str | None = None,
    timeout: float | None = None,
) -> ChatResponse:
    """Send a chat completion request to the configured provider and return the decoded response."""

    chat_request = build_chat_request(provider, tool_call, messages=messages, model=model)
    base_url = (base_url_override or provider.base_url or "").rstrip("/")
    if not base_url:
        raise ValueError(f"Provider '{provider.name}' is missing a base_url")
    url = f"{base_url}{chat_request.path}"
    data = json.dumps(chat_request.body, separators=(",", ":")).encode("utf-8")
    request = Request(url, data=data, headers=dict(chat_request.headers), method="POST")
    request_timeout = timeout or _DEFAULT_TIMEOUT_SECONDS
    try:
        with urlopen(request, timeout=request_timeout) as response:
            payload = response.read()
            status = getattr(response, "status", response.getcode())
    except HTTPError as exc:  # pragma: no cover - exercised with live providers
        error_body = exc.read().decode("utf-8", "replace") if exc.fp else ""
        message = (
            f"Provider '{provider.name}' returned HTTP {exc.code} for {url}: {error_body}"
        )
        raise RuntimeError(message) from exc
    except URLError as exc:  # pragma: no cover - exercised with live providers
        raise RuntimeError(f"Failed to reach provider '{provider.name}' at {url}: {exc}") from exc

    body_text = payload.decode("utf-8") if payload else "{}"
    try:
        parsed = json.loads(body_text) if body_text.strip() else {}
    except json.JSONDecodeError as exc:  # pragma: no cover - malformed upstream payload
        raise RuntimeError(f"Provider '{provider.name}' returned invalid JSON: {body_text}") from exc
    _LOGGER.info(
        "provider=%s status=%s finish_reason=%s",
        provider.name,
        status,
        parsed.get("choices", [{}])[0].get("finish_reason"),
    )
    return ChatResponse(status_code=status, body=parsed)


__all__ = ["ChatRequest", "ChatResponse", "build_chat_request", "invoke_chat_completion"]
