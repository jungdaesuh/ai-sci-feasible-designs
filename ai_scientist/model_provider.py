"""Helpers for building provider chat calls with codex-native compatibility."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ai_scientist.auth_profile import (
    record_profile_failure,
    record_profile_success,
    resolve_auth_candidates,
    resolve_runtime_auth_header,
)
from ai_scientist.config import ProviderConfig

_LOGGER = logging.getLogger(__name__)
_DEFAULT_TIMEOUT_SECONDS = 60.0
_DEFAULT_CODEX_INSTRUCTIONS = "You are a helpful assistant."


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
    return resolve_runtime_auth_header(provider.name, provider.auth_env)


def _is_retryable_profile_status(status_code: int) -> bool:
    return status_code in {401, 403, 407, 429} or status_code >= 500


def _normalize_message(role: Any, content: Any) -> dict[str, str]:
    resolved_role = str(role) if role not in (None, "") else "user"
    resolved_content = str(content) if content not in (None, "") else ""
    return {"role": resolved_role, "content": resolved_content}


def _normalize_codex_model(model: str) -> str:
    normalized = model.strip()
    if "/" not in normalized:
        return normalized
    return normalized.split("/")[-1]


def _build_codex_payload(
    *,
    messages: Sequence[Mapping[str, str]] | None,
    model: str | None,
    default_model: str | None,
) -> Mapping[str, Any]:
    selected_model = _normalize_codex_model(str(model or default_model or ""))
    if not selected_model:
        raise ValueError("codex_native requires a model name")

    instruction_parts: list[str] = []
    input_messages: list[dict[str, str]] = []
    for message in messages or []:
        normalized = _normalize_message(message.get("role"), message.get("content"))
        if normalized["role"] == "system":
            if normalized["content"]:
                instruction_parts.append(normalized["content"])
            continue
        input_messages.append(normalized)
    if not input_messages:
        input_messages.append({"role": "user", "content": "tool request"})

    instructions = (
        "\n\n".join(part for part in instruction_parts if part).strip()
        or _DEFAULT_CODEX_INSTRUCTIONS
    )
    return {
        "model": selected_model,
        "instructions": instructions,
        "input": input_messages,
        "store": False,
        "stream": True,
    }


def _extract_codex_text(completed_response: Mapping[str, Any]) -> str:
    output = completed_response.get("output")
    if not isinstance(output, list):
        return ""
    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        text_parts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if str(part.get("type", "")) != "output_text":
                continue
            text = part.get("text")
            if text not in (None, ""):
                text_parts.append(str(text))
        if text_parts:
            return "".join(text_parts)
    return ""


def _parse_json_object(raw: str) -> Mapping[str, Any] | None:
    if not raw.startswith("{"):
        return None
    try:
        candidate = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(candidate, dict):
        return None
    return candidate


def _codex_response_to_chat(
    completed_response: Mapping[str, Any],
    *,
    model_name: str,
    fallback_text: str = "",
) -> Mapping[str, Any]:
    assistant_text = _extract_codex_text(completed_response) or fallback_text
    status = str(completed_response.get("status", ""))
    finish_reason = "stop" if status == "completed" else "length"
    created_at = completed_response.get("created_at")
    created = (
        int(created_at) if isinstance(created_at, (int, float)) else int(time.time())
    )
    response_id = str(completed_response.get("id", "codex-response"))
    response_model = str(completed_response.get("model", model_name)) or model_name
    chat_body: dict[str, Any] = {
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": response_model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": assistant_text},
                "finish_reason": finish_reason,
            }
        ],
    }
    usage = completed_response.get("usage")
    if isinstance(usage, dict):
        chat_body["usage"] = usage
    return chat_body


def _direct_codex_payload_to_chat(
    payload: Mapping[str, Any] | None,
    *,
    model_name: str,
) -> Mapping[str, Any] | None:
    if payload is None:
        return None
    if "choices" in payload:
        return payload
    response_payload = payload.get("response")
    if isinstance(response_payload, dict):
        return _codex_response_to_chat(response_payload, model_name=model_name)
    if "output" in payload or "status" in payload:
        return _codex_response_to_chat(payload, model_name=model_name)
    return None


def _parse_codex_stream_as_chat(
    body_text: str,
    *,
    model_name: str,
) -> Mapping[str, Any]:
    direct_chat = _direct_codex_payload_to_chat(
        _parse_json_object(body_text.strip()),
        model_name=model_name,
    )
    if direct_chat is not None:
        return direct_chat

    completed_response: Mapping[str, Any] | None = None
    delta_parts: list[str] = []
    for line in body_text.splitlines():
        normalized = line.strip()
        if not normalized.startswith("data:"):
            continue
        payload_text = normalized.split(":", 1)[1].strip()
        if not payload_text or payload_text == "[DONE]":
            continue
        try:
            event = json.loads(payload_text)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        event_type = str(event.get("type", ""))
        if event_type == "response.output_text.delta":
            delta = event.get("delta")
            if delta not in (None, ""):
                delta_parts.append(str(delta))
            continue
        if event_type == "response.output_text.done":
            full_text = event.get("text")
            if full_text not in (None, ""):
                delta_parts = [str(full_text)]
            continue
        if event_type == "response.completed":
            response_payload = event.get("response")
            if isinstance(response_payload, dict):
                completed_response = response_payload

    if completed_response is None:
        raise RuntimeError("Codex response stream missing completion event")

    return _codex_response_to_chat(
        completed_response,
        model_name=model_name,
        fallback_text="".join(delta_parts),
    )


def build_chat_request(
    provider: ProviderConfig,
    tool_call: Mapping[str, Any],
    *,
    messages: Sequence[Mapping[str, str]] | None = None,
    model: str | None = None,
    authorization_header: str | None = None,
) -> ChatRequest:
    """Return the per-provider path, headers, and body for a chat completion call."""

    headers: dict[str, str] = {
        "Authorization": authorization_header or _resolve_auth_header(provider),
        "Content-Type": "application/json",
        "X-AI-Scientist-Tool-Name": str(tool_call.get("name", "")),
    }
    for key, value in provider.extra_headers:
        headers[key] = value
    if provider.name == "codex_native":
        payload = _build_codex_payload(
            messages=messages,
            model=model,
            default_model=provider.default_model,
        )
    else:
        payload = {
            "model": model or provider.default_model,
            "messages": list(messages)
            if messages
            else [{"role": "user", "content": "tool request"}],
        }
        payload["tool_call"] = tool_call
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

    base_url = (base_url_override or provider.base_url or "").rstrip("/")
    if not base_url:
        raise ValueError(f"Provider '{provider.name}' is missing a base_url")
    request_timeout = timeout or _DEFAULT_TIMEOUT_SECONDS
    auth_candidates = resolve_auth_candidates(provider.name, provider.auth_env)
    for index, auth_candidate in enumerate(auth_candidates):
        chat_request = build_chat_request(
            provider,
            tool_call,
            messages=messages,
            model=model,
            authorization_header=auth_candidate.authorization_header,
        )
        url = f"{base_url}{chat_request.path}"
        data = json.dumps(chat_request.body, separators=(",", ":")).encode("utf-8")
        request = Request(
            url, data=data, headers=dict(chat_request.headers), method="POST"
        )
        try:
            with urlopen(request, timeout=request_timeout) as response:
                payload = response.read()
                status = getattr(response, "status", response.getcode())
        except HTTPError as exc:  # pragma: no cover - exercised with live providers
            profile_id = auth_candidate.profile_id
            retryable_profile_failure = (
                profile_id is not None and _is_retryable_profile_status(exc.code)
            )
            if retryable_profile_failure and profile_id is not None:
                record_profile_failure(profile_id)
            if retryable_profile_failure and index < len(auth_candidates) - 1:
                _LOGGER.warning(
                    "provider=%s profile=%s status=%s failover=next",
                    provider.name,
                    profile_id,
                    exc.code,
                )
                continue
            error_body = exc.read().decode("utf-8", "replace") if exc.fp else ""
            message = (
                f"Provider '{provider.name}' returned HTTP {exc.code} for {url}: "
                f"{error_body}"
            )
            raise RuntimeError(message) from exc
        except URLError as exc:  # pragma: no cover - exercised with live providers
            if auth_candidate.profile_id:
                record_profile_failure(auth_candidate.profile_id)
            if auth_candidate.profile_id and index < len(auth_candidates) - 1:
                _LOGGER.warning(
                    "provider=%s profile=%s transport_error=%s failover=next",
                    provider.name,
                    auth_candidate.profile_id,
                    exc,
                )
                continue
            raise RuntimeError(
                f"Failed to reach provider '{provider.name}' at {url}: {exc}"
            ) from exc

        body_text = payload.decode("utf-8") if payload else "{}"
        if provider.name == "codex_native":
            resolved_model = _normalize_codex_model(
                str(
                    chat_request.body.get(
                        "model", model or provider.default_model or ""
                    )
                )
            )
            parsed = _parse_codex_stream_as_chat(body_text, model_name=resolved_model)
        else:
            try:
                parsed = json.loads(body_text) if body_text.strip() else {}
            except (
                json.JSONDecodeError
            ) as exc:  # pragma: no cover - malformed upstream payload
                raise RuntimeError(
                    f"Provider '{provider.name}' returned invalid JSON: {body_text}"
                ) from exc
        if auth_candidate.profile_id:
            record_profile_success(auth_candidate.profile_id)
        _LOGGER.info(
            "provider=%s auth_source=%s status=%s finish_reason=%s",
            provider.name,
            auth_candidate.source,
            status,
            parsed.get("choices", [{}])[0].get("finish_reason"),
        )
        return ChatResponse(status_code=status, body=parsed)
    raise RuntimeError(f"Provider '{provider.name}' has no auth candidates")


__all__ = [
    "ChatRequest",
    "ChatResponse",
    "build_chat_request",
    "invoke_chat_completion",
]
