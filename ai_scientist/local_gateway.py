"""Local OpenAI-compatible gateway bridge for chat completions."""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler
from socketserver import TCPServer, ThreadingMixIn
from typing import Any, Iterator, Mapping
from urllib.parse import urlparse, urlunparse

from ai_scientist.config import ModelConfig, ProviderConfig, load_model_config
from ai_scientist.model_provider import invoke_chat_completion


@dataclass(frozen=True)
class GatewayEndpoint:
    url: str
    provider_name: str
    upstream_base_url: str


class _ThreadedServer(ThreadingMixIn, TCPServer):
    allow_reuse_address = True


class _Handler(BaseHTTPRequestHandler):
    provider: ProviderConfig
    upstream_base_url: str
    timeout_seconds: float

    def log_message(
        self, *_: object, **__: object
    ) -> None:  # pragma: no cover - suppress request log noise
        return

    def do_GET(self) -> None:
        if self.path not in {"/health", "/healthz"}:
            self.send_error(404)
            return
        self._respond(200, {"status": "ok", "provider": self.provider.name})

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            self._respond(400, {"error": "invalid JSON body"})
            return
        if not isinstance(payload, dict):
            self._respond(400, {"error": "JSON body must be an object"})
            return

        messages = payload.get("messages", [])
        if not isinstance(messages, list):
            self._respond(400, {"error": "messages must be a list"})
            return
        normalized_messages = _normalize_messages(messages)
        model_name = payload.get("model")
        model = str(model_name) if model_name not in (None, "") else None

        tool_payload = payload.get("tool_call")
        if isinstance(tool_payload, dict):
            tool_call: Mapping[str, Any] = tool_payload
        else:
            tool_name = self.headers.get("X-AI-Scientist-Tool-Name", "gateway_chat")
            tool_call = {"name": tool_name, "arguments": {}}
        try:
            response = invoke_chat_completion(
                self.provider,
                tool_call,
                messages=normalized_messages,
                model=model,
                base_url_override=self.upstream_base_url,
                timeout=self.timeout_seconds,
            )
        except RuntimeError as exc:
            self._respond(502, {"error": str(exc)})
            return
        self._respond(response.status_code, dict(response.body))

    def _respond(self, status: int, payload: Mapping[str, Any]) -> None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _normalize_messages(
    payload_messages: list[Any],
) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for message in payload_messages:
        if not isinstance(message, dict):
            continue
        role_raw = message.get("role")
        content_raw = message.get("content")
        role = str(role_raw) if role_raw not in (None, "") else "user"
        content = str(content_raw) if content_raw not in (None, "") else ""
        normalized.append({"role": role, "content": content})
    if normalized:
        return normalized
    return [{"role": "user", "content": "tool request"}]


def _build_handler(
    provider: ProviderConfig,
    upstream_base_url: str,
    timeout_seconds: float,
) -> type[_Handler]:
    return type(
        "_GatewayHandler",
        (_Handler,),
        {
            "provider": provider,
            "upstream_base_url": upstream_base_url,
            "timeout_seconds": timeout_seconds,
        },
    )


def _origin(value: str) -> tuple[str, str, int]:
    parsed = urlparse(value)
    scheme = parsed.scheme or "http"
    host = _canonical_host(parsed.hostname or "127.0.0.1")
    port = parsed.port
    if port is None:
        port = 443 if scheme == "https" else 80
    return (scheme, host, port)


def _canonical_host(host: str) -> str:
    lowered = host.strip().lower()
    if lowered in {"127.0.0.1", "localhost", "::1", "0.0.0.0"}:
        return "loopback"
    return lowered


@contextmanager
def run_local_gateway(
    *,
    config: ModelConfig | None = None,
    provider_name: str = "codex_native",
    host: str = "127.0.0.1",
    port: int = 18790,
    upstream_base_url: str,
    timeout_seconds: float | None = None,
) -> Iterator[GatewayEndpoint]:
    resolved = config or load_model_config()
    provider = resolved.get_provider(provider_name)
    handler = _build_handler(
        provider,
        upstream_base_url.rstrip("/"),
        float(timeout_seconds or resolved.request_timeout_seconds),
    )
    with _ThreadedServer((host, port), handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        local_netloc = f"{server.server_address[0]}:{server.server_address[1]}"
        local_url = urlunparse(("http", local_netloc, "", "", "", ""))
        if _origin(local_url) == _origin(upstream_base_url):
            server.shutdown()
            thread.join()
            raise ValueError(
                "gateway upstream cannot point to the same local gateway origin"
            )
        try:
            yield GatewayEndpoint(
                url=local_url,
                provider_name=provider.name,
                upstream_base_url=upstream_base_url.rstrip("/"),
            )
        finally:
            server.shutdown()
            thread.join()


__all__ = ["GatewayEndpoint", "run_local_gateway"]
