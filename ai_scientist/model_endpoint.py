"""Minimal OpenAI-style K2 endpoint to satisfy Phase 1 model-serving (docs/MASTER_PLAN_AI_SCIENTIST.md:99-110, docs/TASKS_CODEX_MINI.md:247-368)."""

from __future__ import annotations

import json
import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler
from socketserver import TCPServer, ThreadingMixIn
from typing import Iterator
from urllib.parse import ParseResult, urlparse, urlunparse

from ai_scientist.config import ModelConfig, load_model_config

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EndpointMetadata:
    """Metadata that mirrors the Phase 1 serving config."""

    context_length: int
    dtype: str
    tensor_parallel: int


@dataclass(frozen=True)
class ModelEndpoint:
    """Information about the reachable K2 endpoint."""

    url: str
    metadata: EndpointMetadata
    provider_name: str
    chat_path: str


class _ThreadedServer(ThreadingMixIn, TCPServer):
    """TCP server that handles requests in threads and reuses its address."""

    allow_reuse_address = True


class _Handler(BaseHTTPRequestHandler):
    metadata: EndpointMetadata
    model_alias: str
    scheme: str
    chat_path: str

    def log_message(
        self, *_: object, **__: object
    ) -> None:  # pragma: no cover - suppress noise
        return

    def do_GET(self) -> None:
        if self.path not in {"/health", "/healthz"}:
            self.send_error(404)
            return
        payload = {
            "status": "ready",
            "model": self.model_alias,
            "metadata": self.metadata.__dict__,
        }
        self._respond(200, payload)

    def do_POST(self) -> None:
        if self.path != self.chat_path:
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b"{}"
        payload = json.loads(body.decode("utf-8"))
        response = {
            "id": "mock-k2",
            "object": "chat.completion",
            "model": self.model_alias,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": {
                            "type": "tool_call",
                            "tool": payload.get("tool_call", {}).get("name"),
                            "confirmation": "endpoint-ready",
                        },
                    },
                    "finish_reason": "function_call",
                    "function_call": payload.get("tool_call"),
                }
            ],
            "usage": {
                "context_length": self.metadata.context_length,
                "dtype": self.metadata.dtype,
                "tensor_parallel": self.metadata.tensor_parallel,
            },
        }
        self._respond(200, response)

    def _respond(self, status: int, payload: dict) -> None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _build_handler(
    metadata: EndpointMetadata, model_alias: str, scheme: str, chat_path: str
) -> type[_Handler]:
    handler = type(
        "_ModelHandler",
        (_Handler,),
        {
            "metadata": metadata,
            "model_alias": model_alias,
            "scheme": scheme,
            "chat_path": chat_path,
        },
    )
    return handler


def _normalize_base_url(base_url: str) -> ParseResult:
    parsed = urlparse(base_url)
    scheme = parsed.scheme or "http"
    netloc = parsed.netloc or parsed.path
    if not netloc:
        netloc = "127.0.0.1"
    return parsed._replace(
        scheme=scheme, netloc=netloc, path="", params="", query="", fragment=""
    )


@contextmanager
def run_model_endpoint(
    config: ModelConfig | None = None,
    provider_name: str | None = None,
) -> Iterator[ModelEndpoint]:
    """Start a lightweight K2 endpoint and yield its URL plus metadata."""

    resolved = config or load_model_config()
    provider = resolved.get_provider(provider_name)
    parsed = _normalize_base_url(resolved.base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 0
    metadata = EndpointMetadata(
        context_length=resolved.context_length,
        dtype=resolved.dtype,
        tensor_parallel=resolved.tensor_parallel,
    )
    handler = _build_handler(
        metadata, resolved.instruct_model, parsed.scheme, provider.chat_path
    )
    with _ThreadedServer((host, port), handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        live_netloc = f"{server.server_address[0]}:{server.server_address[1]}"
        endpoint_url = urlunparse((parsed.scheme, live_netloc, "", "", "", ""))
        _LOGGER.info(
            "Launched K2 endpoint %s (context=%d, dtype=%s, tp=%d)",
            endpoint_url,
            metadata.context_length,
            metadata.dtype,
            metadata.tensor_parallel,
        )
        try:
            yield ModelEndpoint(
                url=endpoint_url,
                metadata=metadata,
                provider_name=provider.name,
                chat_path=provider.chat_path,
            )
        finally:
            server.shutdown()
            thread.join()


__all__ = ["EndpointMetadata", "ModelEndpoint", "run_model_endpoint"]
