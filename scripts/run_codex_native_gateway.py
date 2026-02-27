#!/usr/bin/env python3
"""Run a local OpenAI-compatible gateway for codex-native provider calls."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run local /v1/chat/completions bridge for codex-native."
    )
    parser.add_argument(
        "--provider",
        default="codex_native",
        help="Model provider alias from configs/model*.yaml",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Gateway bind host")
    parser.add_argument("--port", type=int, default=18790, help="Gateway bind port")
    parser.add_argument(
        "--upstream-base-url",
        required=True,
        help="Upstream Codex Responses base URL (for example https://chatgpt.com/backend-api).",
    )
    parser.add_argument(
        "--model-config-path",
        default=None,
        help="Optional model config override path (same as AI_SCIENTIST_MODEL_CONFIG_PATH).",
    )
    return parser


def _resolve_model_config_path(path: str | None) -> Path | None:
    if path:
        return Path(path)
    return None


def main() -> None:
    from ai_scientist.config import load_model_config
    from ai_scientist.local_gateway import run_local_gateway

    args = _parser().parse_args()
    config = load_model_config(_resolve_model_config_path(args.model_config_path))
    with run_local_gateway(
        config=config,
        provider_name=args.provider,
        host=args.host,
        port=args.port,
        upstream_base_url=args.upstream_base_url,
    ) as gateway:
        print(
            "[codex-native-gateway] running at"
            f" {gateway.url}/v1/chat/completions -> {gateway.upstream_base_url}"
        )
        print("[codex-native-gateway] press Ctrl+C to stop")
        while True:
            time.sleep(3600)


if __name__ == "__main__":
    main()
