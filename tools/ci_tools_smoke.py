"""Smoke driver that walks the Phase 1 gateway, config, and tool schemas.

Docs: docs/MASTER_PLAN_AI_SCIENTIST.md:99-110, docs/TASKS_CODEX_MINI.md:247-368.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from ai_scientist import agent, model_endpoint
from ai_scientist.config import ModelConfig, ProviderConfig, load_model_config
from ai_scientist.model_provider import ChatResponse, invoke_chat_completion
from ai_scientist.tools_api_smoke import run_smoke

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_LOGGER = logging.getLogger(__name__)


def _exercise_gates(
    config: ModelConfig,
    provider: ProviderConfig,
    *,
    base_url_override: str | None,
) -> None:
    for role in ("short_loop", "planning", "report"):
        gate = agent.provision_model_tier(role=role, config=config)
        tool_call = {
            "name": "make_boundary",
            "arguments": {
                "params": {
                    "n_field_periods": 1,
                    "r_cos": [[1.5]],
                    "z_sin": [[0.0]],
                }
            },
        }
        response: ChatResponse = invoke_chat_completion(
            provider,
            tool_call,
            model=gate.provider_model,
            messages=[
                {
                    "role": "user",
                    "content": f"echo tool call from gate {gate.model_alias}",
                }
            ],
            base_url_override=base_url_override,
            timeout=float(config.request_timeout_seconds),
        )
        choice = (response.body.get("choices") or [{}])[0]
        finish_reason = choice.get("finish_reason")
        _LOGGER.info(
            "provider=%s gate=%s status=%s finish=%s",
            provider.name,
            gate.model_alias,
            response.status_code,
            finish_reason,
        )
        print(
            f"[ci_tools_smoke] provisioned {gate.model_alias} for role {role}"
            f" (status {response.status_code}, finish={finish_reason})"
        )


def _should_use_remote_provider() -> bool:
    raw = os.getenv("AI_SCIENTIST_REMOTE_PROVIDER", "")
    return raw.lower() in {"1", "true", "yes", "on"}


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    config = load_model_config()
    provider_name = os.getenv("MODEL_PROVIDER", config.default_provider)
    provider = config.get_provider(provider_name)

    manual_endpoint = os.getenv("AI_SCIENTIST_ENDPOINT_URL")
    if manual_endpoint:
        endpoint = manual_endpoint.rstrip("/")
        print(
            f"[ci_tools_smoke] using manual endpoint override at {endpoint} for {provider.name}"
        )
        _exercise_gates(config, provider, base_url_override=endpoint)
        run_smoke()
        return

    if _should_use_remote_provider():
        print(
            f"[ci_tools_smoke] calling provider {provider.name} at {provider.base_url}"
        )
        _exercise_gates(config, provider, base_url_override=None)
        run_smoke()
        return

    with model_endpoint.run_model_endpoint(
        config=config, provider_name=provider.name
    ) as server:
        print(f"[ci_tools_smoke] K2 mock endpoint ready at {server.url}")
        print(
            f"[ci_tools_smoke] provider={provider.name} path={provider.chat_path} metadata={server.metadata}"
        )
        _exercise_gates(config, provider, base_url_override=server.url)
        run_smoke()


if __name__ == "__main__":
    main()
