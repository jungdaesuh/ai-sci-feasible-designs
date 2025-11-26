"""Exercise the OpenAI tool schemas listed in ai_scientist/tools_api.py (Wave 8 checklist in docs/TASKS_CODEX_MINI.md:157-190)."""

from __future__ import annotations

import logging
from typing import Any, Mapping

from ai_scientist.tools_api import list_tool_schemas

_LOGGER = logging.getLogger(__name__)
SAMPLE_PARAMS: Mapping[str, Any] = {
    "r_cos": [[1.5, 0.0], [0.0, 0.05]],
    "z_sin": [[0.0, 0.0], [0.0, 0.05]],
    "n_field_periods": 1,
}


def _build_sample_payload(tool_name: str) -> Mapping[str, Any]:
    if tool_name == "make_boundary":
        return {"params": SAMPLE_PARAMS}
    if tool_name == "propose_boundary":
        return {"params": SAMPLE_PARAMS, "perturbation_scale": 0.05, "seed": 0}
    if tool_name == "recombine_designs":
        return {"parent_a": SAMPLE_PARAMS, "parent_b": SAMPLE_PARAMS, "alpha": 0.5}
    if tool_name in {"evaluate_p1", "evaluate_p2", "evaluate_p3"}:
        return {
            "params": SAMPLE_PARAMS,
            "problem": tool_name.replace("evaluate_", ""),
            "stage": "screen",
        }
    if tool_name == "log_citation":
        return {
            "source_path": "docs/MASTER_PLAN_AI_SCIENTIST.md",
            "anchor": "Phase 1",
            "quote": "Tiered K2 gate ensures deterministic tooling.",
        }
    if tool_name == "write_report":
        return {
            "title": "Smoke Report",
            "sections": [
                {"heading": "Summary", "body": "This report confirms tool schemas."},
                {
                    "heading": "Next Steps",
                    "body": "Log this output and proceed with Phase 9.",
                },
            ],
            "references": [
                "docs/TASKS_CODEX_MINI.md:157-190",
                "docs/MASTER_PLAN_AI_SCIENTIST.md:247-368",
            ],
        }
    if tool_name == "retrieve_rag":
        return {"query": "Phase 3 planning guidance", "k": 2}
    if tool_name == "write_note":
        return {
            "content": "Smoke note content for tool schema validation.",
            "experiment_id": 0,
            "cycle": 0,
        }
    raise ValueError(f"No smoke sample defined for tool '{tool_name}'")


def _validate_schema(schema: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
    parameters = schema.get("parameters", {})
    required = parameters.get("required", [])
    for field in required:
        if field not in payload:
            raise AssertionError(
                f"Payload for {schema['name']} is missing required field '{field}'"
            )


def run_smoke() -> None:
    """Ensure every tool schema exposes a minimal payload (Phase 1 smoke test)."""

    for schema in list_tool_schemas():
        name = schema["name"]
        sample = _build_sample_payload(name)
        _validate_schema(schema, sample)
        _LOGGER.info("Tool schema '%s' accepts sample payload %s", name, sample)


def smoke_entrypoint() -> None:
    logging.basicConfig(level=logging.INFO)
    run_smoke()


if __name__ == "__main__":
    smoke_entrypoint()
