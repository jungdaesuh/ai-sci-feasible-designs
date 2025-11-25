"""OpenAI-style tool catalog for K2 models (Wave 8 checklist: docs/TASKS_CODEX_MINI.md:157-190)."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

ToolSchema = Mapping[str, Any]

_TOOL_DEFINITIONS: Sequence[ToolSchema] = (
    {
        "name": "make_boundary",
        "description": (
            "Build a SurfaceRZFourier boundary before submitting it to any evaluator. "
            "Input matches ai_scientist.tools.make_boundary_from_params and keeps r/z coefficients, symmetry, and nfp. "
            "Reference: docs/TASKS_CODEX_MINI.md:157-190."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "params": {
                    "type": "object",
                    "description": "Dictionary with r_cos/z_sin arrays, n_field_periods (aka nfp), and optional symmetry flag.",
                    "additionalProperties": True,
                }
            },
            "required": ["params"],
            "additionalProperties": False,
        },
    },
    {
        "name": "propose_boundary",
        "description": (
            "Perturb an existing boundary parameter set to generate a new candidate near the original. "
            "Useful for refining a promising seed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "params": {
                    "type": "object",
                    "description": "Base boundary parameters to perturb.",
                },
                "perturbation_scale": {
                    "type": "number",
                    "description": "Standard deviation of the Gaussian noise (default 0.05).",
                },
                "seed": {
                    "type": "integer",
                    "description": "Optional random seed for reproducibility.",
                },
            },
            "required": ["params"],
            "additionalProperties": False,
        },
    },
    {
        "name": "evaluate_p1",
        "description": (
            "Evaluate the low-fidelity P1 problem (minimize max elongation). "
            "Caller must provide the same params dict used in make_boundary."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "params": {"type": "object", "description": "Surface coefficients."},
                "problem": {
                    "type": "string",
                    "enum": ["p1"],
                    "description": "Explicitly declare the target problem.",
                },
                "stage": {
                    "type": "string",
                    "description": "Screening stage (default 'screen').",
                },
            },
            "required": ["params"],
            "additionalProperties": False,
        },
    },
    {
        "name": "evaluate_p2",
        "description": (
            "Run the high-fidelity QI-focused P2 evaluation via forward_model(ConstellarationSettings.default_high_fidelity())."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "params": {"type": "object", "description": "Boundary spec."},
                "problem": {
                    "type": "string",
                    "enum": ["p2"],
                    "description": "Explicit problem identifier.",
                },
                "stage": {"type": "string", "description": "Call tag (default 'p2')."},
            },
            "required": ["params"],
            "additionalProperties": False,
        },
    },
    {
        "name": "evaluate_p3",
        "description": (
            "Run the P3 metrics (aspect ratio + gradient) with high-fidelity settings and return hv-ready metrics."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "params": {"type": "object", "description": "Boundary parameters."},
                "problem": {
                    "type": "string",
                    "enum": ["p3"],
                    "description": "Explicit problem identifier.",
                },
                "stage": {"type": "string", "description": "Stage tag (default 'p3')."},
            },
            "required": ["params"],
            "additionalProperties": False,
        },
    },
    {
        "name": "retrieve_rag",
        "description": (
            "Fetch K relevant chunks from the ai_scientist/rag_index.db knowledge index."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query that guides retrieval.",
                },
                "k": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Number of snippets to return (default 3).",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "write_note",
        "description": (
            "Write a literature note or analysis insight to the world model and reports/notes/."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Markdown content of the note.",
                },
                "filename": {
                    "type": "string",
                    "description": "Optional filename (e.g. 'analysis_cycle_1.md').",
                },
                "experiment_id": {
                    "type": "integer",
                    "description": "Optional experiment id for world-model persistence.",
                },
                "cycle": {
                    "type": "integer",
                    "description": "Optional cycle number for world-model persistence.",
                },
            },
            "required": ["content", "experiment_id", "cycle"],
            "additionalProperties": False,
        },
    },
    {
        "name": "log_citation",
        "description": (
            "Register a doc citation for the current report draft (anchor optional, but include repo path)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "source_path": {
                    "type": "string",
                    "description": "File path, e.g. docs/MASTER_PLAN_AI_SCIENTIST.md",
                },
                "anchor": {
                    "type": "string",
                    "description": "Section or line anchor for the claim.",
                },
                "quote": {
                    "type": "string",
                    "description": "Textual quote or paraphrase.",
                },
            },
            "required": ["source_path", "quote"],
            "additionalProperties": False,
        },
    },
    {
        "name": "write_report",
        "description": (
            "Generate or extend a Markdown report with structured sections and references (docs/TASKS_CODEX_MINI.md:157-190)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Document title."},
                "sections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "heading": {"type": "string"},
                            "body": {"type": "string"},
                        },
                        "required": ["heading", "body"],
                        "additionalProperties": False,
                    },
                },
                "references": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["title", "sections"],
            "additionalProperties": False,
        },
    },
)

TOOL_SCHEMA_BY_NAME = {schema["name"]: schema for schema in _TOOL_DEFINITIONS}


def list_tool_schemas() -> tuple[ToolSchema, ...]:
    """Return all declared OpenAI-style tool definitions."""

    return tuple(_TOOL_DEFINITIONS)


def get_tool_schema(name: str) -> ToolSchema | None:
    """Lookup a schema by tool name."""

    return TOOL_SCHEMA_BY_NAME.get(name)
