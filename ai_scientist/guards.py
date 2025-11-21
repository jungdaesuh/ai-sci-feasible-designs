"""Repository guardrails for the AI Scientist stack."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence


REQUIRED_PATHS: Sequence[Path] = (
    Path("AGENTS.md"),
    Path("docs/TASKS_CODEX_MINI.md"),
    Path("docs/MASTER_PLAN_AI_SCIENTIST.md"),
    Path("configs/model.yaml"),
    Path("ai_scientist/__init__.py"),
    Path("ai_scientist/runner.py"),
)


class GuardViolation(Exception):
    """Raised when a repository guardrail is violated."""


def verify() -> None:
    """Ensure the repository satisfies the declared guardrails."""

    missing = [str(path) for path in REQUIRED_PATHS if not path.exists()]
    if missing:
        raise GuardViolation(f"missing required paths: {', '.join(missing)}")

    tasks = Path("docs/TASKS_CODEX_MINI.md").read_text()
    if "Task 0.3" not in tasks:
        raise GuardViolation("docs/TASKS_CODEX_MINI.md must mention Task 0.3")

    master_plan = Path("docs/MASTER_PLAN_AI_SCIENTIST.md").read_text()
    if "Score-hacking guardrails" not in master_plan:
        raise GuardViolation(
            "MASTER_PLAN_AI_SCIENTIST.md must mention score guardrails"
        )
