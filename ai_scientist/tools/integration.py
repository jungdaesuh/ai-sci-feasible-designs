"""Integration tools for RAG and Memory."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from ai_scientist import memory, rag


def retrieve_rag(
    query: str, *, k: int = 3, index_path: Path | str | None = None
) -> list[dict[str, str]]:
    """Expose RAG retrieval via the ai_scientist/rag_index.db index (Phase 3)."""

    index = Path(index_path) if index_path is not None else rag.DEFAULT_INDEX_PATH
    return rag.retrieve(query=query, k=k, index_path=index)


def write_note(
    content: str,
    *,
    filename: str | None = None,
    out_dir: Path | str | None = None,
    world_model: Any | None = None,
    experiment_id: int,
    cycle: int,
    memory_db: str | Path | None = None,
) -> str:
    """Write a literature note to disk and, if context is provided, persist it in the world model."""

    target_dir = Path(out_dir) if out_dir else Path("reports/notes")
    target_dir.mkdir(parents=True, exist_ok=True)

    if not filename:
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:8]
        filename = f"note_{digest}.md"

    path = target_dir / filename
    path.write_text(content, encoding="utf-8")

    target_wm = world_model
    owned = False
    if target_wm is None and memory_db is not None:
        target_wm = memory.WorldModel(memory_db)
        owned = True
    if target_wm is None:
        raise ValueError(
            "write_note requires world_model or memory_db for persistence."
        )
    try:
        target_wm.log_note(experiment_id=experiment_id, cycle=cycle, content=content)
    except Exception as exc:  # pragma: no cover - safety for agent path
        print(f"[write_note] failed to log note to world_model: {exc}")
    finally:
        if owned:
            target_wm.close()

    return f"Note saved to {path}: {content[:50]}..."
