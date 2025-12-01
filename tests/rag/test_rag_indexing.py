from __future__ import annotations

from pathlib import Path

from ai_scientist import rag


def test_rag_index_covers_both_papers(tmp_path: Path) -> None:
    """Unified RAG index must surface content from both arXiv papers."""

    sources = (
        "docs/papers/2506.19583v1.md",
        "docs/papers/2511.02824v2.md",
        "docs/MASTER_PLAN_AI_SCIENTIST.md",
        "docs/AI_SCIENTIST_PRODUCTION_PLAN.md",
        "docs/AI_SCIENTIST_UPDATED_PLAN.md",
        "docs/AI_SCIENTIST_UNIFIED_ROADMAP.md",
    )
    index_db = tmp_path / "rag_index.db"
    summary = rag.build_index(sources, index_path=index_db)
    assert summary.chunks_indexed > 0

    hits_qi = rag.retrieve(
        "quasi isodynamic stellarator dataset optimization benchmark",
        k=3,
        index_path=index_db,
    )
    assert any("2506.19583v1.md" in hit["source"] for hit in hits_qi)

    hits_kosmos = rag.retrieve(
        "Kosmos AI scientist autonomous discovery cycles parallel data analysis",
        k=3,
        index_path=index_db,
    )
    assert any("2511.02824v2.md" in hit["source"] for hit in hits_kosmos)
