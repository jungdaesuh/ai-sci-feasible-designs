import sqlite3
import textwrap

from ai_scientist import rag


def test_build_index_and_retrieve(tmp_path):
    source = tmp_path / "ref.md"
    source.write_text(
        textwrap.dedent(
            """
            # Anchor One

            The quick brown fox jumps over lazy dogs.

            # Anchor Two

            Another paragraph that mentions jumping.
            """
        ).strip()
    )
    index_db = tmp_path / "rag.db"
    first_idx = rag.build_index([str(source)], index_path=index_db)
    assert first_idx.chunks_indexed >= 2

    with sqlite3.connect(index_db) as conn:
        row = conn.execute(
            "SELECT chunk_count FROM rag_index_meta WHERE source = ?", (str(source),)
        ).fetchone()
    assert row and row[0] == first_idx.chunks_indexed

    cached = rag.ensure_index(index_path=index_db, sources=[str(source)])
    assert cached.chunks_indexed == first_idx.chunks_indexed

    hits = rag.retrieve("jumping over lazy", k=1, index_path=index_db)
    assert hits, "Expected at least one retrieval"
    assert hits[0]["anchor"] == "Anchor One"
    assert hits[0]["chunk"].startswith("The quick brown fox")
    assert hits[0]["start_line"].isdigit()

    source.write_text(source.read_text() + "\n\nNew chunk to trigger rebuild.")
    updated = rag.ensure_index(index_path=index_db, sources=[str(source)])
    assert updated.chunks_indexed > first_idx.chunks_indexed
    metadata_row = rag.ensure_index(index_path=index_db, sources=[str(source)])
    assert metadata_row.chunks_indexed == updated.chunks_indexed


def test_retrieve_without_index(tmp_path):
    missing = tmp_path / "missing.db"
    assert rag.retrieve("anything", index_path=missing) == []
