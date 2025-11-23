"""Local retrieval helpers for AI Scientist Wave 3 (RAG)."""

from __future__ import annotations

import os
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, List, Sequence

DEFAULT_INDEX_PATH = Path("ai_scientist/rag_index.db")
DEFAULT_INDEX_SOURCES = (
    "docs/papers/2506.19583v1.md",
    "docs/papers/2511.02824v2.md",
    "ConStellaration Fusion Challenge_ Benchmarks and Solution Strategies.md",
    "docs/MASTER_PLAN_AI_SCIENTIST.md",
    "docs/AI_SCIENTIST_PRODUCTION_PLAN.md",
    "docs/AI_SCIENTIST_UPDATED_PLAN.md",
    "docs/AI_SCIENTIST_UNIFIED_ROADMAP.md",
)
INDEX_TABLE_NAME = "rag_references"
META_TABLE_NAME = "rag_index_meta"


@dataclass(frozen=True)
class DocumentChunk:
    source: str
    anchor: str
    chunk: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class IndexSummary:
    index_path: str
    chunks_indexed: int


@dataclass(frozen=True)
class SourceMeta:
    source: str
    mtime: float
    chunk_count: int


def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"\b\w+\b", text.lower())
    return [token for token in tokens if len(token) > 1]


def _file_mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _write_metadata(
    conn: sqlite3.Connection,
    sources: Sequence[str],
    chunk_counts: dict[str, int],
) -> None:
    conn.execute(f"DELETE FROM {META_TABLE_NAME}")
    report_lines = []
    for source in sources:
        mtime = _file_mtime(Path(source)) or 0.0
        conn.execute(
            f"INSERT INTO {META_TABLE_NAME} (source, mtime, chunk_count) VALUES (?, ?, ?)",
            (source, mtime, chunk_counts.get(source, 0)),
        )
        report_lines.append(
            f"{source}: {chunk_counts.get(source, 0)} chunks (mtime={mtime:.0f})"
        )
    print(
        "[rag] indexed sources:\n" + "\n".join(f"  - {line}" for line in report_lines)
    )


def _ensure_index(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {INDEX_TABLE_NAME} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            anchor TEXT,
            chunk TEXT NOT NULL,
            start_line INTEGER NOT NULL,
            end_line INTEGER NOT NULL,
            tokens TEXT NOT NULL
        )
        """
    )
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {META_TABLE_NAME} (
            source TEXT PRIMARY KEY,
            mtime REAL NOT NULL,
            chunk_count INTEGER NOT NULL
        )
        """
    )


def _iter_document_chunks(source: str, lines: Sequence[str]) -> Iterable[DocumentChunk]:
    anchor = ""
    chunk: List[str] = []
    chunk_start = 1

    def flush(end_line: int) -> DocumentChunk | None:
        if not chunk:
            return None
        text = " ".join(line.strip() for line in chunk if line.strip())
        if not text:
            return None
        return DocumentChunk(
            source=source,
            anchor=anchor,
            chunk=text,
            start_line=chunk_start,
            end_line=end_line,
        )

    for line_number, raw in enumerate(lines, start=1):
        stripped = raw.strip()
        if stripped.startswith("#"):
            anchor = stripped.lstrip("#").strip()
            if chunk:
                flushed = flush(line_number - 1)
                if flushed:
                    chunk.clear()
                    chunk_start = line_number
                    yield flushed
        if not chunk:
            chunk_start = line_number
        chunk.append(raw)
        if stripped == "":
            flushed = flush(line_number)
            if flushed:
                chunk.clear()
                chunk_start = line_number + 1
                yield flushed
    flushed = flush(len(lines))
    if flushed:
        yield flushed


def ensure_index(
    sources: Sequence[str] | None = None,
    index_path: Path | str | None = None,
    *,
    force_rebuild: bool = False,
) -> IndexSummary:
    """Ensure an index exists and reuse it unless a rebuild is requested."""

    index_path = Path(index_path) if index_path is not None else DEFAULT_INDEX_PATH
    sources = tuple(sources or DEFAULT_INDEX_SOURCES)
    env_force = os.environ.get("AI_SCIENTIST_RAG_FORCE_REBUILD", "").lower() in {
        "1",
        "true",
        "yes",
    }
    should_rebuild = force_rebuild or env_force or not index_path.exists()

    if not should_rebuild and index_path.exists():
        conn = sqlite3.connect(index_path)
        conn.row_factory = sqlite3.Row
        try:
            _ensure_index(conn)
            existing_meta = {
                row["source"]: SourceMeta(
                    source=row["source"],
                    mtime=row["mtime"],
                    chunk_count=int(row["chunk_count"]),
                )
                for row in conn.execute(
                    f"SELECT source, mtime, chunk_count FROM {META_TABLE_NAME}"
                )
            }
            needs_rebuild = False
            current_total = 0
            for source in sources:
                path = Path(source)
                meta = existing_meta.get(source)
                if path.exists():
                    current_mtime = _file_mtime(path)
                    if (
                        meta is None
                        or current_mtime is None
                        or meta.mtime != current_mtime
                    ):
                        needs_rebuild = True
                        break
                    current_total += meta.chunk_count
                else:
                    if meta is not None and meta.chunk_count > 0:
                        needs_rebuild = True
                        break
            if not needs_rebuild and current_total > 0:
                print(f"[rag] reusing index {index_path} ({current_total} chunks)")
                return IndexSummary(str(index_path), current_total)
            print("[rag] source changes detected; rebuilding index")
            should_rebuild = True
        finally:
            conn.close()
    if should_rebuild:
        return build_index(sources, index_path)
    return IndexSummary(str(index_path), 0)


def build_index(
    sources: Sequence[str],
    index_path: Path | str | None = None,
) -> IndexSummary:
    """Persist a simple sqlite-backed index derived from Markdown sources."""

    index_path = Path(index_path) if index_path is not None else DEFAULT_INDEX_PATH
    index_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(index_path)
    conn.row_factory = sqlite3.Row
    try:
        _ensure_index(conn)
        conn.execute(f"DELETE FROM {INDEX_TABLE_NAME}")
        total_chunks = 0
        chunk_counts: dict[str, int] = defaultdict(int)
        for source in sources:
            path = Path(source)
            if not path.exists():
                continue
            lines = path.read_text(encoding="utf-8").splitlines()
            for chunk in _iter_document_chunks(source, lines):
                tokens = " ".join(_tokenize(chunk.chunk))
                conn.execute(
                    f"INSERT INTO {INDEX_TABLE_NAME} (source, anchor, chunk, start_line, end_line, tokens) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        chunk.source,
                        chunk.anchor,
                        chunk.chunk,
                        chunk.start_line,
                        chunk.end_line,
                        tokens,
                    ),
                )
                chunk_counts[source] += 1
                total_chunks += 1
        _write_metadata(conn, sources, chunk_counts)
        conn.commit()
    finally:
        conn.close()
    return IndexSummary(str(index_path), total_chunks)


def _fuzzy_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def retrieve(
    query: str,
    k: int = 3,
    index_path: Path | str | None = None,
    similarity_weight: float = 0.5,
) -> List[dict[str, str]]:
    """Retrieve the top-k chunks using token overlap + fuzzy matching."""

    index_path = Path(index_path) if index_path is not None else DEFAULT_INDEX_PATH
    if not index_path.exists():
        return []
    tokens = set(_tokenize(query))
    if not tokens:
        return []
    conn = sqlite3.connect(index_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(f"SELECT * FROM {INDEX_TABLE_NAME}").fetchall()
    finally:
        conn.close()
    scored: List[tuple[float, sqlite3.Row]] = []
    for row in rows:
        row_tokens = set(str(row["tokens"]).split())
        overlap_score = len(tokens & row_tokens)
        if overlap_score == 0 and similarity_weight <= 0:
            continue
        similarity_score = _fuzzy_similarity(query, row["chunk"])
        score = overlap_score + similarity_weight * similarity_score
        if score == 0:
            continue
        scored.append((score, row))
    scored.sort(key=lambda item: (-item[0], item[1]["source"], item[1]["start_line"]))
    results: List[dict[str, str]] = []
    for _, row in scored[:k]:
        results.append(
            {
                "source": row["source"],
                "anchor": row["anchor"],
                "chunk": row["chunk"],
                "start_line": str(row["start_line"]),
                "end_line": str(row["end_line"]),
            }
        )
    return results
