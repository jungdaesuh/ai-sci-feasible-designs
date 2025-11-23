"""SQLite-backed boundary/evaluation log for AI Scientist cycles.

Implements the world-model slice from docs/AI_SCIENTIST_UNIFIED_ROADMAP.md
step 2: boundaries + evaluations + cycles with hash canonicalization and
helpers for feasibility queries and archive lookups.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS boundaries (
    hash TEXT PRIMARY KEY,
    schema_version TEXT NOT NULL,
    p INTEGER,
    nfp INTEGER,
    r_cos_blob BLOB,
    z_sin_blob BLOB,
    source TEXT,
    parent_id TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    boundary_id TEXT REFERENCES boundaries(hash),
    stage TEXT,
    vmec_status TEXT,
    runtime_sec REAL,
    metrics_json TEXT,
    margins_json TEXT,
    l2_margin REAL,
    is_feasible INTEGER,
    schema_version TEXT NOT NULL,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS cycles (
    cycle_idx INTEGER PRIMARY KEY,
    phase TEXT,
    p INTEGER,
    new_evals INTEGER,
    new_feasible INTEGER,
    cumulative_feasible INTEGER,
    hv REAL,
    notes TEXT,
    created_at TEXT
);
"""


def _normalize_to_json(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(key): _normalize_to_json(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_to_json(val) for val in value]
    return str(value)


def _decimal_places(rounding: float) -> int:
    if rounding <= 0:
        return 0
    return max(0, int(round(-math.log10(rounding))))


def _round_value(value: float, rounding: float) -> float:
    decimals = _decimal_places(rounding)
    return round(value, decimals)


def _round_matrix(matrix: Sequence[Sequence[float]], rounding: float) -> list[list[float]]:
    return [[_round_value(float(val), rounding) for val in row] for row in matrix]


def _encode_matrix(matrix: Sequence[Sequence[float]]) -> sqlite3.Binary:
    n_rows = len(matrix)
    n_cols = len(matrix[0]) if n_rows > 0 else 0
    flattened: list[float] = [float(val) for row in matrix for val in row]
    payload = {"shape": [n_rows, n_cols], "data": flattened}
    return sqlite3.Binary(json.dumps(payload, separators=(",", ":")).encode("utf-8"))


def _decode_matrix(blob: bytes | None) -> list[list[float]]:
    if blob is None:
        return []
    payload = json.loads(blob.decode("utf-8"))
    shape = payload.get("shape", [0, 0])
    data = payload.get("data", [])
    rows, cols = int(shape[0]), int(shape[1]) if len(shape) > 1 else 0
    matrix: list[list[float]] = []
    for row_idx in range(rows):
        start = row_idx * cols
        end = start + cols
        matrix.append([float(val) for val in data[start:end]])
    return matrix


def hash_boundary(
    r_cos: Sequence[Sequence[float]],
    z_sin: Sequence[Sequence[float]],
    schema_version: str,
    rounding: float = 1e-6,
) -> str:
    rounded_r = _round_matrix(r_cos, rounding)
    rounded_z = _round_matrix(z_sin, rounding)
    payload = {
        "schema_version": schema_version,
        "r_cos": rounded_r,
        "z_sin": rounded_z,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class BoundaryWorldModel:
    """SQLite wrapper for boundary archives and evaluations."""

    def __init__(self, path: str | Path) -> None:
        self.db_path = Path(path)
        _init_db(self.db_path)
        self._conn = _connect(self.db_path)

    def __enter__(self) -> "BoundaryWorldModel":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - context helper
        self.close()

    def close(self) -> None:
        self._conn.close()

    def add_boundary(
        self,
        *,
        r_cos: Sequence[Sequence[float]],
        z_sin: Sequence[Sequence[float]],
        schema_version: str,
        p: int,
        nfp: int,
        source: str | None = None,
        parent_id: str | None = None,
        rounding: float = 1e-6,
        created_at: str | None = None,
    ) -> str:
        design_hash = hash_boundary(r_cos, z_sin, schema_version, rounding)
        timestamp = created_at or datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT OR IGNORE INTO boundaries
            (hash, schema_version, p, nfp, r_cos_blob, z_sin_blob, source, parent_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                design_hash,
                schema_version,
                p,
                nfp,
                _encode_matrix(r_cos),
                _encode_matrix(z_sin),
                source,
                parent_id,
                timestamp,
            ),
        )
        self._conn.commit()
        return design_hash

    def add_evaluation(
        self,
        *,
        boundary_hash: str,
        stage: str,
        vmec_status: str,
        runtime_sec: float | None,
        metrics: Mapping[str, Any],
        margins: Mapping[str, float],
        is_feasible: bool,
        schema_version: str,
        created_at: str | None = None,
    ) -> int:
        timestamp = created_at or datetime.now(timezone.utc).isoformat()
        l2_margin = _l2_margin(margins)
        cursor = self._conn.execute(
            """
            INSERT INTO evaluations
            (boundary_id, stage, vmec_status, runtime_sec, metrics_json, margins_json, l2_margin, is_feasible, schema_version, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                boundary_hash,
                stage,
                vmec_status,
                runtime_sec,
                json.dumps(_normalize_to_json(metrics), separators=(",", ":")),
                json.dumps(_normalize_to_json(margins), separators=(",", ":")),
                l2_margin,
                int(is_feasible),
                schema_version,
                timestamp,
            ),
        )
        self._conn.commit()
        rowid = cursor.lastrowid
        assert rowid is not None
        return int(rowid)

    def log_cycle(
        self,
        *,
        cycle_idx: int,
        phase: str,
        p: int,
        new_evals: int,
        new_feasible: int,
        cumulative_feasible: int,
        hv: float | None,
        notes: str | None = None,
        created_at: str | None = None,
    ) -> None:
        timestamp = created_at or datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT OR REPLACE INTO cycles
            (cycle_idx, phase, p, new_evals, new_feasible, cumulative_feasible, hv, notes, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cycle_idx,
                phase,
                p,
                new_evals,
                new_feasible,
                cumulative_feasible,
                hv,
                notes,
                timestamp,
            ),
        )
        self._conn.commit()

    def get_feasible(self, *, p: int | None = None) -> list[Mapping[str, Any]]:
        return self._fetch_evaluations(is_feasible=True, p=p)

    def get_near_feasible(
        self, *, max_l2_margin: float, p: int | None = None
    ) -> list[Mapping[str, Any]]:
        return self._fetch_evaluations(is_feasible=None, p=p, l2_ceiling=max_l2_margin)

    def latest_archive(self, *, p: int | None = None) -> list[Mapping[str, Any]]:
        return self._fetch_evaluations(is_feasible=True, p=p, latest_only=True)

    def cache_stats(self) -> Mapping[str, int]:
        boundaries = self._conn.execute("SELECT COUNT(*) AS n FROM boundaries").fetchone()
        evaluations = self._conn.execute("SELECT COUNT(*) AS n FROM evaluations").fetchone()
        feasible = self._conn.execute(
            "SELECT COUNT(*) AS n FROM evaluations WHERE is_feasible = 1"
        ).fetchone()
        cycles = self._conn.execute("SELECT COUNT(*) AS n FROM cycles").fetchone()
        return {
            "boundaries": int(boundaries["n"]) if boundaries else 0,
            "evaluations": int(evaluations["n"]) if evaluations else 0,
            "feasible": int(feasible["n"]) if feasible else 0,
            "cycles": int(cycles["n"]) if cycles else 0,
        }

    def _fetch_evaluations(
        self,
        *,
        is_feasible: bool | None,
        p: int | None,
        l2_ceiling: float | None = None,
        latest_only: bool = False,
    ) -> list[Mapping[str, Any]]:
        where_clauses = ["1=1"]
        params: list[Any] = []
        if is_feasible is not None:
            where_clauses.append("evaluations.is_feasible = ?")
            params.append(1 if is_feasible else 0)
        if p is not None:
            where_clauses.append("boundaries.p = ?")
            params.append(p)
        if l2_ceiling is not None:
            where_clauses.append("evaluations.l2_margin <= ?")
            params.append(l2_ceiling)
        where_sql = " AND ".join(where_clauses)
        query = f"""
        SELECT evaluations.*, boundaries.schema_version AS boundary_schema, boundaries.p, boundaries.nfp,
               boundaries.r_cos_blob, boundaries.z_sin_blob, boundaries.source, boundaries.parent_id, boundaries.created_at AS boundary_created_at
        FROM evaluations
        JOIN boundaries ON evaluations.boundary_id = boundaries.hash
        WHERE {where_sql}
        ORDER BY evaluations.created_at DESC, evaluations.id DESC
        """
        rows = self._conn.execute(query, params).fetchall()
        if latest_only:
            if not rows:
                return []
            newest_hash = rows[0]["boundary_id"]
            filtered = [row for row in rows if row["boundary_id"] == newest_hash]
            return [_row_to_eval_dict(row) for row in filtered]
        return [_row_to_eval_dict(row) for row in rows]


def _l2_margin(margins: Mapping[str, float]) -> float:
    if not margins:
        return float("inf")
    positives = [max(0.0, float(value)) for value in margins.values()]
    squared = [val * val for val in positives]
    return float(math.sqrt(sum(squared)))


def _row_to_eval_dict(row: sqlite3.Row) -> Mapping[str, Any]:
    metrics = json.loads(row["metrics_json"]) if row["metrics_json"] else {}
    margins = json.loads(row["margins_json"]) if row["margins_json"] else {}
    return {
        "id": int(row["id"]),
        "boundary_hash": row["boundary_id"],
        "stage": row["stage"],
        "vmec_status": row["vmec_status"],
        "runtime_sec": float(row["runtime_sec"]) if row["runtime_sec"] is not None else None,
        "metrics": metrics,
        "margins": margins,
        "l2_margin": float(row["l2_margin"]) if row["l2_margin"] is not None else None,
        "is_feasible": bool(row["is_feasible"]),
        "schema_version": row["schema_version"],
        "created_at": row["created_at"],
        "boundary": {
            "hash": row["boundary_id"],
            "schema_version": row["boundary_schema"],
            "p": row["p"],
            "nfp": row["nfp"],
            "r_cos": _decode_matrix(row["r_cos_blob"]),
            "z_sin": _decode_matrix(row["z_sin_blob"]),
            "source": row["source"],
            "parent_id": row["parent_id"],
            "created_at": row["boundary_created_at"],
        },
    }


def _init_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = _connect(path)
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn
