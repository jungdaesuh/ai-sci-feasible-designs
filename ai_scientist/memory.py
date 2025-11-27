"""SQLite-backed world model for AI Scientist budgeting + logging.

This is the authoritative world model implementation for the AI Scientist.
It fulfills the Unified Roadmap Workstream 6 / Autonomy Plan requirement
for a per-cycle PropertyGraph and the Kosmos-style structured world model
(statements, citations, candidates, cycles → graph snapshot). The legacy
boundary-only world_model.py has been removed.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass
class PropertyGraph:
    nodes: dict[str, Mapping[str, Any]] = field(default_factory=dict)
    edges: list[tuple[str, str, Mapping[str, Any]]] = field(default_factory=list)

    def add_node(self, node_id: str, **attrs: Any) -> None:
        self.nodes[node_id] = attrs

    def add_edge(self, src: str, dst: str, **attrs: Any) -> None:
        self.edges.append((src, dst, attrs))

    def has_node(self, node_id: str) -> bool:
        return node_id in self.nodes


@dataclass(frozen=True)
class StatementRecord:
    id: int
    experiment_id: int
    cycle: int
    stage: str
    text: str
    status: str
    metrics_id: int | None
    tool_name: str
    tool_input_hash: str
    seed: int | None
    git_sha: str
    repro_cmd: str
    created_at: str


@dataclass(frozen=True)
class StageHistoryEntry:
    cycle: int
    stage: str
    selected_at: str


SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    config_json TEXT NOT NULL,
    git_sha TEXT NOT NULL,
    constellaration_sha TEXT NOT NULL,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    problem TEXT NOT NULL,
    params_json TEXT NOT NULL,
    seed INTEGER NOT NULL,
    status TEXT NOT NULL,
    design_hash TEXT NOT NULL,
    FOREIGN KEY(experiment_id) REFERENCES experiments(id)
);

CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id INTEGER NOT NULL,
    raw_json TEXT NOT NULL,
    feasibility REAL NOT NULL,
    objective REAL,
    hv REAL,
    is_feasible INTEGER NOT NULL,
    FOREIGN KEY(candidate_id) REFERENCES candidates(id)
);

CREATE TABLE IF NOT EXISTS pareto (
    experiment_id INTEGER NOT NULL,
    candidate_id INTEGER NOT NULL,
    PRIMARY KEY (experiment_id, candidate_id),
    FOREIGN KEY(experiment_id) REFERENCES experiments(id),
    FOREIGN KEY(candidate_id) REFERENCES candidates(id)
);

CREATE TABLE IF NOT EXISTS citations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    source_path TEXT NOT NULL,
    anchor TEXT,
    quote TEXT,
    FOREIGN KEY(experiment_id) REFERENCES experiments(id)
);

CREATE TABLE IF NOT EXISTS artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    path TEXT NOT NULL,
    kind TEXT NOT NULL,
    FOREIGN KEY(experiment_id) REFERENCES experiments(id)
);

CREATE TABLE IF NOT EXISTS budgets (
    experiment_id INTEGER NOT NULL,
    cycle INTEGER NOT NULL,
    screen_evals INTEGER NOT NULL,
    promoted_evals INTEGER NOT NULL,
    high_fidelity_evals INTEGER NOT NULL,
    wall_seconds REAL NOT NULL,
    best_objective REAL,
    best_feasibility REAL,
    best_score REAL,
    best_stage TEXT,
    PRIMARY KEY (experiment_id, cycle),
    FOREIGN KEY(experiment_id) REFERENCES experiments(id)
);

CREATE TABLE IF NOT EXISTS cycles (
    experiment_id INTEGER NOT NULL,
    cycle INTEGER NOT NULL,
    stage TEXT NOT NULL,
    feasible_count INTEGER NOT NULL,
    hv_score REAL,
    hv_exists INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    PRIMARY KEY (experiment_id, cycle),
    FOREIGN KEY(experiment_id) REFERENCES experiments(id)
);

CREATE TABLE IF NOT EXISTS cycle_stats (
    experiment_id INTEGER NOT NULL,
    cycle INTEGER NOT NULL,
    hv_score REAL NOT NULL,
    reference_point TEXT NOT NULL,
    pareto_json TEXT NOT NULL,
    PRIMARY KEY (experiment_id, cycle),
    FOREIGN KEY(experiment_id) REFERENCES experiments(id)
);

CREATE TABLE IF NOT EXISTS cycle_hv (
    experiment_id INTEGER NOT NULL,
    cycle INTEGER NOT NULL,
    hv_value REAL NOT NULL,
    hv_delta REAL,
    hv_delta_moving_avg REAL,
    n_feasible INTEGER NOT NULL,
    n_archive INTEGER NOT NULL,
    PRIMARY KEY (experiment_id, cycle),
    FOREIGN KEY(experiment_id) REFERENCES experiments(id)
);

CREATE TABLE IF NOT EXISTS pareto_archive (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    cycle INTEGER NOT NULL,
    design_hash TEXT NOT NULL,
    fidelity TEXT NOT NULL,
    gradient REAL NOT NULL,
    aspect REAL NOT NULL,
    metrics_id INTEGER,
    git_sha TEXT,
    constellaration_sha TEXT,
    settings_json TEXT NOT NULL,
    seed INTEGER NOT NULL,
    UNIQUE(experiment_id, cycle, design_hash, fidelity),
    FOREIGN KEY(experiment_id) REFERENCES experiments(id),
    FOREIGN KEY(metrics_id) REFERENCES metrics(id)
);

CREATE TABLE IF NOT EXISTS statements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    cycle INTEGER NOT NULL,
    stage TEXT NOT NULL,
    text TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'PENDING',
    metrics_id INTEGER,
    tool_name TEXT NOT NULL,
    tool_input_hash TEXT NOT NULL,
    seed INTEGER,
    git_sha TEXT NOT NULL,
    repro_cmd TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(experiment_id) REFERENCES experiments(id),
    FOREIGN KEY(metrics_id) REFERENCES metrics(id),
    UNIQUE(experiment_id, cycle, tool_input_hash)
);

CREATE TABLE IF NOT EXISTS stage_history (
    experiment_id INTEGER NOT NULL,
    cycle INTEGER NOT NULL,
    stage TEXT NOT NULL,
    selected_at TEXT NOT NULL,
    PRIMARY KEY (experiment_id, cycle),
    FOREIGN KEY(experiment_id) REFERENCES experiments(id)
);

CREATE TABLE IF NOT EXISTS deterministic_snapshots (
    experiment_id INTEGER NOT NULL,
    cycle INTEGER NOT NULL,
    snapshot_json TEXT NOT NULL,
    constellaration_sha TEXT NOT NULL,
    seed INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (experiment_id, cycle),
    FOREIGN KEY(experiment_id) REFERENCES experiments(id)
);

CREATE TABLE IF NOT EXISTS literature_notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    cycle INTEGER NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(experiment_id) REFERENCES experiments(id)
);

CREATE TABLE IF NOT EXISTS optimization_state (
    experiment_id INTEGER NOT NULL,
    cycle INTEGER NOT NULL,
    alm_multipliers_json TEXT NOT NULL,
    penalty_parameter REAL NOT NULL,
    optimizer_state_json TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (experiment_id, cycle),
    FOREIGN KEY(experiment_id) REFERENCES experiments(id)
);

CREATE TABLE IF NOT EXISTS alm_state_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    cycle INTEGER NOT NULL,
    step_index INTEGER NOT NULL,
    constraint_name TEXT NOT NULL,
    multiplier_value REAL NOT NULL,
    penalty_parameter REAL NOT NULL,
    violation_magnitude REAL,
    FOREIGN KEY(experiment_id) REFERENCES experiments(id)
);

CREATE TABLE IF NOT EXISTS surrogate_registry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    cycle INTEGER NOT NULL,
    backend_type TEXT NOT NULL,
    training_samples INTEGER NOT NULL,
    validation_mse REAL,
    model_hash TEXT NOT NULL,
    weights_path TEXT NOT NULL,
    FOREIGN KEY(experiment_id) REFERENCES experiments(id)
);

CREATE TABLE IF NOT EXISTS surrogate_checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    cycle INTEGER NOT NULL,
    backend TEXT NOT NULL,
    filepath TEXT NOT NULL,
    metrics_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(experiment_id) REFERENCES experiments(id)
);
"""

DEFAULT_RELATIVE_TOLERANCE = 1e-2


@dataclass(frozen=True)
class BudgetUsage:
    screen_evals: int
    promoted_evals: int
    high_fidelity_evals: int
    wall_seconds: float


def _normalize_to_json(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(key): _normalize_to_json(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_to_json(val) for val in value]
    return str(value)


def _hash_payload(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        _normalize_to_json(payload), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def hash_payload(payload: Mapping[str, Any]) -> str:
    """Return a deterministic digest that mirrors the statements table hashing."""

    return _hash_payload(payload)


class WorldModel:
    """Simple SQLite wrapper for experiments, candidates, and budgets."""

    def __init__(self, path: str | Path) -> None:
        db_path = Path(path)
        init_db(db_path)
        self.db_path = db_path
        self._conn: sqlite3.Connection = sqlite3.connect(
            str(self.db_path), check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row

    def __enter__(self) -> "WorldModel":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - context helper
        self.close()

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def transaction(self):
        """Context manager that wraps multiple writes in a single atomic transaction."""

        try:
            self._conn.execute("BEGIN")
            yield
        except Exception:
            self._conn.rollback()
            raise
        else:
            self._conn.commit()

    def start_experiment(
        self,
        config_payload: Mapping[str, Any],
        git_sha: str,
        constellaration_sha: str | None = None,
        notes: str | None = None,
    ) -> int:
        payload = json.dumps(_normalize_to_json(config_payload), separators=(",", ":"))
        cursor = self._conn.execute(
            "INSERT INTO experiments (started_at, config_json, git_sha, constellaration_sha, notes) VALUES (?, ?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(),
                payload,
                git_sha,
                constellaration_sha or "unknown",
                notes,
            ),
        )
        lastrowid = cursor.lastrowid
        assert lastrowid is not None
        self._conn.commit()
        return lastrowid

    def record_cycle(
        self,
        experiment_id: int,
        cycle_number: int,
        screen_evals: int,
        promoted_evals: int,
        high_fidelity_evals: int,
        wall_seconds: float,
        best_params: Mapping[str, Any],
        best_evaluation: Mapping[str, Any],
        seed: int,
        problem: str,
        *,
        log_best_candidate: bool = True,
        commit: bool = True,
    ) -> None:
        payload = (
            experiment_id,
            cycle_number,
            screen_evals,
            promoted_evals,
            high_fidelity_evals,
            wall_seconds,
            best_evaluation.get("objective"),
            best_evaluation.get("feasibility"),
            best_evaluation.get("score"),
            best_evaluation.get("stage", ""),
        )

        def _write() -> None:
            self._conn.execute(
                "INSERT OR REPLACE INTO budgets (experiment_id, cycle, screen_evals, promoted_evals, high_fidelity_evals, wall_seconds, best_objective, best_feasibility, best_score, best_stage) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                payload,
            )
            if log_best_candidate:
                self.log_candidate(
                    experiment_id=experiment_id,
                    problem=problem,
                    params=best_params,
                    seed=seed,
                    status=best_evaluation.get("stage", "unknown"),
                    evaluation=best_evaluation,
                    design_hash=best_evaluation.get("design_hash", ""),
                    commit=False,
                )

        if commit:
            with self._conn:
                _write()
        else:
            _write()

    def record_deterministic_snapshot(
        self,
        experiment_id: int,
        cycle_number: int,
        snapshot: Mapping[str, Any],
        *,
        constellaration_sha: str,
        seed: int,
        created_at: str | None = None,
        commit: bool = True,
    ) -> None:
        payload = json.dumps(_normalize_to_json(snapshot), separators=(",", ":"))
        timestamp = created_at or datetime.now(timezone.utc).isoformat()

        def _write() -> None:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO deterministic_snapshots
                (experiment_id, cycle, snapshot_json, constellaration_sha, seed, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    cycle_number,
                    payload,
                    constellaration_sha,
                    seed,
                    timestamp,
                ),
            )

        if commit:
            with self._conn:
                _write()
        else:
            _write()

    def record_optimization_state(
        self,
        experiment_id: int,
        cycle: int,
        alm_multipliers: Mapping[str, float],
        penalty_parameter: float,
        optimizer_state: Mapping[str, Any] | None = None,
        *,
        created_at: str | None = None,
        commit: bool = True,
    ) -> None:
        timestamp = created_at or datetime.now(timezone.utc).isoformat()
        multipliers_json = json.dumps(
            _normalize_to_json(alm_multipliers), separators=(",", ":")
        )
        optimizer_state_json = (
            json.dumps(_normalize_to_json(optimizer_state), separators=(",", ":"))
            if optimizer_state is not None
            else None
        )

        def _write() -> None:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO optimization_state
                (experiment_id, cycle, alm_multipliers_json, penalty_parameter, optimizer_state_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    cycle,
                    multipliers_json,
                    float(penalty_parameter),
                    optimizer_state_json,
                    timestamp,
                ),
            )

        if commit:
            with self._conn:
                _write()
        else:
            _write()

    def log_alm_state(
        self,
        experiment_id: int,
        cycle: int,
        step_index: int,
        constraint_name: str,
        multiplier_value: float,
        penalty_parameter: float,
        violation_magnitude: float | None = None,
        *,
        commit: bool = True,
    ) -> None:
        """Record the trajectory of ALM multipliers/penalties per step."""
        def _write() -> None:
            self._conn.execute(
                """
                INSERT INTO alm_state_history
                (experiment_id, cycle, step_index, constraint_name, multiplier_value, penalty_parameter, violation_magnitude)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    cycle,
                    step_index,
                    constraint_name,
                    multiplier_value,
                    penalty_parameter,
                    violation_magnitude,
                ),
            )
        
        if commit:
            with self._conn:
                _write()
        else:
            _write()

    def register_surrogate(
        self,
        experiment_id: int,
        cycle: int,
        backend_type: str,
        training_samples: int,
        model_hash: str,
        weights_path: str,
        *,
        validation_mse: float | None = None,
        commit: bool = True,
    ) -> int:
        """Register a trained surrogate model version."""
        def _write() -> None:
            nonlocal row_id
            cursor = self._conn.execute(
                """
                INSERT INTO surrogate_registry
                (experiment_id, cycle, backend_type, training_samples, validation_mse, model_hash, weights_path)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    cycle,
                    backend_type,
                    training_samples,
                    validation_mse,
                    model_hash,
                    weights_path,
                ),
            )
            row_id = cursor.lastrowid
            assert row_id is not None

        row_id: int | None = None
        if commit:
            with self._conn:
                _write()
        else:
            _write()
        assert row_id is not None
        return row_id

    def record_surrogate_checkpoint(
        self,
        experiment_id: int,
        cycle: int,
        backend: str,
        filepath: str | Path,
        metrics: Mapping[str, Any] | None = None,
        *,
        created_at: str | None = None,
        commit: bool = True,
    ) -> int:
        timestamp = created_at or datetime.now(timezone.utc).isoformat()
        metrics_json = (
            json.dumps(_normalize_to_json(metrics), separators=(",", ":"))
            if metrics is not None
            else None
        )

        def _write() -> None:
            nonlocal checkpoint_id
            cursor = self._conn.execute(
                """
                INSERT INTO surrogate_checkpoints
                (experiment_id, cycle, backend, filepath, metrics_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    cycle,
                    backend,
                    str(filepath),
                    metrics_json,
                    timestamp,
                ),
            )
            checkpoint_id = cursor.lastrowid
            assert checkpoint_id is not None

        checkpoint_id: int | None = None
        if commit:
            with self._conn:
                _write()
        else:
            _write()

        assert checkpoint_id is not None
        return checkpoint_id

    def record_cycle_summary(
        self,
        experiment_id: int,
        cycle_number: int,
        stage: str,
        feasible_count: int,
        hv_score: float | None,
        *,
        created_at: str | None = None,
        commit: bool = True,
    ) -> None:
        timestamp = created_at or datetime.now(timezone.utc).isoformat()
        hv_exists = 0 if hv_score is None else 1

        def _write() -> None:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO cycles (experiment_id, cycle, stage, feasible_count, hv_score, hv_exists, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    cycle_number,
                    stage,
                    int(feasible_count),
                    hv_score,
                    hv_exists,
                    timestamp,
                ),
            )

        if commit:
            with self._conn:
                _write()
        else:
            _write()

    def record_cycle_hv(
        self,
        experiment_id: int,
        cycle_number: int,
        hv_score: float,
        reference_point: Sequence[float],
        pareto_entries: Sequence[Mapping[str, float]],
        *,
        n_feasible: int,
        n_archive: int,
        hv_lookback: int | None = None,
        commit: bool = True,
    ) -> None:
        snapshot = {
            "reference_point": list(reference_point),
            "pareto": [_normalize_to_json(entry) for entry in pareto_entries],
        }

        payload_stats = (
            experiment_id,
            cycle_number,
            hv_score,
            json.dumps(list(reference_point)),
            json.dumps(snapshot, separators=(",", ":")),
        )

        prev_row = self._conn.execute(
            "SELECT hv_value FROM cycle_hv WHERE experiment_id = ? ORDER BY cycle DESC LIMIT 1",
            (experiment_id,),
        ).fetchone()
        previous_hv = float(prev_row["hv_value"]) if prev_row else None
        hv_delta: float | None = None
        if previous_hv is not None:
            hv_delta = float(abs(hv_score - previous_hv))
        hv_delta_moving_avg: float | None = None
        if hv_delta is not None and hv_lookback is not None and hv_lookback > 0:
            lookback_limit = hv_lookback - 1
            delta_rows = []
            if lookback_limit > 0:
                delta_rows = self._conn.execute(
                    "SELECT hv_delta FROM cycle_hv WHERE experiment_id = ? AND hv_delta IS NOT NULL ORDER BY cycle DESC LIMIT ?",
                    (experiment_id, lookback_limit),
                ).fetchall()
            recent_deltas = [
                row["hv_delta"] for row in delta_rows if row["hv_delta"] is not None
            ]
            moving_window = [hv_delta] + recent_deltas[:lookback_limit]
            if len(moving_window) >= hv_lookback:
                hv_delta_moving_avg = float(sum(moving_window) / len(moving_window))

        payload_hv = (
            experiment_id,
            cycle_number,
            hv_score,
            hv_delta,
            hv_delta_moving_avg,
            n_feasible,
            n_archive,
        )

        def _write() -> None:
            self._conn.execute(
                "INSERT OR REPLACE INTO cycle_stats (experiment_id, cycle, hv_score, reference_point, pareto_json) VALUES (?, ?, ?, ?, ?)",
                payload_stats,
            )
            self._conn.execute(
                "INSERT OR REPLACE INTO cycle_hv (experiment_id, cycle, hv_value, hv_delta, hv_delta_moving_avg, n_feasible, n_archive) VALUES (?, ?, ?, ?, ?, ?, ?)",
                payload_hv,
            )

        if commit:
            with self._conn:
                _write()
        else:
            _write()

    def average_recent_hv_delta(
        self,
        experiment_id: int,
        lookback: int,
    ) -> float | None:
        if lookback <= 0:
            return None
        rows = self._conn.execute(
            "SELECT hv_delta FROM cycle_hv WHERE experiment_id = ? AND hv_delta IS NOT NULL ORDER BY cycle DESC LIMIT ?",
            (experiment_id, lookback),
        ).fetchall()
        deltas = [row["hv_delta"] for row in rows if row["hv_delta"] is not None]
        if len(deltas) < lookback:
            return None
        return float(sum(deltas) / len(deltas))

    def log_candidate(
        self,
        experiment_id: int,
        problem: str,
        params: Mapping[str, Any],
        seed: int,
        status: str,
        evaluation: Mapping[str, Any],
        *,
        design_hash: str,
        commit: bool = True,
    ) -> tuple[int, int]:
        params_json = json.dumps(_normalize_to_json(params), separators=(",", ":"))
        cursor = self._conn.execute(
            "INSERT INTO candidates (experiment_id, problem, params_json, seed, status, design_hash) VALUES (?, ?, ?, ?, ?, ?)",
            (experiment_id, problem, params_json, seed, status, design_hash),
        )
        candidate_id = cursor.lastrowid
        assert candidate_id is not None
        metrics_id = self.log_metrics(
            candidate_id,
            evaluation.get("metrics", {}),
            feasibility=float(evaluation.get("feasibility", 0.0)),
            objective=evaluation.get("objective"),
            hv=evaluation.get("hv"),
            commit=False,
        )
        if commit:
            self._conn.commit()
        return candidate_id, metrics_id

    def log_artifact(
        self,
        experiment_id: int,
        path: str | Path,
        kind: str,
        *,
        commit: bool = True,
    ) -> None:
        """Record artifacts such as metrics snapshots or Pareto figures."""

        self._conn.execute(
            "INSERT INTO artifacts (experiment_id, path, kind) VALUES (?, ?, ?)",
            (experiment_id, str(path), kind),
        )
        if commit:
            self._conn.commit()

    def log_statement(
        self,
        experiment_id: int,
        cycle: int,
        stage: str,
        text: str,
        status: str,
        tool_name: str,
        tool_input: Mapping[str, Any],
        *,
        metrics_id: int | None = None,
        seed: int | None = None,
        git_sha: str,
        repro_cmd: str,
        created_at: str | None = None,
        commit: bool = True,
    ) -> int:
        digest = _hash_payload(tool_input)
        timestamp = created_at or datetime.now(timezone.utc).isoformat()
        cursor = self._conn.execute(
            """
            INSERT INTO statements
            (experiment_id, cycle, stage, text, status, metrics_id, tool_name, tool_input_hash, seed, git_sha, repro_cmd, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_id,
                cycle,
                stage,
                text,
                status,
                metrics_id,
                tool_name,
                digest,
                int(seed) if seed is not None else None,
                git_sha,
                repro_cmd,
                timestamp,
            ),
        )
        statement_id = cursor.lastrowid
        assert statement_id is not None
        if commit:
            self._conn.commit()
        return statement_id

    def log_note(
        self,
        experiment_id: int,
        cycle: int,
        content: str,
        *,
        created_at: str | None = None,
        commit: bool = True,
    ) -> int:
        timestamp = created_at or datetime.now(timezone.utc).isoformat()
        cursor = self._conn.execute(
            "INSERT INTO literature_notes (experiment_id, cycle, content, created_at) VALUES (?, ?, ?, ?)",
            (experiment_id, cycle, content, timestamp),
        )
        note_id = cursor.lastrowid
        assert note_id is not None
        if commit:
            self._conn.commit()
        return note_id

    def notes(
        self, experiment_id: int, cycle: int | None = None
    ) -> list[Mapping[str, Any]]:
        """Retrieve stored literature notes, optionally filtered by cycle."""
        if cycle is not None:
            rows = self._conn.execute(
                "SELECT * FROM literature_notes WHERE experiment_id = ? AND cycle = ? ORDER BY id ASC",
                (experiment_id, cycle),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM literature_notes WHERE experiment_id = ? ORDER BY id ASC",
                (experiment_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def statements_for_cycle(
        self, experiment_id: int, cycle: int
    ) -> list[StatementRecord]:
        rows = self._conn.execute(
            """
            SELECT *
            FROM statements
            WHERE experiment_id = ? AND cycle = ?
            ORDER BY id ASC
            """,
            (experiment_id, cycle),
        ).fetchall()
        history: list[StatementRecord] = []
        for row in rows:
            history.append(
                StatementRecord(
                    id=row["id"],
                    experiment_id=row["experiment_id"],
                    cycle=row["cycle"],
                    stage=row["stage"],
                    text=row["text"],
                    status=row["status"],
                    metrics_id=row["metrics_id"],
                    tool_name=row["tool_name"],
                    tool_input_hash=row["tool_input_hash"],
                    seed=row["seed"],
                    git_sha=row["git_sha"],
                    repro_cmd=row["repro_cmd"],
                    created_at=row["created_at"],
                )
            )
        return history

    def stage_history(self, experiment_id: int) -> list[StageHistoryEntry]:
        rows = self._conn.execute(
            """
            SELECT cycle, stage, selected_at
            FROM stage_history
            WHERE experiment_id = ?
            ORDER BY cycle ASC
            """,
            (experiment_id,),
        ).fetchall()
        return [
            StageHistoryEntry(
                cycle=row["cycle"],
                stage=row["stage"],
                selected_at=row["selected_at"],
            )
            for row in rows
        ]

    def cycle_summaries(self, experiment_id: int) -> list[dict[str, Any]]:
        """Return cycle-level summaries combining stage history, budgets, and cycles tables."""

        rows = self._conn.execute(
            """
            SELECT s.cycle,
                   s.stage,
                   b.best_objective,
                   b.best_feasibility,
                   c.hv_score
            FROM stage_history s
            LEFT JOIN budgets b
              ON s.experiment_id = b.experiment_id AND s.cycle = b.cycle
            LEFT JOIN cycles c
              ON s.experiment_id = c.experiment_id AND s.cycle = c.cycle
            WHERE s.experiment_id = ?
            ORDER BY s.cycle ASC
            """,
            (experiment_id,),
        ).fetchall()
        return [
            {
                "cycle": row["cycle"],
                "stage": row["stage"],
                "objective": row["best_objective"],
                "feasibility": row["best_feasibility"],
                "hv": row["hv_score"],
            }
            for row in rows
        ]

    def record_stage_history(
        self,
        experiment_id: int,
        cycle: int,
        stage: str,
        *,
        selected_at: str | None = None,
        commit: bool = True,
    ) -> None:
        timestamp = selected_at or datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT OR REPLACE INTO stage_history (experiment_id, cycle, stage, selected_at)
            VALUES (?, ?, ?, ?)
            """,
            (experiment_id, cycle, stage, timestamp),
        )
        if commit:
            self._conn.commit()

    def recent_stage_candidates(
        self,
        experiment_id: int,
        problem: str,
        stage: str,
        *,
        limit: int = 64,
    ) -> list[tuple[Mapping[str, Any], float]]:
        """Return parameters + feasibility for the most recent candidates of a given stage."""

        rows = self._conn.execute(
            """
            SELECT c.params_json, m.feasibility
            FROM candidates c
            JOIN metrics m ON m.candidate_id = c.id
            WHERE c.experiment_id = ? AND c.problem = ? AND c.status = ?
            ORDER BY m.id DESC
            LIMIT ?
            """,
            (experiment_id, problem, stage, limit),
        ).fetchall()
        records: list[tuple[Mapping[str, Any], float]] = []
        for row in rows:
            raw_params = row["params_json"]
            feasibility = row["feasibility"]
            if feasibility is None:
                continue
            params = json.loads(raw_params)
            records.append((params, float(feasibility)))
        return records

    def recent_failures(
        self,
        experiment_id: int,
        problem: str,
        *,
        limit: int = 5,
    ) -> list[Mapping[str, Any]]:
        """Return detailed info on recent failed candidates to help the agent learn."""
        
        rows = self._conn.execute(
            """
            SELECT c.params_json, m.raw_json, m.feasibility
            FROM metrics m
            JOIN candidates c ON m.candidate_id = c.id
            WHERE c.experiment_id = ? AND c.problem = ? AND m.is_feasible = 0
            ORDER BY m.id DESC
            LIMIT ?
            """,
            (experiment_id, problem, limit),
        ).fetchall()
        
        failures: list[Mapping[str, Any]] = []
        for row in rows:
            try:
                params = json.loads(row["params_json"])
                metrics_raw = json.loads(row["raw_json"])
                constraint_margins = metrics_raw.get("constraint_margins", {})
                # Only include significant violations to keep context clean
                violations = {k: v for k, v in constraint_margins.items() if v > 0}
                
                failures.append({
                    "params": params,
                    "feasibility": float(row["feasibility"]),
                    "violations": violations
                })
            except (json.JSONDecodeError, ValueError):
                continue
                
        return failures

    def previous_best_hv(self, experiment_id: int, cycle_number: int) -> float | None:
        row = self._conn.execute(
            """
            SELECT MAX(hv_value) AS max_hv
            FROM cycle_hv
            WHERE experiment_id = ? AND cycle < ?
            """,
            (experiment_id, cycle_number),
        ).fetchone()
        if row is None:
            return None
        max_hv = row["max_hv"]
        return float(max_hv) if max_hv is not None else None

    def log_metrics(
        self,
        candidate_id: int,
        metrics_payload: Mapping[str, Any],
        *,
        feasibility: float | None = None,
        objective: float | None = None,
        hv: float | None = None,
        commit: bool = True,
    ) -> int:
        payload = json.dumps(_normalize_to_json(metrics_payload), separators=(",", ":"))
        feasibility_value = float(
            feasibility
            if feasibility is not None
            else metrics_payload.get("feasibility", 0.0)
        )

        def _coerce_float(value: Any | None) -> float | None:
            if value is None:
                return None
            return float(value)

        objective_value = (
            _coerce_float(objective)
            if objective is not None
            else _coerce_float(metrics_payload.get("objective"))
        )
        hv_value = (
            _coerce_float(hv)
            if hv is not None
            else _coerce_float(metrics_payload.get("hv"))
        )
        cursor = self._conn.execute(
            "INSERT INTO metrics (candidate_id, raw_json, feasibility, objective, hv, is_feasible) VALUES (?, ?, ?, ?, ?, ?)",
            (
                candidate_id,
                payload,
                feasibility_value,
                objective_value,
                hv_value,
                1 if feasibility_value <= DEFAULT_RELATIVE_TOLERANCE else 0,
            ),
        )
        metrics_id = cursor.lastrowid
        assert metrics_id is not None
        if commit:
            self._conn.commit()
        return metrics_id

    def record_pareto_archive(
        self,
        experiment_id: int,
        cycle_number: int,
        entries: Sequence[Mapping[str, Any]],
        *,
        commit: bool = True,
    ) -> None:
        def _write() -> None:
            for entry in entries:
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO pareto_archive
                    (experiment_id, cycle, design_hash, fidelity, gradient, aspect, metrics_id, git_sha, constellaration_sha, settings_json, seed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        experiment_id,
                        cycle_number,
                        entry["design_hash"],
                        entry["fidelity"],
                        float(entry["gradient"]),
                        float(entry["aspect"]),
                        entry.get("metrics_id"),
                        entry.get("git_sha"),
                        entry.get("constellaration_sha"),
                        entry.get("settings_json"),
                        int(entry.get("seed", -1)),
                    ),
                )

        if commit:
            with self._conn:
                _write()
        else:
            _write()

    def budget_usage(self, experiment_id: int) -> BudgetUsage:
        row = self._conn.execute(
            "SELECT COALESCE(SUM(screen_evals), 0), COALESCE(SUM(promoted_evals), 0), COALESCE(SUM(high_fidelity_evals), 0), COALESCE(SUM(wall_seconds), 0.0) FROM budgets WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone()
        return BudgetUsage(
            screen_evals=int(row[0]),
            promoted_evals=int(row[1]),
            high_fidelity_evals=int(row[2]),
            wall_seconds=float(row[3]),
        )

    def cycles_completed(self, experiment_id: int) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM budgets WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone()
        return int(row[0])

    def upsert_pareto(
        self, experiment_id: int, candidate_id: int, *, commit: bool = True
    ) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO pareto (experiment_id, candidate_id) VALUES (?, ?)",
            (experiment_id, candidate_id),
        )
        if commit:
            self._conn.commit()

    def to_networkx(self, experiment_id: int) -> PropertyGraph:
        graph = PropertyGraph()
        experiment = self._conn.execute(
            "SELECT * FROM experiments WHERE id = ?",
            (experiment_id,),
        ).fetchone()
        if experiment is None:
            raise ValueError(f"experiment {experiment_id} does not exist")
        exp_node = f"experiment:{experiment_id}"
        graph.add_node(
            exp_node,
            type="experiment",
            started_at=experiment["started_at"],
            git_sha=experiment["git_sha"],
            notes=experiment["notes"],
        )
        candidate_rows = self._conn.execute(
            "SELECT * FROM candidates WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchall()
        for candidate in candidate_rows:
            candidate_node = f"candidate:{candidate['id']}"
            graph.add_node(
                candidate_node,
                type="candidate",
                problem=candidate["problem"],
                status=candidate["status"],
                seed=candidate["seed"],
                params=candidate["params_json"],
            )
            graph.add_edge(exp_node, candidate_node, relation="contains")
        metrics_rows = self._conn.execute(
            "SELECT m.*, c.problem FROM metrics m JOIN candidates c ON m.candidate_id = c.id WHERE c.experiment_id = ?",
            (experiment_id,),
        ).fetchall()
        for metrics in metrics_rows:
            metrics_node = f"metrics:{metrics['id']}"
            graph.add_node(
                metrics_node,
                type="metrics",
                feasibility=metrics["feasibility"],
                objective=metrics["objective"],
                hv=metrics["hv"],
                raw=metrics["raw_json"],
                problem=metrics["problem"],
            )
            candidate_node = f"candidate:{metrics['candidate_id']}"
            graph.add_edge(candidate_node, metrics_node, relation="evaluated_as")
        citations = self._conn.execute(
            "SELECT * FROM citations WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchall()
        for citation in citations:
            citation_node = f"citation:{citation['id']}"
            graph.add_node(
                citation_node,
                type="citation",
                source_path=citation["source_path"],
                anchor=citation["anchor"],
                quote=citation["quote"],
            )
            graph.add_edge(exp_node, citation_node, relation="cites")
        artifacts = self._conn.execute(
            "SELECT * FROM artifacts WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchall()
        for artifact in artifacts:
            artifact_node = f"artifact:{artifact['id']}"
            graph.add_node(
                artifact_node,
                type="artifact",
                path=artifact["path"],
                kind=artifact["kind"],
            )
            graph.add_edge(exp_node, artifact_node, relation="produces")
        budgets = self._conn.execute(
            "SELECT * FROM budgets WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchall()
        for budget in budgets:
            budget_node = f"budget:{experiment_id}:{budget['cycle']}"
            graph.add_node(
                budget_node,
                type="budget",
                cycle=budget["cycle"],
                screen_evals=budget["screen_evals"],
                promoted_evals=budget["promoted_evals"],
                high_fidelity_evals=budget["high_fidelity_evals"],
                wall_seconds=budget["wall_seconds"],
                best_stage=budget["best_stage"],
            )
            graph.add_edge(exp_node, budget_node, relation="cycle")
        stage_rows = self._conn.execute(
            "SELECT * FROM stage_history WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchall()
        for stage_row in stage_rows:
            stage_node = f"stage:{experiment_id}:{stage_row['cycle']}"
            graph.add_node(
                stage_node,
                type="stage",
                cycle=stage_row["cycle"],
                stage=stage_row["stage"],
                selected_at=stage_row["selected_at"],
            )
            graph.add_edge(exp_node, stage_node, relation="governance")
        pareto_rows = self._conn.execute(
            "SELECT candidate_id FROM pareto WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchall()
        for pareto in pareto_rows:
            candidate_node = f"candidate:{pareto['candidate_id']}"
            if graph.has_node(candidate_node):
                graph.add_edge(exp_node, candidate_node, relation="pareto_member")
        note_rows = self._conn.execute(
            "SELECT * FROM literature_notes WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchall()
        for note in note_rows:
            note_node = f"note:{note['id']}"
            graph.add_node(
                note_node,
                type="note",
                cycle=note["cycle"],
                content=note["content"],
                created_at=note["created_at"],
            )
            graph.add_edge(exp_node, note_node, relation="literature_note")
        return graph

    def property_graph_summary(self, experiment_id: int) -> Mapping[str, Any]:
        """Return node/edge counts and citation anchors for reporting."""

        graph = self.to_networkx(experiment_id)
        citations = [
            {
                "source_path": attrs.get("source_path"),
                "anchor": attrs.get("anchor"),
                "quote": attrs.get("quote"),
            }
            for attrs in graph.nodes.values()
            if isinstance(attrs, Mapping) and attrs.get("type") == "citation"
        ]
        return {
            "node_count": len(graph.nodes),
            "edge_count": len(graph.edges),
            "citation_count": len(citations),
            "citations": citations,
        }

    def surrogate_training_data(
        self,
        *,
        target: str = "hv",
        problem: str | None = None,
    ) -> list[tuple[Mapping[str, Any], float]]:
        """Return cached metrics + target values usable by the surrogate ranker."""

        allowed_targets = {"hv", "objective", "feasibility"}
        target_column = target if target in allowed_targets else "hv"
        rows = self._conn.execute(
            """
            SELECT c.problem, c.params_json, m.raw_json, m.hv, m.objective, m.feasibility
            FROM metrics m
            JOIN candidates c ON m.candidate_id = c.id
            ORDER BY m.id ASC
            """
        ).fetchall()
        history: list[tuple[Mapping[str, Any], float]] = []
        for row in rows:
            if problem is not None and row["problem"] != problem:
                continue
            value = row[target_column]
            if value is None:
                continue
            metrics_payload = json.loads(row["raw_json"])
            try:
                params_payload = json.loads(row["params_json"])
            except (TypeError, ValueError):
                params_payload = None
            if params_payload is not None:
                metrics_payload["candidate_params"] = params_payload
            # Inject feasibility so runner's restoration logic sees it (P1 fix)
            if row["feasibility"] is not None:
                metrics_payload["feasibility"] = float(row["feasibility"])
            history.append((metrics_payload, float(value)))
        return history


def init_db(path: str | Path) -> None:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path))
    try:
        con.executescript(SCHEMA)
        try:
            con.execute(
                "ALTER TABLE candidates ADD COLUMN design_hash TEXT NOT NULL DEFAULT ''"
            )
        except sqlite3.OperationalError:
            pass
        try:
            con.execute(
                "ALTER TABLE experiments ADD COLUMN constellaration_sha TEXT NOT NULL DEFAULT 'unknown'"
            )
        except sqlite3.OperationalError:
            pass
        try:
            con.execute("ALTER TABLE cycle_hv ADD COLUMN hv_delta REAL")
        except sqlite3.OperationalError:
            pass
        try:
            con.execute("ALTER TABLE cycle_hv ADD COLUMN hv_delta_moving_avg REAL")
        except sqlite3.OperationalError:
            pass
        con.commit()
    finally:
        con.close()
