"""Database schema and initialization for the AI Scientist world model."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path


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


@dataclass(frozen=True)
class BudgetUsage:
    screen_evals: int
    promoted_evals: int
    high_fidelity_evals: int
    wall_seconds: float


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
