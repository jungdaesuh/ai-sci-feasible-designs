"""Repository implementation for the AI Scientist world model."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..p3_data_plane import DataPlaneSample, summarize_data_plane
from .graph import PropertyGraph
from .schema import BudgetUsage, StageHistoryEntry, StatementRecord, init_db

DEFAULT_RELATIVE_TOLERANCE = 1e-2


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


def _parse_lineage_hashes(raw: Any) -> list[str]:
    if not isinstance(raw, str) or not raw:
        return []
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    parsed: list[str] = []
    for item in payload:
        if not isinstance(item, str):
            continue
        normalized = item.strip()
        if not normalized or normalized.lower() in {"none", "null"}:
            continue
        parsed.append(normalized)
    return parsed


def _flatten_boundary_params(params: Mapping[str, Any]) -> list[float]:
    values: list[float] = []
    for key in ("r_cos", "z_sin", "r_sin", "z_cos"):
        matrix = params.get(key)
        if not isinstance(matrix, list):
            continue
        for row in matrix:
            if not isinstance(row, list):
                continue
            for entry in row:
                if isinstance(entry, bool) or not isinstance(entry, (int, float)):
                    continue
                value = float(entry)
                if math.isfinite(value):
                    values.append(value)
    return values


def _vector_l2_distance(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right:
        return float("inf")
    length = min(len(left), len(right))
    acc = 0.0
    for idx in range(length):
        delta = float(left[idx]) - float(right[idx])
        acc += delta * delta
    if len(left) > length:
        for value in left[length:]:
            acc += float(value) * float(value)
    if len(right) > length:
        for value in right[length:]:
            acc += float(value) * float(value)
    return math.sqrt(acc)


def _delta_summary(
    left: Sequence[float],
    right: Sequence[float],
) -> tuple[float, float, int]:
    if not left and not right:
        return 0.0, 0.0, 0
    length = min(len(left), len(right))
    max_abs = 0.0
    total_abs = 0.0
    changed = 0
    for idx in range(length):
        delta = abs(float(left[idx]) - float(right[idx]))
        total_abs += delta
        if delta > max_abs:
            max_abs = delta
        if delta > 0.0:
            changed += 1
    for value in left[length:]:
        delta = abs(float(value))
        total_abs += delta
        if delta > max_abs:
            max_abs = delta
        if delta > 0.0:
            changed += 1
    for value in right[length:]:
        delta = abs(float(value))
        total_abs += delta
        if delta > max_abs:
            max_abs = delta
        if delta > 0.0:
            changed += 1
    denom = max(len(left), len(right), 1)
    return total_abs / float(denom), max_abs, changed


def _extract_delta_recipe(
    *,
    source_params: Mapping[str, Any],
    target_params: Mapping[str, Any],
    max_terms: int = 12,
) -> Mapping[str, Any] | None:
    terms: list[tuple[float, str, int, int, float]] = []
    for field in ("r_cos", "z_sin", "r_sin", "z_cos"):
        source_matrix = source_params.get(field)
        target_matrix = target_params.get(field)
        if not isinstance(source_matrix, list) or not isinstance(target_matrix, list):
            continue
        row_count = min(len(source_matrix), len(target_matrix))
        for row_idx in range(row_count):
            source_row = source_matrix[row_idx]
            target_row = target_matrix[row_idx]
            if not isinstance(source_row, list) or not isinstance(target_row, list):
                continue
            col_count = min(len(source_row), len(target_row))
            for col_idx in range(col_count):
                source_val = source_row[col_idx]
                target_val = target_row[col_idx]
                if (
                    isinstance(source_val, bool)
                    or isinstance(target_val, bool)
                    or not isinstance(source_val, (int, float))
                    or not isinstance(target_val, (int, float))
                ):
                    continue
                delta = float(target_val) - float(source_val)
                if not math.isfinite(delta) or delta == 0.0:
                    continue
                terms.append((abs(delta), field, row_idx, col_idx, delta))
    if not terms:
        return None
    terms.sort(key=lambda item: item[0], reverse=True)
    top_terms = terms[: max(1, int(max_terms))]
    return {
        "type": "sparse_additive_delta",
        "changes": [
            {
                "field": field,
                "row": row_idx,
                "col": col_idx,
                "delta": delta,
            }
            for _, field, row_idx, col_idx, delta in top_terms
        ],
        "max_abs_delta": float(top_terms[0][0]),
        "term_count": len(top_terms),
    }


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
        hv_exists = 1 if feasible_count > 0 else 0

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
        lineage_parent_hashes: Sequence[str] | None = None,
        novelty_score: float | None = None,
        operator_family: str | None = None,
        model_route: str | None = None,
        commit: bool = True,
    ) -> tuple[int, int]:
        params_json = json.dumps(_normalize_to_json(params), separators=(",", ":"))
        lineage_json = json.dumps(
            [str(parent) for parent in (lineage_parent_hashes or [])],
            separators=(",", ":"),
        )
        novelty_value = float(novelty_score) if novelty_score is not None else None
        operator_family_value = str(operator_family or "")
        model_route_value = str(model_route or "")
        cursor = self._conn.execute(
            """
            INSERT INTO candidates
            (experiment_id, problem, params_json, seed, status, design_hash, lineage_parent_hashes_json, novelty_score, operator_family, model_route)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_id,
                problem,
                params_json,
                seed,
                status,
                design_hash,
                lineage_json,
                novelty_value,
                operator_family_value,
                model_route_value,
            ),
        )
        candidate_id = cursor.lastrowid
        assert candidate_id is not None
        # H3 FIX: Merge constraint_margins into metrics payload so recent_failures()
        # can access constraint violation data for the "learning from failures" loop.
        # Previously only evaluation["metrics"] was stored, but recent_failures()
        # expects constraint_margins to be inside raw_json.
        metrics_payload = dict(evaluation.get("metrics", {}))
        if "constraint_margins" in evaluation:
            metrics_payload["constraint_margins"] = evaluation["constraint_margins"]
        for key in (
            "error",
            "error_normalized",
            "failure_label",
            "failure_source",
            "failure_signature",
            "vmec_status",
            "stage",
        ):
            value = evaluation.get(key)
            if value is not None:
                metrics_payload[key] = value
        metrics_id = self.log_metrics(
            candidate_id,
            metrics_payload,
            feasibility=float(evaluation.get("feasibility", 0.0)),
            objective=evaluation.get("objective"),
            hv=evaluation.get(
                "gradient_proxy", evaluation.get("hv")
            ),  # H2 FIX: Use gradient_proxy with fallback
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

    def log_model_router_reward_event(
        self,
        *,
        experiment_id: int,
        problem: str,
        model_route: str,
        window_size: int,
        previous_feasible_yield: float,
        current_feasible_yield: float,
        previous_hv: float,
        current_hv: float,
        reward: float,
        reward_components: Mapping[str, Any] | None = None,
        created_at: str | None = None,
        commit: bool = True,
    ) -> int:
        timestamp = created_at or datetime.now(timezone.utc).isoformat()
        payload = json.dumps(
            _normalize_to_json(reward_components or {}), separators=(",", ":")
        )
        cursor = self._conn.execute(
            """
            INSERT INTO model_router_reward_events
            (experiment_id, problem, model_route, window_size, previous_feasible_yield,
             current_feasible_yield, previous_hv, current_hv, reward, reward_components_json,
             created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_id,
                str(problem),
                str(model_route),
                int(window_size),
                float(previous_feasible_yield),
                float(current_feasible_yield),
                float(previous_hv),
                float(current_hv),
                float(reward),
                payload,
                timestamp,
            ),
        )
        event_id = cursor.lastrowid
        assert event_id is not None
        if commit:
            self._conn.commit()
        return event_id

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

    def log_scratchpad_event(
        self,
        experiment_id: int,
        cycle: int,
        step: int,
        planner_intent: Mapping[str, Any] | None,
        aso_action: str,
        intent_agreement: str,
        override_reason: str | None,
        diagnostics: Mapping[str, Any],
        outcome: Mapping[str, Any],
        *,
        created_at: str | None = None,
        commit: bool = True,
    ) -> int:
        timestamp = created_at or datetime.now(timezone.utc).isoformat()
        cursor = self._conn.execute(
            """
            INSERT INTO scratchpad_events (
                experiment_id,
                cycle,
                step,
                planner_intent_json,
                aso_action,
                intent_agreement,
                override_reason,
                diagnostics_json,
                outcome_json,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_id,
                cycle,
                step,
                json.dumps(
                    _normalize_to_json(planner_intent or {}), separators=(",", ":")
                ),
                aso_action,
                intent_agreement,
                override_reason,
                json.dumps(_normalize_to_json(diagnostics), separators=(",", ":")),
                json.dumps(_normalize_to_json(outcome), separators=(",", ":")),
                timestamp,
            ),
        )
        event_id = cursor.lastrowid
        assert event_id is not None
        if commit:
            self._conn.commit()
        return event_id

    def scratchpad_cycle_summary(
        self,
        experiment_id: int,
        cycle: int,
        *,
        limit: int = 20,
    ) -> Mapping[str, Any]:
        rows = self._conn.execute(
            """
            SELECT step,
                   planner_intent_json,
                   aso_action,
                   intent_agreement,
                   override_reason,
                   diagnostics_json,
                   outcome_json,
                   created_at
            FROM scratchpad_events
            WHERE experiment_id = ? AND cycle = ?
            ORDER BY step DESC, id DESC
            LIMIT ?
            """,
            (experiment_id, cycle, limit),
        ).fetchall()
        events = [
            {
                "step": int(row["step"]),
                "planner_intent": json.loads(row["planner_intent_json"]),
                "aso_action": row["aso_action"],
                "intent_agreement": row["intent_agreement"],
                "override_reason": row["override_reason"],
                "diagnostics": json.loads(row["diagnostics_json"]),
                "outcome": json.loads(row["outcome_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]
        events.reverse()
        action_counts: dict[str, int] = {}
        for event in events:
            action = str(event["aso_action"])
            action_counts[action] = action_counts.get(action, 0) + 1

        return {
            "cycle": cycle,
            "event_count": len(events),
            "action_counts": action_counts,
            "events": events,
        }

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

    def recent_candidate_snapshots(
        self,
        experiment_id: int,
        problem: str,
        *,
        limit: int = 128,
    ) -> list[Mapping[str, Any]]:
        """Return recent candidate snapshots with params + metrics for staged planning."""

        rows = self._conn.execute(
            """
            SELECT c.id,
                   c.design_hash,
                   c.seed,
                   c.params_json,
                   c.lineage_parent_hashes_json,
                   c.novelty_score,
                   c.operator_family,
                   c.model_route,
                   m.feasibility,
                   m.objective,
                   m.is_feasible,
                   m.raw_json
            FROM metrics m
            JOIN candidates c ON m.candidate_id = c.id
            WHERE c.experiment_id = ? AND c.problem = ?
            ORDER BY m.id DESC
            LIMIT ?
            """,
            (experiment_id, problem, int(limit)),
        ).fetchall()

        snapshots: list[Mapping[str, Any]] = []
        for row in rows:
            params = json.loads(str(row["params_json"]))
            raw_metrics = json.loads(str(row["raw_json"]))
            nested_metrics = raw_metrics.get("metrics")
            metrics = (
                nested_metrics
                if isinstance(nested_metrics, Mapping)
                else {
                    key: value
                    for key, value in raw_metrics.items()
                    if key != "constraint_margins"
                }
            )
            margins = raw_metrics.get("constraint_margins", {})

            constraint_margins: dict[str, float] = {}
            if isinstance(margins, Mapping):
                for key, value in margins.items():
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        continue
                    value_f = float(value)
                    if value_f > 0.0 and math.isfinite(value_f):
                        constraint_margins[str(key)] = value_f

            lineage_parent_hashes = _parse_lineage_hashes(
                row["lineage_parent_hashes_json"]
            )

            snapshots.append(
                {
                    "candidate_id": int(row["id"]),
                    "design_hash": str(row["design_hash"] or ""),
                    "seed": int(row["seed"] or 0),
                    "params": params if isinstance(params, Mapping) else {},
                    "feasibility": float(row["feasibility"]),
                    "objective": (
                        float(row["objective"])
                        if row["objective"] is not None
                        else None
                    ),
                    "is_feasible": int(row["is_feasible"]) == 1,
                    "constraint_margins": constraint_margins,
                    "metrics": metrics if isinstance(metrics, Mapping) else {},
                    "operator_family": str(row["operator_family"] or ""),
                    "model_route": str(row["model_route"] or ""),
                    "lineage_parent_hashes": lineage_parent_hashes,
                    "novelty_score": (
                        float(row["novelty_score"])
                        if row["novelty_score"] is not None
                        else None
                    ),
                }
            )
        return snapshots

    def select_parent_group_performance_novelty(
        self,
        experiment_id: int,
        problem: str,
        *,
        group_size: int = 2,
        limit: int = 128,
        near_feasibility_threshold: float = 0.25,
        worst_constraint: str | None = None,
        focus_constraint_margin: float | None = None,
        leverage_weight: float = 0.35,
    ) -> list[Mapping[str, Any]]:
        """Select parent group by combined performance + novelty score."""

        snapshots = self.recent_candidate_snapshots(
            experiment_id=experiment_id,
            problem=problem,
            limit=max(limit, group_size * 4),
        )
        if not snapshots:
            return []

        problem_key = (problem or "").lower()
        focus_margin = (
            float(focus_constraint_margin)
            if isinstance(focus_constraint_margin, (int, float))
            else None
        )
        leverage_w = max(0.0, float(leverage_weight))
        scored: list[tuple[float, float, Mapping[str, Any]]] = []
        for snapshot in snapshots:
            feasibility = float(snapshot.get("feasibility", float("inf")))
            objective = snapshot.get("objective")
            is_feasible = bool(snapshot.get("is_feasible", False))
            novelty = snapshot.get("novelty_score")
            novelty_score = float(novelty) if isinstance(novelty, (int, float)) else 0.0
            leverage_score = 0.0
            margins = snapshot.get("constraint_margins")
            if isinstance(worst_constraint, str) and worst_constraint:
                margin_value = 0.0
                if isinstance(margins, Mapping):
                    raw_margin = margins.get(worst_constraint)
                    if isinstance(raw_margin, (int, float)) and not isinstance(
                        raw_margin, bool
                    ):
                        margin_value = max(0.0, float(raw_margin))
                if focus_margin is not None and focus_margin > 0.0:
                    leverage_score = (
                        max(0.0, focus_margin - margin_value) / focus_margin
                    )
                else:
                    leverage_score = 1.0 / (1.0 + margin_value)
            if is_feasible:
                if objective is None:
                    performance_score = 1.0
                else:
                    objective_value = float(objective)
                    if problem_key.startswith("p1"):
                        performance_score = 1.0 / (1.0 + max(0.0, objective_value))
                    else:
                        performance_score = max(0.0, objective_value)
            elif feasibility <= near_feasibility_threshold:
                performance_score = 0.5 / max(feasibility, 1e-6)
            else:
                performance_score = 1.0 / (1.0 + max(feasibility, 0.0))
            combined = (
                performance_score
                + (0.25 * max(0.0, novelty_score))
                + (leverage_w * max(0.0, leverage_score))
            )
            scored.append((combined, leverage_score, snapshot))

        scored.sort(key=lambda item: float(item[0]), reverse=True)
        selected: list[Mapping[str, Any]] = []
        seen_hashes: set[str] = set()
        for score, leverage_score, snapshot in scored:
            design_hash = str(snapshot.get("design_hash", ""))
            if not design_hash or design_hash in seen_hashes:
                continue
            enriched = dict(snapshot)
            enriched["parent_selection_score"] = float(score)
            enriched["constraint_leverage_score"] = float(leverage_score)
            selected.append(enriched)
            seen_hashes.add(design_hash)
            if len(selected) >= group_size:
                break
        return selected

    def nearest_case_deltas(
        self,
        experiment_id: int,
        problem: str,
        *,
        seed_params: Mapping[str, Any],
        limit: int = 4,
        near_feasibility_threshold: float = 0.25,
        include_recipes: bool = True,
    ) -> list[Mapping[str, Any]]:
        """Return nearest feasible/near-feasible historical cases with parameter deltas."""

        baseline_vector = _flatten_boundary_params(seed_params)
        if not baseline_vector:
            return []

        snapshots = self.recent_candidate_snapshots(
            experiment_id=experiment_id,
            problem=problem,
            limit=max(64, limit * 24),
        )
        nearest: list[tuple[float, Mapping[str, Any], str]] = []
        for snapshot in snapshots:
            snapshot_params = snapshot.get("params")
            if not isinstance(snapshot_params, Mapping):
                continue
            feasibility = float(snapshot.get("feasibility", float("inf")))
            is_feasible = bool(snapshot.get("is_feasible", False))
            if is_feasible:
                bucket = "feasible"
            elif feasibility <= near_feasibility_threshold:
                bucket = "near_feasible"
            else:
                continue
            vector = _flatten_boundary_params(snapshot_params)
            if not vector:
                continue
            distance = _vector_l2_distance(baseline_vector, vector)
            mean_abs_delta, max_abs_delta, changed_count = _delta_summary(
                baseline_vector, vector
            )
            nearest.append(
                (
                    distance,
                    {
                        "design_hash": str(snapshot.get("design_hash", "")),
                        "feasibility": feasibility,
                        "objective": snapshot.get("objective"),
                        "worst_constraint": snapshot.get("constraint_margins", {}),
                        "lineage_parent_hashes": list(
                            snapshot.get("lineage_parent_hashes", [])
                        ),
                        "distance_l2": distance,
                        "delta_summary": {
                            "mean_abs_delta": mean_abs_delta,
                            "max_abs_delta": max_abs_delta,
                            "changed_coefficients": changed_count,
                        },
                        "delta_recipe": (
                            _extract_delta_recipe(
                                source_params=seed_params,
                                target_params=snapshot_params,
                            )
                            if include_recipes
                            else None
                        ),
                    },
                    bucket,
                )
            )

        nearest.sort(key=lambda item: float(item[0]))
        selected: list[Mapping[str, Any]] = []
        feasible_taken = 0
        near_taken = 0
        for _, payload, bucket in nearest:
            if bucket == "feasible" and feasible_taken >= max(1, limit // 2):
                continue
            if bucket == "near_feasible" and near_taken >= max(1, limit // 2):
                continue
            item = dict(payload)
            item["bucket"] = bucket
            selected.append(item)
            if bucket == "feasible":
                feasible_taken += 1
            else:
                near_taken += 1
            if len(selected) >= limit:
                break
        return selected

    def ancestor_chains(
        self,
        experiment_id: int,
        problem: str,
        design_hashes: Sequence[str],
        *,
        max_depth: int = 3,
        max_ancestors_per_design: int = 8,
    ) -> list[Mapping[str, Any]]:
        """Return compact parent-chain context for selected design hashes."""

        rows = self._conn.execute(
            """
            SELECT c.design_hash,
                   c.seed,
                   c.operator_family,
                   c.model_route,
                   c.lineage_parent_hashes_json,
                   m.feasibility,
                   m.objective,
                   m.raw_json
            FROM candidates c
            LEFT JOIN metrics m ON m.candidate_id = c.id
            WHERE c.experiment_id = ? AND c.problem = ?
            ORDER BY c.id DESC, m.id DESC
            """,
            (experiment_id, problem),
        ).fetchall()

        by_hash: dict[str, dict[str, Any]] = {}
        for row in rows:
            design_hash = str(row["design_hash"] or "")
            if not design_hash or design_hash in by_hash:
                continue
            raw_payload = row["raw_json"]
            if isinstance(raw_payload, str) and raw_payload:
                try:
                    metrics_raw = json.loads(raw_payload)
                except (TypeError, json.JSONDecodeError):
                    metrics_raw = {}
            else:
                metrics_raw = {}
            margins_raw = (
                metrics_raw.get("constraint_margins", {})
                if isinstance(metrics_raw, Mapping)
                else {}
            )
            violations: dict[str, float] = {}
            if isinstance(margins_raw, Mapping):
                for key, value in margins_raw.items():
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        continue
                    value_f = float(value)
                    if value_f > 0.0:
                        violations[str(key)] = value_f
            if violations:
                worst_constraint = max(
                    violations,
                    key=lambda name: violations[name],
                )
                worst_violation = float(violations[worst_constraint])
            else:
                worst_constraint = None
                worst_violation = 0.0

            by_hash[design_hash] = {
                "design_hash": design_hash,
                "seed": int(row["seed"] or 0),
                "operator_family": str(row["operator_family"] or ""),
                "model_route": str(row["model_route"] or ""),
                "lineage_parent_hashes": _parse_lineage_hashes(
                    row["lineage_parent_hashes_json"]
                ),
                "feasibility": (
                    float(row["feasibility"])
                    if row["feasibility"] is not None
                    else None
                ),
                "objective": (
                    float(row["objective"]) if row["objective"] is not None else None
                ),
                "worst_constraint": worst_constraint,
                "worst_constraint_violation": worst_violation,
            }

        unique_hashes: list[str] = []
        seen_targets: set[str] = set()
        for design_hash in design_hashes:
            key = str(design_hash)
            if key and key not in seen_targets:
                unique_hashes.append(key)
                seen_targets.add(key)

        chains: list[Mapping[str, Any]] = []
        for target_hash in unique_hashes:
            target = by_hash.get(target_hash)
            parent_frontier: list[tuple[str, int]] = []
            if target is not None:
                for parent_hash in target["lineage_parent_hashes"]:
                    parent_frontier.append((str(parent_hash), 1))
            ancestors: list[Mapping[str, Any]] = []
            visited: set[str] = set()
            while parent_frontier and len(ancestors) < max_ancestors_per_design:
                current_hash, depth = parent_frontier.pop(0)
                if current_hash in visited or depth > max_depth:
                    continue
                visited.add(current_hash)
                record = by_hash.get(current_hash)
                if record is None:
                    ancestors.append(
                        {
                            "design_hash": current_hash,
                            "depth": depth,
                            "status": "missing",
                        }
                    )
                    continue
                ancestors.append(
                    {
                        "design_hash": record["design_hash"],
                        "depth": depth,
                        "feasibility": record["feasibility"],
                        "objective": record["objective"],
                        "worst_constraint": record["worst_constraint"],
                        "worst_constraint_violation": record[
                            "worst_constraint_violation"
                        ],
                        "operator_family": record["operator_family"],
                        "model_route": record["model_route"],
                    }
                )
                if depth < max_depth:
                    for parent_hash in record["lineage_parent_hashes"]:
                        parent_frontier.append((str(parent_hash), depth + 1))
            chains.append(
                {
                    "target_design_hash": target_hash,
                    "ancestors": ancestors,
                }
            )
        return chains

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

                failure_entry: dict[str, Any] = {
                    "params": params,
                    "feasibility": float(row["feasibility"]),
                    "violations": violations,
                }
                for key in (
                    "error",
                    "error_normalized",
                    "failure_label",
                    "failure_source",
                    "failure_signature",
                    "vmec_status",
                    "stage",
                ):
                    value = metrics_raw.get(key)
                    if value is not None:
                        failure_entry[key] = value

                failures.append(failure_entry)
            except (json.JSONDecodeError, ValueError):
                continue

        return failures

    def recent_experience_pack(
        self,
        experiment_id: int,
        problem: str,
        *,
        limit_per_bucket: int = 3,
        near_feasibility_threshold: float = 0.1,
        delta_window: int = 5,
    ) -> Mapping[str, Any]:
        """Return balanced recent experience slices plus compact feedback adapters."""

        rows = self._conn.execute(
            """
            SELECT c.design_hash,
                   c.seed,
                   c.operator_family,
                   c.model_route,
                   m.raw_json,
                   m.feasibility,
                   m.objective,
                   m.is_feasible
            FROM metrics m
            JOIN candidates c ON m.candidate_id = c.id
            WHERE c.experiment_id = ? AND c.problem = ?
            ORDER BY m.id DESC
            LIMIT ?
            """,
            (experiment_id, problem, max(24, limit_per_bucket * 20)),
        ).fetchall()

        recent_successes: list[Mapping[str, Any]] = []
        recent_near_successes: list[Mapping[str, Any]] = []
        recent_failures: list[Mapping[str, Any]] = []
        timeline: list[Mapping[str, Any]] = []

        for row in rows:
            try:
                metrics_raw = json.loads(row["raw_json"])
            except (json.JSONDecodeError, TypeError):
                metrics_raw = {}

            constraint_margins = metrics_raw.get("constraint_margins", {})
            violations: dict[str, float] = {}
            for name, value in dict(constraint_margins).items():
                try:
                    magnitude = float(value)
                except (TypeError, ValueError):
                    continue
                if magnitude > 0.0:
                    violations[str(name)] = magnitude

            total_violation = sum(violations.values())
            if total_violation > 0.0:
                normalized_violations = {
                    name: magnitude / total_violation
                    for name, magnitude in sorted(
                        violations.items(), key=lambda item: item[1], reverse=True
                    )
                }
                worst_constraint = max(violations.items(), key=lambda item: item[1])[0]
                worst_constraint_violation = float(violations[worst_constraint])
            else:
                normalized_violations = {}
                worst_constraint = None
                worst_constraint_violation = 0.0

            objective_raw = row["objective"]
            objective = float(objective_raw) if objective_raw is not None else None
            feasibility = float(row["feasibility"])
            entry: dict[str, Any] = {
                "design_hash": str(row["design_hash"] or ""),
                "seed": int(row["seed"] or 0),
                "feasibility": feasibility,
                "objective": objective,
                "normalized_violations": normalized_violations,
                "worst_constraint": worst_constraint,
                "worst_constraint_violation": worst_constraint_violation,
                "stage": metrics_raw.get("stage"),
                "operator_family": str(row["operator_family"] or ""),
                "model_route": str(row["model_route"] or ""),
            }
            for key in (
                "error",
                "error_normalized",
                "failure_label",
                "failure_source",
                "failure_signature",
                "vmec_status",
            ):
                value = metrics_raw.get(key)
                if value is not None:
                    entry[key] = value

            timeline.append(entry)

            is_feasible = int(row["is_feasible"]) == 1
            if is_feasible:
                if len(recent_successes) < limit_per_bucket:
                    recent_successes.append(entry)
                continue

            if feasibility <= near_feasibility_threshold:
                if len(recent_near_successes) < limit_per_bucket:
                    recent_near_successes.append(entry)
                continue

            if len(recent_failures) < limit_per_bucket:
                recent_failures.append(entry)

            if (
                len(recent_successes) >= limit_per_bucket
                and len(recent_near_successes) >= limit_per_bucket
                and len(recent_failures) >= limit_per_bucket
            ):
                break

        worst_constraint_sequence = [
            str(item["worst_constraint"])
            for item in timeline
            if item.get("worst_constraint")
        ]
        worst_constraint_counts: dict[str, int] = {}
        for name in worst_constraint_sequence:
            worst_constraint_counts[name] = worst_constraint_counts.get(name, 0) + 1

        recent_effective_deltas: list[Mapping[str, Any]] = []
        for newer, older in zip(timeline, timeline[1:]):
            feasibility_delta = float(newer["feasibility"]) - float(
                older["feasibility"]
            )
            objective_new = newer.get("objective")
            objective_old = older.get("objective")
            objective_delta = None
            if objective_new is not None and objective_old is not None:
                objective_delta = float(objective_new) - float(objective_old)
            violation_delta = float(newer["worst_constraint_violation"]) - float(
                older["worst_constraint_violation"]
            )
            recent_effective_deltas.append(
                {
                    "from_design_hash": older.get("design_hash"),
                    "to_design_hash": newer.get("design_hash"),
                    "feasibility_delta": feasibility_delta,
                    "objective_delta": objective_delta,
                    "worst_constraint_delta": violation_delta,
                }
            )
            if len(recent_effective_deltas) >= delta_window:
                break

        return {
            "recent_successes": recent_successes,
            "recent_near_successes": recent_near_successes,
            "recent_failures": recent_failures,
            "feedback_adapter": {
                "worst_constraint_trend": {
                    "sequence": worst_constraint_sequence[:delta_window],
                    "counts": worst_constraint_counts,
                },
                "recent_effective_deltas": recent_effective_deltas,
            },
        }

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
        node_by_design_hash: dict[str, str] = {}
        for candidate in candidate_rows:
            candidate_node = f"candidate:{candidate['id']}"
            graph.add_node(
                candidate_node,
                type="candidate",
                problem=candidate["problem"],
                status=candidate["status"],
                seed=candidate["seed"],
                params=candidate["params_json"],
                operator_family=candidate["operator_family"],
                model_route=candidate["model_route"],
                novelty_score=candidate["novelty_score"],
                lineage_parent_hashes_json=candidate["lineage_parent_hashes_json"],
            )
            design_hash = str(candidate["design_hash"] or "")
            if design_hash:
                node_by_design_hash[design_hash] = candidate_node
            graph.add_edge(exp_node, candidate_node, relation="contains")
        for candidate in candidate_rows:
            child_node = f"candidate:{candidate['id']}"
            for parent_hash in _parse_lineage_hashes(
                candidate["lineage_parent_hashes_json"]
            ):
                parent_node = node_by_design_hash.get(parent_hash)
                if parent_node is None:
                    parent_node = f"lineage_hash:{parent_hash}"
                    if not graph.has_node(parent_node):
                        graph.add_node(
                            parent_node,
                            type="lineage_stub",
                            design_hash=parent_hash,
                        )
                graph.add_edge(parent_node, child_node, relation="lineage_parent_of")
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

    def candidate_data_plane_summary(
        self,
        experiment_id: int,
        *,
        problem: str = "p3",
        limit: int = 500,
        novelty_reject_threshold: float = 0.05,
    ) -> Mapping[str, Any]:
        rows = self._conn.execute(
            """
            SELECT lineage_parent_hashes_json, novelty_score, operator_family, model_route
            FROM candidates
            WHERE experiment_id = ? AND problem = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (experiment_id, problem, int(limit)),
        ).fetchall()

        samples: list[DataPlaneSample] = []

        for row in rows:
            lineage_raw = row["lineage_parent_hashes_json"]
            try:
                lineage_payload = json.loads(lineage_raw) if lineage_raw else []
            except (TypeError, json.JSONDecodeError):
                lineage_payload = []

            novelty = row["novelty_score"]
            novelty_value = float(novelty) if novelty is not None else None
            has_lineage = isinstance(lineage_payload, list) and len(lineage_payload) > 0
            samples.append(
                DataPlaneSample(
                    has_lineage=has_lineage,
                    novelty_score=novelty_value,
                    operator_family=str(row["operator_family"] or "unknown"),
                    model_route=str(row["model_route"] or "unknown"),
                )
            )

        summary = summarize_data_plane(
            samples,
            novelty_reject_threshold=float(novelty_reject_threshold),
        )
        summary["model_router_reward"] = self.model_router_reward_summary(
            experiment_id,
            problem=problem,
        )
        return summary

    def model_router_reward_summary(
        self,
        experiment_id: int,
        *,
        problem: str = "p3",
        limit: int = 200,
        reward_eligible_only: bool = False,
    ) -> Mapping[str, Any]:
        total_row = self._conn.execute(
            """
            SELECT COUNT(*) AS n
            FROM model_router_reward_events
            WHERE experiment_id = ? AND problem = ?
            """,
            (experiment_id, str(problem)),
        ).fetchone()
        total_events_all = int(total_row["n"]) if total_row is not None else 0
        rows = self._conn.execute(
            """
            SELECT model_route, reward, reward_components_json
            FROM model_router_reward_events
            WHERE experiment_id = ? AND problem = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (experiment_id, str(problem), int(limit)),
        ).fetchall()

        eligible_rows: list[sqlite3.Row] = []
        if reward_eligible_only:
            eligible_total_rows = self._conn.execute(
                """
                SELECT reward_components_json
                FROM model_router_reward_events
                WHERE experiment_id = ? AND problem = ?
                """,
                (experiment_id, str(problem)),
            ).fetchall()
            total_events = 0
            for row in eligible_total_rows:
                payload_raw = row["reward_components_json"]
                if isinstance(payload_raw, str) and payload_raw:
                    try:
                        payload = json.loads(payload_raw)
                    except (TypeError, json.JSONDecodeError):
                        payload = {}
                else:
                    payload = {}
                if isinstance(payload, Mapping) and bool(
                    payload.get("reward_eligible")
                ):
                    total_events += 1
            for row in rows:
                payload_raw = row["reward_components_json"]
                if isinstance(payload_raw, str) and payload_raw:
                    try:
                        payload = json.loads(payload_raw)
                    except (TypeError, json.JSONDecodeError):
                        payload = {}
                else:
                    payload = {}
                if isinstance(payload, Mapping) and bool(
                    payload.get("reward_eligible")
                ):
                    eligible_rows.append(row)
            rows_to_use = eligible_rows
        else:
            total_events = total_events_all
            rows_to_use = list(rows)
        if not rows_to_use:
            return {
                "event_count": total_events,
                "sampled_event_count": 0,
                "avg_reward": None,
                "last_reward": None,
                "model_routes": {},
            }

        rewards: list[float] = []
        route_counts: dict[str, int] = {}
        for row in rows_to_use:
            reward = float(row["reward"])
            rewards.append(reward)
            route = str(row["model_route"] or "unknown")
            route_counts[route] = route_counts.get(route, 0) + 1

        return {
            "event_count": total_events,
            "sampled_event_count": len(rows_to_use),
            "avg_reward": float(sum(rewards) / len(rewards)),
            "last_reward": rewards[0],
            "model_routes": dict(
                sorted(route_counts.items(), key=lambda item: item[1], reverse=True)
            ),
        }

    def model_router_reward_eligible_history(
        self,
        experiment_id: int,
        *,
        problem: str = "p3",
        limit: int = 200,
    ) -> list[Mapping[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT model_route, window_size, reward, reward_components_json, created_at
            FROM model_router_reward_events
            WHERE experiment_id = ? AND problem = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (experiment_id, str(problem), int(limit)),
        ).fetchall()
        history: list[Mapping[str, Any]] = []
        for row in rows:
            payload_raw = row["reward_components_json"]
            if isinstance(payload_raw, str) and payload_raw:
                try:
                    payload = json.loads(payload_raw)
                except (TypeError, json.JSONDecodeError):
                    payload = {}
            else:
                payload = {}
            if not isinstance(payload, Mapping):
                payload = {}
            if not bool(payload.get("reward_eligible")):
                continue
            history.append(
                {
                    "model_route": str(row["model_route"] or "unknown"),
                    "window_size": int(row["window_size"]),
                    "reward": float(row["reward"]),
                    "reward_components": dict(payload),
                    "created_at": str(row["created_at"] or ""),
                }
            )
        return history

    def surrogate_training_data(
        self,
        *,
        target: str = "hv",
        problem: str | None = None,
        experiment_id: int | None = None,
    ) -> list[tuple[Mapping[str, Any], float]]:
        """Return cached metrics + target values usable by the surrogate ranker.

        A7 FIX: Query now applies WHERE filters in SQL to avoid fetching
        the entire metrics table, preventing OOM for large experiment history.
        """

        # H2 FIX: Accept "gradient_proxy" as alias for DB column "hv"
        # The DB column remains "hv" for backward compat with existing data
        target_column = "hv" if target in {"hv", "gradient_proxy"} else target
        allowed_columns = {"hv", "objective", "feasibility"}
        target_column = target_column if target_column in allowed_columns else "hv"

        # A7 FIX: Build query with SQL-side filtering instead of Python-side
        where_clauses: list[str] = []
        params: list[Any] = []

        if problem is not None:
            where_clauses.append("c.problem = ?")
            params.append(problem)

        if experiment_id is not None:
            where_clauses.append("c.experiment_id = ?")
            params.append(experiment_id)

        where_sql = (" WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

        query = f"""
            SELECT c.problem, c.params_json, c.status, m.raw_json, m.hv, m.objective, m.feasibility,
                   c.experiment_id
            FROM metrics m
            JOIN candidates c ON m.candidate_id = c.id{where_sql}
            ORDER BY m.id ASC
            """
        rows = self._conn.execute(query, params).fetchall()

        history: list[tuple[Mapping[str, Any], float]] = []
        for row in rows:
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

            # A4.3 FIX Helper: Inject stage/status so surrogate knows fidelity
            if row["status"] is not None:
                metrics_payload["_stage"] = row["status"]

            history.append((metrics_payload, float(value)))
        return history
