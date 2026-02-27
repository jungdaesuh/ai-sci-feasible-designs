#!/usr/bin/env python
# ruff: noqa: E402
"""Fixed-budget A/B validation for P3 governor static vs adaptive modes.

This script compares two P3 experiments in the WorldModel DB:
- static governor arm (baseline)
- adaptive governor arm (candidate)

It computes fixed-budget metrics from deterministic high-fidelity eval rows:
- feasible yield per 100 evaluations
- hypervolume at budget and final hypervolume
- best feasible L_gradB at budget

Outputs can be written as JSON and Markdown.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ai_scientist.memory.schema import init_db
from ai_scientist.p3_data_plane import DataPlaneSample, summarize_data_plane

_NOVELTY_REJECT_THRESHOLD = 0.05
_M24_MIN_BUDGET = 20


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect(db_path: Path) -> sqlite3.Connection:
    init_db(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


@dataclass(frozen=True)
class EvalRow:
    metric_id: int
    is_feasible: bool
    feasibility: float
    objective: float | None
    aspect: float | None
    novelty_score: float | None
    operator_family: str
    model_route: str
    has_lineage: bool
    has_eval_error: bool


@dataclass(frozen=True)
class ArmSummary:
    experiment_id: int
    eval_count_total: int
    eval_count_budget: int
    feasible_count_budget: int
    feasible_yield_per_100: float
    hv_at_budget: float
    hv_final: float
    best_lgradb_feasible_at_budget: float | None
    route_summary_at_budget: dict
    error_count_budget: int = 0


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _has_lineage(lineage_json: object) -> bool:
    if not isinstance(lineage_json, str) or not lineage_json:
        return False
    try:
        payload = json.loads(lineage_json)
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    return isinstance(payload, list) and len(payload) > 0


def _extract_aspect(raw_json: str) -> float | None:
    try:
        payload = json.loads(raw_json)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        return None
    return _safe_float(metrics.get("aspect_ratio"))


def _has_eval_error(raw_json: str) -> bool:
    try:
        payload = json.loads(raw_json)
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    error = payload.get("error")
    return isinstance(error, str) and bool(error.strip())


def _fetch_eval_rows(conn: sqlite3.Connection, *, experiment_id: int) -> list[EvalRow]:
    rows = conn.execute(
        """
        SELECT
            m.id AS metric_id,
            m.is_feasible AS is_feasible,
            m.feasibility AS feasibility,
            m.objective AS objective,
            m.raw_json AS raw_json,
            c.novelty_score AS novelty_score,
            c.operator_family AS operator_family,
            c.model_route AS model_route,
            c.lineage_parent_hashes_json AS lineage_parent_hashes_json
        FROM metrics m
        JOIN candidates c ON c.id = m.candidate_id
        WHERE c.experiment_id = ?
        ORDER BY m.id ASC
        """,
        (int(experiment_id),),
    ).fetchall()

    out: list[EvalRow] = []
    for row in rows:
        raw_json = str(row["raw_json"])
        out.append(
            EvalRow(
                metric_id=int(row["metric_id"]),
                is_feasible=bool(int(row["is_feasible"])),
                feasibility=float(row["feasibility"]),
                objective=_safe_float(row["objective"]),
                aspect=_extract_aspect(raw_json),
                novelty_score=_safe_float(row["novelty_score"]),
                operator_family=str(row["operator_family"] or "unknown"),
                model_route=str(row["model_route"] or "unknown"),
                has_lineage=_has_lineage(row["lineage_parent_hashes_json"]),
                has_eval_error=_has_eval_error(raw_json),
            )
        )
    return out


def _pareto_front_minimize(
    points: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    valid = sorted(points, key=lambda pair: (pair[0], pair[1]))
    front: list[tuple[float, float]] = []
    best_y = float("inf")
    for x, y in valid:
        if y < best_y:
            front.append((x, y))
            best_y = y
    return front


def _hypervolume_2d_minimize(
    points: list[tuple[float, float]], *, ref_x: float, ref_y: float
) -> float:
    clipped = [(x, y) for x, y in points if x < ref_x and y < ref_y]
    if not clipped:
        return 0.0
    front = _pareto_front_minimize(clipped)
    area = 0.0
    prev_y = ref_y
    for x, y in front:
        width = ref_x - x
        height = prev_y - y
        if width > 0.0 and height > 0.0:
            area += width * height
        if y < prev_y:
            prev_y = y
    return float(area)


def _compute_hv(
    points: list[tuple[float, float]], *, ref_x: float, ref_y: float
) -> float:
    transformed = [(-float(lgradb), float(aspect)) for lgradb, aspect in points]
    return _hypervolume_2d_minimize(transformed, ref_x=ref_x, ref_y=ref_y)


def _route_summary(rows: list[EvalRow]) -> dict:
    samples = [
        DataPlaneSample(
            has_lineage=row.has_lineage,
            novelty_score=row.novelty_score,
            operator_family=row.operator_family or "unknown",
            model_route=row.model_route or "unknown",
        )
        for row in rows
    ]
    return summarize_data_plane(
        samples,
        novelty_reject_threshold=_NOVELTY_REJECT_THRESHOLD,
    )


def _build_arm_summary(
    *,
    experiment_id: int,
    rows: list[EvalRow],
    budget: int,
    ref_x: float,
    ref_y: float,
) -> ArmSummary:
    budget_rows = rows[:budget]
    feasible_budget = [row for row in budget_rows if row.is_feasible]
    feasible_points_budget = [
        (row.objective, row.aspect)
        for row in feasible_budget
        if row.objective is not None and row.aspect is not None
    ]
    feasible_points_final = [
        (row.objective, row.aspect)
        for row in rows
        if row.is_feasible and row.objective is not None and row.aspect is not None
    ]
    best_budget = (
        max(point[0] for point in feasible_points_budget)
        if feasible_points_budget
        else None
    )
    feasible_count_budget = len(feasible_budget)
    error_count_budget = sum(1 for row in budget_rows if row.has_eval_error)
    feasible_yield_per_100 = (
        (100.0 * float(feasible_count_budget) / float(budget)) if budget > 0 else 0.0
    )
    return ArmSummary(
        experiment_id=int(experiment_id),
        eval_count_total=len(rows),
        eval_count_budget=budget,
        feasible_count_budget=feasible_count_budget,
        feasible_yield_per_100=feasible_yield_per_100,
        hv_at_budget=_compute_hv(feasible_points_budget, ref_x=ref_x, ref_y=ref_y),
        hv_final=_compute_hv(feasible_points_final, ref_x=ref_x, ref_y=ref_y),
        best_lgradb_feasible_at_budget=best_budget,
        route_summary_at_budget=_route_summary(budget_rows),
        error_count_budget=error_count_budget,
    )


def _best_delta(adaptive: float | None, static: float | None) -> float | None:
    if adaptive is None or static is None:
        return None
    return float(adaptive - static)


def _build_report(
    *,
    static_summary: ArmSummary,
    adaptive_summary: ArmSummary,
    budget_requested: int,
    budget_used: int,
    ref_x: float,
    ref_y: float,
    m24_min_budget: int = _M24_MIN_BUDGET,
) -> dict:
    delta_hv_budget = adaptive_summary.hv_at_budget - static_summary.hv_at_budget
    delta_hv_final = adaptive_summary.hv_final - static_summary.hv_final
    delta_feasible_yield = (
        adaptive_summary.feasible_yield_per_100 - static_summary.feasible_yield_per_100
    )
    delta_best = _best_delta(
        adaptive_summary.best_lgradb_feasible_at_budget,
        static_summary.best_lgradb_feasible_at_budget,
    )
    m23_contract_pass = bool(
        adaptive_summary.hv_at_budget >= static_summary.hv_at_budget
        and adaptive_summary.feasible_yield_per_100
        >= static_summary.feasible_yield_per_100
    )
    budget_meets_m24_minimum = bool(int(budget_used) >= int(m24_min_budget))
    non_trivial_feasible_evidence = bool(
        int(static_summary.feasible_count_budget) > 0
        or int(adaptive_summary.feasible_count_budget) > 0
    )
    m24_performance_evidence_pass = bool(
        m23_contract_pass and budget_meets_m24_minimum and non_trivial_feasible_evidence
    )
    gates = {
        "hv_at_budget_non_regression": bool(
            adaptive_summary.hv_at_budget >= static_summary.hv_at_budget
        ),
        "feasible_yield_non_regression": bool(
            adaptive_summary.feasible_yield_per_100
            >= static_summary.feasible_yield_per_100
        ),
        "m23_contract_pass": m23_contract_pass,
        "m24_min_budget": int(m24_min_budget),
        "budget_meets_m24_minimum": budget_meets_m24_minimum,
        "non_trivial_feasible_evidence": non_trivial_feasible_evidence,
        "m24_performance_evidence_pass": m24_performance_evidence_pass,
    }
    gates["pass"] = m23_contract_pass
    return {
        "generated_at": _utc_now_iso(),
        "budget_requested": int(budget_requested),
        "budget_used": int(budget_used),
        "ref_point": {"x": float(ref_x), "y": float(ref_y)},
        "static": asdict(static_summary),
        "adaptive": asdict(adaptive_summary),
        "deltas": {
            "hv_at_budget": float(delta_hv_budget),
            "hv_final": float(delta_hv_final),
            "feasible_yield_per_100": float(delta_feasible_yield),
            "best_lgradb_feasible_at_budget": delta_best,
        },
        "gates": gates,
    }


def _render_markdown(report: dict) -> str:
    static = report["static"]
    adaptive = report["adaptive"]
    deltas = report["deltas"]
    gates = report["gates"]
    lines = [
        "# P3 Governor A/B Validation",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Budget requested: `{report['budget_requested']}`",
        f"- Budget used: `{report['budget_used']}`",
        f"- Static experiment: `{static['experiment_id']}`",
        f"- Adaptive experiment: `{adaptive['experiment_id']}`",
        "",
        "## Fixed-budget comparison",
        "",
        "| Metric | Static | Adaptive | Delta (Adaptive-Static) |",
        "|---|---:|---:|---:|",
        f"| Eval error rows at budget | {static['error_count_budget']} | {adaptive['error_count_budget']} | {adaptive['error_count_budget'] - static['error_count_budget']} |",
        f"| Feasible yield / 100 evals | {static['feasible_yield_per_100']:.4f} | {adaptive['feasible_yield_per_100']:.4f} | {deltas['feasible_yield_per_100']:.4f} |",
        f"| Hypervolume at budget | {static['hv_at_budget']:.6f} | {adaptive['hv_at_budget']:.6f} | {deltas['hv_at_budget']:.6f} |",
        f"| Hypervolume final | {static['hv_final']:.6f} | {adaptive['hv_final']:.6f} | {deltas['hv_final']:.6f} |",
        f"| Best feasible L_gradB at budget | {static['best_lgradb_feasible_at_budget']} | {adaptive['best_lgradb_feasible_at_budget']} | {deltas['best_lgradb_feasible_at_budget']} |",
        "",
        "## Gate result",
        "",
        f"- `hv_at_budget_non_regression`: `{gates['hv_at_budget_non_regression']}`",
        f"- `feasible_yield_non_regression`: `{gates['feasible_yield_non_regression']}`",
        f"- `m23_contract_pass`: `{gates['m23_contract_pass']}`",
        (
            f"- `budget_meets_m24_minimum` "
            f"(budget>={gates['m24_min_budget']}): "
            f"`{gates['budget_meets_m24_minimum']}`"
        ),
        f"- `non_trivial_feasible_evidence`: `{gates['non_trivial_feasible_evidence']}`",
        f"- `m24_performance_evidence_pass`: `{gates['m24_performance_evidence_pass']}`",
        f"- `pass` (legacy alias for M2.3): `{gates['pass']}`",
    ]
    return "\n".join(lines) + "\n"


def _resolve_budget(
    *,
    budget_requested: int,
    static_count: int,
    adaptive_count: int,
) -> int:
    if budget_requested < 0:
        raise ValueError("--budget must be >= 0.")
    max_shared = min(static_count, adaptive_count)
    if max_shared <= 0:
        raise ValueError(
            "Both experiments must have at least one evaluated metric row."
        )
    if budget_requested == 0:
        return max_shared
    return min(int(budget_requested), max_shared)


def _validate_m24_min_budget(m24_min_budget: int) -> None:
    if int(m24_min_budget) < int(_M24_MIN_BUDGET):
        raise ValueError(
            f"--m24-min-budget must be >= {_M24_MIN_BUDGET} "
            "(the M2.4 minimum meaningful budget)."
        )


def _validate_ab_contract(
    *,
    static_summary: ArmSummary,
    adaptive_summary: ArmSummary,
    allow_legacy_route_metadata: bool,
    max_error_rows_budget: int = 0,
) -> None:
    if static_summary.experiment_id == adaptive_summary.experiment_id:
        raise ValueError(
            "Static and adaptive experiment IDs must be distinct for valid A/B."
        )
    if max_error_rows_budget < 0:
        raise ValueError("--max-error-rows-budget must be >= 0.")
    if static_summary.error_count_budget > max_error_rows_budget:
        raise ValueError(
            "Static arm exceeded allowed eval-error rows at budget. "
            "Fix worker/runtime errors or increase --max-error-rows-budget."
        )
    if adaptive_summary.error_count_budget > max_error_rows_budget:
        raise ValueError(
            "Adaptive arm exceeded allowed eval-error rows at budget. "
            "Fix worker/runtime errors or increase --max-error-rows-budget."
        )
    if allow_legacy_route_metadata:
        return

    static_rows = int(static_summary.route_summary_at_budget.get("static_path_rows", 0))
    static_adaptive_rows = int(
        static_summary.route_summary_at_budget.get("adaptive_path_rows", 0)
    )
    adaptive_rows = int(
        adaptive_summary.route_summary_at_budget.get("adaptive_path_rows", 0)
    )
    adaptive_static_rows = int(
        adaptive_summary.route_summary_at_budget.get("static_path_rows", 0)
    )
    adaptive_fallback_rows = int(
        adaptive_summary.route_summary_at_budget.get("fallback_static_delegate_rows", 0)
    )
    adaptive_non_fallback_rows = adaptive_rows - adaptive_fallback_rows

    if static_rows <= 0:
        raise ValueError(
            "Static arm has zero static-route rows at budget. "
            "Use the correct static experiment or pass --allow-legacy-route-metadata."
        )
    if adaptive_rows <= 0:
        raise ValueError(
            "Adaptive arm has zero adaptive-route rows at budget. "
            "Use the correct adaptive experiment or pass --allow-legacy-route-metadata."
        )
    if (
        static_rows != int(static_summary.eval_count_budget)
        or static_adaptive_rows != 0
    ):
        raise ValueError(
            "Static arm route contamination detected at budget. "
            "Strict mode requires all rows to be static-route."
        )
    if (
        adaptive_rows != int(adaptive_summary.eval_count_budget)
        or adaptive_static_rows != 0
    ):
        raise ValueError(
            "Adaptive arm route contamination detected at budget. "
            "Strict mode requires all rows to be adaptive-route."
        )
    if adaptive_non_fallback_rows <= 0:
        raise ValueError(
            "Adaptive arm contains only static-delegate fallback rows at budget. "
            "Strict mode requires at least one non-fallback adaptive row."
        )


def _enforce_required_gates(
    report: dict,
    *,
    require_m23_pass: bool,
    require_m24_pass: bool,
) -> None:
    gates = report.get("gates", {})
    if require_m23_pass and not bool(gates.get("m23_contract_pass")):
        raise ValueError(
            "M2.3 contract gate failed: expected "
            "`hv_at_budget_non_regression && feasible_yield_non_regression`."
        )
    if require_m24_pass and not bool(gates.get("m24_performance_evidence_pass")):
        raise ValueError(
            "M2.4 performance-evidence gate failed: requires M2.3 pass, "
            "budget>=m24_min_budget, and non-trivial feasible evidence."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="P3 fixed-budget governor A/B validator."
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("reports/p3_world_model.sqlite"),
        help="SQLite DB path for P3 runs.",
    )
    parser.add_argument("--static-experiment-id", type=int, required=True)
    parser.add_argument("--adaptive-experiment-id", type=int, required=True)
    parser.add_argument(
        "--budget",
        type=int,
        default=0,
        help="Fixed eval budget per arm; 0 uses the max shared budget.",
    )
    parser.add_argument("--ref-point-x", type=float, default=1.0)
    parser.add_argument("--ref-point-y", type=float, default=20.0)
    parser.add_argument(
        "--m24-min-budget",
        type=int,
        default=_M24_MIN_BUDGET,
        help=(
            "Minimum fixed budget required for M2.4 performance-evidence gate. "
            f"Must be >= {_M24_MIN_BUDGET}."
        ),
    )
    parser.add_argument(
        "--max-error-rows-budget",
        type=int,
        default=0,
        help="Maximum allowed eval rows with runtime error per arm at budget.",
    )
    parser.add_argument(
        "--allow-legacy-route-metadata",
        action="store_true",
        help=(
            "Allow A/B runs where model_route labels are missing/legacy; "
            "disables strict static/adaptive route-family checks."
        ),
    )
    parser.add_argument(
        "--require-m23-pass",
        action="store_true",
        help="Fail with non-zero exit if M2.3 contract gate is false.",
    )
    parser.add_argument(
        "--require-m24-pass",
        action="store_true",
        help="Fail with non-zero exit if M2.4 performance-evidence gate is false.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    args = parser.parse_args()

    conn = _connect(args.db)
    try:
        static_rows = _fetch_eval_rows(
            conn,
            experiment_id=int(args.static_experiment_id),
        )
        adaptive_rows = _fetch_eval_rows(
            conn,
            experiment_id=int(args.adaptive_experiment_id),
        )
    finally:
        conn.close()

    budget = _resolve_budget(
        budget_requested=int(args.budget),
        static_count=len(static_rows),
        adaptive_count=len(adaptive_rows),
    )
    _validate_m24_min_budget(int(args.m24_min_budget))
    static_summary = _build_arm_summary(
        experiment_id=int(args.static_experiment_id),
        rows=static_rows,
        budget=budget,
        ref_x=float(args.ref_point_x),
        ref_y=float(args.ref_point_y),
    )
    adaptive_summary = _build_arm_summary(
        experiment_id=int(args.adaptive_experiment_id),
        rows=adaptive_rows,
        budget=budget,
        ref_x=float(args.ref_point_x),
        ref_y=float(args.ref_point_y),
    )
    _validate_ab_contract(
        static_summary=static_summary,
        adaptive_summary=adaptive_summary,
        allow_legacy_route_metadata=bool(args.allow_legacy_route_metadata),
        max_error_rows_budget=int(args.max_error_rows_budget),
    )
    report = _build_report(
        static_summary=static_summary,
        adaptive_summary=adaptive_summary,
        budget_requested=int(args.budget),
        budget_used=budget,
        ref_x=float(args.ref_point_x),
        ref_y=float(args.ref_point_y),
        m24_min_budget=int(args.m24_min_budget),
    )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2))
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(_render_markdown(report))

    _enforce_required_gates(
        report,
        require_m23_pass=bool(args.require_m23_pass),
        require_m24_pass=bool(args.require_m24_pass),
    )

    if args.output_json is None and args.output_md is None:
        print(json.dumps(report, indent=2))
    else:
        print(
            "A/B report complete: budget=%d static_exp=%d adaptive_exp=%d pass=%s"
            % (
                budget,
                int(args.static_experiment_id),
                int(args.adaptive_experiment_id),
                str(report["gates"]["pass"]),
            )
        )


if __name__ == "__main__":
    main()
