#!/usr/bin/env python
"""M3.6 fixed-budget validator for cross-problem policy hardening."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai_scientist.memory.schema import init_db

_MIN_MEANINGFUL_BUDGET = 20


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"Missing report file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON report: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Report must be a JSON object: {path}")
    return payload


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


def _safe_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number


def _required_int(report: dict[str, Any], key: str) -> int:
    value = _safe_int(report.get(key))
    if value is None:
        raise ValueError(f"Missing/invalid integer field: {key}")
    return int(value)


def _required_dict(report: dict[str, Any], key: str) -> dict[str, Any]:
    value = report.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing/invalid object field: {key}")
    return value


def _optional_bool(
    container: dict[str, Any], key: str, *, field_name: str | None = None
) -> bool:
    value = container.get(key)
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    name = field_name if field_name is not None else key
    raise ValueError(f"Field must be boolean when provided: {name}")


def _model_routes(route_summary: dict[str, Any]) -> dict[str, int]:
    payload = route_summary.get("model_routes")
    if not isinstance(payload, dict):
        return {}
    out: dict[str, int] = {}
    for key, value in payload.items():
        if not isinstance(key, str):
            continue
        count = _safe_int(value)
        if count is None or count <= 0:
            continue
        out[key] = int(count)
    return out


def _extract_p1_p2_summary(
    *,
    report: dict[str, Any],
    expected_problem: str,
    min_budget: int,
) -> dict[str, Any]:
    problem = str(report.get("problem", "")).strip().lower()
    if problem != expected_problem:
        raise ValueError(
            f"Expected {expected_problem.upper()} report, got: {report.get('problem')!r}"
        )
    gates = _required_dict(report, "gates")
    budget_used = _required_int(report, "budget_used")
    return {
        "problem": expected_problem,
        "budget_used": budget_used,
        "budget_meets_minimum": bool(budget_used >= min_budget),
        "m33_contract_pass": _optional_bool(
            gates, "m33_contract_pass", field_name="gates.m33_contract_pass"
        ),
        "non_trivial_feasible_evidence": _optional_bool(
            gates,
            "non_trivial_feasible_evidence",
            field_name="gates.non_trivial_feasible_evidence",
        ),
        "best_feasible_metric_non_regression": _optional_bool(
            gates,
            "best_feasible_metric_non_regression",
            field_name="gates.best_feasible_metric_non_regression",
        ),
        "feasible_yield_non_regression": _optional_bool(
            gates,
            "feasible_yield_non_regression",
            field_name="gates.feasible_yield_non_regression",
        ),
        "source_generated_at": report.get("generated_at"),
    }


def _extract_p3_summary(*, report: dict[str, Any], min_budget: int) -> dict[str, Any]:
    gates = _required_dict(report, "gates")
    static = _required_dict(report, "static")
    adaptive = _required_dict(report, "adaptive")
    static_route = _required_dict(static, "route_summary_at_budget")
    adaptive_route = _required_dict(adaptive, "route_summary_at_budget")
    budget_used = _required_int(report, "budget_used")
    static_experiment_id = _required_int(static, "experiment_id")
    adaptive_experiment_id = _required_int(adaptive, "experiment_id")
    static_novelty_reject_rate = _safe_float(static_route.get("novelty_reject_rate"))
    adaptive_novelty_reject_rate = _safe_float(
        adaptive_route.get("novelty_reject_rate")
    )
    static_routes = _model_routes(static_route)
    adaptive_routes = _model_routes(adaptive_route)
    return {
        "budget_used": budget_used,
        "budget_meets_minimum": bool(budget_used >= min_budget),
        "m24_performance_evidence_pass": _optional_bool(
            gates,
            "m24_performance_evidence_pass",
            field_name="gates.m24_performance_evidence_pass",
        ),
        "static_experiment_id": static_experiment_id,
        "adaptive_experiment_id": adaptive_experiment_id,
        "static_novelty_reject_rate": static_novelty_reject_rate,
        "adaptive_novelty_reject_rate": adaptive_novelty_reject_rate,
        "novelty_reject_rate_available": (
            static_novelty_reject_rate is not None
            and adaptive_novelty_reject_rate is not None
        ),
        "static_model_routes": static_routes,
        "adaptive_model_routes": adaptive_routes,
        "router_decisions_available": bool(static_routes) and bool(adaptive_routes),
        "source_generated_at": report.get("generated_at"),
    }


def _connect(db_path: Path) -> sqlite3.Connection:
    init_db(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _router_reward_summary(
    conn: sqlite3.Connection, *, experiment_id: int
) -> dict[str, Any]:
    rows = conn.execute(
        """
        SELECT reward, model_route
        FROM model_router_reward_events
        WHERE experiment_id = ? AND problem = 'p3'
        ORDER BY id DESC
        """,
        (int(experiment_id),),
    ).fetchall()
    if not rows:
        return {
            "event_count": 0,
            "avg_reward": None,
            "last_reward": None,
            "model_routes": {},
        }
    rewards: list[float] = []
    route_counts: dict[str, int] = {}
    for row in rows:
        reward = _safe_float(row["reward"])
        if reward is None:
            continue
        rewards.append(reward)
        route = str(row["model_route"] or "unknown")
        route_counts[route] = route_counts.get(route, 0) + 1
    if not rewards:
        return {
            "event_count": len(rows),
            "avg_reward": None,
            "last_reward": None,
            "model_routes": dict(sorted(route_counts.items())),
        }
    return {
        "event_count": len(rows),
        "avg_reward": float(sum(rewards) / len(rewards)),
        "last_reward": float(rewards[0]),
        "model_routes": dict(
            sorted(route_counts.items(), key=lambda item: item[1], reverse=True)
        ),
    }


def _reward_deltas(
    static_reward: dict[str, Any], adaptive_reward: dict[str, Any]
) -> dict[str, Any]:
    static_avg = _safe_float(static_reward.get("avg_reward"))
    adaptive_avg = _safe_float(adaptive_reward.get("avg_reward"))
    static_last = _safe_float(static_reward.get("last_reward"))
    adaptive_last = _safe_float(adaptive_reward.get("last_reward"))
    return {
        "avg_reward": (
            float(adaptive_avg - static_avg)
            if static_avg is not None and adaptive_avg is not None
            else None
        ),
        "last_reward": (
            float(adaptive_last - static_last)
            if static_last is not None and adaptive_last is not None
            else None
        ),
        "event_count": int(adaptive_reward.get("event_count", 0))
        - int(static_reward.get("event_count", 0)),
    }


def _validate_min_budget(min_budget: int) -> None:
    if int(min_budget) < _MIN_MEANINGFUL_BUDGET:
        raise ValueError(
            f"--min-budget must be >= {_MIN_MEANINGFUL_BUDGET} "
            "(meaningful fixed-budget threshold)."
        )


def _build_report(
    *,
    p1_summary: dict[str, Any],
    p2_summary: dict[str, Any],
    p3_summary: dict[str, Any],
    static_reward: dict[str, Any],
    adaptive_reward: dict[str, Any],
    min_budget: int,
    require_reward_telemetry: bool,
) -> dict[str, Any]:
    reward_delta = _reward_deltas(static_reward, adaptive_reward)
    reward_delta_available = (
        reward_delta["avg_reward"] is not None
        and reward_delta["last_reward"] is not None
    )
    gates = {
        "p1_budget_meets_minimum": bool(p1_summary["budget_meets_minimum"]),
        "p2_budget_meets_minimum": bool(p2_summary["budget_meets_minimum"]),
        "p3_budget_meets_minimum": bool(p3_summary["budget_meets_minimum"]),
        "p1_m33_contract_pass": bool(p1_summary["m33_contract_pass"]),
        "p2_m33_contract_pass": bool(p2_summary["m33_contract_pass"]),
        "p3_m24_performance_evidence_pass": bool(
            p3_summary["m24_performance_evidence_pass"]
        ),
        "p3_novelty_reject_rate_available": bool(
            p3_summary["novelty_reject_rate_available"]
        ),
        "p3_router_decisions_available": bool(p3_summary["router_decisions_available"]),
        "p3_reward_delta_available": bool(reward_delta_available),
        "require_reward_telemetry": bool(require_reward_telemetry),
    }
    required_gate_keys = [
        "p1_budget_meets_minimum",
        "p2_budget_meets_minimum",
        "p3_budget_meets_minimum",
        "p1_m33_contract_pass",
        "p2_m33_contract_pass",
        "p3_m24_performance_evidence_pass",
        "p3_novelty_reject_rate_available",
        "p3_router_decisions_available",
    ]
    if require_reward_telemetry:
        required_gate_keys.append("p3_reward_delta_available")
    unmet = [key for key in required_gate_keys if not bool(gates.get(key))]
    gates["m36_contract_pass"] = len(unmet) == 0
    gates["pass"] = gates["m36_contract_pass"]
    return {
        "generated_at": _utc_now_iso(),
        "min_budget": int(min_budget),
        "p1": p1_summary,
        "p2": p2_summary,
        "p3": p3_summary,
        "router_reward": {
            "static": static_reward,
            "adaptive": adaptive_reward,
            "deltas": reward_delta,
        },
        "gates": gates,
        "unmet_gates": unmet,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    p1 = report["p1"]
    p2 = report["p2"]
    p3 = report["p3"]
    reward = report["router_reward"]
    gates = report["gates"]
    lines = [
        "# M3.6 Fixed-Budget Policy Hardening Validation",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Minimum required budget: `{report['min_budget']}`",
        "",
        "## Problem gates",
        "",
        "| Problem | Budget used | Budget>=min | Gate |",
        "|---|---:|---:|---:|",
        f"| P1 | {p1['budget_used']} | {p1['budget_meets_minimum']} | {p1['m33_contract_pass']} |",
        f"| P2 | {p2['budget_used']} | {p2['budget_meets_minimum']} | {p2['m33_contract_pass']} |",
        (
            f"| P3 | {p3['budget_used']} | {p3['budget_meets_minimum']} | "
            f"{p3['m24_performance_evidence_pass']} |"
        ),
        "",
        "## P3 policy telemetry",
        "",
        (
            f"- Novelty reject rates (static/adaptive): "
            f"`{p3['static_novelty_reject_rate']}` / `{p3['adaptive_novelty_reject_rate']}`"
        ),
        f"- Router decision families (static): `{p3['static_model_routes']}`",
        f"- Router decision families (adaptive): `{p3['adaptive_model_routes']}`",
        (
            f"- Router reward deltas (adaptive-static): "
            f"`avg={reward['deltas']['avg_reward']}`, "
            f"`last={reward['deltas']['last_reward']}`"
        ),
        "",
        "## Final gate",
        "",
        f"- `m36_contract_pass`: `{gates['m36_contract_pass']}`",
        f"- `pass` (legacy alias): `{gates['pass']}`",
    ]
    unmet_gates = report.get("unmet_gates", [])
    if unmet_gates:
        lines.append(f"- Unmet gates: `{unmet_gates}`")
    return "\n".join(lines) + "\n"


def _enforce_required_gates(report: dict[str, Any], *, require_m36_pass: bool) -> None:
    if require_m36_pass and not bool(report.get("gates", {}).get("m36_contract_pass")):
        unmet = report.get("unmet_gates", [])
        raise ValueError(
            "M3.6 fixed-budget gate failed; unmet gates: "
            f"{unmet if unmet else '<unknown>'}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="M3.6 fixed-budget validator for policy hardening."
    )
    parser.add_argument("--p1-report-json", type=Path, required=True)
    parser.add_argument("--p2-report-json", type=Path, required=True)
    parser.add_argument("--p3-report-json", type=Path, required=True)
    parser.add_argument(
        "--p3-db",
        type=Path,
        default=Path("reports/p3_world_model.sqlite"),
        help="P3 world-model DB path used for model_router_reward_events telemetry.",
    )
    parser.add_argument(
        "--min-budget",
        type=int,
        default=_MIN_MEANINGFUL_BUDGET,
        help=(
            "Minimum required fixed budget for P1/P2/P3 reports; "
            f"must be >= {_MIN_MEANINGFUL_BUDGET}."
        ),
    )
    parser.add_argument(
        "--allow-missing-reward-telemetry",
        action="store_true",
        help=(
            "Allow M3.6 gate to pass without router reward delta telemetry. "
            "Use only for legacy runs."
        ),
    )
    parser.add_argument(
        "--require-m36-pass",
        action="store_true",
        help="Fail with non-zero exit if M3.6 contract gate is false.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    args = parser.parse_args()

    min_budget = int(args.min_budget)
    _validate_min_budget(min_budget)

    p1_report = _load_json(args.p1_report_json)
    p2_report = _load_json(args.p2_report_json)
    p3_report = _load_json(args.p3_report_json)

    p1_summary = _extract_p1_p2_summary(
        report=p1_report,
        expected_problem="p1",
        min_budget=min_budget,
    )
    p2_summary = _extract_p1_p2_summary(
        report=p2_report,
        expected_problem="p2",
        min_budget=min_budget,
    )
    p3_summary = _extract_p3_summary(report=p3_report, min_budget=min_budget)

    conn = _connect(args.p3_db)
    try:
        static_reward = _router_reward_summary(
            conn,
            experiment_id=int(p3_summary["static_experiment_id"]),
        )
        adaptive_reward = _router_reward_summary(
            conn,
            experiment_id=int(p3_summary["adaptive_experiment_id"]),
        )
    finally:
        conn.close()

    report = _build_report(
        p1_summary=p1_summary,
        p2_summary=p2_summary,
        p3_summary=p3_summary,
        static_reward=static_reward,
        adaptive_reward=adaptive_reward,
        min_budget=min_budget,
        require_reward_telemetry=not bool(args.allow_missing_reward_telemetry),
    )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(_render_markdown(report), encoding="utf-8")

    _enforce_required_gates(report, require_m36_pass=bool(args.require_m36_pass))

    if args.output_json is None and args.output_md is None:
        print(json.dumps(report, indent=2))
    else:
        print(
            "M3.6 report complete: pass=%s unmet=%s"
            % (
                str(report["gates"]["m36_contract_pass"]),
                str(report.get("unmet_gates", [])),
            )
        )


if __name__ == "__main__":
    main()
