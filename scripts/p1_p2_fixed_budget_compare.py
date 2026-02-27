#!/usr/bin/env python
"""Fixed-budget comparison validator for P1/P2 ALM-NGOpt runs."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, cast

Problem = Literal["p1", "p2"]
_HISTORY_FILE = "history.jsonl"
_VIOLATION_KEYS = (
    "feasibility_official",
    "constraint_violation_inf",
    "constraint_violation_l2",
)


@dataclass(frozen=True)
class HistoryRow:
    index: int
    metric_value: float | None
    violation: float | None
    has_error: bool
    restart_seed: str | None


@dataclass(frozen=True)
class ArmSummary:
    run_dir: str
    eval_count_total: int
    eval_count_budget: int
    feasible_count_budget: int
    feasible_yield_per_100: float
    best_feasible_metric_at_budget: float | None
    error_count_budget: int
    restart_seed_labeled_rows_budget: int


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _extract_metric(payload: dict, *, problem: Problem) -> float | None:
    if problem == "p1":
        return _safe_float(payload.get("objective"))
    lgradb = _safe_float(payload.get("lgradb"))
    if lgradb is not None:
        return lgradb
    objective = _safe_float(payload.get("objective"))
    if objective is None:
        return None
    return -objective


def _extract_violation(payload: dict) -> float | None:
    for key in _VIOLATION_KEYS:
        value = _safe_float(payload.get(key))
        if value is not None:
            return value
    return None


def _extract_error(payload: dict) -> bool:
    error = payload.get("error")
    return isinstance(error, str) and bool(error.strip())


def _extract_restart_seed(payload: dict) -> str | None:
    restart_seed = payload.get("restart_seed")
    if not isinstance(restart_seed, str):
        return None
    token = restart_seed.strip()
    return token if token else None


def _load_history_rows(run_dir: Path, *, problem: Problem) -> list[HistoryRow]:
    history_path = run_dir / _HISTORY_FILE
    if not history_path.exists():
        raise ValueError(f"Missing history file: {history_path}")

    rows: list[HistoryRow] = []
    for line_idx, line in enumerate(
        history_path.read_text(encoding="utf-8").splitlines()
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {history_path}:{line_idx + 1}") from exc
        if not isinstance(payload, dict):
            raise ValueError(
                f"History row must be a JSON object at {history_path}:{line_idx + 1}"
            )
        rows.append(
            HistoryRow(
                index=line_idx,
                metric_value=_extract_metric(payload, problem=problem),
                violation=_extract_violation(payload),
                has_error=_extract_error(payload),
                restart_seed=_extract_restart_seed(payload),
            )
        )
    return rows


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
        raise ValueError("Both runs must have at least one evaluation row.")
    if budget_requested == 0:
        return max_shared
    return min(int(budget_requested), max_shared)


def _is_feasible(row: HistoryRow) -> bool:
    if row.has_error:
        return False
    if row.metric_value is None:
        return False
    if row.violation is None:
        return False
    return row.violation <= 0.0


def _best_feasible_metric(rows: list[HistoryRow], *, problem: Problem) -> float | None:
    feasible_metrics: list[float] = []
    for row in rows:
        metric = row.metric_value
        if _is_feasible(row) and metric is not None:
            feasible_metrics.append(metric)
    if not feasible_metrics:
        return None
    if problem == "p1":
        return min(feasible_metrics)
    return max(feasible_metrics)


def _build_arm_summary(
    *,
    run_dir: Path,
    rows: list[HistoryRow],
    budget: int,
    problem: Problem,
) -> ArmSummary:
    budget_rows = rows[:budget]
    feasible_count_budget = 0
    error_count_budget = 0
    restart_seed_labeled_rows_budget = 0
    for row in budget_rows:
        if _is_feasible(row):
            feasible_count_budget += 1
        if row.has_error:
            error_count_budget += 1
        if row.restart_seed is not None:
            restart_seed_labeled_rows_budget += 1
    feasible_yield_per_100 = (
        100.0 * float(feasible_count_budget) / float(budget) if budget > 0 else 0.0
    )
    return ArmSummary(
        run_dir=str(run_dir),
        eval_count_total=len(rows),
        eval_count_budget=budget,
        feasible_count_budget=feasible_count_budget,
        feasible_yield_per_100=feasible_yield_per_100,
        best_feasible_metric_at_budget=_best_feasible_metric(
            budget_rows, problem=problem
        ),
        error_count_budget=error_count_budget,
        restart_seed_labeled_rows_budget=restart_seed_labeled_rows_budget,
    )


def _metric_delta(adaptive: float | None, static: float | None) -> float | None:
    if adaptive is None or static is None:
        return None
    return float(adaptive - static)


def _metric_non_regression(
    *,
    problem: Problem,
    static: float | None,
    adaptive: float | None,
) -> bool:
    if static is None:
        return True
    if adaptive is None:
        return False
    if problem == "p1":
        return adaptive <= static
    return adaptive >= static


def _metric_contract(problem: Problem) -> tuple[str, str]:
    if problem == "p1":
        return "max_elongation", "minimize"
    return "minimum_normalized_magnetic_gradient_scale_length", "maximize"


def _build_report(
    *,
    problem: Problem,
    static_summary: ArmSummary,
    adaptive_summary: ArmSummary,
    budget_requested: int,
    budget_used: int,
    allow_legacy_restart_metadata: bool = False,
    max_error_rows_budget: int = 0,
) -> dict:
    metric_name, direction = _metric_contract(problem)
    best_non_regression = _metric_non_regression(
        problem=problem,
        static=static_summary.best_feasible_metric_at_budget,
        adaptive=adaptive_summary.best_feasible_metric_at_budget,
    )
    feasible_yield_non_regression = (
        adaptive_summary.feasible_yield_per_100 >= static_summary.feasible_yield_per_100
    )
    non_trivial_feasible_evidence = (
        static_summary.feasible_count_budget > 0
        or adaptive_summary.feasible_count_budget > 0
    )
    gates = {
        "best_feasible_metric_non_regression": bool(best_non_regression),
        "feasible_yield_non_regression": bool(feasible_yield_non_regression),
        "non_trivial_feasible_evidence": bool(non_trivial_feasible_evidence),
    }
    gates["m33_contract_pass"] = bool(
        gates["best_feasible_metric_non_regression"]
        and gates["feasible_yield_non_regression"]
        and gates["non_trivial_feasible_evidence"]
    )
    gates["pass"] = gates["m33_contract_pass"]
    deltas = {
        "feasible_yield_per_100": float(
            adaptive_summary.feasible_yield_per_100
            - static_summary.feasible_yield_per_100
        ),
        "best_feasible_metric_at_budget": _metric_delta(
            adaptive_summary.best_feasible_metric_at_budget,
            static_summary.best_feasible_metric_at_budget,
        ),
        "eval_error_rows_at_budget": int(
            adaptive_summary.error_count_budget - static_summary.error_count_budget
        ),
    }
    return {
        "generated_at": _utc_now_iso(),
        "problem": problem,
        "metric": {
            "name": metric_name,
            "direction": direction,
        },
        "budget_requested": int(budget_requested),
        "budget_used": int(budget_used),
        "run_contract": {
            "allow_legacy_restart_metadata": bool(allow_legacy_restart_metadata),
            "strict_restart_seed_metadata_enforced": not bool(
                allow_legacy_restart_metadata
            ),
            "max_error_rows_budget": int(max_error_rows_budget),
        },
        "static": asdict(static_summary),
        "adaptive": asdict(adaptive_summary),
        "deltas": deltas,
        "gates": gates,
    }


def _render_markdown(report: dict) -> str:
    metric = report["metric"]
    static = report["static"]
    adaptive = report["adaptive"]
    deltas = report["deltas"]
    gates = report["gates"]
    lines = [
        f"# {report['problem'].upper()} Fixed-Budget Comparison",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Metric: `{metric['name']}` (`{metric['direction']}`)",
        f"- Budget requested: `{report['budget_requested']}`",
        f"- Budget used: `{report['budget_used']}`",
        (
            "- Run contract: "
            f"`allow_legacy_restart_metadata={report['run_contract']['allow_legacy_restart_metadata']}`, "
            f"`strict_restart_seed_metadata_enforced={report['run_contract']['strict_restart_seed_metadata_enforced']}`, "
            f"`max_error_rows_budget={report['run_contract']['max_error_rows_budget']}`"
        ),
        "",
        "## Fixed-budget comparison",
        "",
        "| Metric | Static | Adaptive | Delta (Adaptive-Static) |",
        "|---|---:|---:|---:|",
        (
            f"| Eval error rows at budget | {static['error_count_budget']} | "
            f"{adaptive['error_count_budget']} | {deltas['eval_error_rows_at_budget']} |"
        ),
        (
            f"| Feasible yield / 100 evals | {static['feasible_yield_per_100']:.4f} | "
            f"{adaptive['feasible_yield_per_100']:.4f} | "
            f"{deltas['feasible_yield_per_100']:.4f} |"
        ),
        (
            f"| Best feasible metric at budget | {static['best_feasible_metric_at_budget']} | "
            f"{adaptive['best_feasible_metric_at_budget']} | "
            f"{deltas['best_feasible_metric_at_budget']} |"
        ),
        "",
        "## Gate result",
        "",
        (
            f"- `best_feasible_metric_non_regression` "
            f"({metric['direction']}): `{gates['best_feasible_metric_non_regression']}`"
        ),
        f"- `feasible_yield_non_regression`: `{gates['feasible_yield_non_regression']}`",
        f"- `non_trivial_feasible_evidence`: `{gates['non_trivial_feasible_evidence']}`",
        f"- `m33_contract_pass`: `{gates['m33_contract_pass']}`",
        f"- `pass` (legacy alias): `{gates['pass']}`",
    ]
    return "\n".join(lines) + "\n"


def _validate_run_contract(
    *,
    static_run_dir: Path,
    adaptive_run_dir: Path,
    static_summary: ArmSummary,
    adaptive_summary: ArmSummary,
    max_error_rows_budget: int,
    allow_legacy_restart_metadata: bool,
) -> None:
    if static_run_dir.resolve() == adaptive_run_dir.resolve():
        raise ValueError("Static and adaptive run directories must be distinct.")
    if max_error_rows_budget < 0:
        raise ValueError("--max-error-rows-budget must be >= 0.")
    for arm_name, arm_summary in (
        ("Static", static_summary),
        ("Adaptive", adaptive_summary),
    ):
        if arm_summary.error_count_budget > max_error_rows_budget:
            raise ValueError(
                f"{arm_name} arm exceeded allowed eval-error rows at budget. "
                "Fix run errors or increase --max-error-rows-budget."
            )
    if allow_legacy_restart_metadata:
        return
    if static_summary.restart_seed_labeled_rows_budget > 0:
        raise ValueError(
            "Static arm contains restart-seed-labeled rows at budget. "
            "Strict mode requires static arm rows to have null/absent restart labels."
        )
    if adaptive_summary.restart_seed_labeled_rows_budget <= 0:
        raise ValueError(
            "Adaptive arm has zero restart-seed-labeled rows at budget. "
            "Use an adaptive run or pass --allow-legacy-restart-metadata."
        )


def _enforce_required_gates(report: dict, *, require_m33_pass: bool) -> None:
    if require_m33_pass and not bool(report.get("gates", {}).get("m33_contract_pass")):
        raise ValueError(
            "M3.3 fixed-budget gate failed: expected "
            "`best_feasible_metric_non_regression && feasible_yield_non_regression "
            "&& non_trivial_feasible_evidence`."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="P1/P2 fixed-budget comparator.")
    parser.add_argument("--problem", choices=["p1", "p2"], required=True)
    parser.add_argument("--static-run-dir", type=Path, required=True)
    parser.add_argument("--adaptive-run-dir", type=Path, required=True)
    parser.add_argument(
        "--budget",
        type=int,
        default=0,
        help="Fixed eval budget per arm; 0 uses the max shared budget.",
    )
    parser.add_argument(
        "--max-error-rows-budget",
        type=int,
        default=0,
        help="Maximum allowed eval rows with runtime error per arm at budget.",
    )
    parser.add_argument(
        "--allow-legacy-restart-metadata",
        action="store_true",
        help=(
            "Allow runs without strict restart-seed metadata checks. "
            "Use only for legacy history files."
        ),
    )
    parser.add_argument(
        "--require-m33-pass",
        action="store_true",
        help="Fail with non-zero exit if M3.3 fixed-budget gate is false.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    args = parser.parse_args()

    problem = cast(Problem, args.problem)
    static_rows = _load_history_rows(args.static_run_dir, problem=problem)
    adaptive_rows = _load_history_rows(args.adaptive_run_dir, problem=problem)
    budget = _resolve_budget(
        budget_requested=int(args.budget),
        static_count=len(static_rows),
        adaptive_count=len(adaptive_rows),
    )
    static_summary = _build_arm_summary(
        run_dir=args.static_run_dir,
        rows=static_rows,
        budget=budget,
        problem=problem,
    )
    adaptive_summary = _build_arm_summary(
        run_dir=args.adaptive_run_dir,
        rows=adaptive_rows,
        budget=budget,
        problem=problem,
    )
    _validate_run_contract(
        static_run_dir=args.static_run_dir,
        adaptive_run_dir=args.adaptive_run_dir,
        static_summary=static_summary,
        adaptive_summary=adaptive_summary,
        max_error_rows_budget=int(args.max_error_rows_budget),
        allow_legacy_restart_metadata=bool(args.allow_legacy_restart_metadata),
    )
    report = _build_report(
        problem=problem,
        static_summary=static_summary,
        adaptive_summary=adaptive_summary,
        budget_requested=int(args.budget),
        budget_used=budget,
        allow_legacy_restart_metadata=bool(args.allow_legacy_restart_metadata),
        max_error_rows_budget=int(args.max_error_rows_budget),
    )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(_render_markdown(report), encoding="utf-8")

    _enforce_required_gates(report, require_m33_pass=bool(args.require_m33_pass))

    if args.output_json is None and args.output_md is None:
        print(json.dumps(report, indent=2))
    else:
        print(
            "P1/P2 report complete: problem=%s budget=%d pass=%s"
            % (
                problem,
                budget,
                str(report["gates"]["pass"]),
            )
        )


if __name__ == "__main__":
    main()
