#!/usr/bin/env python
# ruff: noqa: E402
"""P3 Governor: decide what to propose next based on DB progress.

This is the "intervention" layer:
- reads recent outcomes from the WorldModel SQLite DB
- identifies the highest-leverage near-feasible candidate and its worst constraint
- proposes the next batch by calling `scripts/p3_propose.py`

By default it prints the commands it would run. Use --execute to actually enqueue.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ai_scientist.model_router_reward import compute_model_router_reward
from ai_scientist.novelty_gate import NoveltyCandidate, apply_two_stage_novelty_gate
from ai_scientist.p3_data_plane import DataPlaneSample, summarize_data_plane
from ai_scientist.memory.schema import init_db


_DEFAULT_RECORD_HV = 135.15669906272515
_NOVELTY_REJECT_THRESHOLD = 0.05
_ADAPTIVE_NOVELTY_GATE = 0.03
_ADAPTIVE_NEAR_DUPLICATE_GATE = 0.05
_ADAPTIVE_MAX_COMMANDS = 8
_ADAPTIVE_EXPLORATION_WEIGHT = 0.35
_ADAPTIVE_PARENT_SATURATION_PENALTY = 0.15
_ROUTER_REWARD_WINDOW = 20
_ROUTER_REWARD_FEASIBLE_WEIGHT = 0.5
_ROUTER_REWARD_HV_WEIGHT = 0.5


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def _connect(db_path: Path) -> sqlite3.Connection:
    init_db(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _status_bucket(status: str) -> str:
    return status.split(":", 1)[0] if status else "unknown"


def _compute_hv(points: list[tuple[float, float]]) -> float:
    if not points:
        return 0.0
    from pymoo.indicators import hv

    X = np.array([(-lgradb, aspect) for lgradb, aspect in points], dtype=float)
    indicator = hv.Hypervolume(ref_point=np.array([1.0, 20.0], dtype=float))
    out = indicator(X)
    assert out is not None
    return float(out)


def _load_boundary(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise TypeError("Boundary JSON must be an object.")
    return payload


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _next_batch_id(run_dir: Path) -> int:
    candidates_dir = run_dir / "candidates"
    max_batch = 0
    for meta_path in candidates_dir.glob("*_meta.json"):
        meta = json.loads(meta_path.read_text())
        if not isinstance(meta, dict):
            continue
        batch_id = meta.get("batch_id")
        if isinstance(batch_id, int) and batch_id > max_batch:
            max_batch = batch_id
    return max_batch + 1


@dataclass(frozen=True)
class CandidateRow:
    candidate_id: int
    design_hash: str
    seed: int
    feasibility: float
    is_feasible: bool
    lgradb: float | None
    aspect: float | None
    violations: dict[str, float]
    metrics: dict
    meta: dict
    lineage_parent_hashes: list[str]
    novelty_score: float | None
    operator_family: str
    model_route: str


def _fetch_candidates(
    conn: sqlite3.Connection, *, experiment_id: int, limit: int
) -> list[CandidateRow]:
    rows = conn.execute(
        """
        SELECT c.id AS candidate_id, c.design_hash AS design_hash, c.seed AS seed,
               c.lineage_parent_hashes_json AS lineage_parent_hashes_json,
               c.novelty_score AS novelty_score,
               c.operator_family AS operator_family,
               c.model_route AS model_route,
               m.feasibility AS feasibility, m.is_feasible AS is_feasible,
               m.objective AS objective, m.raw_json AS raw_json
        FROM metrics m
        JOIN candidates c ON c.id = m.candidate_id
        WHERE c.experiment_id = ?
        ORDER BY m.id DESC
        LIMIT ?
        """,
        (experiment_id, int(limit)),
    ).fetchall()

    out: list[CandidateRow] = []
    for row in rows:
        payload = json.loads(str(row["raw_json"]))
        metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
        violations = payload.get("violations", {}) if isinstance(payload, dict) else {}
        meta = payload.get("meta", {}) if isinstance(payload, dict) else {}

        aspect = None
        if isinstance(metrics, dict) and "aspect_ratio" in metrics:
            try:
                aspect = float(metrics["aspect_ratio"])
            except (TypeError, ValueError):
                aspect = None

        lgradb = None
        if row["objective"] is not None:
            lgradb = float(row["objective"])
        elif isinstance(metrics, dict) and "lgradB" in metrics:
            try:
                lgradb = float(metrics["lgradB"])
            except (TypeError, ValueError):
                lgradb = None

        vio_clean: dict[str, float] = {}
        if isinstance(violations, dict):
            for k, v in violations.items():
                try:
                    vio_clean[str(k)] = float(v)
                except (TypeError, ValueError):
                    continue
        lineage_payload = []
        lineage_raw = row["lineage_parent_hashes_json"]
        if isinstance(lineage_raw, str) and lineage_raw:
            try:
                loaded_lineage = json.loads(lineage_raw)
                if isinstance(loaded_lineage, list):
                    lineage_payload = [str(item) for item in loaded_lineage]
            except (TypeError, ValueError, json.JSONDecodeError):
                lineage_payload = []

        novelty_score = None
        novelty_raw = row["novelty_score"]
        if novelty_raw is not None:
            try:
                novelty_score = float(novelty_raw)
            except (TypeError, ValueError):
                novelty_score = None

        out.append(
            CandidateRow(
                candidate_id=int(row["candidate_id"]),
                design_hash=str(row["design_hash"]),
                seed=int(row["seed"]),
                feasibility=float(row["feasibility"]),
                is_feasible=bool(int(row["is_feasible"])),
                lgradb=lgradb,
                aspect=aspect,
                violations=vio_clean,
                metrics=metrics if isinstance(metrics, dict) else {},
                meta=meta if isinstance(meta, dict) else {},
                lineage_parent_hashes=lineage_payload,
                novelty_score=novelty_score,
                operator_family=str(row["operator_family"] or ""),
                model_route=str(row["model_route"] or ""),
            )
        )
    return out


def _worst_constraint(violations: dict[str, float]) -> tuple[str | None, float]:
    if not violations:
        return None, 0.0
    worst_name = None
    worst_val = -float("inf")
    for name, val in violations.items():
        if val > worst_val:
            worst_val = val
            worst_name = name
    return worst_name, float(worst_val)


def _potential_area(lgradb: float, aspect: float) -> float:
    return max(0.0, 1.0 + float(lgradb)) * max(0.0, 20.0 - float(aspect))


def _focus_score(row: CandidateRow) -> float:
    if row.lgradb is None or row.aspect is None:
        return -float("inf")
    return _potential_area(row.lgradb, row.aspect) / max(row.feasibility, 1e-3)


def _choose_focus(
    candidates: list[CandidateRow], *, max_feasibility: float
) -> CandidateRow | None:
    pool = [
        c
        for c in candidates
        if (not c.is_feasible)
        and math.isfinite(c.feasibility)
        and c.feasibility > 1e-2
        and c.feasibility <= max_feasibility
        and c.lgradb is not None
        and c.aspect is not None
    ]
    if not pool:
        return None
    return max(pool, key=_focus_score)


def _choose_partner(
    candidates: list[CandidateRow],
    *,
    worst_constraint: str,
) -> CandidateRow | None:
    feasible = [
        c
        for c in candidates
        if c.is_feasible and c.lgradb is not None and c.aspect is not None
    ]
    pool = (
        feasible
        if feasible
        else [c for c in candidates if c.lgradb is not None and c.aspect is not None]
    )
    if not pool:
        return None

    key_name = {
        "mirror": "mirror",
        "log10_qi": "log10_qi",
        "flux": "flux_compression",
        "vacuum": "vacuum_well",
        "iota": "iota_edge",
    }.get(worst_constraint, "")

    if not key_name:
        return max(pool, key=lambda c: float(c.lgradb or -1e9))

    def metric_value(c: CandidateRow) -> float:
        if key_name not in c.metrics:
            return float("inf") if worst_constraint != "vacuum" else -float("inf")
        try:
            return float(c.metrics[key_name])
        except (TypeError, ValueError):
            return float("inf") if worst_constraint != "vacuum" else -float("inf")

    if worst_constraint == "vacuum":
        return max(pool, key=metric_value)
    return min(pool, key=metric_value)


def _lineage_child_counts(candidates: list[CandidateRow]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        for parent_hash in candidate.lineage_parent_hashes:
            counts[parent_hash] = counts.get(parent_hash, 0) + 1
    return counts


def _candidate_parent_score(
    candidate: CandidateRow,
    *,
    child_counts: dict[str, int],
) -> float:
    if candidate.lgradb is None or candidate.aspect is None:
        return -float("inf")
    if candidate.is_feasible:
        base = _potential_area(candidate.lgradb, candidate.aspect)
    else:
        base = _focus_score(candidate)
    saturation_penalty = (
        float(child_counts.get(candidate.design_hash, 0))
        * _ADAPTIVE_PARENT_SATURATION_PENALTY
    )
    return float(base) - saturation_penalty


@dataclass(frozen=True)
class ParentGroupSelection:
    group: str
    focus: CandidateRow
    partner: CandidateRow | None
    score: float
    candidate_count: int


def _select_parent_group(
    candidates: list[CandidateRow],
    *,
    max_feasibility: float,
) -> ParentGroupSelection | None:
    child_counts = _lineage_child_counts(candidates)

    near_feasible = [
        c
        for c in candidates
        if (not c.is_feasible)
        and c.lgradb is not None
        and c.aspect is not None
        and c.feasibility > 1e-2
        and c.feasibility <= max_feasibility
    ]
    feasible = [
        c
        for c in candidates
        if c.is_feasible and c.lgradb is not None and c.aspect is not None
    ]
    assigned_ids = {
        *(candidate.candidate_id for candidate in near_feasible),
        *(candidate.candidate_id for candidate in feasible),
    }
    broad = [
        c
        for c in candidates
        if c.lgradb is not None
        and c.aspect is not None
        and c.candidate_id not in assigned_ids
    ]

    groups: list[tuple[str, list[CandidateRow]]] = [
        ("near_feasible", near_feasible),
        ("feasible", feasible),
        ("broad", broad),
    ]

    best_group: ParentGroupSelection | None = None
    for group_name, pool in groups:
        if not pool:
            continue
        focus = max(
            pool,
            key=lambda candidate: _candidate_parent_score(
                candidate, child_counts=child_counts
            ),
        )
        focus_score = _candidate_parent_score(focus, child_counts=child_counts)
        worst_name, _ = _worst_constraint(focus.violations)
        worst_constraint = str(worst_name) if worst_name is not None else ""
        partner = _choose_partner(candidates, worst_constraint=worst_constraint)
        if partner is not None and partner.design_hash == focus.design_hash:
            partner = None
        selection = ParentGroupSelection(
            group=group_name,
            focus=focus,
            partner=partner,
            score=focus_score,
            candidate_count=len(pool),
        )
        if best_group is None or selection.score > best_group.score:
            best_group = selection
    return best_group


@dataclass(frozen=True)
class ProposalCommand:
    argv: list[str]


def _cmd_str(cmd: ProposalCommand) -> str:
    return " ".join(cmd.argv)


def _cmd_flag_value(cmd: ProposalCommand, flag: str) -> str | None:
    argv = cmd.argv
    if flag not in argv:
        return None
    idx = argv.index(flag)
    if idx + 1 >= len(argv):
        return None
    return str(argv[idx + 1])


def _command_family(cmd: ProposalCommand) -> str:
    value = _cmd_flag_value(cmd, "--family")
    return str(value) if value is not None else "unknown"


def _command_novelty(cmd: ProposalCommand) -> float:
    family = _command_family(cmd)
    if family == "blend":
        t_min = _cmd_flag_value(cmd, "--t-min")
        t_max = _cmd_flag_value(cmd, "--t-max")
        t_values = [abs(float(value)) for value in (t_min, t_max) if value is not None]
        return float(max(t_values) if t_values else 0.0)

    if family != "scale_groups":
        return 0.0

    novelty = 0.0
    axisym_z = _cmd_flag_value(cmd, "--axisym-z")
    axisym_r = _cmd_flag_value(cmd, "--axisym-r")
    if axisym_z is not None:
        novelty = max(novelty, abs(float(axisym_z) - 1.0))
    if axisym_r is not None:
        novelty = max(novelty, abs(float(axisym_r) - 1.0))

    argv = cmd.argv
    for idx, token in enumerate(argv):
        if token == "--scale-abs-n" and idx + 2 < len(argv):
            novelty = max(novelty, abs(float(argv[idx + 2]) - 1.0))
        if token == "--scale-m-ge" and idx + 2 < len(argv):
            novelty = max(novelty, abs(float(argv[idx + 2]) - 1.0))
    return float(novelty)


def _allow_near_duplicate_command(candidate: NoveltyCandidate) -> bool:
    novelty = candidate.novelty_score
    if novelty is None:
        novelty = candidate.embedding_distance
    if novelty is None:
        return False
    novelty_value = float(novelty)
    if not math.isfinite(novelty_value):
        return False
    midpoint = 0.5 * (
        float(_ADAPTIVE_NOVELTY_GATE) + float(_ADAPTIVE_NEAR_DUPLICATE_GATE)
    )
    return novelty_value >= midpoint


def _gate_adaptive_commands(
    cmds: list[ProposalCommand],
) -> tuple[list[tuple[ProposalCommand, str, float]], dict]:
    command_by_label: dict[str, tuple[ProposalCommand, str, float]] = {}
    novelty_candidates: list[NoveltyCandidate] = []
    for idx, cmd in enumerate(cmds):
        label = str(idx)
        family = _command_family(cmd)
        novelty = _command_novelty(cmd)
        command_by_label[label] = (cmd, family, novelty)
        novelty_candidates.append(
            NoveltyCandidate(
                label=label,
                embedding_distance=novelty,
                novelty_score=novelty,
                feasibility=0.0,
            )
        )

    near_duplicate_distance = max(
        float(_ADAPTIVE_NOVELTY_GATE),
        float(_ADAPTIVE_NEAR_DUPLICATE_GATE),
    )
    selected_candidates, diagnostics = apply_two_stage_novelty_gate(
        novelty_candidates,
        embedding_prefilter_min_distance=float(_ADAPTIVE_NOVELTY_GATE),
        near_duplicate_distance=near_duplicate_distance,
        judge=_allow_near_duplicate_command,
        judge_label="heuristic",
        fallback_to_ungated=False,
    )
    selected = [
        command_by_label[candidate.label]
        for candidate in selected_candidates
        if candidate.label in command_by_label
    ]
    return selected, diagnostics


def _feasible_yield_per_100(rows: list[CandidateRow]) -> float:
    if not rows:
        return 0.0
    feasible_count = sum(1 for row in rows if row.is_feasible)
    return 100.0 * float(feasible_count) / float(len(rows))


def _route_hv(rows: list[CandidateRow]) -> float:
    points = [
        (float(row.lgradb), float(row.aspect))
        for row in rows
        if row.is_feasible and row.lgradb is not None and row.aspect is not None
    ]
    return _compute_hv(points)


def _compute_model_router_reward_event(
    *,
    history_candidates: list[CandidateRow],
    model_route: str,
    window_size: int,
) -> dict:
    route_key = str(model_route)
    window = int(window_size)
    route_rows = [
        row
        for row in history_candidates
        if str(row.model_route or "unknown") == route_key
    ]
    current_window = route_rows[:window]
    previous_window = route_rows[window : (2 * window)]
    previous_feasible_yield = _feasible_yield_per_100(previous_window)
    current_feasible_yield = _feasible_yield_per_100(current_window)
    previous_hv = _route_hv(previous_window)
    current_hv = _route_hv(current_window)
    reward_eligible = len(current_window) == window and len(previous_window) == window
    payload = compute_model_router_reward(
        previous_feasible_yield=previous_feasible_yield,
        current_feasible_yield=current_feasible_yield,
        previous_hv=previous_hv,
        current_hv=current_hv,
        feasible_weight=float(_ROUTER_REWARD_FEASIBLE_WEIGHT),
        hv_weight=float(_ROUTER_REWARD_HV_WEIGHT),
    )
    if not reward_eligible:
        payload.update(
            {
                "delta_feasible_yield": 0.0,
                "relative_feasible_yield": 0.0,
                "delta_hv": 0.0,
                "relative_hv": 0.0,
                "reward_raw": float(payload.get("reward", 0.0)),
                "reward": 0.0,
                "reward_eligible": False,
                "eligibility_reason": "insufficient_route_history",
            }
        )
    else:
        payload["reward_eligible"] = True
        payload["eligibility_reason"] = "ok"
    payload["model_route"] = route_key
    payload["window_size"] = window
    payload["current_window_rows"] = len(current_window)
    payload["previous_window_rows"] = len(previous_window)
    payload["route_rows_available"] = len(route_rows)
    return payload


def _operator_reward(candidate: CandidateRow) -> float:
    if candidate.lgradb is None or candidate.aspect is None:
        return 0.0
    area = _potential_area(candidate.lgradb, candidate.aspect)
    if candidate.is_feasible:
        return area
    return (0.1 * area) - (10.0 * float(candidate.feasibility))


def _operator_bandit_scores(
    candidates: list[CandidateRow],
    *,
    families: set[str],
) -> dict[str, float]:
    counts: dict[str, int] = {}
    rewards: dict[str, float] = {}
    for candidate in candidates:
        family = str(candidate.operator_family or "unknown")
        counts[family] = counts.get(family, 0) + 1
        rewards[family] = rewards.get(family, 0.0) + _operator_reward(candidate)

    total = sum(counts.values())
    family_count = max(len(families), 1)
    scores: dict[str, float] = {}
    for family in sorted(families):
        c = counts.get(family, 0)
        reward_sum = rewards.get(family, 0.0)
        mean_reward = reward_sum / float(c) if c > 0 else 0.0
        exploration = _ADAPTIVE_EXPLORATION_WEIGHT * math.sqrt(
            math.log(float(total + family_count + 1)) / float(c + 1)
        )
        scores[family] = float(mean_reward + exploration)
    return scores


def _build_blend_cmd(
    *,
    db: Path,
    experiment_id: int,
    run_dir: Path,
    batch_id: int,
    seed_base: int,
    parent_a: Path,
    parent_b: Path,
    t_min: float,
    t_max: float,
    t_step: float,
    model_route: str,
) -> ProposalCommand:
    return ProposalCommand(
        argv=[
            "python",
            "scripts/p3_propose.py",
            "--db",
            str(db),
            "--experiment-id",
            str(experiment_id),
            "--run-dir",
            str(run_dir),
            "--batch-id",
            str(batch_id),
            "--seed-base",
            str(seed_base),
            "--family",
            "blend",
            "--model-route",
            str(model_route),
            "--parent-a",
            str(parent_a),
            "--parent-b",
            str(parent_b),
            "--t-min",
            f"{t_min:.6f}",
            "--t-max",
            f"{t_max:.6f}",
            "--t-step",
            f"{t_step:.6f}",
        ]
    )


def _build_scale_cmd(
    *,
    db: Path,
    experiment_id: int,
    run_dir: Path,
    batch_id: int,
    seed_base: int,
    parent: Path,
    axisym_z: float | None = None,
    axisym_r: float | None = None,
    scale_abs_n: tuple[int, float] | None = None,
    scale_m_ge: tuple[int, float] | None = None,
    model_route: str = "governor_static_recipe",
) -> ProposalCommand:
    argv = [
        "python",
        "scripts/p3_propose.py",
        "--db",
        str(db),
        "--experiment-id",
        str(experiment_id),
        "--run-dir",
        str(run_dir),
        "--batch-id",
        str(batch_id),
        "--seed-base",
        str(seed_base),
        "--family",
        "scale_groups",
        "--model-route",
        str(model_route),
        "--parent",
        str(parent),
    ]
    if axisym_z is not None:
        argv += ["--axisym-z", f"{axisym_z:.6f}"]
    if axisym_r is not None:
        argv += ["--axisym-r", f"{axisym_r:.6f}"]
    if scale_abs_n is not None:
        abs_n, factor = scale_abs_n
        argv += ["--scale-abs-n", str(int(abs_n)), f"{float(factor):.6f}"]
    if scale_m_ge is not None:
        m_min, factor = scale_m_ge
        argv += ["--scale-m-ge", str(int(m_min)), f"{float(factor):.6f}"]
    return ProposalCommand(argv=argv)


def _recent_data_plane_summary(
    candidates: list[CandidateRow], *, novelty_reject_threshold: float
) -> dict:
    samples = [
        DataPlaneSample(
            has_lineage=bool(candidate.lineage_parent_hashes),
            novelty_score=float(candidate.novelty_score)
            if candidate.novelty_score is not None
            else None,
            operator_family=candidate.operator_family or "unknown",
            model_route=candidate.model_route or "unknown",
        )
        for candidate in candidates
    ]
    return summarize_data_plane(
        samples,
        novelty_reject_threshold=float(novelty_reject_threshold),
    )


def _bootstrap_route_label(*, adaptive: bool) -> str:
    if adaptive:
        return "governor_adaptive/bootstrap"
    return "governor_static_recipe/bootstrap"


def _ensure_parent_file(run_dir: Path, *, design_hash: str, boundary: dict) -> Path:
    path = run_dir / "candidates" / f"{design_hash}.json"
    if not path.exists():
        path.write_text(json.dumps(boundary, indent=2))
    return path


def _select_static_recipe(
    *,
    db: Path,
    experiment_id: int,
    run_dir: Path,
    batch_id: int,
    seed_base: int,
    focus: CandidateRow,
    partner: CandidateRow | None,
    route_prefix: str = "governor_static_recipe",
) -> tuple[list[ProposalCommand], dict]:
    # Ensure parent boundary JSONs exist (we can pull from DB if needed).
    conn = _connect(db)
    try:
        row = conn.execute(
            "SELECT params_json FROM candidates WHERE experiment_id = ? AND design_hash = ? LIMIT 1",
            (experiment_id, focus.design_hash),
        ).fetchone()
        if row is None:
            raise ValueError("Focus candidate missing from candidates table.")
        focus_boundary = json.loads(str(row["params_json"]))
    finally:
        conn.close()

    focus_path = _ensure_parent_file(
        run_dir, design_hash=focus.design_hash, boundary=focus_boundary
    )

    worst_name, worst_val = _worst_constraint(focus.violations)
    worst = str(worst_name) if worst_name is not None else ""

    cmds: list[ProposalCommand] = []
    route_label = f"{str(route_prefix)}/{worst or 'unknown'}"

    # Scale sweep around the focus point (1D, ~4-6 candidates).
    if worst == "mirror":
        axisym_z_values = [
            0.95,
            0.96,
            0.965,
            0.97,
            0.975,
            0.98,
            0.985,
            0.99,
            0.995,
            1.0,
        ]
        for i, sz in enumerate(axisym_z_values):
            cmds.append(
                _build_scale_cmd(
                    db=db,
                    experiment_id=experiment_id,
                    run_dir=run_dir,
                    batch_id=batch_id,
                    seed_base=seed_base + 10 + i,
                    parent=focus_path,
                    axisym_z=sz,
                    model_route=route_label,
                )
            )
        for j, factor in enumerate([0.85, 0.9, 0.95]):
            cmds.append(
                _build_scale_cmd(
                    db=db,
                    experiment_id=experiment_id,
                    run_dir=run_dir,
                    batch_id=batch_id,
                    seed_base=seed_base + 40 + j,
                    parent=focus_path,
                    scale_m_ge=(3, factor),
                    model_route=route_label,
                )
            )
        combo_index = 0
        for sz in [0.96, 0.965, 0.97]:
            for factor in [0.9, 0.95]:
                cmds.append(
                    _build_scale_cmd(
                        db=db,
                        experiment_id=experiment_id,
                        run_dir=run_dir,
                        batch_id=batch_id,
                        seed_base=seed_base + 60 + combo_index,
                        parent=focus_path,
                        axisym_z=sz,
                        scale_m_ge=(3, factor),
                        model_route=route_label,
                    )
                )
                combo_index += 1
    elif worst == "log10_qi":
        for i, factor in enumerate([0.85, 0.9, 0.95]):
            cmds.append(
                _build_scale_cmd(
                    db=db,
                    experiment_id=experiment_id,
                    run_dir=run_dir,
                    batch_id=batch_id,
                    seed_base=seed_base + 10 + i,
                    parent=focus_path,
                    scale_m_ge=(3, factor),
                    model_route=route_label,
                )
            )
        for i, factor in enumerate([0.96, 0.98]):
            cmds.append(
                _build_scale_cmd(
                    db=db,
                    experiment_id=experiment_id,
                    run_dir=run_dir,
                    batch_id=batch_id,
                    seed_base=seed_base + 20 + i,
                    parent=focus_path,
                    scale_abs_n=(1, factor),
                    model_route=route_label,
                )
            )
        for i, factor in enumerate([1.02, 1.04, 1.06]):
            cmds.append(
                _build_scale_cmd(
                    db=db,
                    experiment_id=experiment_id,
                    run_dir=run_dir,
                    batch_id=batch_id,
                    seed_base=seed_base + 30 + i,
                    parent=focus_path,
                    scale_abs_n=(3, factor),
                    model_route=route_label,
                )
            )
    elif worst == "flux":
        for i, factor in enumerate([0.7, 0.8, 0.9]):
            cmds.append(
                _build_scale_cmd(
                    db=db,
                    experiment_id=experiment_id,
                    run_dir=run_dir,
                    batch_id=batch_id,
                    seed_base=seed_base + 10 + i,
                    parent=focus_path,
                    scale_m_ge=(3, factor),
                    model_route=route_label,
                )
            )
    elif worst == "vacuum":
        for i, factor in enumerate([0.9, 0.95, 0.98]):
            cmds.append(
                _build_scale_cmd(
                    db=db,
                    experiment_id=experiment_id,
                    run_dir=run_dir,
                    batch_id=batch_id,
                    seed_base=seed_base + 10 + i,
                    parent=focus_path,
                    scale_m_ge=(2, factor),
                    model_route=route_label,
                )
            )
    elif worst == "iota":
        for i, factor in enumerate([1.02, 1.04, 1.06]):
            cmds.append(
                _build_scale_cmd(
                    db=db,
                    experiment_id=experiment_id,
                    run_dir=run_dir,
                    batch_id=batch_id,
                    seed_base=seed_base + 10 + i,
                    parent=focus_path,
                    scale_abs_n=(1, factor),
                    model_route=route_label,
                )
            )

    # Objective-only exploration fallback when no explicit worst-constraint recipe
    # applies (for example, feasible parents with empty violations).
    if not cmds:
        objective_sweep: list[tuple[float | None, tuple[int, float] | None]] = [
            (0.96, None),
            (1.04, None),
            (None, (3, 0.92)),
            (None, (3, 1.08)),
        ]
        for i, (axisym_z, scale_m_ge) in enumerate(objective_sweep):
            cmds.append(
                _build_scale_cmd(
                    db=db,
                    experiment_id=experiment_id,
                    run_dir=run_dir,
                    batch_id=batch_id,
                    seed_base=seed_base + 10 + i,
                    parent=focus_path,
                    axisym_z=axisym_z,
                    scale_m_ge=scale_m_ge,
                    model_route=route_label,
                )
            )

    # Blend sweep towards a partner that is better on the worst constraint.
    if partner is not None:
        conn2 = _connect(db)
        try:
            row2 = conn2.execute(
                "SELECT params_json FROM candidates WHERE experiment_id = ? AND design_hash = ? LIMIT 1",
                (experiment_id, partner.design_hash),
            ).fetchone()
            partner_boundary = (
                json.loads(str(row2["params_json"])) if row2 is not None else None
            )
        finally:
            conn2.close()

        if partner_boundary is not None:
            partner_path = _ensure_parent_file(
                run_dir, design_hash=partner.design_hash, boundary=partner_boundary
            )
            cmds.append(
                _build_blend_cmd(
                    db=db,
                    experiment_id=experiment_id,
                    run_dir=run_dir,
                    batch_id=batch_id,
                    seed_base=seed_base,
                    parent_a=focus_path,
                    parent_b=partner_path,
                    t_min=0.0,
                    t_max=0.1,
                    t_step=0.02,
                    model_route=route_label,
                )
            )

    decision = {
        "batch_id": batch_id,
        "focus": {
            "design_hash": focus.design_hash,
            "candidate_id": focus.candidate_id,
            "feasibility": focus.feasibility,
            "aspect": focus.aspect,
            "lgradb": focus.lgradb,
            "worst_constraint": worst,
            "worst_violation": worst_val,
        },
        "partner": None
        if partner is None
        else {
            "design_hash": partner.design_hash,
            "candidate_id": partner.candidate_id,
            "aspect": partner.aspect,
            "lgradb": partner.lgradb,
        },
        "commands": [_cmd_str(c) for c in cmds],
        "model_route": route_label,
        "recipe_mode": "static",
        "created_at": _utc_now_iso(),
    }
    return cmds, decision


def _select_adaptive_recipe(
    *,
    db: Path,
    experiment_id: int,
    run_dir: Path,
    batch_id: int,
    seed_base: int,
    focus: CandidateRow,
    partner: CandidateRow | None,
    history_candidates: list[CandidateRow],
    parent_group: str,
) -> tuple[list[ProposalCommand], dict]:
    cmds, decision = _select_static_recipe(
        db=db,
        experiment_id=experiment_id,
        run_dir=run_dir,
        batch_id=batch_id,
        seed_base=seed_base,
        focus=focus,
        partner=partner,
        route_prefix=f"governor_adaptive/{str(parent_group)}",
    )
    families = {_command_family(cmd) for cmd in cmds}
    bandit_scores = _operator_bandit_scores(
        history_candidates,
        families=families,
    )

    gated_commands, novelty_gate = _gate_adaptive_commands(cmds)
    scored: list[tuple[float, ProposalCommand, str, float]] = []
    for cmd, family, novelty in gated_commands:
        score = bandit_scores.get(family, 0.0) + novelty
        scored.append((score, cmd, family, novelty))

    if not scored:
        fallback_cmds, fallback_decision = _select_static_recipe(
            db=db,
            experiment_id=experiment_id,
            run_dir=run_dir,
            batch_id=batch_id,
            seed_base=seed_base,
            focus=focus,
            partner=partner,
            route_prefix="governor_adaptive_scaffold/static_delegate",
        )
        fallback_decision["recipe_mode"] = "adaptive"
        fallback_decision["adaptive_policy"] = {
            "version": "v1",
            "strategy": "static_delegate_fallback",
            "parent_group": str(parent_group),
            "bandit_scores": bandit_scores,
            "novelty_gate": novelty_gate,
            "novelty_reject_count": int(novelty_gate.get("rejected_count", 0)),
            "candidate_command_count": len(cmds),
            "selected_command_count": len(fallback_cmds),
        }
        return fallback_cmds, fallback_decision

    scored.sort(key=lambda item: item[0], reverse=True)
    selected = scored[: min(_ADAPTIVE_MAX_COMMANDS, len(scored))]
    selected_cmds = [item[1] for item in selected]
    selected_families = sorted({item[2] for item in selected})
    selected_novelties = [item[3] for item in selected]

    decision["commands"] = [_cmd_str(cmd) for cmd in selected_cmds]
    decision["recipe_mode"] = "adaptive"
    decision["adaptive_policy"] = {
        "version": "v1",
        "strategy": "parent_group_operator_bandit_novelty_gate",
        "parent_group": str(parent_group),
        "bandit_scores": bandit_scores,
        "novelty_gate": novelty_gate,
        "novelty_reject_count": int(novelty_gate.get("rejected_count", 0)),
        "candidate_command_count": len(cmds),
        "selected_command_count": len(selected_cmds),
        "selected_operator_families": selected_families,
        "selected_avg_novelty": (
            float(sum(selected_novelties) / len(selected_novelties))
            if selected_novelties
            else None
        ),
    }
    return selected_cmds, decision


def _run_cmds(cmds: Iterable[ProposalCommand]) -> None:
    for cmd in cmds:
        subprocess.run(cmd.argv, check=True)


def _log_model_router_reward_event(
    conn: sqlite3.Connection,
    *,
    experiment_id: int,
    problem: str,
    event: dict,
) -> None:
    conn.execute(
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
            str(event.get("model_route", "unknown")),
            int(event.get("window_size", 0)),
            float(event.get("previous_feasible_yield", 0.0)),
            float(event.get("current_feasible_yield", 0.0)),
            float(event.get("previous_hv", 0.0)),
            float(event.get("current_hv", 0.0)),
            float(event.get("reward", 0.0)),
            json.dumps(event, separators=(",", ":")),
            _utc_now_iso(),
        ),
    )


def _log_governor_artifact(
    conn: sqlite3.Connection, *, experiment_id: int, path: Path
) -> None:
    conn.execute(
        "INSERT INTO artifacts (experiment_id, path, kind) VALUES (?, ?, ?)",
        (experiment_id, str(path), "governor_decision"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="P3 proposal governor.")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("reports/p3_world_model.sqlite"),
        help="SQLite DB path for P3 runs.",
    )
    parser.add_argument("--experiment-id", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--queue-multiplier", type=int, default=2)
    parser.add_argument("--max-focus-feas", type=float, default=0.25)
    parser.add_argument("--recent-limit", type=int, default=500)
    parser.add_argument("--record-hv", type=float, default=_DEFAULT_RECORD_HV)
    parser.add_argument("--sleep-sec", type=float, default=15.0)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually enqueue proposals by invoking p3_propose.py.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Run continuously: keep queue filled and adjust batches over time.",
    )
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Enable adaptive parent-group + operator-bandit + novelty-gate policy.",
    )
    parser.add_argument(
        "--bootstrap-parent-a",
        type=Path,
        default=None,
        help="Optional bootstrap: parent A boundary JSON for an initial blend sweep.",
    )
    parser.add_argument(
        "--bootstrap-parent-b",
        type=Path,
        default=None,
        help="Optional bootstrap: parent B boundary JSON for an initial blend sweep.",
    )
    parser.add_argument("--bootstrap-t-min", type=float, default=0.85)
    parser.add_argument("--bootstrap-t-max", type=float, default=0.95)
    parser.add_argument("--bootstrap-t-step", type=float, default=0.005)
    args = parser.parse_args()

    target_queue = int(args.workers) * int(args.queue_multiplier)

    conn = _connect(args.db)
    try:
        while True:
            exp_id = int(args.experiment_id)
            pending = conn.execute(
                "SELECT COUNT(*) AS n FROM candidates WHERE experiment_id = ? AND status = 'pending'",
                (exp_id,),
            ).fetchone()
            pending_n = int(pending["n"]) if pending is not None else 0

            # Compute current feasible HV (best estimate from DB).
            feasible_rows = conn.execute(
                """
                SELECT m.objective AS lgradb, m.raw_json AS raw
                FROM metrics m
                JOIN candidates c ON m.candidate_id = c.id
                WHERE c.experiment_id = ? AND m.is_feasible = 1
                """,
                (exp_id,),
            ).fetchall()
            feasible_points: list[tuple[float, float]] = []
            for row in feasible_rows:
                payload = json.loads(str(row["raw"]))
                metrics = (
                    payload.get("metrics", {}) if isinstance(payload, dict) else {}
                )
                if not isinstance(metrics, dict):
                    continue
                if row["lgradb"] is None or "aspect_ratio" not in metrics:
                    continue
                feasible_points.append(
                    (float(row["lgradb"]), float(metrics["aspect_ratio"]))
                )
            hv_value = _compute_hv(feasible_points)

            print(
                f"[{_utc_now_iso()}] exp={exp_id} pending={pending_n}/{target_queue} hv={hv_value:.6f} record={float(args.record_hv):.6f}"
            )

            if pending_n >= target_queue:
                if not args.loop:
                    break
                time.sleep(float(args.sleep_sec))
                continue

            candidates = _fetch_candidates(
                conn, experiment_id=exp_id, limit=int(args.recent_limit)
            )
            if not candidates:
                if args.bootstrap_parent_a is None or args.bootstrap_parent_b is None:
                    print(
                        "No metrics yet and no bootstrap parents provided; nothing to propose."
                    )
                    if not args.loop:
                        break
                    time.sleep(float(args.sleep_sec))
                    continue

                batch_id = _next_batch_id(args.run_dir)
                seed_base = int(args.experiment_id) * 10_000_000 + int(batch_id) * 1_000
                bootstrap_route = _bootstrap_route_label(adaptive=bool(args.adaptive))
                cmd = _build_blend_cmd(
                    db=args.db,
                    experiment_id=exp_id,
                    run_dir=args.run_dir,
                    batch_id=batch_id,
                    seed_base=seed_base,
                    parent_a=args.bootstrap_parent_a,
                    parent_b=args.bootstrap_parent_b,
                    t_min=float(args.bootstrap_t_min),
                    t_max=float(args.bootstrap_t_max),
                    t_step=float(args.bootstrap_t_step),
                    model_route=bootstrap_route,
                )
                decision = {
                    "batch_id": batch_id,
                    "bootstrap": True,
                    "commands": [_cmd_str(cmd)],
                    "model_route": bootstrap_route,
                    "governor_mode": "adaptive" if bool(args.adaptive) else "static",
                    "created_at": _utc_now_iso(),
                }
                artifact_path = (
                    args.run_dir
                    / "governor"
                    / f"governor_bootstrap_batch_{batch_id:03}_{_utc_stamp()}.json"
                )
                _write_json(artifact_path, decision)
                with conn:
                    _log_governor_artifact(
                        conn, experiment_id=exp_id, path=artifact_path
                    )
                print(_cmd_str(cmd))
                if args.execute:
                    _run_cmds([cmd])
                if not args.loop:
                    break
                time.sleep(float(args.sleep_sec))
                continue

            parent_group_meta: dict[str, object] | None = None
            if args.adaptive:
                parent_selection = _select_parent_group(
                    candidates,
                    max_feasibility=float(args.max_focus_feas),
                )
                if parent_selection is None:
                    print("No parent-group candidate found in recent window.")
                    if not args.loop:
                        break
                    time.sleep(float(args.sleep_sec))
                    continue
                focus = parent_selection.focus
                partner = parent_selection.partner
                parent_group_meta = {
                    "group": parent_selection.group,
                    "score": parent_selection.score,
                    "candidate_count": parent_selection.candidate_count,
                }
            else:
                focus = _choose_focus(
                    candidates, max_feasibility=float(args.max_focus_feas)
                )
                if focus is None:
                    print("No near-feasible focus candidate found in recent window.")
                    if not args.loop:
                        break
                    time.sleep(float(args.sleep_sec))
                    continue
                worst_name, _worst_val = _worst_constraint(focus.violations)
                worst = str(worst_name) if worst_name is not None else ""
                partner = _choose_partner(candidates, worst_constraint=worst)

            batch_id = _next_batch_id(args.run_dir)
            seed_base = int(args.experiment_id) * 10_000_000 + int(batch_id) * 1_000
            if args.adaptive:
                assert parent_group_meta is not None
                cmds, decision = _select_adaptive_recipe(
                    db=args.db,
                    experiment_id=exp_id,
                    run_dir=args.run_dir,
                    batch_id=batch_id,
                    seed_base=seed_base,
                    focus=focus,
                    partner=partner,
                    history_candidates=candidates,
                    parent_group=str(parent_group_meta["group"]),
                )
            else:
                cmds, decision = _select_static_recipe(
                    db=args.db,
                    experiment_id=exp_id,
                    run_dir=args.run_dir,
                    batch_id=batch_id,
                    seed_base=seed_base,
                    focus=focus,
                    partner=partner,
                )

            decision["focus_meta"] = asdict(
                focus
            )  # traceability; includes violations/metrics/meta
            decision["partner_meta"] = None if partner is None else asdict(partner)
            decision["recent_data_plane"] = _recent_data_plane_summary(
                candidates,
                novelty_reject_threshold=_NOVELTY_REJECT_THRESHOLD,
            )
            if parent_group_meta is not None:
                decision["parent_group"] = parent_group_meta
            decision["hv_at_decision"] = hv_value
            decision["record_hv"] = float(args.record_hv)
            decision["governor_mode"] = "adaptive" if bool(args.adaptive) else "static"
            model_route = str(decision.get("model_route", "unknown"))
            reward_event = _compute_model_router_reward_event(
                history_candidates=candidates,
                model_route=model_route,
                window_size=int(_ROUTER_REWARD_WINDOW),
            )
            decision["model_router_reward"] = reward_event

            artifact_path = (
                args.run_dir
                / "governor"
                / f"governor_batch_{batch_id:03}_{_utc_stamp()}.json"
            )
            _write_json(artifact_path, decision)
            with conn:
                _log_governor_artifact(conn, experiment_id=exp_id, path=artifact_path)
                _log_model_router_reward_event(
                    conn,
                    experiment_id=exp_id,
                    problem="p3",
                    event=reward_event,
                )

            for cmd in cmds:
                print(_cmd_str(cmd))
            if args.execute and cmds:
                _run_cmds(cmds)

            if not args.loop:
                break
            time.sleep(float(args.sleep_sec))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
