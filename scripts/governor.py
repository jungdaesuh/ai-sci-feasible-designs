#!/usr/bin/env python
# ruff: noqa: E402
"""P3 Governor: decide what to propose next based on DB progress.

This is the "intervention" layer:
- reads recent outcomes from the WorldModel SQLite DB
- identifies the highest-leverage near-feasible candidate and its worst constraint
- proposes the next batch by calling a problem-scoped proposal backend script

By default it prints the commands it would run. Use --execute to actually enqueue.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shlex
import sqlite3
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence, TextIO

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ai_scientist.llm_controller import (
    build_observation,
    decide,
    evaluate_phase_transition,
    select_restart_plan,
)
from ai_scientist.model_router_reward import compute_model_router_reward
from ai_scientist.novelty_gate import NoveltyCandidate, apply_two_stage_novelty_gate
from ai_scientist.p3_data_plane import DataPlaneSample, summarize_data_plane
from ai_scientist.p3_enqueue import candidate_seed, sanitize_candidate_boundary
from ai_scientist.problem_profiles import (
    FrontierRecipeConfig,
    ProblemProfile,
    RunSurgeryPolicy,
    get_problem_profile,
)
from ai_scientist.staged_governor import (
    build_staged_seed_plan_from_snapshots,
    worst_constraint_from_violations,
)
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
_RESUME_MANIFEST_VERSION = 1


@dataclass
class WorkerRuntime:
    worker_id: int
    process: subprocess.Popen[str]
    log_handle: TextIO
    cmd: list[str]
    log_path: Path


@dataclass(frozen=True)
class AutonomyProgress:
    last_metric_count: int
    best_feasibility_seen: float | None
    best_objective_feasible_seen: float | None
    evals_since_feasibility_improve: int
    evals_since_objective_improve: int


@dataclass
class RunSurgeryState:
    no_objective_progress_windows: int = 0
    no_feasibility_progress_windows: int = 0
    previous_action_signature: str | None = None
    last_action_signature: str | None = None
    same_action_windows: int = 0
    operator_shift_lock_remaining: int = 0
    operator_shift_index: int = 0
    autoscale_cooldown_remaining: int = 0
    last_rebootstrap_pair_hash: str | None = None


def _default_proposal_script(problem: str) -> str:
    candidate = _REPO_ROOT / "scripts" / f"{problem}_propose.py"
    if candidate.exists():
        return str(candidate.relative_to(_REPO_ROOT))
    return "scripts/p3_propose.py"


def _default_run_seed(*, experiment_id: int, problem: str) -> int:
    offsets = {"p1": 101_001, "p2": 202_002, "p3": 303_003}
    return int(experiment_id) * 10_000_000 + int(offsets.get(problem, 0))


def _resolve_manifest_path(*, run_dir: Path, override: Path | None) -> Path:
    if override is not None:
        return override
    return run_dir / "governor" / "resume_manifest.json"


def _load_or_init_resume_manifest(
    *,
    path: Path,
    experiment_id: int,
    problem: str,
    proposal_script: str,
    run_seed_override: int | None,
) -> dict:
    if path.exists():
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid resume manifest object at {path}.")
        version = int(payload.get("version", -1))
        if version != int(_RESUME_MANIFEST_VERSION):
            raise ValueError(
                f"Resume manifest version mismatch at {path}: {version} != {_RESUME_MANIFEST_VERSION}."
            )
        manifest_experiment = int(payload.get("experiment_id", -1))
        if manifest_experiment != int(experiment_id):
            raise ValueError(
                "Resume manifest experiment_id mismatch: "
                f"{manifest_experiment} != {int(experiment_id)}."
            )
        manifest_problem = str(payload.get("problem", ""))
        if manifest_problem != str(problem):
            raise ValueError(
                f"Resume manifest problem mismatch: {manifest_problem!r} != {str(problem)!r}."
            )
        manifest_run_seed = int(payload.get("run_seed", -1))
        if run_seed_override is not None and manifest_run_seed != int(
            run_seed_override
        ):
            raise ValueError(
                "Resume manifest run_seed mismatch: "
                f"{manifest_run_seed} != {int(run_seed_override)}."
            )
        cycles = payload.get("cycles")
        if not isinstance(cycles, dict):
            raise ValueError(f"Resume manifest cycles must be an object at {path}.")
        payload["proposal_script"] = str(
            payload.get("proposal_script", proposal_script)
        )
        manifest_phase = (
            str(payload.get("phase", "feasibility_recovery")).strip().lower()
        )
        if manifest_phase not in {"feasibility_recovery", "frontier_improvement"}:
            manifest_phase = "feasibility_recovery"
        payload["phase"] = manifest_phase
        payload["cycles"] = cycles
        for cycle_payload in payload["cycles"].values():
            if not isinstance(cycle_payload, dict):
                continue
            command_argvs = cycle_payload.get("command_argvs")
            if not isinstance(command_argvs, list):
                cycle_payload["command_argvs"] = []
        return payload

    run_seed = (
        int(run_seed_override)
        if run_seed_override is not None
        else _default_run_seed(experiment_id=experiment_id, problem=problem)
    )
    payload: dict = {
        "version": int(_RESUME_MANIFEST_VERSION),
        "experiment_id": int(experiment_id),
        "problem": str(problem),
        "proposal_script": str(proposal_script),
        "run_seed": int(run_seed),
        "phase": "feasibility_recovery",
        "last_cycle": 0,
        "cycles": {},
        "created_at": _utc_now_iso(),
        "updated_at": _utc_now_iso(),
    }
    _write_json(path, payload)
    return payload


def _command_fingerprint(cmd: ProposalCommand) -> str:
    encoded = json.dumps(cmd.argv, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _command_seed_base(cmd: ProposalCommand) -> int:
    value = _cmd_flag_value(cmd, "--seed-base")
    if value is None:
        raise ValueError("Proposal command missing --seed-base.")
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError("Proposal command has invalid --seed-base.") from exc


def _expected_candidates_for_command(cmd: ProposalCommand) -> int:
    family = _command_family(cmd)
    if family != "blend":
        return 1

    explicit_t_count = sum(1 for token in cmd.argv if token == "--t")
    if explicit_t_count > 0:
        return int(explicit_t_count)

    t_min = _cmd_flag_value(cmd, "--t-min")
    t_max = _cmd_flag_value(cmd, "--t-max")
    t_step = _cmd_flag_value(cmd, "--t-step")
    if t_min is None or t_max is None or t_step is None:
        return 1
    try:
        t_min_f = float(t_min)
        t_max_f = float(t_max)
        t_step_f = float(t_step)
    except ValueError:
        return 1
    if t_step_f <= 0:
        return 1
    if t_min_f > t_max_f + 1e-12:
        return 0
    count = int(math.floor((t_max_f - t_min_f) / t_step_f + 1e-12)) + 1
    return max(count, 0)


def _expected_candidate_volume(cmds: Sequence[ProposalCommand]) -> int:
    return int(sum(_expected_candidates_for_command(cmd) for cmd in cmds))


def _command_knob_signature(cmds: Sequence[ProposalCommand]) -> str:
    fingerprints = sorted(_command_fingerprint(cmd) for cmd in cmds)
    payload = json.dumps(fingerprints, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _consecutive_no_progress_action_repeats(
    conn: sqlite3.Connection,
    *,
    experiment_id: int,
    action: str,
    target_constraint: str,
    knob_signature: str,
    limit: int = 20,
) -> int:
    rows = conn.execute(
        """
        SELECT
            r.cycle AS cycle,
            r.outcome_json AS reflection_outcome_json,
            p.diagnostics_json AS plan_diagnostics_json
        FROM scratchpad_events r
        LEFT JOIN scratchpad_events p
          ON p.experiment_id = r.experiment_id
         AND p.cycle = r.cycle
         AND p.step = 0
        WHERE r.experiment_id = ?
          AND r.step = 1
          AND r.aso_action = 'reflection'
        ORDER BY r.id DESC
        LIMIT ?
        """,
        (int(experiment_id), int(limit)),
    ).fetchall()
    if not rows:
        return 0

    count = 0
    for row in rows:
        try:
            plan_diag = json.loads(str(row["plan_diagnostics_json"]))
            outcome = json.loads(str(row["reflection_outcome_json"]))
        except (TypeError, ValueError, json.JSONDecodeError):
            break
        if not isinstance(plan_diag, dict) or not isinstance(outcome, dict):
            break

        row_action = str(plan_diag.get("action", ""))
        row_target = str(plan_diag.get("predicted_target_metric", ""))
        row_signature = str(plan_diag.get("knob_signature", ""))
        if (
            row_action != str(action)
            or row_target != str(target_constraint)
            or row_signature != str(knob_signature)
        ):
            break

        realized_feas = _as_finite_float(outcome.get("realized_feasibility_delta"))
        realized_hv = _as_finite_float(outcome.get("realized_hv_delta"))
        has_progress = (realized_feas is not None and realized_feas > 0.0) or (
            realized_hv is not None and realized_hv > 0.0
        )
        if has_progress:
            break
        count += 1

    return count


def _record_cycle_manifest(
    *,
    manifest: dict,
    path: Path,
    batch_id: int,
    seed_base: int,
    cmds: list[ProposalCommand],
    model_route: str,
) -> None:
    cycle_key = str(int(batch_id))
    command_fingerprints = [_command_fingerprint(cmd) for cmd in cmds]
    command_seed_bases = [_command_seed_base(cmd) for cmd in cmds]
    expected_candidates = int(
        sum(_expected_candidates_for_command(cmd) for cmd in cmds)
    )
    cycle_payload = {
        "batch_id": int(batch_id),
        "seed_base": int(seed_base),
        "command_count": len(cmds),
        "command_fingerprints": command_fingerprints,
        "command_seed_bases": command_seed_bases,
        "expected_candidates": expected_candidates,
        "command_argvs": [list(cmd.argv) for cmd in cmds],
        "model_route": str(model_route),
        "created_at": _utc_now_iso(),
    }
    cycles = manifest.get("cycles")
    if not isinstance(cycles, dict):
        raise ValueError("Resume manifest cycles must be a mapping.")
    existing = cycles.get(cycle_key)
    if existing is not None:
        if not isinstance(existing, dict):
            raise ValueError(
                f"Resume manifest cycle entry must be an object: {cycle_key}."
            )
        existing_seed_base = int(existing.get("seed_base", -1))
        existing_fingerprints = existing.get("command_fingerprints", [])
        existing_seed_bases = existing.get("command_seed_bases", [])
        existing_argvs = existing.get("command_argvs", [])
        existing_expected_candidates_raw = existing.get("expected_candidates")
        expected_argvs = [list(cmd.argv) for cmd in cmds]
        argv_mismatch = bool(existing_argvs) and existing_argvs != expected_argvs
        expected_candidates_mismatch = (
            existing_expected_candidates_raw is not None
            and int(existing_expected_candidates_raw) != int(expected_candidates)
        )
        if (
            existing_seed_base != int(seed_base)
            or existing_fingerprints != command_fingerprints
            or existing_seed_bases != command_seed_bases
            or argv_mismatch
            or expected_candidates_mismatch
        ):
            raise ValueError(
                "Replay identity mismatch before enqueue for cycle "
                f"{cycle_key}: existing manifest entry does not match planned commands."
            )
        return

    cycles[cycle_key] = cycle_payload
    manifest["last_cycle"] = max(int(manifest.get("last_cycle", 0)), int(batch_id))
    manifest["updated_at"] = _utc_now_iso()
    _write_json(path, manifest)


def _update_manifest_phase(*, manifest: dict, path: Path, phase: str) -> None:
    normalized_phase = str(phase).strip().lower()
    if normalized_phase not in {"feasibility_recovery", "frontier_improvement"}:
        normalized_phase = "feasibility_recovery"
    manifest["phase"] = normalized_phase
    manifest["updated_at"] = _utc_now_iso()
    _write_json(path, manifest)


def _meta_batch_counts(run_dir: Path) -> dict[int, int]:
    counts: dict[int, int] = {}
    candidates_dir = run_dir / "candidates"
    for meta_path in candidates_dir.glob("*_meta.json"):
        try:
            payload = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        batch_raw = payload.get("batch_id")
        if isinstance(batch_raw, bool) or not isinstance(batch_raw, int):
            continue
        batch_id = int(batch_raw)
        counts[batch_id] = counts.get(batch_id, 0) + 1
    return counts


def _meta_batch_design_hashes(run_dir: Path) -> dict[int, set[str]]:
    hashes: dict[int, set[str]] = {}
    candidates_dir = run_dir / "candidates"
    for meta_path in candidates_dir.glob("*_meta.json"):
        try:
            payload = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        batch_raw = payload.get("batch_id")
        if isinstance(batch_raw, bool) or not isinstance(batch_raw, int):
            continue
        name = meta_path.name
        if not name.endswith("_meta.json"):
            continue
        design_hash = name[: -len("_meta.json")]
        if not design_hash:
            continue
        batch_id = int(batch_raw)
        if batch_id not in hashes:
            hashes[batch_id] = set()
        hashes[batch_id].add(design_hash)
    return hashes


def _meta_batch_seeds(run_dir: Path) -> dict[int, set[int]]:
    seeds: dict[int, set[int]] = {}
    candidates_dir = run_dir / "candidates"
    for meta_path in candidates_dir.glob("*_meta.json"):
        try:
            payload = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        batch_raw = payload.get("batch_id")
        seed_raw = payload.get("seed")
        if (
            isinstance(batch_raw, bool)
            or not isinstance(batch_raw, int)
            or isinstance(seed_raw, bool)
            or not isinstance(seed_raw, int)
        ):
            continue
        batch_id = int(batch_raw)
        if batch_id not in seeds:
            seeds[batch_id] = set()
        seeds[batch_id].add(int(seed_raw))
    return seeds


def _cmd_flag_values(cmd: ProposalCommand, flag: str) -> list[str]:
    values: list[str] = []
    argv = cmd.argv
    i = 0
    while i < len(argv):
        token = argv[i]
        if token != flag:
            i += 1
            continue
        if i + 1 < len(argv):
            values.append(str(argv[i + 1]))
            i += 2
            continue
        i += 1
    return values


def _blend_t_values_for_command(cmd: ProposalCommand) -> list[float]:
    explicit_values = _cmd_flag_values(cmd, "--t")
    if explicit_values:
        out: list[float] = []
        for value in explicit_values:
            try:
                out.append(float(value))
            except ValueError:
                return []
        return out
    t_min = _cmd_flag_value(cmd, "--t-min")
    t_max = _cmd_flag_value(cmd, "--t-max")
    t_step = _cmd_flag_value(cmd, "--t-step")
    if t_min is None or t_max is None or t_step is None:
        return []
    try:
        t_min_f = float(t_min)
        t_max_f = float(t_max)
        t_step_f = float(t_step)
    except ValueError:
        return []
    if t_step_f <= 0.0:
        return []
    out: list[float] = []
    t = t_min_f
    while t <= t_max_f + 1e-12:
        out.append(float(t))
        t += t_step_f
    return out


@dataclass(frozen=True)
class _ReplayCandidateSpec:
    cmd: ProposalCommand
    seed: int
    replay_seed_base: int
    t_value: float | None


def _candidate_replay_specs_for_command(
    *,
    cmd: ProposalCommand,
    batch_id: int,
) -> list[_ReplayCandidateSpec]:
    try:
        seed_base = _command_seed_base(cmd)
    except ValueError:
        return []
    expected = _expected_candidates_for_command(cmd)
    if expected <= 0:
        return []
    if _command_family(cmd) != "blend":
        return [
            _ReplayCandidateSpec(
                cmd=cmd,
                seed=candidate_seed(seed_base=seed_base, batch_id=batch_id, index=0),
                replay_seed_base=int(seed_base),
                t_value=None,
            )
        ]
    t_values = _blend_t_values_for_command(cmd)
    if len(t_values) != expected:
        return []
    out: list[_ReplayCandidateSpec] = []
    for idx, t_value in enumerate(t_values):
        out.append(
            _ReplayCandidateSpec(
                cmd=cmd,
                seed=candidate_seed(seed_base=seed_base, batch_id=batch_id, index=idx),
                replay_seed_base=int(seed_base) + int(idx),
                t_value=float(t_value),
            )
        )
    return out


def _is_replay_command_valid(cmd: ProposalCommand) -> bool:
    seed_base = _cmd_flag_value(cmd, "--seed-base")
    if seed_base is not None:
        try:
            int(seed_base)
        except ValueError:
            return False
    expected = _expected_candidates_for_command(cmd)
    if expected <= 0:
        return False
    if _command_family(cmd) != "blend":
        return True
    has_explicit_t = bool(_cmd_flag_values(cmd, "--t"))
    has_t_range = (
        _cmd_flag_value(cmd, "--t-min") is not None
        or _cmd_flag_value(cmd, "--t-max") is not None
        or _cmd_flag_value(cmd, "--t-step") is not None
    )
    if not has_explicit_t and not has_t_range:
        return True
    t_values = _blend_t_values_for_command(cmd)
    return len(t_values) == int(expected)


def _build_replay_command_for_spec(spec: _ReplayCandidateSpec) -> ProposalCommand:
    family = _command_family(spec.cmd)
    consumed_flags = {"--seed-base"}
    if family == "blend":
        consumed_flags.update({"--t", "--t-min", "--t-max", "--t-step"})
    rebuilt: list[str] = []
    argv = spec.cmd.argv
    i = 0
    while i < len(argv):
        token = argv[i]
        if token in consumed_flags:
            i += 2
            continue
        rebuilt.append(token)
        i += 1
    rebuilt.extend(["--seed-base", str(int(spec.replay_seed_base))])
    if family == "blend":
        if spec.t_value is None:
            raise ValueError("Blend replay candidate is missing a concrete t value.")
        rebuilt.extend(["--t", f"{float(spec.t_value):.6f}"])
    return ProposalCommand(argv=rebuilt)


def _candidate_status_by_design_hash(
    conn: sqlite3.Connection,
    *,
    experiment_id: int,
    design_hashes: set[str],
) -> dict[str, str]:
    if not design_hashes:
        return {}
    status_by_hash: dict[str, str] = {}
    design_list = sorted(design_hashes)
    chunk_size = 200
    for i in range(0, len(design_list), chunk_size):
        chunk = design_list[i : i + chunk_size]
        placeholders = ",".join("?" for _ in chunk)
        rows = conn.execute(
            f"""
            SELECT design_hash, status
            FROM candidates
            WHERE experiment_id = ? AND design_hash IN ({placeholders})
            """,
            (int(experiment_id), *chunk),
        ).fetchall()
        for row in rows:
            status_by_hash[str(row["design_hash"])] = str(row["status"])
    return status_by_hash


def _startup_replay_commands(
    *,
    manifest: Mapping[str, object],
    run_dir: Path,
    conn: sqlite3.Connection | None = None,
) -> tuple[list[ProposalCommand], dict]:
    cycles_raw = manifest.get("cycles")
    if not isinstance(cycles_raw, Mapping):
        return [], {
            "replay_batches": [],
            "partial_replay_batches": [],
            "partial_skipped_batches": [],
            "pending_skipped_batches": [],
            "replay_command_count": 0,
        }
    batch_counts = _meta_batch_counts(run_dir)
    batch_hashes = _meta_batch_design_hashes(run_dir)
    batch_seeds = _meta_batch_seeds(run_dir)
    status_by_hash: dict[str, str] = {}
    experiment_id_raw = manifest.get("experiment_id")
    if (
        conn is not None
        and isinstance(experiment_id_raw, int)
        and not isinstance(experiment_id_raw, bool)
    ):
        all_hashes: set[str] = set()
        for values in batch_hashes.values():
            all_hashes.update(values)
        status_by_hash = _candidate_status_by_design_hash(
            conn,
            experiment_id=int(experiment_id_raw),
            design_hashes=all_hashes,
        )
    replay_cmds: list[ProposalCommand] = []
    replay_batches: list[int] = []
    partial_replay_batches: list[int] = []
    partial_skipped_batches: list[int] = []
    pending_skipped_batches: list[int] = []
    sortable_cycle_keys: list[tuple[int, object]] = []
    for key in cycles_raw.keys():
        try:
            sort_key = int(str(key))
        except (TypeError, ValueError):
            continue
        sortable_cycle_keys.append((sort_key, key))
    for _sort_key, key in sorted(sortable_cycle_keys, key=lambda item: item[0]):
        cycle_payload = cycles_raw.get(key)
        if not isinstance(cycle_payload, Mapping):
            continue
        batch_id_raw = cycle_payload.get("batch_id")
        if isinstance(batch_id_raw, bool) or not isinstance(batch_id_raw, int):
            continue
        batch_id = int(batch_id_raw)
        argv_payload = cycle_payload.get("command_argvs")
        if not isinstance(argv_payload, list) or not argv_payload:
            continue
        existing_count = int(batch_counts.get(batch_id, 0))
        cycle_cmds: list[ProposalCommand] = []
        valid_cycle = True
        for item in argv_payload:
            if not isinstance(item, list) or not item:
                valid_cycle = False
                break
            tokens: list[str] = []
            for token in item:
                if not isinstance(token, str) or not token:
                    valid_cycle = False
                    break
                tokens.append(token)
            if not valid_cycle:
                break
            cycle_cmds.append(ProposalCommand(argv=tokens))
        if not valid_cycle:
            continue
        if not cycle_cmds:
            continue
        expected_candidates_raw = cycle_payload.get("expected_candidates")
        if isinstance(expected_candidates_raw, bool):
            expected_candidates_raw = None
        expected_candidates = (
            int(expected_candidates_raw)
            if isinstance(expected_candidates_raw, int)
            else int(sum(_expected_candidates_for_command(cmd) for cmd in cycle_cmds))
        )
        batch_design_hashes = batch_hashes.get(batch_id, set())
        batch_statuses = [
            status_by_hash[h] for h in batch_design_hashes if h in status_by_hash
        ]
        has_pending_or_running = any(
            status.startswith("pending") or status.startswith("running")
            for status in batch_statuses
        )
        if has_pending_or_running:
            pending_skipped_batches.append(batch_id)
            continue
        observed_progress = len(batch_statuses) if batch_statuses else existing_count
        if observed_progress >= expected_candidates:
            continue
        if observed_progress > 0:
            remaining = max(int(expected_candidates) - int(observed_progress), 0)
            if remaining <= 0:
                continue
            specs: list[_ReplayCandidateSpec] = []
            for cmd in cycle_cmds:
                specs.extend(
                    _candidate_replay_specs_for_command(cmd=cmd, batch_id=batch_id)
                )
            if not specs:
                partial_skipped_batches.append(batch_id)
                continue
            existing_batch_seeds = batch_seeds.get(batch_id, set())
            replay_specs: list[_ReplayCandidateSpec]
            overlap = sum(1 for spec in specs if spec.seed in existing_batch_seeds)
            if not existing_batch_seeds or overlap == 0:
                replay_specs = specs[int(observed_progress) :]
            else:
                replay_specs = [
                    spec for spec in specs if spec.seed not in existing_batch_seeds
                ]
            replay_specs = replay_specs[:remaining]
            if not replay_specs:
                partial_skipped_batches.append(batch_id)
                continue
            for spec in replay_specs:
                replay_cmds.append(_build_replay_command_for_spec(spec))
            partial_replay_batches.append(batch_id)
            continue
        if any(not _is_replay_command_valid(cmd) for cmd in cycle_cmds):
            partial_skipped_batches.append(batch_id)
            continue
        replay_cmds.extend(cycle_cmds)
        replay_batches.append(batch_id)
    diagnostics = {
        "replay_batches": replay_batches,
        "partial_replay_batches": partial_replay_batches,
        "partial_skipped_batches": partial_skipped_batches,
        "pending_skipped_batches": pending_skipped_batches,
        "replay_command_count": len(replay_cmds),
    }
    return replay_cmds, diagnostics


def _replay_cycle_emissions(diagnostics: Mapping[str, object]) -> int:
    emitted: set[int] = set()
    for key in ("replay_batches", "partial_replay_batches"):
        values = diagnostics.get(key)
        if not isinstance(values, list):
            continue
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int):
                continue
            emitted.add(int(value))
    return len(emitted)


def _table_columns(conn: sqlite3.Connection, *, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {str(row["name"]) for row in rows}


def _schema_compatibility_status(conn: sqlite3.Connection) -> tuple[bool, str | None]:
    required = {
        "candidates": {
            "id",
            "experiment_id",
            "problem",
            "params_json",
            "seed",
            "status",
            "design_hash",
            "operator_family",
            "model_route",
        },
        "metrics": {
            "id",
            "candidate_id",
            "raw_json",
            "feasibility",
            "objective",
            "is_feasible",
        },
        "artifacts": {"id", "experiment_id", "path", "kind"},
        "model_router_reward_events": {
            "experiment_id",
            "problem",
            "model_route",
            "window_size",
            "previous_feasible_yield",
            "current_feasible_yield",
            "previous_hv",
            "current_hv",
            "reward",
            "reward_components_json",
            "created_at",
        },
        "scratchpad_events": {
            "experiment_id",
            "cycle",
            "step",
            "planner_intent_json",
            "aso_action",
            "intent_agreement",
            "override_reason",
            "diagnostics_json",
            "outcome_json",
            "created_at",
        },
    }
    for table, columns in required.items():
        present = _table_columns(conn, table=table)
        if not present:
            return False, f"schema_incompatible:missing_table:{table}"
        missing = sorted(columns - present)
        if missing:
            return (
                False,
                f"schema_incompatible:missing_columns:{table}:{','.join(missing)}",
            )
    return True, None


def _resolve_llm_transport(
    args: argparse.Namespace,
) -> tuple[Path | None, str | None, str | None]:
    if not bool(args.llm_enabled):
        return None, None, None

    if args.llm_decision_file is not None:
        if not bool(args.llm_allow_decision_file):
            raise ValueError(
                "--llm-decision-file requires --llm-allow-decision-file in codex-only mode."
            )
        return args.llm_decision_file, None, "file"

    command = None
    if args.llm_codex_command is not None:
        command = str(args.llm_codex_command).strip()
    elif args.llm_decision_command is not None:
        command = str(args.llm_decision_command).strip()
    if not command:
        raise ValueError(
            "--llm-enabled requires --llm-codex-command "
            "(or --llm-decision-file with --llm-allow-decision-file)."
        )
    command_tokens = shlex.split(command)
    if not command_tokens:
        raise ValueError("--llm-codex-command must not be empty.")
    command_head = Path(command_tokens[0]).name.lower()
    if command_head != "codex":
        raise ValueError("codex-only transport requires argv[0] to be 'codex'.")
    return None, command, "codex_command"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def _connect(db_path: Path) -> sqlite3.Connection:
    init_db(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _resolve_worker_script_path(worker_script: str) -> Path:
    script_path = Path(worker_script)
    if not script_path.is_absolute():
        script_path = _REPO_ROOT / script_path
    return script_path.resolve()


def _spawn_worker_runtime(
    *,
    worker_id: int,
    problem: str,
    db: Path,
    experiment_id: int,
    run_dir: Path,
    worker_script_path: Path,
    worker_limit: int,
    worker_sleep_sec: float,
) -> WorkerRuntime:
    worker_log_dir = run_dir / "governor" / "workers"
    worker_log_dir.mkdir(parents=True, exist_ok=True)
    log_path = worker_log_dir / f"worker_{int(worker_id):03}.log"
    log_handle = log_path.open("a", encoding="utf-8")
    cmd = [
        sys.executable,
        str(worker_script_path),
        "--problem",
        str(problem),
        "--db",
        str(db),
        "--experiment-id",
        str(int(experiment_id)),
        "--run-dir",
        str(run_dir),
        "--worker-id",
        str(int(worker_id)),
        "--sleep-sec",
        str(float(worker_sleep_sec)),
        "--limit",
        str(int(worker_limit)),
    ]
    process = subprocess.Popen(
        cmd,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return WorkerRuntime(
        worker_id=int(worker_id),
        process=process,
        log_handle=log_handle,
        cmd=cmd,
        log_path=log_path,
    )


def _start_worker_pool(
    *,
    workers: int,
    problem: str,
    db: Path,
    experiment_id: int,
    run_dir: Path,
    worker_script_path: Path,
    worker_limit: int,
    worker_sleep_sec: float,
) -> dict[int, WorkerRuntime]:
    pool: dict[int, WorkerRuntime] = {}
    for worker_id in range(1, int(workers) + 1):
        pool[worker_id] = _spawn_worker_runtime(
            worker_id=worker_id,
            problem=problem,
            db=db,
            experiment_id=experiment_id,
            run_dir=run_dir,
            worker_script_path=worker_script_path,
            worker_limit=worker_limit,
            worker_sleep_sec=worker_sleep_sec,
        )
    return pool


def _supervise_worker_pool(
    *,
    pool: dict[int, WorkerRuntime],
    problem: str,
    db: Path,
    experiment_id: int,
    run_dir: Path,
    worker_script_path: Path,
    worker_limit: int,
    worker_sleep_sec: float,
) -> None:
    for worker_id, runtime in list(pool.items()):
        return_code = runtime.process.poll()
        if return_code is None:
            continue
        runtime.log_handle.write(
            f"[{_utc_now_iso()}] worker={int(worker_id)} exited rc={int(return_code)}; restarting.\n"
        )
        runtime.log_handle.flush()
        runtime.log_handle.close()
        pool[worker_id] = _spawn_worker_runtime(
            worker_id=worker_id,
            problem=problem,
            db=db,
            experiment_id=experiment_id,
            run_dir=run_dir,
            worker_script_path=worker_script_path,
            worker_limit=worker_limit,
            worker_sleep_sec=worker_sleep_sec,
        )


def _stop_worker_pool(*, pool: dict[int, WorkerRuntime]) -> None:
    for runtime in pool.values():
        if runtime.process.poll() is None:
            runtime.process.terminate()
            try:
                runtime.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                runtime.process.kill()
                runtime.process.wait(timeout=5.0)
        runtime.log_handle.flush()
        runtime.log_handle.close()


def _status_bucket(status: str) -> str:
    return status.split(":", 1)[0] if status else "unknown"


def _canonical_objective_utility(
    *, profile: ProblemProfile, objective_value: float
) -> float:
    value = float(objective_value)
    if profile.objective.direction == "minimize":
        return -value
    return value


def _objective_area_gain(objective_utility: float) -> float:
    value = float(objective_utility)
    if value >= 0.0:
        return 1.0 + value
    return 1.0 / (1.0 + abs(value))


def _compute_hv(points: list[tuple[float, float]]) -> float:
    if not points:
        return 0.0
    from pymoo.indicators import hv

    X = np.array([(-objective, aspect) for objective, aspect in points], dtype=float)
    ref_x = max(1.0, float(np.max(X[:, 0])) + 1.0)
    ref_y = max(20.0, float(np.max(X[:, 1])) + 1.0)
    indicator = hv.Hypervolume(ref_point=np.array([ref_x, ref_y], dtype=float))
    out = indicator(X)
    assert out is not None
    return float(out)


def _hv_gap_perturbation_scale(
    *,
    hv_value: float,
    record_hv: float,
    frontier_recipe: FrontierRecipeConfig,
) -> float:
    if not math.isfinite(record_hv) or record_hv <= 0.0:
        return float(frontier_recipe.base_perturbation_scale)
    gap_frac = max(0.0, (float(record_hv) - float(hv_value)) / float(record_hv))
    raw = float(frontier_recipe.base_perturbation_scale) * (
        1.0 + float(frontier_recipe.hv_gap_sensitivity) * gap_frac
    )
    return max(
        float(frontier_recipe.min_perturbation_scale),
        min(raw, float(frontier_recipe.max_perturbation_scale)),
    )


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


def _is_invalid_llm_override_reason(reason: object) -> bool:
    if reason is None:
        return False
    text = str(reason).strip().lower()
    return text.startswith("invalid_llm_output")


def _remaining_run_budget(
    *,
    proposal_cycles_emitted: int,
    max_cycles: int,
    elapsed_runtime_sec: float,
    max_runtime_sec: float,
) -> float:
    remaining_cycles = math.inf
    if int(max_cycles) > 0:
        remaining_cycles = float(max(int(max_cycles) - int(proposal_cycles_emitted), 0))
    remaining_runtime = math.inf
    if float(max_runtime_sec) > 0.0:
        remaining_runtime = max(
            float(max_runtime_sec) - float(elapsed_runtime_sec), 0.0
        )
    return float(min(remaining_cycles, remaining_runtime))


def _deterministic_restart_action(
    *,
    profile_problem: str,
    policy_restart_plan: str,
    partner_available: bool,
) -> str | None:
    profile = get_problem_profile(profile_problem)
    if policy_restart_plan == "global_restart":
        return "global_restart"
    if policy_restart_plan == "soft_retry":
        return "repair"
    if policy_restart_plan == "degraded_restart":
        if profile.allows_action("bridge") and partner_available:
            return "bridge"
        if profile.allows_action("jump"):
            return "jump"
        return "global_restart"
    return None


def _log_scratchpad_event(
    conn: sqlite3.Connection,
    *,
    experiment_id: int,
    cycle: int,
    step: int,
    planner_intent: Mapping[str, object],
    aso_action: str,
    intent_agreement: str,
    override_reason: str | None,
    diagnostics: Mapping[str, object],
    outcome: Mapping[str, object],
) -> None:
    conn.execute(
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
            int(experiment_id),
            int(cycle),
            int(step),
            json.dumps(planner_intent, separators=(",", ":")),
            str(aso_action),
            str(intent_agreement),
            override_reason,
            json.dumps(diagnostics, separators=(",", ":")),
            json.dumps(outcome, separators=(",", ":")),
            _utc_now_iso(),
        ),
    )


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


def _next_batch_id_from_manifest(
    *, manifest: Mapping[str, object], run_dir: Path
) -> int:
    """Monotonic cycle clock: manifest first, filesystem only as compatibility floor."""
    manifest_last_cycle = manifest.get("last_cycle")
    manifest_next = 1
    if isinstance(manifest_last_cycle, int) and not isinstance(
        manifest_last_cycle, bool
    ):
        manifest_next = max(1, int(manifest_last_cycle) + 1)
    fs_next = _next_batch_id(run_dir)
    return max(manifest_next, int(fs_next))


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
    params: dict | None = None


def _sanitize_candidate_params(candidate: CandidateRow) -> CandidateRow | None:
    if not isinstance(candidate.params, dict):
        return None
    try:
        sanitized = sanitize_candidate_boundary(candidate.params)
    except ValueError:
        return None
    return replace(candidate, params=sanitized)


def _boundary_vector(payload: Mapping[str, object]) -> list[float] | None:
    try:
        sanitized = sanitize_candidate_boundary(payload)
    except ValueError:
        return None
    vector: list[float] = []
    for matrix_key in ("r_cos", "z_sin"):
        matrix = sanitized.get(matrix_key)
        if not isinstance(matrix, list):
            return None
        for row in matrix:
            if not isinstance(row, list):
                return None
            for value in row:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    return None
                vector.append(float(value))
    return vector


def _boundary_distance(left: Sequence[float], right: Sequence[float]) -> float:
    min_len = min(len(left), len(right))
    max_len = max(len(left), len(right))
    if min_len == 0:
        return float(max_len)
    total = 0.0
    for idx in range(min_len):
        diff = float(left[idx]) - float(right[idx])
        total += diff * diff
    total += float(max_len - min_len)
    return float(math.sqrt(total))


def _nearest_valid_parent(
    *,
    reference: CandidateRow,
    valid_candidates: Sequence[CandidateRow],
    exclude_hashes: set[str] | None = None,
) -> CandidateRow | None:
    excluded = exclude_hashes or set()
    pool = [
        candidate
        for candidate in valid_candidates
        if candidate.design_hash not in excluded
    ]
    if not pool:
        return None

    ref_vector = None
    if isinstance(reference.params, dict):
        ref_vector = _boundary_vector(reference.params)

    if ref_vector is None:
        return min(
            pool,
            key=lambda candidate: (
                0 if candidate.is_feasible else 1,
                float(candidate.feasibility),
                -float(
                    candidate.novelty_score
                    if candidate.novelty_score is not None
                    else 0.0
                ),
            ),
        )

    best: CandidateRow | None = None
    best_dist = float("inf")
    for candidate in pool:
        if not isinstance(candidate.params, dict):
            continue
        vector = _boundary_vector(candidate.params)
        if vector is None:
            continue
        dist = _boundary_distance(ref_vector, vector)
        if dist < best_dist:
            best = candidate
            best_dist = dist
    if best is not None:
        return best
    return min(
        pool,
        key=lambda candidate: (
            0 if candidate.is_feasible else 1,
            float(candidate.feasibility),
        ),
    )


def _sanitize_cycle_parents(
    *,
    focus: CandidateRow,
    partner: CandidateRow | None,
    candidates: Sequence[CandidateRow],
) -> tuple[CandidateRow | None, CandidateRow | None, dict]:
    valid_candidates: list[CandidateRow] = []
    seen_hashes: set[str] = set()
    for candidate in candidates:
        if candidate.design_hash in seen_hashes:
            continue
        seen_hashes.add(candidate.design_hash)
        sanitized = _sanitize_candidate_params(candidate)
        if sanitized is not None:
            valid_candidates.append(sanitized)

    details = {
        "focus_source": "original",
        "partner_source": "original" if partner is not None else "none",
        "invalid_parent_detected": False,
        "valid_parent_pool_size": len(valid_candidates),
    }

    sanitized_focus = _sanitize_candidate_params(focus)
    if sanitized_focus is None:
        details["invalid_parent_detected"] = True
        swapped_focus = _nearest_valid_parent(
            reference=focus,
            valid_candidates=valid_candidates,
        )
        if swapped_focus is None:
            details["focus_source"] = "missing"
            return None, None, details
        sanitized_focus = swapped_focus
        details["focus_source"] = "swapped_nearest_valid"

    sanitized_partner: CandidateRow | None = None
    if partner is not None:
        sanitized_partner = _sanitize_candidate_params(partner)
        if sanitized_partner is None:
            details["invalid_parent_detected"] = True
            swapped_partner = _nearest_valid_parent(
                reference=partner,
                valid_candidates=valid_candidates,
                exclude_hashes={sanitized_focus.design_hash},
            )
            if swapped_partner is None:
                details["partner_source"] = "dropped_invalid"
            else:
                sanitized_partner = swapped_partner
                details["partner_source"] = "swapped_nearest_valid"
        if (
            sanitized_partner is not None
            and sanitized_partner.design_hash == sanitized_focus.design_hash
        ):
            sanitized_partner = None
            details["partner_source"] = "dropped_duplicate_focus"

    return sanitized_focus, sanitized_partner, details


def _fetch_candidates(
    conn: sqlite3.Connection,
    *,
    profile: ProblemProfile,
    experiment_id: int,
    limit: int,
) -> list[CandidateRow]:
    rows = conn.execute(
        """
        SELECT c.id AS candidate_id, c.design_hash AS design_hash, c.seed AS seed,
               c.params_json AS params_json,
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
        try:
            payload = json.loads(str(row["raw_json"]))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        try:
            params_payload = json.loads(str(row["params_json"]))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            nested_metrics = payload.get("metrics")
            if isinstance(nested_metrics, dict):
                metrics = nested_metrics
            else:
                metrics = {
                    key: value
                    for key, value in payload.items()
                    if key not in {"constraint_margins", "violations", "meta"}
                }
        else:
            metrics = {}
        margins = (
            payload.get("constraint_margins", {}) if isinstance(payload, dict) else {}
        )
        violations_payload = (
            payload.get("violations", {}) if isinstance(payload, dict) else {}
        )
        if isinstance(margins, dict) and margins:
            violations = margins
        elif isinstance(violations_payload, dict):
            violations = violations_payload
        elif isinstance(margins, dict):
            violations = margins
        else:
            violations = {}
        meta = payload.get("meta", {}) if isinstance(payload, dict) else {}

        aspect = None
        if isinstance(metrics, dict) and "aspect_ratio" in metrics:
            try:
                aspect = float(metrics["aspect_ratio"])
            except (TypeError, ValueError):
                aspect = None

        lgradb = None
        if row["objective"] is not None:
            try:
                lgradb = _canonical_objective_utility(
                    profile=profile,
                    objective_value=float(row["objective"]),
                )
            except (TypeError, ValueError):
                lgradb = None
        elif isinstance(metrics, dict) and "lgradB" in metrics:
            try:
                lgradb = _canonical_objective_utility(
                    profile=profile,
                    objective_value=float(metrics["lgradB"]),
                )
            except (TypeError, ValueError):
                lgradb = None

        vio_clean: dict[str, float] = {}
        if isinstance(violations, dict):
            for k, v in violations.items():
                if isinstance(v, bool):
                    continue
                try:
                    value_f = float(v)
                except (TypeError, ValueError):
                    continue
                if value_f > 0.0 and math.isfinite(value_f):
                    vio_clean[str(k)] = value_f
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
                params=params_payload if isinstance(params_payload, dict) else None,
            )
        )
    return out


def _worst_constraint(violations: dict[str, float]) -> tuple[str | None, float]:
    return worst_constraint_from_violations(violations)


def _potential_area(lgradb: float, aspect: float) -> float:
    return _objective_area_gain(float(lgradb)) * max(0.0, 20.0 - float(aspect))


def _objective_for_profile(
    row: CandidateRow, *, profile: ProblemProfile
) -> float | None:
    metric_name = str(profile.frontier_recipe.objective_metric).strip().lower()
    if metric_name in {"lgradb", "objective"} and row.lgradb is not None:
        return float(row.lgradb)

    metric_value = row.metrics.get(profile.frontier_recipe.objective_metric)
    if metric_value is None and metric_name == "lgradb":
        metric_value = row.metrics.get("lgradB")
    if metric_value is not None:
        try:
            raw_value = float(metric_value)
        except (TypeError, ValueError):
            raw_value = None
        if raw_value is not None:
            if profile.frontier_recipe.objective_direction == "minimize":
                return -raw_value
            return raw_value

    if row.lgradb is not None:
        return float(row.lgradb)
    return None


def _aspect_for_profile(row: CandidateRow, *, profile: ProblemProfile) -> float | None:
    metric_name = profile.frontier_recipe.aspect_metric
    if metric_name is None:
        return None
    if metric_name == "aspect_ratio" and row.aspect is not None:
        return float(row.aspect)
    metric_value = row.metrics.get(metric_name)
    if metric_value is None:
        return None
    try:
        return float(metric_value)
    except (TypeError, ValueError):
        return None


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
        "iota_edge": "iota_edge",
        "aspect_ratio": "aspect_ratio",
        "average_triangularity": "average_triangularity",
        "max_elongation": "max_elongation",
    }.get(worst_constraint, "")

    if not key_name:
        return max(pool, key=lambda c: float(c.lgradb or -1e9))

    def metric_value(c: CandidateRow) -> float:
        if key_name not in c.metrics:
            return (
                -float("inf")
                if worst_constraint in {"vacuum", "iota", "iota_edge"}
                else float("inf")
            )
        try:
            return float(c.metrics[key_name])
        except (TypeError, ValueError):
            return (
                -float("inf")
                if worst_constraint in {"vacuum", "iota", "iota_edge"}
                else float("inf")
            )

    if worst_constraint in {"vacuum", "iota", "iota_edge"}:
        return max(pool, key=metric_value)
    return min(pool, key=metric_value)


def _choose_frontier_focus(
    candidates: list[CandidateRow],
    *,
    profile: ProblemProfile,
) -> CandidateRow | None:
    if profile.frontier_recipe.aspect_metric is not None:
        scored_with_aspect: list[tuple[CandidateRow, float, float]] = []
        for candidate in candidates:
            if not candidate.is_feasible:
                continue
            objective_utility = _objective_for_profile(candidate, profile=profile)
            aspect_value = _aspect_for_profile(candidate, profile=profile)
            if objective_utility is None or aspect_value is None:
                continue
            scored_with_aspect.append((candidate, objective_utility, aspect_value))
        if not scored_with_aspect:
            return None
        best_candidate, _obj, _asp = max(
            scored_with_aspect,
            key=lambda item: _potential_area(item[1], item[2]),
        )
        return best_candidate

    scored: list[tuple[CandidateRow, float]] = []
    for candidate in candidates:
        if not candidate.is_feasible:
            continue
        objective_utility = _objective_for_profile(candidate, profile=profile)
        if objective_utility is None:
            continue
        scored.append((candidate, objective_utility))
    if not scored:
        return None
    best_candidate, _score = max(scored, key=lambda item: item[1])
    return best_candidate


def _choose_frontier_partner(
    candidates: list[CandidateRow],
    *,
    focus: CandidateRow,
    profile: ProblemProfile,
) -> CandidateRow | None:
    if profile.frontier_recipe.aspect_metric is None:
        return None
    focus_obj = _objective_for_profile(focus, profile=profile)
    focus_asp = _aspect_for_profile(focus, profile=profile)
    if focus_obj is None or focus_asp is None:
        return None
    feasible: list[tuple[CandidateRow, float, float]] = []
    for candidate in candidates:
        if not candidate.is_feasible or candidate.candidate_id == focus.candidate_id:
            continue
        candidate_obj = _objective_for_profile(candidate, profile=profile)
        candidate_asp = _aspect_for_profile(candidate, profile=profile)
        if candidate_obj is None or candidate_asp is None:
            continue
        feasible.append((candidate, candidate_obj, candidate_asp))
    if not feasible:
        return None

    obj_values = [item[1] for item in feasible] + [focus_obj]
    asp_values = [item[2] for item in feasible] + [focus_asp]
    obj_range = max(obj_values) - min(obj_values) if len(obj_values) > 1 else 1.0
    asp_range = max(asp_values) - min(asp_values) if len(asp_values) > 1 else 1.0

    def pareto_distance(item: tuple[CandidateRow, float, float]) -> float:
        candidate_obj = item[1]
        candidate_asp = item[2]
        d_obj = abs(candidate_obj - focus_obj) / max(obj_range, 1e-6)
        d_asp = abs(candidate_asp - focus_asp) / max(asp_range, 1e-6)
        return d_obj + d_asp

    best_candidate, _obj, _asp = max(feasible, key=pareto_distance)
    return best_candidate


def _update_autonomy_progress(
    *,
    snapshot: Mapping[str, object],
    progress: AutonomyProgress,
) -> AutonomyProgress:
    metric_count_raw = snapshot.get("metric_count")
    metric_count = (
        int(metric_count_raw)
        if isinstance(metric_count_raw, int) and not isinstance(metric_count_raw, bool)
        else int(progress.last_metric_count)
    )
    delta_evals = max(int(metric_count) - int(progress.last_metric_count), 0)

    best_feasibility = _as_finite_float(snapshot.get("best_feasibility"))
    best_objective_feasible = _as_finite_float(snapshot.get("best_objective_feasible"))
    best_feasibility_seen = progress.best_feasibility_seen
    best_objective_feasible_seen = progress.best_objective_feasible_seen

    evals_since_feasibility_improve = int(progress.evals_since_feasibility_improve)
    if best_feasibility is not None:
        if (
            best_feasibility_seen is None
            or float(best_feasibility) < float(best_feasibility_seen) - 1e-12
        ):
            best_feasibility_seen = float(best_feasibility)
            evals_since_feasibility_improve = 0
        else:
            evals_since_feasibility_improve += int(delta_evals)
    else:
        evals_since_feasibility_improve += int(delta_evals)

    evals_since_objective_improve = int(progress.evals_since_objective_improve)
    if best_objective_feasible is not None:
        if (
            best_objective_feasible_seen is None
            or float(best_objective_feasible)
            > float(best_objective_feasible_seen) + 1e-12
        ):
            best_objective_feasible_seen = float(best_objective_feasible)
            evals_since_objective_improve = 0
        else:
            evals_since_objective_improve += int(delta_evals)
    else:
        evals_since_objective_improve += int(delta_evals)

    return AutonomyProgress(
        last_metric_count=int(metric_count),
        best_feasibility_seen=best_feasibility_seen,
        best_objective_feasible_seen=best_objective_feasible_seen,
        evals_since_feasibility_improve=int(evals_since_feasibility_improve),
        evals_since_objective_improve=int(evals_since_objective_improve),
    )


def _empty_run_surgery_event() -> dict[str, object]:
    return {
        "action": "none",
        "reason": None,
        "parent_pair": None,
        "backlog_pruned_count": 0,
        "operator_shift_lock_cycles": 0,
        "forced_action": None,
        "autoscale": None,
    }


def _compute_frontier_progress(
    conn: sqlite3.Connection,
    *,
    experiment_id: int,
    eval_window: int,
) -> dict[str, float | int | bool | None]:
    window = max(int(eval_window), 1)
    metric_id_rows = conn.execute(
        """
        SELECT m.id AS metric_id
        FROM metrics m
        JOIN candidates c ON c.id = m.candidate_id
        WHERE c.experiment_id = ?
        ORDER BY m.id DESC
        LIMIT ?
        """,
        (int(experiment_id), int(window) + 1),
    ).fetchall()
    if not metric_id_rows:
        return {
            "metric_delta": 0,
            "objective_delta": 0.0,
            "feasibility_delta": 0.0,
            "window_ready": False,
            "anchor_metric_id": None,
        }
    anchor_metric_id: int | None = None
    if len(metric_id_rows) > int(window):
        anchor_metric_id = int(metric_id_rows[-1]["metric_id"])

    current = conn.execute(
        """
        SELECT
            COUNT(*) AS metric_count,
            MIN(m.feasibility) AS best_feasibility,
            MAX(CASE WHEN m.feasibility <= 0.01 THEN m.objective END) AS best_objective_feasible
        FROM metrics m
        JOIN candidates c ON c.id = m.candidate_id
        WHERE c.experiment_id = ?
        """,
        (int(experiment_id),),
    ).fetchone()
    current_count = int(current["metric_count"]) if current is not None else 0
    current_best_feasibility = (
        _as_finite_float(current["best_feasibility"]) if current is not None else None
    )
    current_best_objective = (
        _as_finite_float(current["best_objective_feasible"])
        if current is not None
        else None
    )

    previous_count = 0
    previous_best_feasibility: float | None = None
    previous_best_objective: float | None = None
    if anchor_metric_id is not None and anchor_metric_id > 0:
        previous = conn.execute(
            """
            SELECT
                COUNT(*) AS metric_count,
                MIN(m.feasibility) AS best_feasibility,
                MAX(CASE WHEN m.feasibility <= 0.01 THEN m.objective END) AS best_objective_feasible
            FROM metrics m
            JOIN candidates c ON c.id = m.candidate_id
            WHERE c.experiment_id = ? AND m.id <= ?
            """,
            (int(experiment_id), int(anchor_metric_id)),
        ).fetchone()
        if previous is not None:
            previous_count = int(previous["metric_count"])
            previous_best_feasibility = _as_finite_float(previous["best_feasibility"])
            previous_best_objective = _as_finite_float(
                previous["best_objective_feasible"]
            )

    feasibility_delta = 0.0
    if previous_best_feasibility is not None and current_best_feasibility is not None:
        feasibility_delta = float(previous_best_feasibility) - float(
            current_best_feasibility
        )
    objective_delta = 0.0
    if previous_best_objective is not None and current_best_objective is not None:
        objective_delta = float(current_best_objective) - float(previous_best_objective)
    metric_delta = max(int(current_count) - int(previous_count), 0)
    return {
        "metric_delta": int(metric_delta),
        "objective_delta": float(objective_delta),
        "feasibility_delta": float(feasibility_delta),
        "window_ready": bool(anchor_metric_id is not None),
        "anchor_metric_id": anchor_metric_id,
    }


def _detect_stall(
    progress: Mapping[str, object],
    *,
    policy: RunSurgeryPolicy,
) -> bool:
    metric_delta = int(progress.get("metric_delta", 0))
    objective_delta = float(progress.get("objective_delta", 0.0))
    feasibility_delta = float(progress.get("feasibility_delta", 0.0))
    if metric_delta < int(policy.eval_window):
        return False
    return objective_delta < float(
        policy.min_objective_delta
    ) and feasibility_delta < float(policy.min_feasibility_delta)


def _rebootstrap_rows(
    conn: sqlite3.Connection,
    *,
    experiment_id: int,
    where_clause: str,
    args: tuple[object, ...],
    limit: int,
) -> list[sqlite3.Row]:
    return conn.execute(
        f"""
        SELECT
            c.id AS candidate_id,
            c.design_hash AS design_hash,
            c.params_json AS params_json,
            m.feasibility AS feasibility,
            m.objective AS objective,
            m.is_feasible AS is_feasible
        FROM metrics m
        JOIN candidates c ON c.id = m.candidate_id
        WHERE c.experiment_id = ? AND {where_clause}
        ORDER BY m.objective DESC, m.feasibility ASC, c.id ASC
        LIMIT ?
        """,
        (int(experiment_id), *args, int(limit)),
    ).fetchall()


def _decode_boundary_json(params_json: object) -> dict | None:
    if not isinstance(params_json, str) or not params_json:
        return None
    try:
        payload = json.loads(params_json)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    try:
        return sanitize_candidate_boundary(payload)
    except ValueError:
        return None


def _select_rebootstrap_pair(
    conn: sqlite3.Connection,
    *,
    experiment_id: int,
    policy: RunSurgeryPolicy,
    run_dir: Path,
) -> tuple[Path, Path, str, str]:
    feasible_rows = _rebootstrap_rows(
        conn,
        experiment_id=experiment_id,
        where_clause="m.is_feasible = 1 AND m.objective IS NOT NULL",
        args=(),
        limit=int(policy.rebootstrap_top_feasible_k),
    )
    near_rows = _rebootstrap_rows(
        conn,
        experiment_id=experiment_id,
        where_clause="m.feasibility >= ? AND m.feasibility <= ? AND m.objective IS NOT NULL",
        args=(float(policy.nearfeasible_min), float(policy.nearfeasible_max)),
        limit=int(policy.rebootstrap_top_nearfeasible_k),
    )

    def _first_valid(
        rows: Sequence[sqlite3.Row], excluded_hashes: set[str]
    ) -> dict | None:
        for row in rows:
            design_hash = str(row["design_hash"] or "")
            if not design_hash or design_hash in excluded_hashes:
                continue
            boundary = _decode_boundary_json(row["params_json"])
            if boundary is None:
                continue
            return {
                "design_hash": design_hash,
                "boundary": boundary,
                "candidate_id": int(row["candidate_id"]),
            }
        return None

    parent_a = _first_valid(near_rows, excluded_hashes=set())
    source = "nearfeasible+feasible"
    if parent_a is None:
        parent_a = _first_valid(feasible_rows, excluded_hashes=set())
        source = "feasible+feasible"
    if parent_a is None:
        raise ValueError("run_surgery_rebootstrap_unavailable:missing_parent_a")

    parent_b = _first_valid(
        feasible_rows,
        excluded_hashes={str(parent_a["design_hash"])},
    )
    if parent_b is None:
        parent_b = _first_valid(
            near_rows,
            excluded_hashes={str(parent_a["design_hash"])},
        )
        if parent_b is not None and source == "nearfeasible+feasible":
            source = "nearfeasible+nearfeasible"
    if parent_b is None:
        raise ValueError("run_surgery_rebootstrap_unavailable:missing_parent_b")

    parent_a_path = _ensure_parent_file(
        run_dir,
        design_hash=str(parent_a["design_hash"]),
        boundary=parent_a["boundary"],
    )
    parent_b_path = _ensure_parent_file(
        run_dir,
        design_hash=str(parent_b["design_hash"]),
        boundary=parent_b["boundary"],
    )
    pair_hash_payload = "|".join(
        sorted(
            [
                str(parent_a["design_hash"]),
                str(parent_b["design_hash"]),
            ]
        )
    )
    pair_hash = hashlib.sha256(pair_hash_payload.encode("utf-8")).hexdigest()
    return parent_a_path, parent_b_path, source, pair_hash


def _prune_backlog(
    conn: sqlite3.Connection,
    *,
    experiment_id: int,
    policy: RunSurgeryPolicy,
    reason: str,
) -> int:
    dominant = conn.execute(
        """
        SELECT model_route, operator_family, COUNT(*) AS n
        FROM candidates
        WHERE experiment_id = ? AND status = 'pending'
        GROUP BY model_route, operator_family
        ORDER BY n DESC, model_route ASC, operator_family ASC
        LIMIT 1
        """,
        (int(experiment_id),),
    ).fetchone()
    if dominant is None:
        return 0
    dominant_count = int(dominant["n"])
    if dominant_count < int(policy.backlog_prune_min_pending):
        return 0
    prune_count = int(math.floor(dominant_count * float(policy.backlog_prune_fraction)))
    prune_count = max(prune_count, 1)
    ids = conn.execute(
        """
        SELECT id
        FROM candidates
        WHERE experiment_id = ?
          AND status = 'pending'
          AND model_route = ?
          AND operator_family = ?
        ORDER BY id ASC
        LIMIT ?
        """,
        (
            int(experiment_id),
            str(dominant["model_route"] or ""),
            str(dominant["operator_family"] or ""),
            int(prune_count),
        ),
    ).fetchall()
    if not ids:
        return 0
    deferred_status = f"deferred:auto_prune:{_utc_stamp()}:{reason}"
    for row in ids:
        conn.execute(
            "UPDATE candidates SET status = ? WHERE id = ?",
            (deferred_status, int(row["id"])),
        )
    return int(len(ids))


def _apply_operator_shift_lock(
    *,
    state: RunSurgeryState,
    profile: ProblemProfile,
    partner_available: bool,
) -> str | None:
    if int(state.operator_shift_lock_remaining) <= 0:
        return None
    allowed: list[str] = []
    if profile.allows_action("repair"):
        allowed.append("repair")
    if partner_available and profile.allows_action("bridge"):
        allowed.append("bridge")
    if profile.allows_action("jump"):
        allowed.append("jump")
    if not allowed:
        state.operator_shift_lock_remaining = max(
            int(state.operator_shift_lock_remaining) - 1,
            0,
        )
        return "global_restart"
    index = int(state.operator_shift_index) % len(allowed)
    selected = allowed[index]
    state.operator_shift_index = int(state.operator_shift_index) + 1
    state.operator_shift_lock_remaining = max(
        int(state.operator_shift_lock_remaining) - 1,
        0,
    )
    return selected


def _preferred_post_rebootstrap_action(
    *,
    profile: ProblemProfile,
    partner_available: bool,
) -> str:
    if partner_available and profile.allows_action("bridge"):
        return "bridge"
    if profile.allows_action("jump"):
        return "jump"
    if profile.allows_action("repair"):
        return "repair"
    return "global_restart"


def _terminate_worker_runtime(runtime: WorkerRuntime) -> None:
    if runtime.process.poll() is None:
        runtime.process.terminate()
        try:
            runtime.process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            runtime.process.kill()
            runtime.process.wait(timeout=5.0)
    runtime.log_handle.flush()
    runtime.log_handle.close()


def _resize_worker_pool(
    *,
    pool: dict[int, WorkerRuntime],
    desired_workers: int,
    problem: str,
    db: Path,
    experiment_id: int,
    run_dir: Path,
    worker_script_path: Path,
    worker_limit: int,
    worker_sleep_sec: float,
) -> None:
    desired = max(int(desired_workers), 0)
    current = len(pool)
    if desired == current:
        return
    if desired > current:
        next_worker_id = max(pool.keys(), default=0) + 1
        for worker_id in range(next_worker_id, next_worker_id + (desired - current)):
            pool[int(worker_id)] = _spawn_worker_runtime(
                worker_id=int(worker_id),
                problem=problem,
                db=db,
                experiment_id=experiment_id,
                run_dir=run_dir,
                worker_script_path=worker_script_path,
                worker_limit=worker_limit,
                worker_sleep_sec=worker_sleep_sec,
            )
        return
    removable = sorted(pool.keys(), reverse=True)[: current - desired]
    for worker_id in removable:
        runtime = pool.pop(int(worker_id))
        _terminate_worker_runtime(runtime)


def _maybe_run_surgery(
    *,
    conn: sqlite3.Connection,
    experiment_id: int,
    profile: ProblemProfile,
    run_dir: Path,
    state: RunSurgeryState,
    progress: Mapping[str, object],
    pending_n: int,
    target_queue: int,
    current_workers: int,
    invalid_parent_failure_streak: int,
    partner_available: bool,
    disable_run_surgery: bool,
    disable_autoscale: bool,
) -> dict | None:
    policy = profile.autonomy_policy.run_surgery
    metric_delta = int(progress.get("metric_delta", 0))
    objective_delta = float(progress.get("objective_delta", 0.0))
    feasibility_delta = float(progress.get("feasibility_delta", 0.0))
    window_ready = bool(progress.get("window_ready", False))
    if window_ready and metric_delta >= int(policy.eval_window):
        if objective_delta < float(policy.min_objective_delta):
            state.no_objective_progress_windows += 1
        else:
            state.no_objective_progress_windows = 0
        if feasibility_delta < float(policy.min_feasibility_delta):
            state.no_feasibility_progress_windows += 1
        else:
            state.no_feasibility_progress_windows = 0
        if objective_delta < float(
            policy.min_objective_delta
        ) and feasibility_delta < float(policy.min_feasibility_delta):
            if (
                state.previous_action_signature is not None
                and state.previous_action_signature == state.last_action_signature
            ):
                state.same_action_windows += 1
            else:
                state.same_action_windows = 1
        else:
            state.same_action_windows = 0
    if state.autoscale_cooldown_remaining > 0:
        state.autoscale_cooldown_remaining -= 1

    if not disable_run_surgery:
        if int(invalid_parent_failure_streak) >= int(
            policy.invalid_basin_failure_limit
        ):
            try:
                parent_a, parent_b, pair_source, pair_hash = _select_rebootstrap_pair(
                    conn,
                    experiment_id=int(experiment_id),
                    policy=policy,
                    run_dir=run_dir,
                )
                state.operator_shift_lock_remaining = int(
                    policy.operator_shift_lock_cycles
                )
                state.operator_shift_index = 0
                state.last_rebootstrap_pair_hash = str(pair_hash)
                return {
                    "action": "rebootstrap_pair",
                    "reason": "invalid_basin_escape",
                    "parent_a": parent_a,
                    "parent_b": parent_b,
                    "parent_pair": {
                        "hash": str(pair_hash),
                        "source": str(pair_source),
                        "parent_a": str(parent_a),
                        "parent_b": str(parent_b),
                    },
                    "backlog_pruned_count": 0,
                    "operator_shift_lock_cycles": int(
                        state.operator_shift_lock_remaining
                    ),
                    "forced_action": "global_restart",
                    "post_restart_forced_action": _preferred_post_rebootstrap_action(
                        profile=profile,
                        partner_available=partner_available,
                    ),
                }
            except ValueError:
                state.operator_shift_lock_remaining = max(
                    int(state.operator_shift_lock_remaining),
                    int(policy.operator_shift_lock_cycles),
                )
                state.operator_shift_index = 0
                return {
                    "action": "invalid_basin_escape",
                    "reason": "invalid_basin_escape_no_rebootstrap_pair",
                    "parent_pair": None,
                    "backlog_pruned_count": 0,
                    "operator_shift_lock_cycles": int(
                        state.operator_shift_lock_remaining
                    ),
                    "forced_action": "global_restart",
                    "post_restart_forced_action": _preferred_post_rebootstrap_action(
                        profile=profile,
                        partner_available=partner_available,
                    ),
                }

        stalled = _detect_stall(progress, policy=policy)
        if stalled and (
            state.no_objective_progress_windows >= int(policy.objective_stall_windows)
            or state.no_feasibility_progress_windows
            >= int(policy.feasibility_stall_windows)
        ):
            try:
                parent_a, parent_b, pair_source, pair_hash = _select_rebootstrap_pair(
                    conn,
                    experiment_id=int(experiment_id),
                    policy=policy,
                    run_dir=run_dir,
                )
                state.operator_shift_lock_remaining = int(
                    policy.operator_shift_lock_cycles
                )
                state.operator_shift_index = 0
                state.last_rebootstrap_pair_hash = str(pair_hash)
                return {
                    "action": "rebootstrap_pair",
                    "reason": "stall_no_frontier_progress",
                    "parent_a": parent_a,
                    "parent_b": parent_b,
                    "parent_pair": {
                        "hash": str(pair_hash),
                        "source": str(pair_source),
                        "parent_a": str(parent_a),
                        "parent_b": str(parent_b),
                    },
                    "backlog_pruned_count": 0,
                    "operator_shift_lock_cycles": int(
                        state.operator_shift_lock_remaining
                    ),
                    "forced_action": "global_restart",
                    "post_restart_forced_action": _preferred_post_rebootstrap_action(
                        profile=profile,
                        partner_available=partner_available,
                    ),
                }
            except ValueError:
                pass
            if int(pending_n) >= int(policy.backlog_prune_min_pending):
                pruned_count = _prune_backlog(
                    conn,
                    experiment_id=int(experiment_id),
                    policy=policy,
                    reason="stall",
                )
                if pruned_count > 0:
                    return {
                        "action": "backlog_prune",
                        "reason": "stall_backlog_prune",
                        "parent_pair": None,
                        "backlog_pruned_count": int(pruned_count),
                        "operator_shift_lock_cycles": int(
                            state.operator_shift_lock_remaining
                        ),
                        "forced_action": None,
                    }
            if int(state.same_action_windows) >= int(policy.max_same_action_windows):
                state.operator_shift_lock_remaining = max(
                    int(state.operator_shift_lock_remaining),
                    int(policy.operator_shift_lock_cycles),
                )
                state.operator_shift_index = 0
                forced_action = _apply_operator_shift_lock(
                    state=state,
                    profile=profile,
                    partner_available=partner_available,
                )
                return {
                    "action": "operator_shift_lock",
                    "reason": "stall_same_action_repeat",
                    "parent_pair": None,
                    "backlog_pruned_count": 0,
                    "operator_shift_lock_cycles": int(
                        state.operator_shift_lock_remaining
                    ),
                    "forced_action": forced_action,
                }

        if int(state.operator_shift_lock_remaining) > 0:
            forced_action = _apply_operator_shift_lock(
                state=state,
                profile=profile,
                partner_available=partner_available,
            )
            return {
                "action": "operator_shift_lock",
                "reason": "lock_active",
                "parent_pair": None,
                "backlog_pruned_count": 0,
                "operator_shift_lock_cycles": int(state.operator_shift_lock_remaining),
                "forced_action": forced_action,
            }

    if (
        not disable_autoscale
        and bool(policy.autoscale_enabled)
        and int(state.autoscale_cooldown_remaining) <= 0
    ):
        queue_ratio = float(pending_n) / float(max(target_queue, 1))
        if queue_ratio >= float(policy.autoscale_up_pending_ratio) and int(
            current_workers
        ) < int(policy.autoscale_max_workers):
            desired_workers = min(
                int(policy.autoscale_max_workers),
                int(current_workers) + int(policy.autoscale_step),
            )
            state.autoscale_cooldown_remaining = int(policy.autoscale_cooldown_cycles)
            return {
                "action": "autoscale_up",
                "reason": "queue_pressure_high",
                "desired_workers": int(desired_workers),
                "parent_pair": None,
                "backlog_pruned_count": 0,
                "operator_shift_lock_cycles": int(state.operator_shift_lock_remaining),
                "forced_action": None,
            }
        if queue_ratio <= float(policy.autoscale_down_pending_ratio) and int(
            current_workers
        ) > int(policy.autoscale_min_workers):
            desired_workers = max(
                int(policy.autoscale_min_workers),
                int(current_workers) - int(policy.autoscale_step),
            )
            state.autoscale_cooldown_remaining = int(policy.autoscale_cooldown_cycles)
            return {
                "action": "autoscale_down",
                "reason": "queue_pressure_low",
                "desired_workers": int(desired_workers),
                "parent_pair": None,
                "backlog_pruned_count": 0,
                "operator_shift_lock_cycles": int(state.operator_shift_lock_remaining),
                "forced_action": None,
            }
    return None


def _action_signature(
    *,
    action: str | None,
    model_route: str | None,
    parent_pair_hash: str | None,
) -> str:
    action_name = str(action).strip().lower() if action is not None else "none"
    route_name = str(model_route).strip().lower() if model_route is not None else "none"
    pair_hash = str(parent_pair_hash).strip().lower() if parent_pair_hash else "none"
    return f"{action_name}|{route_name}|{pair_hash}"


def _best_feasibility_candidate(
    candidates: Sequence[CandidateRow],
) -> CandidateRow | None:
    pool = [
        candidate
        for candidate in candidates
        if math.isfinite(float(candidate.feasibility))
    ]
    if not pool:
        return None
    return min(pool, key=lambda candidate: float(candidate.feasibility))


def _best_feasible_objective_candidate(
    candidates: Sequence[CandidateRow],
    *,
    profile: ProblemProfile,
) -> CandidateRow | None:
    feasible_pool: list[tuple[CandidateRow, float]] = []
    for candidate in candidates:
        if not candidate.is_feasible:
            continue
        objective_utility = _objective_for_profile(candidate, profile=profile)
        if objective_utility is None:
            continue
        feasible_pool.append((candidate, float(objective_utility)))
    if not feasible_pool:
        return None
    best_candidate, _best_utility = max(feasible_pool, key=lambda item: item[1])
    return best_candidate


def _best_objective_candidate(
    candidates: Sequence[CandidateRow],
    *,
    profile: ProblemProfile,
    exclude_hashes: set[str] | None = None,
) -> CandidateRow | None:
    excluded = exclude_hashes or set()
    pool: list[tuple[CandidateRow, float]] = []
    for candidate in candidates:
        if candidate.design_hash in excluded:
            continue
        objective_utility = _objective_for_profile(candidate, profile=profile)
        if objective_utility is None:
            continue
        pool.append((candidate, float(objective_utility)))
    if not pool:
        return None
    best_candidate, _best_utility = max(pool, key=lambda item: item[1])
    return best_candidate


def _autonomous_focus_partner(
    *,
    candidates: list[CandidateRow],
    profile: ProblemProfile,
    strategy: str,
) -> tuple[CandidateRow, CandidateRow | None] | None:
    if strategy == "bridge_obj":
        focus = _best_feasible_objective_candidate(candidates, profile=profile)
        if focus is None:
            focus = _best_feasibility_candidate(candidates)
        if focus is None:
            return None
        partner = _best_objective_candidate(
            candidates,
            profile=profile,
            exclude_hashes={focus.design_hash},
        )
        return focus, partner

    if strategy == "exploit_feas":
        pool = [
            candidate
            for candidate in candidates
            if math.isfinite(float(candidate.feasibility))
        ]
        if not pool:
            return None
        ranked = sorted(pool, key=lambda candidate: float(candidate.feasibility))
        focus = ranked[0]
        partner = None
        for candidate in ranked[1:]:
            if candidate.design_hash != focus.design_hash:
                partner = candidate
                break
        if partner is None:
            partner = _best_feasible_objective_candidate(candidates, profile=profile)
            if partner is not None and partner.design_hash == focus.design_hash:
                partner = None
        return focus, partner

    return None


def _focus_partner_via_shared_staged_plan(
    candidates: list[CandidateRow],
    *,
    max_feasibility: float,
    problem: str,
) -> tuple[CandidateRow, CandidateRow | None] | None:
    snapshots: list[dict] = []
    for candidate in candidates:
        if candidate.params is None:
            continue
        snapshots.append(
            {
                "design_hash": candidate.design_hash,
                "params": candidate.params,
                "feasibility": candidate.feasibility,
                "objective": candidate.lgradb,
                "is_feasible": candidate.is_feasible,
                "constraint_margins": dict(candidate.violations),
                "metrics": dict(candidate.metrics),
            }
        )
    if not snapshots:
        return None
    staged_plan = build_staged_seed_plan_from_snapshots(
        snapshots=snapshots,
        problem=str(problem),
        near_feasibility_threshold=float(max_feasibility),
        max_repair_candidates=1,
        bridge_blend_t=0.86,
    )
    if staged_plan is None:
        return None
    by_hash: dict[str, CandidateRow] = {}
    for candidate in candidates:
        # candidates are fetched newest-first; keep first seen row per hash.
        if candidate.design_hash not in by_hash:
            by_hash[candidate.design_hash] = candidate
    focus = by_hash.get(staged_plan.focus_hash)
    if focus is None:
        return None
    partner = (
        by_hash.get(staged_plan.partner_hash)
        if staged_plan.partner_hash is not None
        else None
    )
    return focus, partner


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
    proposal_script: str,
    problem: str,
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
            str(proposal_script),
            "--problem",
            str(problem),
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


def _derive_restart_parent_payloads(
    *,
    focus: CandidateRow,
    partner: CandidateRow | None,
    candidates: list[CandidateRow],
) -> tuple[dict, dict] | None:
    selected: list[dict] = []
    seen_hashes: set[str] = set()
    for candidate in [focus, partner, *candidates]:
        if candidate is None:
            continue
        if candidate.design_hash in seen_hashes:
            continue
        if not isinstance(candidate.params, dict):
            continue
        try:
            sanitized = sanitize_candidate_boundary(candidate.params)
        except ValueError:
            continue
        selected.append(json.loads(json.dumps(sanitized)))
        seen_hashes.add(candidate.design_hash)
        if len(selected) >= 2:
            break
    if not selected:
        return None
    if len(selected) == 1:
        parent_a = selected[0]
        parent_b = json.loads(json.dumps(parent_a))
        for matrix_key in ("r_cos", "z_sin"):
            matrix = parent_b.get(matrix_key)
            if not isinstance(matrix, list):
                continue
            for m_idx, row in enumerate(matrix):
                if m_idx < 2 or not isinstance(row, list):
                    continue
                for n_idx, value in enumerate(row):
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        continue
                    row[n_idx] = float(value) * 1.05
        return parent_a, parent_b
    return selected[0], selected[1]


def _materialize_restart_parent_pair(
    *,
    run_dir: Path,
    batch_id: int,
    focus: CandidateRow,
    partner: CandidateRow | None,
    candidates: list[CandidateRow],
) -> tuple[Path, Path] | None:
    payload_pair = _derive_restart_parent_payloads(
        focus=focus,
        partner=partner,
        candidates=candidates,
    )
    if payload_pair is None:
        return None
    parent_a_payload, parent_b_payload = payload_pair
    parent_dir = run_dir / "governor" / "restart_parents" / f"batch_{int(batch_id):03d}"
    parent_a_path = parent_dir / "parent_a.json"
    parent_b_path = parent_dir / "parent_b.json"
    _write_json(parent_a_path, parent_a_payload)
    _write_json(parent_b_path, parent_b_payload)
    return parent_a_path, parent_b_path


def _clamp_float(value: float, *, low: float, high: float) -> float:
    return max(float(low), min(float(value), float(high)))


def _restart_blend_schedule(
    *,
    run_seed: int,
    batch_id: int,
    t_min: float,
    t_max: float,
    t_step: float,
) -> tuple[float, float, float]:
    """Deterministic jitter to prevent identical restart replays across cycles."""
    base_min = float(t_min)
    base_max = float(t_max)
    base_step = max(1e-6, float(t_step))
    shift_options = (0.0, 0.01, -0.01, 0.005, -0.005)
    shift_idx = int((int(run_seed) + int(batch_id)) % len(shift_options))
    shift = float(shift_options[shift_idx])
    out_min = _clamp_float(base_min + shift, low=0.0, high=0.99)
    out_max = _clamp_float(base_max + shift, low=out_min + 0.001, high=1.0)
    step_options = (
        base_step,
        max(0.002, base_step * 1.2),
        max(0.002, base_step * 0.8),
    )
    step_idx = int((int(run_seed) + int(batch_id) * 3) % len(step_options))
    out_step = float(step_options[step_idx])
    span = max(0.001, out_max - out_min)
    out_step = min(out_step, span)
    return out_min, out_max, max(out_step, 0.001)


def _resolve_global_restart_parent_pair(
    *,
    run_dir: Path,
    batch_id: int,
    focus: CandidateRow,
    partner: CandidateRow | None,
    candidates: list[CandidateRow],
    bootstrap_parent_a: Path | None,
    bootstrap_parent_b: Path | None,
    run_seed: int,
) -> tuple[Path, Path, str]:
    parent_a = bootstrap_parent_a
    parent_b = bootstrap_parent_b
    source = "bootstrap"
    if parent_a is None or parent_b is None:
        auto_pair = _materialize_restart_parent_pair(
            run_dir=run_dir,
            batch_id=batch_id,
            focus=focus,
            partner=partner,
            candidates=candidates,
        )
        if auto_pair is None:
            raise ValueError(
                "global_restart requested but no restart parents were available."
            )
        parent_a, parent_b = auto_pair
        source = "auto_archive"
    assert parent_a is not None
    assert parent_b is not None
    # Deterministically permute parent order so repeated restart actions are not static.
    if int((int(run_seed) + int(batch_id)) % 2) == 1:
        parent_a, parent_b = parent_b, parent_a
        source = f"{source}_swapped"
    return parent_a, parent_b, source


def _build_scale_cmd(
    *,
    proposal_script: str,
    problem: str,
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
        str(proposal_script),
        "--problem",
        str(problem),
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


def _dominant_violation(candidates: list[CandidateRow], *, limit: int = 20) -> str:
    counts: dict[str, int] = {}
    for row in candidates[:limit]:
        for key in row.violations.keys():
            counts[str(key)] = int(counts.get(str(key), 0)) + 1
    if not counts:
        return "none"
    return max(counts.items(), key=lambda item: item[1])[0]


def _dominant_violation_rate(
    candidates: list[CandidateRow], *, limit: int = 20
) -> float:
    recent = candidates[:limit]
    if not recent:
        return 1.0
    dominant = _dominant_violation(recent, limit=len(recent))
    if dominant == "none":
        return 0.0
    hit_count = 0
    for row in recent:
        value = _as_finite_float(row.violations.get(dominant))
        if value is None or value <= 0.0:
            continue
        hit_count += 1
    return float(hit_count) / float(len(recent))


def _lesson_summary(candidates: list[CandidateRow], *, limit: int = 40) -> dict:
    family_counts: dict[str, int] = {}
    family_feasible: dict[str, int] = {}
    for row in candidates[:limit]:
        family = str(row.operator_family or "unknown")
        family_counts[family] = int(family_counts.get(family, 0)) + 1
        if row.is_feasible:
            family_feasible[family] = int(family_feasible.get(family, 0)) + 1
    items: list[dict] = []
    for family, count in sorted(family_counts.items()):
        feasible = int(family_feasible.get(family, 0))
        items.append(
            {
                "operator_family": family,
                "n": count,
                "feasible_n": feasible,
                "feasible_rate": (float(feasible) / float(count)) if count > 0 else 0.0,
            }
        )
    return {"operator_stats": items[:8]}


def _as_finite_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    value_f = float(value)
    if not math.isfinite(value_f):
        return None
    return value_f


def _current_snapshot(
    conn: sqlite3.Connection,
    *,
    experiment_id: int,
    hv_value: float,
) -> dict:
    row = conn.execute(
        """
        SELECT
            MIN(m.feasibility) AS best_feasibility,
            SUM(CASE WHEN m.is_feasible = 1 THEN 1 ELSE 0 END) AS feasible_count,
            COUNT(*) AS metric_count,
            MAX(CASE WHEN m.feasibility <= 0.01 THEN m.objective ELSE NULL END) AS best_objective_feasible
        FROM metrics m
        JOIN candidates c ON c.id = m.candidate_id
        WHERE c.experiment_id = ?
        """,
        (int(experiment_id),),
    ).fetchone()
    best_feasibility = None
    feasible_count = 0
    metric_count = 0
    best_objective_feasible = None
    if row is not None:
        best_feasibility = _as_finite_float(row["best_feasibility"])
        feasible_raw = row["feasible_count"]
        if feasible_raw is not None:
            feasible_count = int(feasible_raw)
        metric_raw = row["metric_count"]
        if metric_raw is not None:
            metric_count = int(metric_raw)
        best_objective_feasible = _as_finite_float(row["best_objective_feasible"])
    return {
        "best_feasibility": best_feasibility,
        "best_objective_feasible": best_objective_feasible,
        "hv": float(hv_value),
        "feasible_count": int(feasible_count),
        "metric_count": int(metric_count),
    }


def _frontier_integrity_ok(*, snapshot: Mapping[str, object], hv_value: float) -> bool:
    if not math.isfinite(float(hv_value)) or float(hv_value) < 0.0:
        return False
    snapshot_hv = _as_finite_float(snapshot.get("hv"))
    if snapshot_hv is None or snapshot_hv < 0.0:
        return False
    feasible_count = snapshot.get("feasible_count")
    if isinstance(feasible_count, bool) or not isinstance(feasible_count, int):
        return False
    if int(feasible_count) < 0:
        return False
    best_feasibility_raw = snapshot.get("best_feasibility")
    if (
        best_feasibility_raw is not None
        and _as_finite_float(best_feasibility_raw) is None
    ):
        return False
    return True


def _log_reflection_for_previous_cycle(
    conn: sqlite3.Connection,
    *,
    experiment_id: int,
    next_cycle: int,
    snapshot: Mapping[str, object],
) -> None:
    previous_cycle = int(next_cycle) - 1
    if previous_cycle <= 0:
        return
    reflected = conn.execute(
        """
        SELECT 1
        FROM scratchpad_events
        WHERE experiment_id = ? AND cycle = ? AND step = 1
        LIMIT 1
        """,
        (int(experiment_id), int(previous_cycle)),
    ).fetchone()
    if reflected is not None:
        return

    prior = conn.execute(
        """
        SELECT diagnostics_json, outcome_json
        FROM scratchpad_events
        WHERE experiment_id = ? AND cycle = ? AND step = 0
        ORDER BY id DESC
        LIMIT 1
        """,
        (int(experiment_id), int(previous_cycle)),
    ).fetchone()
    if prior is None:
        return

    diagnostics = json.loads(str(prior["diagnostics_json"]))
    outcome = json.loads(str(prior["outcome_json"]))
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    if not isinstance(outcome, dict):
        outcome = {}
    baseline_snapshot = outcome.get("snapshot_at_decision", {})
    if not isinstance(baseline_snapshot, dict):
        baseline_snapshot = {}

    baseline_best = _as_finite_float(baseline_snapshot.get("best_feasibility"))
    baseline_hv = _as_finite_float(baseline_snapshot.get("hv"))
    current_best = _as_finite_float(snapshot.get("best_feasibility"))
    current_hv = _as_finite_float(snapshot.get("hv"))

    realized_feasibility_delta = None
    if baseline_best is not None and current_best is not None:
        realized_feasibility_delta = baseline_best - current_best

    realized_hv_delta = None
    if baseline_hv is not None and current_hv is not None:
        realized_hv_delta = current_hv - baseline_hv

    _log_scratchpad_event(
        conn,
        experiment_id=int(experiment_id),
        cycle=int(previous_cycle),
        step=1,
        planner_intent={},
        aso_action="reflection",
        intent_agreement="n/a",
        override_reason=None,
        diagnostics={
            "action": str(diagnostics.get("action", "unknown")),
            "predicted_target_metric": diagnostics.get("predicted_target_metric"),
            "predicted_direction": diagnostics.get("predicted_direction"),
            "predicted_expected_effect": diagnostics.get("predicted_expected_effect"),
        },
        outcome={
            "snapshot_baseline": baseline_snapshot,
            "snapshot_current": dict(snapshot),
            "realized_feasibility_delta": realized_feasibility_delta,
            "realized_hv_delta": realized_hv_delta,
        },
    )


def _synthesize_recipe_rules(
    conn: sqlite3.Connection,
    *,
    experiment_id: int,
    limit: int = 160,
    top_k: int = 6,
) -> list[dict]:
    rows = conn.execute(
        """
        SELECT diagnostics_json, outcome_json
        FROM scratchpad_events
        WHERE experiment_id = ? AND step = 1 AND aso_action = 'reflection'
        ORDER BY id DESC
        LIMIT ?
        """,
        (int(experiment_id), int(limit)),
    ).fetchall()
    aggregates: dict[tuple[str, str], dict[str, float]] = {}
    for row in rows:
        diagnostics = json.loads(str(row["diagnostics_json"]))
        outcome = json.loads(str(row["outcome_json"]))
        if not isinstance(diagnostics, dict) or not isinstance(outcome, dict):
            continue
        action = str(diagnostics.get("action", "unknown"))
        target = str(diagnostics.get("predicted_target_metric", "unknown"))
        key = (action, target)
        aggregate = aggregates.get(
            key,
            {
                "count": 0.0,
                "success": 0.0,
                "sum_realized_feasibility_delta": 0.0,
                "sum_realized_hv_delta": 0.0,
            },
        )
        realized_feas = _as_finite_float(outcome.get("realized_feasibility_delta"))
        realized_hv = _as_finite_float(outcome.get("realized_hv_delta"))
        aggregate["count"] += 1.0
        if (realized_feas is not None and realized_feas > 0.0) or (
            realized_hv is not None and realized_hv > 0.0
        ):
            aggregate["success"] += 1.0
        if realized_feas is not None:
            aggregate["sum_realized_feasibility_delta"] += realized_feas
        if realized_hv is not None:
            aggregate["sum_realized_hv_delta"] += realized_hv
        aggregates[key] = aggregate

    rules: list[dict] = []
    for (action, target), aggregate in aggregates.items():
        count = aggregate["count"]
        if count <= 0.0:
            continue
        success = aggregate["success"]
        rules.append(
            {
                "action": action,
                "target_constraint": target,
                "samples": int(count),
                "success_rate": float(success / count),
                "mean_realized_feasibility_delta": float(
                    aggregate["sum_realized_feasibility_delta"] / count
                ),
                "mean_realized_hv_delta": float(
                    aggregate["sum_realized_hv_delta"] / count
                ),
            }
        )
    rules.sort(
        key=lambda item: (float(item["success_rate"]), int(item["samples"])),
        reverse=True,
    )
    return rules[: int(top_k)]


def _latest_reflection_event_id(conn: sqlite3.Connection, *, experiment_id: int) -> int:
    row = conn.execute(
        """
        SELECT MAX(id) AS max_id
        FROM scratchpad_events
        WHERE experiment_id = ? AND step = 1 AND aso_action = 'reflection'
        """,
        (int(experiment_id),),
    ).fetchone()
    if row is None:
        return 0
    value = row["max_id"]
    if value is None:
        return 0
    return int(value)


def _constraint_direction(profile: object, *, constraint: str | None) -> str | None:
    if constraint is None:
        return None
    name = str(constraint).strip().lower()
    if not name:
        return None
    constraints = getattr(profile, "constraints", ())
    for spec in constraints:
        spec_name = str(getattr(spec, "name", "")).strip().lower()
        if spec_name != name:
            continue
        relation = str(getattr(spec, "relation", "")).strip()
        if relation == "<=":
            return "decrease"
        if relation == ">=":
            return "increase"
    return None


def _bootstrap_route_label(*, adaptive: bool) -> str:
    if adaptive:
        return "governor_adaptive/bootstrap"
    return "governor_static_recipe/bootstrap"


def _ensure_parent_file(run_dir: Path, *, design_hash: str, boundary: dict) -> Path:
    path = run_dir / "candidates" / f"{design_hash}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(json.dumps(boundary, indent=2))
    return path


def _load_candidate_boundary(
    conn: sqlite3.Connection,
    *,
    experiment_id: int,
    design_hash: str,
) -> dict:
    row = conn.execute(
        """
        SELECT params_json
        FROM candidates
        WHERE experiment_id = ? AND design_hash = ?
        LIMIT 1
        """,
        (int(experiment_id), str(design_hash)),
    ).fetchone()
    if row is None:
        raise ValueError(f"Candidate {design_hash!r} not found in candidates table.")
    boundary = json.loads(str(row["params_json"]))
    if not isinstance(boundary, dict):
        raise TypeError("Candidate params_json must be a JSON object.")
    try:
        return sanitize_candidate_boundary(boundary)
    except ValueError as exc:
        raise ValueError(
            f"Candidate {design_hash!r} has invalid boundary payload: {exc}"
        ) from exc


def _build_llm_mutation_cmds(
    *,
    proposal_script: str,
    problem: str,
    db: Path,
    conn: sqlite3.Connection | None,
    experiment_id: int,
    run_dir: Path,
    batch_id: int,
    seed_base: int,
    focus: CandidateRow,
    llm_mutations: Iterable[Mapping[str, object]],
    route_prefix: str,
) -> tuple[list[ProposalCommand], dict]:
    active_conn = conn
    owns_conn = False
    if active_conn is None:
        active_conn = _connect(db)
        owns_conn = True
    try:
        boundary = _load_candidate_boundary(
            active_conn,
            experiment_id=experiment_id,
            design_hash=focus.design_hash,
        )
    finally:
        if owns_conn:
            active_conn.close()
    focus_path = _ensure_parent_file(
        run_dir,
        design_hash=focus.design_hash,
        boundary=boundary,
    )

    cmds: list[ProposalCommand] = []
    applied: list[dict] = []
    rejected: list[dict] = []
    for index, mutation in enumerate(llm_mutations):
        group_raw = mutation.get("parameter_group")
        delta_raw = mutation.get("normalized_delta")
        if not isinstance(group_raw, str) or not group_raw.strip():
            rejected.append({"reason": "invalid_parameter_group", "mutation": mutation})
            continue
        if isinstance(delta_raw, bool) or not isinstance(delta_raw, (int, float)):
            rejected.append(
                {"reason": "invalid_normalized_delta", "mutation": mutation}
            )
            continue
        group = str(group_raw).strip().lower()
        factor = 1.0 + float(delta_raw)
        seed = int(seed_base) + 500 + int(index)

        if group == "axisym_z":
            cmd = _build_scale_cmd(
                proposal_script=proposal_script,
                problem=problem,
                db=db,
                experiment_id=experiment_id,
                run_dir=run_dir,
                batch_id=batch_id,
                seed_base=seed,
                parent=focus_path,
                axisym_z=factor,
                model_route=f"{route_prefix}/mutation",
            )
        elif group == "axisym_r":
            cmd = _build_scale_cmd(
                proposal_script=proposal_script,
                problem=problem,
                db=db,
                experiment_id=experiment_id,
                run_dir=run_dir,
                batch_id=batch_id,
                seed_base=seed,
                parent=focus_path,
                axisym_r=factor,
                model_route=f"{route_prefix}/mutation",
            )
        elif group.startswith("m_ge_"):
            suffix = group.removeprefix("m_ge_")
            if not suffix.isdigit():
                rejected.append({"reason": "invalid_m_ge_group", "mutation": mutation})
                continue
            cmd = _build_scale_cmd(
                proposal_script=proposal_script,
                problem=problem,
                db=db,
                experiment_id=experiment_id,
                run_dir=run_dir,
                batch_id=batch_id,
                seed_base=seed,
                parent=focus_path,
                scale_m_ge=(int(suffix), factor),
                model_route=f"{route_prefix}/mutation",
            )
        elif group.startswith("abs_n_"):
            suffix = group.removeprefix("abs_n_")
            if not suffix.isdigit():
                rejected.append({"reason": "invalid_abs_n_group", "mutation": mutation})
                continue
            cmd = _build_scale_cmd(
                proposal_script=proposal_script,
                problem=problem,
                db=db,
                experiment_id=experiment_id,
                run_dir=run_dir,
                batch_id=batch_id,
                seed_base=seed,
                parent=focus_path,
                scale_abs_n=(int(suffix), factor),
                model_route=f"{route_prefix}/mutation",
            )
        else:
            rejected.append(
                {"reason": "unsupported_parameter_group", "mutation": mutation}
            )
            continue

        cmds.append(cmd)
        applied.append(
            {
                "parameter_group": group,
                "normalized_delta": float(delta_raw),
                "factor": factor,
            }
        )

    return cmds, {"applied": applied, "rejected": rejected}


def _build_jump_recipe_cmds(
    *,
    proposal_script: str,
    problem: str,
    db: Path,
    conn: sqlite3.Connection | None,
    jump_delta_cap: float,
    experiment_id: int,
    run_dir: Path,
    batch_id: int,
    seed_base: int,
    focus: CandidateRow,
    partner: CandidateRow | None,
    route_prefix: str,
) -> tuple[list[ProposalCommand], dict]:
    active_conn = conn
    owns_conn = False
    if active_conn is None:
        active_conn = _connect(db)
        owns_conn = True
    try:
        focus_boundary = _load_candidate_boundary(
            active_conn,
            experiment_id=experiment_id,
            design_hash=focus.design_hash,
        )
        partner_boundary = None
        if partner is not None:
            try:
                partner_boundary = _load_candidate_boundary(
                    active_conn,
                    experiment_id=experiment_id,
                    design_hash=partner.design_hash,
                )
            except ValueError:
                partner_boundary = None
    finally:
        if owns_conn:
            active_conn.close()

    focus_path = _ensure_parent_file(
        run_dir,
        design_hash=focus.design_hash,
        boundary=focus_boundary,
    )
    partner_path = (
        _ensure_parent_file(
            run_dir,
            design_hash=partner.design_hash,
            boundary=partner_boundary,
        )
        if (partner is not None and partner_boundary is not None)
        else None
    )
    route_label = f"{route_prefix}/jump"
    cap = abs(float(jump_delta_cap))
    axisym_z_delta = cap * 0.27
    axisym_r_delta = cap * 0.22
    m_ge_delta = cap * 0.49
    abs_n_delta = cap * 0.40
    jump_scales: list[
        tuple[
            float | None,
            float | None,
            tuple[int, float] | None,
            tuple[int, float] | None,
        ]
    ] = [
        (1.0 - axisym_z_delta, None, None, None),
        (1.0 + axisym_z_delta, None, None, None),
        (None, 1.0 - axisym_r_delta, None, None),
        (None, 1.0 + axisym_r_delta, None, None),
        (None, None, (2, 1.0 - m_ge_delta), None),
        (None, None, (3, 1.0 + m_ge_delta), None),
        (None, None, None, (1, 1.0 + abs_n_delta)),
    ]
    cmds: list[ProposalCommand] = []
    for index, (axisym_z, axisym_r, scale_m_ge, scale_abs_n) in enumerate(jump_scales):
        cmds.append(
            _build_scale_cmd(
                proposal_script=proposal_script,
                problem=problem,
                db=db,
                experiment_id=experiment_id,
                run_dir=run_dir,
                batch_id=batch_id,
                seed_base=seed_base + 200 + index,
                parent=focus_path,
                axisym_z=axisym_z,
                axisym_r=axisym_r,
                scale_m_ge=scale_m_ge,
                scale_abs_n=scale_abs_n,
                model_route=route_label,
            )
        )

    if partner_path is not None:
        cmds.append(
            _build_blend_cmd(
                proposal_script=proposal_script,
                problem=problem,
                db=db,
                experiment_id=experiment_id,
                run_dir=run_dir,
                batch_id=batch_id,
                seed_base=seed_base + 300,
                parent_a=focus_path,
                parent_b=partner_path,
                t_min=0.20,
                t_max=0.80,
                t_step=0.15,
                model_route=route_label,
            )
        )

    diagnostics = {
        "mode": "jump_non_local",
        "focus_design_hash": focus.design_hash,
        "partner_design_hash": None if partner is None else partner.design_hash,
        "command_count": len(cmds),
    }
    return cmds, diagnostics


def _select_frontier_recipe(
    *,
    proposal_script: str,
    problem: str,
    db: Path,
    conn: sqlite3.Connection | None,
    experiment_id: int,
    run_dir: Path,
    batch_id: int,
    seed_base: int,
    focus: CandidateRow,
    partner: CandidateRow | None,
    profile: ProblemProfile,
    hv_value: float,
    record_hv: float,
    route_prefix: str = "governor_frontier_recipe",
) -> tuple[list[ProposalCommand], dict]:
    if profile.problem != "p3":
        return _select_static_recipe(
            proposal_script=proposal_script,
            problem=problem,
            db=db,
            conn=conn,
            experiment_id=experiment_id,
            run_dir=run_dir,
            batch_id=batch_id,
            seed_base=seed_base,
            focus=focus,
            partner=partner,
            route_prefix=route_prefix,
        )

    active_conn = conn
    owns_conn = False
    if active_conn is None:
        active_conn = _connect(db)
        owns_conn = True
    try:
        focus_boundary = _load_candidate_boundary(
            active_conn,
            experiment_id=experiment_id,
            design_hash=focus.design_hash,
        )
        partner_boundary = None
        if partner is not None:
            try:
                partner_boundary = _load_candidate_boundary(
                    active_conn,
                    experiment_id=experiment_id,
                    design_hash=partner.design_hash,
                )
            except ValueError:
                partner_boundary = None
    finally:
        if owns_conn:
            active_conn.close()

    focus_path = _ensure_parent_file(
        run_dir,
        design_hash=focus.design_hash,
        boundary=focus_boundary,
    )
    partner_path = (
        _ensure_parent_file(
            run_dir,
            design_hash=partner.design_hash,
            boundary=partner_boundary,
        )
        if (partner is not None and isinstance(partner_boundary, dict))
        else None
    )
    route_label = f"{str(route_prefix)}/p3_frontier"
    frontier_recipe = profile.frontier_recipe
    scale = _hv_gap_perturbation_scale(
        hv_value=float(hv_value),
        record_hv=float(record_hv),
        frontier_recipe=frontier_recipe,
    )

    cmd_specs: list[
        tuple[
            float | None,
            float | None,
            tuple[int, float] | None,
            tuple[int, float] | None,
        ]
    ] = [
        (1.0 - (0.35 * scale), None, None, None),
        (1.0 + (0.35 * scale), None, None, None),
        (None, 1.0 - (0.25 * scale), None, None),
        (None, None, (2, 1.0 + scale), None),
        (None, None, (3, 1.0 + (0.8 * scale)), None),
        (None, None, None, (1, 1.0 + (0.7 * scale))),
        (None, None, None, (2, 1.0 + (0.5 * scale))),
    ]
    cmds: list[ProposalCommand] = []
    if "scale_groups" in frontier_recipe.frontier_move_families:
        for index, (axisym_z, axisym_r, scale_m_ge, scale_abs_n) in enumerate(
            cmd_specs
        ):
            cmds.append(
                _build_scale_cmd(
                    proposal_script=proposal_script,
                    problem=problem,
                    db=db,
                    experiment_id=experiment_id,
                    run_dir=run_dir,
                    batch_id=batch_id,
                    seed_base=seed_base + 100 + index,
                    parent=focus_path,
                    axisym_z=axisym_z,
                    axisym_r=axisym_r,
                    scale_m_ge=scale_m_ge,
                    scale_abs_n=scale_abs_n,
                    model_route=route_label,
                )
            )

    if partner_path is not None and "blend" in frontier_recipe.frontier_move_families:
        cmds.append(
            _build_blend_cmd(
                proposal_script=proposal_script,
                problem=problem,
                db=db,
                experiment_id=experiment_id,
                run_dir=run_dir,
                batch_id=batch_id,
                seed_base=seed_base + 200,
                parent_a=focus_path,
                parent_b=partner_path,
                t_min=0.20,
                t_max=0.80,
                t_step=0.20,
                model_route=route_label,
            )
        )

    max_candidates = int(profile.mutation_budget.max_candidates_per_cycle)
    if len(cmds) > max_candidates:
        cmds = cmds[:max_candidates]

    hv_gap = max(0.0, float(record_hv) - float(hv_value))
    decision = {
        "batch_id": batch_id,
        "focus": {
            "design_hash": focus.design_hash,
            "candidate_id": focus.candidate_id,
            "feasibility": focus.feasibility,
            "aspect": focus.aspect,
            "lgradb": focus.lgradb,
        },
        "partner": None
        if partner is None
        else {
            "design_hash": partner.design_hash,
            "candidate_id": partner.candidate_id,
            "aspect": partner.aspect,
            "lgradb": partner.lgradb,
        },
        "commands": [_cmd_str(cmd) for cmd in cmds],
        "model_route": route_label,
        "recipe_mode": "frontier",
        "frontier_policy": {
            "scale": scale,
            "hv_at_decision": float(hv_value),
            "record_hv": float(record_hv),
            "hv_gap": hv_gap,
        },
        "created_at": _utc_now_iso(),
    }
    return cmds, decision


def _select_static_recipe(
    *,
    proposal_script: str,
    problem: str,
    db: Path,
    conn: sqlite3.Connection | None,
    experiment_id: int,
    run_dir: Path,
    batch_id: int,
    seed_base: int,
    focus: CandidateRow,
    partner: CandidateRow | None,
    route_prefix: str = "governor_static_recipe",
    target_constraint_override: str | None = None,
) -> tuple[list[ProposalCommand], dict]:
    # Ensure parent boundary JSONs exist (we can pull from DB if needed).
    active_conn = conn
    owns_conn = False
    if active_conn is None:
        active_conn = _connect(db)
        owns_conn = True
    try:
        focus_boundary = _load_candidate_boundary(
            active_conn,
            experiment_id=experiment_id,
            design_hash=focus.design_hash,
        )

        focus_path = _ensure_parent_file(
            run_dir, design_hash=focus.design_hash, boundary=focus_boundary
        )

        if (
            target_constraint_override is not None
            and target_constraint_override.strip()
        ):
            worst = str(target_constraint_override).strip().lower()
            worst_val = float(focus.violations.get(worst, 0.0))
        else:
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
                        proposal_script=proposal_script,
                        problem=problem,
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
                        proposal_script=proposal_script,
                        problem=problem,
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
                            proposal_script=proposal_script,
                            problem=problem,
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
                        proposal_script=proposal_script,
                        problem=problem,
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
                        proposal_script=proposal_script,
                        problem=problem,
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
                        proposal_script=proposal_script,
                        problem=problem,
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
                        proposal_script=proposal_script,
                        problem=problem,
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
                        proposal_script=proposal_script,
                        problem=problem,
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
                        proposal_script=proposal_script,
                        problem=problem,
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
                        proposal_script=proposal_script,
                        problem=problem,
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
            try:
                partner_boundary = _load_candidate_boundary(
                    active_conn,
                    experiment_id=experiment_id,
                    design_hash=partner.design_hash,
                )
            except ValueError:
                partner_boundary = None

            if partner_boundary is not None:
                partner_path = _ensure_parent_file(
                    run_dir, design_hash=partner.design_hash, boundary=partner_boundary
                )
                cmds.append(
                    _build_blend_cmd(
                        proposal_script=proposal_script,
                        problem=problem,
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
    finally:
        if owns_conn:
            active_conn.close()

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
    proposal_script: str,
    problem: str,
    db: Path,
    conn: sqlite3.Connection | None,
    experiment_id: int,
    run_dir: Path,
    batch_id: int,
    seed_base: int,
    focus: CandidateRow,
    partner: CandidateRow | None,
    history_candidates: list[CandidateRow],
    parent_group: str,
    route_prefix: str | None = None,
    target_constraint_override: str | None = None,
) -> tuple[list[ProposalCommand], dict]:
    prefix = (
        str(route_prefix)
        if route_prefix is not None
        else f"governor_adaptive/{str(parent_group)}"
    )
    cmds, decision = _select_static_recipe(
        proposal_script=proposal_script,
        problem=problem,
        db=db,
        conn=conn,
        experiment_id=experiment_id,
        run_dir=run_dir,
        batch_id=batch_id,
        seed_base=seed_base,
        focus=focus,
        partner=partner,
        route_prefix=prefix,
        target_constraint_override=target_constraint_override,
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
            proposal_script=proposal_script,
            problem=problem,
            db=db,
            conn=conn,
            experiment_id=experiment_id,
            run_dir=run_dir,
            batch_id=batch_id,
            seed_base=seed_base,
            focus=focus,
            partner=partner,
            route_prefix="governor_adaptive_scaffold/static_delegate",
            target_constraint_override=target_constraint_override,
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


def _build_bridge_cmd(
    *,
    proposal_script: str,
    problem: str,
    db: Path,
    conn: sqlite3.Connection,
    experiment_id: int,
    run_dir: Path,
    batch_id: int,
    seed_base: int,
    focus: CandidateRow,
    partner: CandidateRow,
    route_prefix: str,
) -> ProposalCommand:
    focus_boundary = _load_candidate_boundary(
        conn,
        experiment_id=experiment_id,
        design_hash=focus.design_hash,
    )
    partner_boundary = _load_candidate_boundary(
        conn,
        experiment_id=experiment_id,
        design_hash=partner.design_hash,
    )
    focus_path = _ensure_parent_file(
        run_dir,
        design_hash=focus.design_hash,
        boundary=focus_boundary,
    )
    partner_path = _ensure_parent_file(
        run_dir,
        design_hash=partner.design_hash,
        boundary=partner_boundary,
    )
    return _build_blend_cmd(
        proposal_script=proposal_script,
        problem=problem,
        db=db,
        experiment_id=experiment_id,
        run_dir=run_dir,
        batch_id=batch_id,
        seed_base=seed_base,
        parent_a=focus_path,
        parent_b=partner_path,
        t_min=0.20,
        t_max=0.80,
        t_step=0.15,
        model_route=f"{route_prefix}/bridge",
    )


def _build_forced_action_plan(
    *,
    action: str,
    proposal_script: str,
    problem: str,
    db: Path,
    conn: sqlite3.Connection,
    profile: ProblemProfile,
    experiment_id: int,
    run_dir: Path,
    batch_id: int,
    run_seed: int,
    seed_base: int,
    focus: CandidateRow,
    partner: CandidateRow | None,
    candidates: list[CandidateRow],
    target_constraint_override: str | None,
    bootstrap_parent_a: Path | None,
    bootstrap_parent_b: Path | None,
    bootstrap_t_min: float,
    bootstrap_t_max: float,
    bootstrap_t_step: float,
    route_prefix: str,
) -> tuple[list[ProposalCommand], dict, str]:
    normalized = str(action).strip().lower()
    if normalized == "repair":
        cmds, decision = _select_static_recipe(
            proposal_script=proposal_script,
            problem=problem,
            db=db,
            conn=conn,
            experiment_id=experiment_id,
            run_dir=run_dir,
            batch_id=batch_id,
            seed_base=seed_base,
            focus=focus,
            partner=None,
            route_prefix=f"{route_prefix}/repair",
            target_constraint_override=target_constraint_override,
        )
        return cmds, decision, "repair"

    if normalized == "jump" and profile.allows_action("jump"):
        cmds, jump_diag = _build_jump_recipe_cmds(
            proposal_script=proposal_script,
            problem=problem,
            db=db,
            conn=conn,
            jump_delta_cap=profile.mutation_budget.jump_delta_cap,
            experiment_id=experiment_id,
            run_dir=run_dir,
            batch_id=batch_id,
            seed_base=seed_base,
            focus=focus,
            partner=partner,
            route_prefix=route_prefix,
        )
        decision = {
            "batch_id": batch_id,
            "commands": [_cmd_str(cmd) for cmd in cmds],
            "model_route": f"{route_prefix}/jump",
            "recipe_mode": "policy_jump",
            "jump_policy": jump_diag,
            "created_at": _utc_now_iso(),
        }
        return cmds, decision, "jump"

    if (
        normalized == "bridge"
        and partner is not None
        and profile.allows_action("bridge")
    ):
        try:
            bridge_cmd = _build_bridge_cmd(
                proposal_script=proposal_script,
                problem=problem,
                db=db,
                conn=conn,
                experiment_id=experiment_id,
                run_dir=run_dir,
                batch_id=batch_id,
                seed_base=seed_base,
                focus=focus,
                partner=partner,
                route_prefix=route_prefix,
            )
        except ValueError:
            bridge_cmd = None
        if bridge_cmd is not None:
            cmds = [bridge_cmd]
            decision = {
                "batch_id": batch_id,
                "commands": [_cmd_str(bridge_cmd)],
                "model_route": f"{route_prefix}/bridge",
                "recipe_mode": "policy_bridge",
                "created_at": _utc_now_iso(),
            }
            return cmds, decision, "bridge"

    restart_parent_a, restart_parent_b, restart_parent_source = (
        _resolve_global_restart_parent_pair(
            run_dir=run_dir,
            batch_id=batch_id,
            focus=focus,
            partner=partner,
            candidates=candidates,
            bootstrap_parent_a=bootstrap_parent_a,
            bootstrap_parent_b=bootstrap_parent_b,
            run_seed=run_seed,
        )
    )
    restart_t_min, restart_t_max, restart_t_step = _restart_blend_schedule(
        run_seed=run_seed,
        batch_id=batch_id,
        t_min=float(bootstrap_t_min),
        t_max=float(bootstrap_t_max),
        t_step=float(bootstrap_t_step),
    )
    restart_cmd = _build_blend_cmd(
        proposal_script=proposal_script,
        problem=problem,
        db=db,
        experiment_id=experiment_id,
        run_dir=run_dir,
        batch_id=batch_id,
        seed_base=seed_base,
        parent_a=restart_parent_a,
        parent_b=restart_parent_b,
        t_min=restart_t_min,
        t_max=restart_t_max,
        t_step=restart_t_step,
        model_route=f"{route_prefix}/global_restart",
    )
    decision = {
        "batch_id": batch_id,
        "commands": [_cmd_str(restart_cmd)],
        "model_route": f"{route_prefix}/global_restart",
        "recipe_mode": "policy_global_restart",
        "restart_parent_source": restart_parent_source,
        "restart_schedule": {
            "t_min": restart_t_min,
            "t_max": restart_t_max,
            "t_step": restart_t_step,
        },
        "created_at": _utc_now_iso(),
    }
    return [restart_cmd], decision, "global_restart"


def _diversity_escalation_order(
    *,
    profile: ProblemProfile,
    current_action: str | None,
    partner_available: bool,
) -> list[str]:
    order: list[str] = []
    if partner_available and profile.allows_action("bridge"):
        order.append("bridge")
    if profile.allows_action("jump"):
        order.append("jump")
    if profile.allows_action("repair"):
        order.append("repair")
    order.append("global_restart")
    normalized_current = str(current_action).strip().lower() if current_action else None
    return [item for item in order if item != normalized_current] or ["global_restart"]


def _distinct_action_for_anti_repeat(
    *,
    profile: ProblemProfile,
    current_action: str | None,
    partner_available: bool,
) -> str:
    candidates = _diversity_escalation_order(
        profile=profile,
        current_action=current_action,
        partner_available=partner_available,
    )
    return candidates[0] if candidates else "global_restart"


def _run_cmds(cmds: Iterable[ProposalCommand]) -> dict[str, int | bool]:
    inserted_total = 0
    skipped_total = 0
    parsed_any = False
    for cmd in cmds:
        result = subprocess.run(
            cmd.argv,
            check=True,
            text=True,
            capture_output=True,
        )
        stdout_text = str(result.stdout or "")
        stderr_text = str(result.stderr or "")
        if stdout_text:
            print(stdout_text, end="")
        if stderr_text:
            print(stderr_text, end="", file=sys.stderr)
        insert_match = re.search(r"(?:^|\s)inserted=(\d+)", stdout_text)
        skip_match = re.search(r"(?:^|\s)skipped=(\d+)", stdout_text)
        if insert_match is not None:
            parsed_any = True
            inserted_total += int(insert_match.group(1))
        if skip_match is not None:
            parsed_any = True
            skipped_total += int(skip_match.group(1))
    return {
        "inserted": int(inserted_total),
        "skipped": int(skipped_total),
        "parsed": bool(parsed_any),
    }


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


def _consecutive_transient_failures(
    conn: sqlite3.Connection, *, experiment_id: int, limit: int = 20
) -> int:
    rows = conn.execute(
        """
        SELECT status
        FROM candidates
        WHERE experiment_id = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (int(experiment_id), int(limit)),
    ).fetchall()
    count = 0
    for row in rows:
        status = str(row["status"] or "")
        bucket = _status_bucket(status)
        if bucket in {"failed", "timeout"}:
            count += 1
            continue
        break
    return count


def _queue_desync_events_last20(
    conn: sqlite3.Connection, *, experiment_id: int, stale_minutes: int = 60
) -> int:
    rows = conn.execute(
        """
        SELECT status
        FROM candidates
        WHERE experiment_id = ?
        ORDER BY id DESC
        LIMIT 20
        """,
        (int(experiment_id),),
    ).fetchall()
    now_utc = datetime.now(timezone.utc)
    stale_count = 0
    for row in rows:
        status = str(row["status"] or "")
        if not status.startswith("running:"):
            continue
        parts = status.split(":")
        if len(parts) < 3:
            stale_count += 1
            continue
        stamp = parts[2]
        try:
            started = datetime.strptime(stamp, "%Y%m%dT%H%M%S").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            stale_count += 1
            continue
        age_minutes = (now_utc - started).total_seconds() / 60.0
        if age_minutes >= float(stale_minutes):
            stale_count += 1
    return stale_count


def _stagnation_cycles(conn: sqlite3.Connection, *, experiment_id: int) -> int:
    rows = conn.execute(
        """
        SELECT outcome_json
        FROM scratchpad_events
        WHERE experiment_id = ? AND step = 1 AND aso_action = 'reflection'
        ORDER BY id DESC
        LIMIT 20
        """,
        (int(experiment_id),),
    ).fetchall()
    if not rows:
        # Bootstrap fallback before reflection events are available.
        metric_rows = conn.execute(
            """
            SELECT m.is_feasible
            FROM metrics m
            JOIN candidates c ON c.id = m.candidate_id
            WHERE c.experiment_id = ?
            ORDER BY m.id DESC
            LIMIT 20
            """,
            (int(experiment_id),),
        ).fetchall()
        fallback_count = 0
        for row in metric_rows:
            if int(row["is_feasible"]) == 1:
                break
            fallback_count += 1
        return fallback_count
    count = 0
    for row in rows:
        try:
            outcome = json.loads(str(row["outcome_json"]))
        except json.JSONDecodeError:
            count += 1
            continue
        if not isinstance(outcome, dict):
            count += 1
            continue
        realized_feasibility_delta = _as_finite_float(
            outcome.get("realized_feasibility_delta")
        )
        realized_hv_delta = _as_finite_float(outcome.get("realized_hv_delta"))
        has_progress = (
            realized_feasibility_delta is not None and realized_feasibility_delta > 0.0
        ) or (realized_hv_delta is not None and realized_hv_delta > 0.0)
        if has_progress:
            break
        count += 1
    return count


def _invalid_llm_outputs_last20(conn: sqlite3.Connection, *, experiment_id: int) -> int:
    rows = conn.execute(
        """
        SELECT override_reason
        FROM scratchpad_events
        WHERE experiment_id = ?
        ORDER BY id DESC
        LIMIT 20
        """,
        (int(experiment_id),),
    ).fetchall()
    return sum(
        1 for row in rows if _is_invalid_llm_override_reason(row["override_reason"])
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="P3 proposal governor.")
    parser.add_argument(
        "--problem",
        choices=["p1", "p2", "p3"],
        default="p3",
        help="Problem profile used for action/constraint contracts.",
    )
    parser.add_argument(
        "--proposal-script",
        type=str,
        default=None,
        help="Proposal backend script path. Defaults to problem-scoped script with p3 fallback.",
    )
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
    parser.add_argument(
        "--run-seed",
        type=int,
        default=None,
        help="Deterministic run seed persisted in the resume manifest.",
    )
    parser.add_argument(
        "--resume-manifest-path",
        type=Path,
        default=None,
        help="Optional path override for resume/replay manifest JSON.",
    )
    parser.add_argument("--max-focus-feas", type=float, default=0.25)
    parser.add_argument("--recent-limit", type=int, default=500)
    parser.add_argument("--record-hv", type=float, default=_DEFAULT_RECORD_HV)
    parser.add_argument("--sleep-sec", type=float, default=15.0)
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=0,
        help="Stop after N proposal cycles (0 disables cap).",
    )
    parser.add_argument(
        "--max-runtime-sec",
        type=float,
        default=0.0,
        help="Stop after this wall-clock runtime in seconds (0 disables cap).",
    )
    parser.add_argument(
        "--max-stagnation-cycles",
        type=int,
        default=0,
        help="Stop when stagnation cycles reach this value (0 uses profile SSOT threshold).",
    )
    parser.add_argument(
        "--recipe-synthesis-interval",
        type=int,
        default=3,
        help="Synthesize compact recipe rules every N cycles (minimum 1).",
    )
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
    parser.add_argument(
        "--llm-enabled",
        action="store_true",
        help="Enable LLM decision contract (action/constraint/mutation schema).",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="codex",
        help="Model label recorded in decision artifacts.",
    )
    parser.add_argument(
        "--llm-session-id",
        type=str,
        default=None,
        help="Optional session id for single-session decision context.",
    )
    parser.add_argument(
        "--llm-decision-file",
        type=Path,
        default=None,
        help="Path to JSON decision payload (dev/tests only; requires --llm-allow-decision-file).",
    )
    parser.add_argument(
        "--llm-codex-command",
        type=str,
        default=None,
        help="Codex command path used as the runtime LLM transport.",
    )
    parser.add_argument(
        "--llm-decision-command",
        type=str,
        default=None,
        help="Deprecated alias for --llm-codex-command; still validated as codex-only.",
    )
    parser.add_argument(
        "--llm-allow-decision-file",
        action="store_true",
        help="Allow file payload transport for tests/dev only.",
    )
    parser.add_argument(
        "--llm-fallback",
        action="store_true",
        help="Allow deterministic fallback when LLM decision payload is missing/invalid.",
    )
    parser.add_argument(
        "--llm-repair-attempts",
        type=int,
        default=2,
        help="Retry count for in-session LLM self-repair on invalid/missing output.",
    )
    parser.add_argument(
        "--autonomous",
        action="store_true",
        help="Enable self-driving loop: supervise workers and apply stall-driven steering.",
    )
    parser.add_argument(
        "--autonomous-worker-script",
        type=str,
        default="scripts/p3_worker.py",
        help="Worker script path used by autonomous supervisor.",
    )
    parser.add_argument(
        "--autonomous-worker-limit",
        type=int,
        default=0,
        help="Per-worker evaluation cap in autonomous mode (0 = unlimited).",
    )
    parser.add_argument(
        "--autonomy-disable-run-surgery",
        action="store_true",
        help="Disable run-surgery controller actions (rebootstrap/prune/operator-shift).",
    )
    parser.add_argument(
        "--autonomy-disable-autoscale",
        action="store_true",
        help="Disable autonomous worker autoscaling.",
    )
    args = parser.parse_args(argv)

    profile = get_problem_profile(args.problem)
    if bool(args.autonomous):
        args.execute = True
        args.loop = True
    proposal_script = (
        str(args.proposal_script)
        if args.proposal_script is not None
        else _default_proposal_script(profile.problem)
    )
    proposal_script_path = (_REPO_ROOT / proposal_script).resolve()
    if not proposal_script_path.exists():
        raise ValueError(f"Proposal backend script does not exist: {proposal_script}")
    manifest_path = _resolve_manifest_path(
        run_dir=args.run_dir,
        override=args.resume_manifest_path,
    )
    resume_manifest = _load_or_init_resume_manifest(
        path=manifest_path,
        experiment_id=int(args.experiment_id),
        problem=profile.problem,
        proposal_script=proposal_script,
        run_seed_override=args.run_seed,
    )
    run_seed = int(resume_manifest["run_seed"])
    llm_decision_file, llm_decision_command, _ = _resolve_llm_transport(args)
    recipe_synthesis_interval = max(1, int(args.recipe_synthesis_interval))
    worker_script_path = _resolve_worker_script_path(str(args.autonomous_worker_script))
    if bool(args.autonomous) and not worker_script_path.exists():
        raise ValueError(
            f"Autonomous worker script does not exist: {worker_script_path}"
        )

    current_workers = int(args.workers)
    if bool(args.autonomous):
        run_surgery_policy = profile.autonomy_policy.run_surgery
        current_workers = max(
            int(run_surgery_policy.autoscale_min_workers),
            min(int(current_workers), int(run_surgery_policy.autoscale_max_workers)),
        )
    target_queue = int(current_workers) * int(args.queue_multiplier)
    max_cycles = max(0, int(args.max_cycles))
    max_runtime_sec = max(0.0, float(args.max_runtime_sec))
    stagnation_stop_cap = int(args.max_stagnation_cycles)
    if stagnation_stop_cap <= 0:
        stagnation_stop_cap = int(
            profile.restart_thresholds.global_restart_min_stagnation_cycles
        )
    runtime_started = time.monotonic()
    proposal_cycles_emitted = 0

    conn = _connect(args.db)
    worker_pool: dict[int, WorkerRuntime] = {}
    try:
        schema_compatible, schema_reason = _schema_compatibility_status(conn)
        if not schema_compatible:
            batch_id = _next_batch_id_from_manifest(
                manifest=resume_manifest,
                run_dir=args.run_dir,
            )
            decision = {
                "batch_id": batch_id,
                "commands": [],
                "model_route": "governor_policy/circuit_break",
                "recipe_mode": "policy_circuit_break",
                "problem": profile.problem,
                "proposal_script": proposal_script,
                "restart_policy": {
                    "selected": "circuit_break",
                    "reason": schema_reason,
                },
                "run_seed": run_seed,
                "created_at": _utc_now_iso(),
            }
            artifact_path = (
                args.run_dir
                / "governor"
                / f"governor_batch_{batch_id:03}_{_utc_stamp()}.json"
            )
            _write_json(artifact_path, decision)
            print(str(schema_reason))
            return
        replay_cmds, replay_diagnostics = _startup_replay_commands(
            manifest=resume_manifest,
            run_dir=args.run_dir,
            conn=conn,
        )
        if replay_cmds:
            replay_batches = replay_diagnostics.get("replay_batches", [])
            replay_cycle_count = _replay_cycle_emissions(replay_diagnostics)
            print(
                "startup_replay:"
                f" batches={replay_batches}"
                f" commands={len(replay_cmds)}"
                f" cycles_counted={replay_cycle_count}"
            )
            for cmd in replay_cmds:
                print(_cmd_str(cmd))
            if args.execute:
                _run_cmds(replay_cmds)
            proposal_cycles_emitted += int(replay_cycle_count)
            if not args.loop:
                return
        recipe_rules_cache: list[dict] = []
        recipe_rules_cache_reflection_id = -1
        invalid_parent_failure_streak = 0
        zero_yield_streak = 0
        force_action_next_cycle: str | None = None
        run_surgery_state = RunSurgeryState()
        autonomy_progress = AutonomyProgress(
            last_metric_count=0,
            best_feasibility_seen=None,
            best_objective_feasible_seen=None,
            evals_since_feasibility_improve=0,
            evals_since_objective_improve=0,
        )
        if bool(args.autonomous):
            worker_pool = _start_worker_pool(
                workers=int(current_workers),
                problem=profile.problem,
                db=args.db,
                experiment_id=int(args.experiment_id),
                run_dir=args.run_dir,
                worker_script_path=worker_script_path,
                worker_limit=int(args.autonomous_worker_limit),
                worker_sleep_sec=max(float(args.sleep_sec) * 0.25, 1.0),
            )
        while True:
            if bool(args.autonomous):
                _supervise_worker_pool(
                    pool=worker_pool,
                    problem=profile.problem,
                    db=args.db,
                    experiment_id=int(args.experiment_id),
                    run_dir=args.run_dir,
                    worker_script_path=worker_script_path,
                    worker_limit=int(args.autonomous_worker_limit),
                    worker_sleep_sec=max(float(args.sleep_sec) * 0.25, 1.0),
                )
            if max_cycles > 0 and proposal_cycles_emitted >= max_cycles:
                print(
                    "stop_policy:max_cycles"
                    f" reached={proposal_cycles_emitted} cap={max_cycles}"
                )
                break
            elapsed_runtime = time.monotonic() - runtime_started
            if max_runtime_sec > 0.0 and elapsed_runtime >= max_runtime_sec:
                print(
                    "stop_policy:max_runtime_sec"
                    f" elapsed={elapsed_runtime:.3f} cap={max_runtime_sec:.3f}"
                )
                break
            exp_id = int(args.experiment_id)
            pending = conn.execute(
                "SELECT COUNT(*) AS n FROM candidates WHERE experiment_id = ? AND status = 'pending'",
                (exp_id,),
            ).fetchone()
            pending_n = int(pending["n"]) if pending is not None else 0

            # Compute current feasible HV (best estimate from DB).
            feasible_rows = conn.execute(
                """
                SELECT m.objective AS objective, m.raw_json AS raw
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
                if row["objective"] is None or "aspect_ratio" not in metrics:
                    continue
                feasible_points.append(
                    (
                        _canonical_objective_utility(
                            profile=profile,
                            objective_value=float(row["objective"]),
                        ),
                        float(metrics["aspect_ratio"]),
                    )
                )
            hv_value = _compute_hv(feasible_points)
            snapshot = _current_snapshot(
                conn,
                experiment_id=exp_id,
                hv_value=hv_value,
            )
            autonomy_progress = _update_autonomy_progress(
                snapshot=snapshot,
                progress=autonomy_progress,
            )
            autonomous_strategy: str | None = None
            if bool(args.autonomous):
                objective_stall_window = int(
                    profile.autonomy_policy.autonomous_objective_stall_eval_window
                )
                feasibility_stall_window = int(
                    profile.autonomy_policy.autonomous_feasibility_stall_eval_window
                )
                feasible_count_snapshot = snapshot.get("feasible_count")
                feasible_count = (
                    int(feasible_count_snapshot)
                    if isinstance(feasible_count_snapshot, int)
                    and not isinstance(feasible_count_snapshot, bool)
                    else 0
                )
                if (
                    int(autonomy_progress.evals_since_objective_improve)
                    >= objective_stall_window
                    and feasible_count > 0
                ):
                    autonomous_strategy = "bridge_obj"
                elif (
                    int(autonomy_progress.evals_since_feasibility_improve)
                    >= feasibility_stall_window
                ):
                    autonomous_strategy = "exploit_feas"

            print(
                f"[{_utc_now_iso()}] exp={exp_id} pending={pending_n}/{target_queue} hv={hv_value:.6f} record={float(args.record_hv):.6f}"
            )

            if pending_n >= target_queue and not bool(args.autonomous):
                if not args.loop:
                    break
                time.sleep(float(args.sleep_sec))
                continue

            candidates = _fetch_candidates(
                conn,
                profile=profile,
                experiment_id=exp_id,
                limit=int(args.recent_limit),
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

                batch_id = _next_batch_id_from_manifest(
                    manifest=resume_manifest,
                    run_dir=args.run_dir,
                )
                seed_base = int(run_seed) + int(batch_id) * 1_000
                with conn:
                    _log_reflection_for_previous_cycle(
                        conn,
                        experiment_id=exp_id,
                        next_cycle=batch_id,
                        snapshot=snapshot,
                    )
                bootstrap_route = _bootstrap_route_label(adaptive=bool(args.adaptive))
                cmd = _build_blend_cmd(
                    proposal_script=proposal_script,
                    problem=profile.problem,
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
                    "proposal_script": proposal_script,
                    "run_seed": run_seed,
                    "seed_base": seed_base,
                    "created_at": _utc_now_iso(),
                }
                _record_cycle_manifest(
                    manifest=resume_manifest,
                    path=manifest_path,
                    batch_id=batch_id,
                    seed_base=seed_base,
                    cmds=[cmd],
                    model_route=bootstrap_route,
                )
                artifact_path = (
                    args.run_dir
                    / "governor"
                    / f"governor_bootstrap_batch_{batch_id:03}_{_utc_stamp()}.json"
                )
                _write_json(artifact_path, decision)
                bootstrap_target_metric = (
                    profile.constraints[0].name if profile.constraints else "unknown"
                )
                with conn:
                    _log_governor_artifact(
                        conn, experiment_id=exp_id, path=artifact_path
                    )
                    _log_scratchpad_event(
                        conn,
                        experiment_id=exp_id,
                        cycle=int(batch_id),
                        step=0,
                        planner_intent={},
                        aso_action="bootstrap",
                        intent_agreement="disabled",
                        override_reason=None,
                        diagnostics={
                            "model_route": bootstrap_route,
                            "problem": profile.problem,
                            "action": "bootstrap",
                            "predicted_target_metric": bootstrap_target_metric,
                            "predicted_direction": _constraint_direction(
                                profile,
                                constraint=bootstrap_target_metric,
                            ),
                            "predicted_expected_effect": "bootstrap_blend_seed_bank",
                            "knob_signature": _command_knob_signature([cmd]),
                            "batch_id": int(batch_id),
                            "command_count": 1,
                            "expected_candidate_count": int(
                                _expected_candidate_volume([cmd])
                            ),
                            "phase_after_policy": str(
                                resume_manifest.get("phase", "feasibility_recovery")
                            ),
                            "restart_selected": "continue",
                        },
                        outcome={
                            "pending_before": int(pending_n),
                            "target_queue": int(target_queue),
                            "hv_at_decision": float(hv_value),
                            "record_hv": float(args.record_hv),
                            "snapshot_at_decision": dict(snapshot),
                        },
                    )
                print(_cmd_str(cmd))
                if args.execute:
                    _run_cmds([cmd])
                proposal_cycles_emitted += 1
                if not args.loop:
                    break
                time.sleep(float(args.sleep_sec))
                continue

            recent_20 = candidates[:20]
            recent_10 = candidates[:10]
            accepted_feasible_last20 = sum(1 for row in recent_20 if row.is_feasible)
            accepted_feasible_last10 = sum(1 for row in recent_10 if row.is_feasible)
            dominant_violation_rate_last20 = _dominant_violation_rate(
                candidates,
                limit=20,
            )
            current_phase = (
                str(resume_manifest.get("phase", "feasibility_recovery"))
                .strip()
                .lower()
            )
            if current_phase not in {"feasibility_recovery", "frontier_improvement"}:
                current_phase = "feasibility_recovery"
            phase_after_policy = evaluate_phase_transition(
                profile=profile,
                current_phase=current_phase,
                accepted_feasible_last20=accepted_feasible_last20,
                dominant_violation_rate_last20=dominant_violation_rate_last20,
                accepted_feasible_last10=accepted_feasible_last10,
            )

            parent_group_meta: dict[str, object] | None = None
            if bool(args.autonomous) and autonomous_strategy is not None:
                autonomous_selection = _autonomous_focus_partner(
                    candidates=candidates,
                    profile=profile,
                    strategy=autonomous_strategy,
                )
                if autonomous_selection is not None:
                    focus, partner = autonomous_selection
                    parent_group_meta = {
                        "group": f"autonomous:{autonomous_strategy}",
                        "score": None,
                        "candidate_count": 0,
                    }
                elif args.adaptive:
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
                        print("No autonomous focus candidate found in recent window.")
                        if not args.loop:
                            break
                        time.sleep(float(args.sleep_sec))
                        continue
                    worst_name, _worst_val = _worst_constraint(focus.violations)
                    worst = str(worst_name) if worst_name is not None else ""
                    partner = _choose_partner(candidates, worst_constraint=worst)
            elif args.adaptive:
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
                if (
                    profile.problem == "p3"
                    and phase_after_policy == "frontier_improvement"
                ):
                    focus = _choose_frontier_focus(candidates, profile=profile)
                    if focus is None:
                        focus = _choose_focus(
                            candidates, max_feasibility=float(args.max_focus_feas)
                        )
                    if focus is None:
                        print("No frontier focus candidate found in recent window.")
                        if not args.loop:
                            break
                        time.sleep(float(args.sleep_sec))
                        continue
                    if focus.is_feasible:
                        partner = _choose_frontier_partner(
                            candidates,
                            focus=focus,
                            profile=profile,
                        )
                    else:
                        worst_name, _worst_val = _worst_constraint(focus.violations)
                        worst = str(worst_name) if worst_name is not None else ""
                        partner = _choose_partner(candidates, worst_constraint=worst)
                else:
                    shared_selection = _focus_partner_via_shared_staged_plan(
                        candidates,
                        max_feasibility=float(args.max_focus_feas),
                        problem=profile.problem,
                    )
                    if shared_selection is None:
                        focus = _choose_focus(
                            candidates, max_feasibility=float(args.max_focus_feas)
                        )
                        if focus is None:
                            print(
                                "No near-feasible focus candidate found in recent window."
                            )
                            if not args.loop:
                                break
                            time.sleep(float(args.sleep_sec))
                            continue
                        worst_name, _worst_val = _worst_constraint(focus.violations)
                        worst = str(worst_name) if worst_name is not None else ""
                        partner = _choose_partner(candidates, worst_constraint=worst)
                    else:
                        focus, partner = shared_selection

            focus, partner, parent_sanitization = _sanitize_cycle_parents(
                focus=focus,
                partner=partner,
                candidates=candidates,
            )
            if bool(parent_sanitization.get("invalid_parent_detected")):
                invalid_parent_failure_streak += 1
            else:
                invalid_parent_failure_streak = 0
            invalid_basin_escape_due = int(invalid_parent_failure_streak) >= int(
                profile.autonomy_policy.invalid_basin_max_consecutive_parent_failures
            )
            if focus is None:
                print(
                    "No valid sanitized focus candidate found; cannot issue local proposals."
                )
                if args.bootstrap_parent_a is None or args.bootstrap_parent_b is None:
                    if not args.loop:
                        break
                    time.sleep(float(args.sleep_sec))
                    continue
                # Placeholder for global restart path that uses explicit bootstrap parents.
                focus = candidates[0]
                partner = None
                invalid_basin_escape_due = True

            batch_id = _next_batch_id_from_manifest(
                manifest=resume_manifest,
                run_dir=args.run_dir,
            )
            seed_base = int(run_seed) + int(batch_id) * 1_000
            with conn:
                _log_reflection_for_previous_cycle(
                    conn,
                    experiment_id=exp_id,
                    next_cycle=batch_id,
                    snapshot=snapshot,
                )
            recipe_rules: list[dict] = []
            if int(batch_id) % int(recipe_synthesis_interval) == 0:
                latest_reflection_id = _latest_reflection_event_id(
                    conn,
                    experiment_id=exp_id,
                )
                if latest_reflection_id != recipe_rules_cache_reflection_id:
                    recipe_rules_cache = _synthesize_recipe_rules(
                        conn,
                        experiment_id=exp_id,
                    )
                    recipe_rules_cache_reflection_id = latest_reflection_id
                recipe_rules = list(recipe_rules_cache)
            run_budget_remaining = _remaining_run_budget(
                proposal_cycles_emitted=proposal_cycles_emitted,
                max_cycles=max_cycles,
                elapsed_runtime_sec=elapsed_runtime,
                max_runtime_sec=max_runtime_sec,
            )
            llm_observation: Mapping[str, object] | None = None
            llm_intent_payload: Mapping[str, object] | None = None
            llm_validated_decision = None
            llm_selected_action: str | None = None
            llm_selected_constraint: str | None = None
            llm_fallback_reason: str | None = None
            llm_input_source: str | None = None
            consecutive_failures = _consecutive_transient_failures(
                conn,
                experiment_id=exp_id,
            )
            queue_desync_events = _queue_desync_events_last20(
                conn,
                experiment_id=exp_id,
            )
            stagnation = _stagnation_cycles(
                conn,
                experiment_id=exp_id,
            )
            invalid_outputs = _invalid_llm_outputs_last20(
                conn,
                experiment_id=exp_id,
            )
            frontier_integrity_ok = _frontier_integrity_ok(
                snapshot=snapshot,
                hv_value=hv_value,
            )
            policy_restart_plan = select_restart_plan(
                profile=profile,
                consecutive_transient_failures=consecutive_failures,
                queue_desync_events_last20=queue_desync_events,
                stagnation_cycles=stagnation,
                budget_remaining=run_budget_remaining,
                invalid_llm_outputs_last20=invalid_outputs,
                schema_compatible=schema_compatible,
                frontier_integrity_ok=frontier_integrity_ok,
            )
            if invalid_basin_escape_due and policy_restart_plan != "circuit_break":
                policy_restart_plan = "global_restart"
            if (
                int(stagnation) >= int(stagnation_stop_cap)
                and policy_restart_plan != "global_restart"
            ):
                print(
                    "stop_policy:max_stagnation_cycles"
                    f" reached={int(stagnation)} cap={int(stagnation_stop_cap)}"
                )
                break

            run_surgery_event = _empty_run_surgery_event()
            run_surgery_forced_action: str | None = None
            run_surgery_parent_a: Path | None = None
            run_surgery_parent_b: Path | None = None
            run_surgery_parent_pair_hash: str | None = None
            run_surgery_post_restart_forced_action: str | None = None
            hard_restart_plan = policy_restart_plan in {
                "global_restart",
                "circuit_break",
            }
            if bool(args.autonomous):
                run_surgery_result = _maybe_run_surgery(
                    conn=conn,
                    experiment_id=exp_id,
                    profile=profile,
                    run_dir=args.run_dir,
                    state=run_surgery_state,
                    progress=_compute_frontier_progress(
                        conn,
                        experiment_id=exp_id,
                        eval_window=profile.autonomy_policy.run_surgery.eval_window,
                    ),
                    pending_n=int(pending_n),
                    target_queue=int(target_queue),
                    current_workers=int(current_workers),
                    invalid_parent_failure_streak=int(invalid_parent_failure_streak),
                    partner_available=partner is not None,
                    disable_run_surgery=bool(args.autonomy_disable_run_surgery)
                    or bool(hard_restart_plan),
                    disable_autoscale=bool(args.autonomy_disable_autoscale),
                )
                if run_surgery_result is not None:
                    run_surgery_event["action"] = str(
                        run_surgery_result.get("action", "none")
                    )
                    run_surgery_event["reason"] = run_surgery_result.get("reason")
                    run_surgery_event["parent_pair"] = run_surgery_result.get(
                        "parent_pair"
                    )
                    run_surgery_event["backlog_pruned_count"] = int(
                        run_surgery_result.get("backlog_pruned_count", 0)
                    )
                    run_surgery_event["operator_shift_lock_cycles"] = int(
                        run_surgery_result.get("operator_shift_lock_cycles", 0)
                    )
                    run_surgery_event["forced_action"] = run_surgery_result.get(
                        "forced_action"
                    )
                    run_surgery_event["autoscale"] = run_surgery_result.get("autoscale")
                    action_name = str(run_surgery_result.get("action", "none"))
                    if action_name == "rebootstrap_pair":
                        parent_a_raw = run_surgery_result.get("parent_a")
                        parent_b_raw = run_surgery_result.get("parent_b")
                        if isinstance(parent_a_raw, (str, Path)) and isinstance(
                            parent_b_raw, (str, Path)
                        ):
                            run_surgery_parent_a = Path(str(parent_a_raw))
                            run_surgery_parent_b = Path(str(parent_b_raw))
                        parent_pair_payload = run_surgery_result.get("parent_pair")
                        if isinstance(parent_pair_payload, Mapping):
                            pair_hash_raw = parent_pair_payload.get("hash")
                            if isinstance(pair_hash_raw, str) and pair_hash_raw.strip():
                                run_surgery_parent_pair_hash = pair_hash_raw.strip()
                    forced_action_raw = run_surgery_result.get("forced_action")
                    if isinstance(forced_action_raw, str) and forced_action_raw.strip():
                        run_surgery_forced_action = forced_action_raw.strip()
                    post_restart_action_raw = run_surgery_result.get(
                        "post_restart_forced_action"
                    )
                    if (
                        isinstance(post_restart_action_raw, str)
                        and post_restart_action_raw.strip()
                    ):
                        run_surgery_post_restart_forced_action = (
                            post_restart_action_raw.strip()
                        )
                    if action_name in {"autoscale_up", "autoscale_down"}:
                        desired_workers_raw = run_surgery_result.get("desired_workers")
                        if (
                            isinstance(desired_workers_raw, int)
                            and not isinstance(desired_workers_raw, bool)
                            and desired_workers_raw > 0
                            and desired_workers_raw != int(current_workers)
                        ):
                            workers_before = int(current_workers)
                            _resize_worker_pool(
                                pool=worker_pool,
                                desired_workers=int(desired_workers_raw),
                                problem=profile.problem,
                                db=args.db,
                                experiment_id=exp_id,
                                run_dir=args.run_dir,
                                worker_script_path=worker_script_path,
                                worker_limit=int(args.autonomous_worker_limit),
                                worker_sleep_sec=max(float(args.sleep_sec) * 0.25, 1.0),
                            )
                            current_workers = int(desired_workers_raw)
                            target_queue = int(current_workers) * int(
                                args.queue_multiplier
                            )
                            run_surgery_event["autoscale"] = {
                                "workers_before": int(workers_before),
                                "workers_after": int(current_workers),
                                "target_queue_after": int(target_queue),
                            }
                    if action_name == "backlog_prune":
                        pending = conn.execute(
                            "SELECT COUNT(*) AS n FROM candidates WHERE experiment_id = ? AND status = 'pending'",
                            (exp_id,),
                        ).fetchone()
                        pending_n = int(pending["n"]) if pending is not None else 0
                        conn.commit()
            if pending_n >= target_queue:
                if str(run_surgery_event.get("action", "none")) != "none":
                    surgery_artifact = {
                        "batch_id": int(batch_id),
                        "commands": [],
                        "model_route": "governor_policy/run_surgery_only",
                        "recipe_mode": "run_surgery_only",
                        "problem": profile.problem,
                        "proposal_script": proposal_script,
                        "run_seed": run_seed,
                        "seed_base": int(seed_base),
                        "created_at": _utc_now_iso(),
                        "run_surgery": {
                            "action": run_surgery_event.get("action", "none"),
                            "reason": run_surgery_event.get("reason"),
                            "parent_pair": run_surgery_event.get("parent_pair"),
                            "backlog_pruned_count": int(
                                run_surgery_event.get("backlog_pruned_count", 0)
                            ),
                            "operator_shift_lock_cycles": int(
                                run_surgery_event.get("operator_shift_lock_cycles", 0)
                            ),
                            "autoscale": run_surgery_event.get("autoscale"),
                        },
                    }
                    artifact_path = (
                        args.run_dir
                        / "governor"
                        / f"governor_batch_{batch_id:03}_{_utc_stamp()}.json"
                    )
                    _write_json(artifact_path, surgery_artifact)
                    with conn:
                        _log_governor_artifact(
                            conn, experiment_id=exp_id, path=artifact_path
                        )
                if not args.loop:
                    break
                time.sleep(float(args.sleep_sec))
                continue

            if args.llm_enabled:
                lesson_summary = _lesson_summary(candidates, limit=40)
                if recipe_rules:
                    lesson_summary["recipe_rules"] = recipe_rules
                observation_rows = [
                    {
                        "is_feasible": bool(row.is_feasible),
                        "feasibility": float(row.feasibility),
                        "violations": dict(row.violations),
                        "operator_family": str(row.operator_family),
                        "model_route": str(row.model_route),
                    }
                    for row in candidates[:100]
                ]
                llm_observation = build_observation(
                    profile=profile,
                    rows=observation_rows,
                    context={
                        "batch_id": int(batch_id),
                        "pending": int(pending_n),
                        "target_queue": int(target_queue),
                        "hv_at_decision": float(hv_value),
                        "record_hv": float(args.record_hv),
                        "remaining_budget": float(run_budget_remaining),
                        "phase": current_phase,
                        "dominant_violation": _dominant_violation(candidates, limit=20),
                        "lesson_summary": lesson_summary,
                        "governor_mode": "adaptive"
                        if bool(args.adaptive)
                        else "static",
                        "consecutive_failures": int(consecutive_failures),
                        "queue_desync_events_last20": int(queue_desync_events),
                        "stagnation_cycles": int(stagnation),
                        "invalid_llm_outputs_last20": int(invalid_outputs),
                        "frontier_integrity_ok": bool(frontier_integrity_ok),
                        "policy_restart_plan": str(policy_restart_plan),
                        "parent_sanitization": dict(parent_sanitization),
                        "invalid_parent_failure_streak": int(
                            invalid_parent_failure_streak
                        ),
                        "invalid_basin_escape_due": bool(invalid_basin_escape_due),
                        "autonomous_enabled": bool(args.autonomous),
                        "autonomous_strategy_hint": autonomous_strategy,
                        "evals_since_feasibility_improve": int(
                            autonomy_progress.evals_since_feasibility_improve
                        ),
                        "evals_since_objective_improve": int(
                            autonomy_progress.evals_since_objective_improve
                        ),
                        "run_surgery_state": {
                            "no_objective_progress_windows": int(
                                run_surgery_state.no_objective_progress_windows
                            ),
                            "no_feasibility_progress_windows": int(
                                run_surgery_state.no_feasibility_progress_windows
                            ),
                            "same_action_windows": int(
                                run_surgery_state.same_action_windows
                            ),
                            "operator_shift_lock_remaining": int(
                                run_surgery_state.operator_shift_lock_remaining
                            ),
                            "autoscale_cooldown_remaining": int(
                                run_surgery_state.autoscale_cooldown_remaining
                            ),
                        },
                        "current_workers": int(current_workers),
                    },
                )
                try:
                    llm_decide_result = decide(
                        profile=profile,
                        observation=llm_observation,
                        model=str(args.llm_model),
                        session_id=args.llm_session_id,
                        decision_file=llm_decision_file,
                        decision_command=llm_decision_command,
                        repair_attempts=int(args.llm_repair_attempts),
                    )
                    llm_intent_payload = dict(llm_decide_result.payload)
                    llm_input_source = llm_decide_result.input_source
                    llm_validated_decision = llm_decide_result.validated_decision
                    llm_selected_action = llm_validated_decision.selected_action
                    llm_selected_constraint = llm_validated_decision.selected_constraint
                except ValueError as exc:
                    if not bool(args.llm_fallback):
                        raise ValueError(f"llm_decision_unresolved:{exc}") from exc
                    llm_fallback_reason = f"invalid_llm_output:{exc}"
                    llm_input_source = "fallback_policy"
            if llm_validated_decision is not None:
                requested_restart = llm_validated_decision.decision.restart_plan
                if requested_restart in {
                    "soft_retry",
                    "degraded_restart",
                    "global_restart",
                    "circuit_break",
                }:
                    policy_hard_restart_lock = policy_restart_plan in {
                        "global_restart",
                        "circuit_break",
                    }
                    if (
                        policy_hard_restart_lock
                        and requested_restart != policy_restart_plan
                    ):
                        llm_fallback_reason = (
                            "policy_override_blocked:hard_restart_trigger"
                        )
                    else:
                        policy_restart_plan = requested_restart

            hard_restart_lock = policy_restart_plan in {
                "global_restart",
                "circuit_break",
            }

            effective_action = llm_selected_action
            autonomous_forced_action: str | None = None
            deterministic_action = _deterministic_restart_action(
                profile_problem=profile.problem,
                policy_restart_plan=policy_restart_plan,
                partner_available=partner is not None,
            )
            zero_yield_forced_action: str | None = None
            if hard_restart_lock:
                if (
                    effective_action is not None
                    and deterministic_action is not None
                    and effective_action != deterministic_action
                ):
                    llm_fallback_reason = "policy_override_blocked:hard_restart_trigger"
                effective_action = deterministic_action
                force_action_next_cycle = None
            elif run_surgery_forced_action is not None:
                effective_action = str(run_surgery_forced_action)
                if llm_fallback_reason is None:
                    run_surgery_reason = str(
                        run_surgery_event.get("reason")
                        or run_surgery_event.get("action")
                    )
                    llm_fallback_reason = (
                        f"policy_override:run_surgery:{run_surgery_reason}"
                    )
            elif force_action_next_cycle is not None:
                zero_yield_forced_action = str(force_action_next_cycle)
                effective_action = zero_yield_forced_action
                if llm_fallback_reason is None:
                    llm_fallback_reason = f"policy_override:zero_yield_recovery:{zero_yield_forced_action}"
                force_action_next_cycle = None
            elif effective_action is None:
                effective_action = deterministic_action
            if (
                bool(args.autonomous)
                and autonomous_strategy is not None
                and not hard_restart_lock
            ):
                desired_action = (
                    "bridge"
                    if profile.allows_action("bridge") and partner is not None
                    else "repair"
                )
                if effective_action != desired_action:
                    autonomous_forced_action = desired_action
                    if llm_fallback_reason is None:
                        llm_fallback_reason = (
                            f"policy_override:autonomous_{autonomous_strategy}"
                        )
                    effective_action = desired_action
            should_circuit_break = policy_restart_plan == "circuit_break"

            route_prefix: str | None = None
            partner_for_recipe = partner
            if effective_action is not None:
                route_prefix = f"governor_llm/{effective_action}"
                if effective_action == "repair":
                    partner_for_recipe = None
                elif effective_action == "jump":
                    # Keep partner when available so jump has broader non-local moves.
                    partner_for_recipe = partner
                elif effective_action == "bridge":
                    if partner is None:
                        partner_for_recipe = None
                        llm_fallback_reason = (
                            "policy_fallback:bridge_requested_without_partner"
                        )
                        effective_action = "repair"
                elif effective_action == "global_restart":
                    partner_for_recipe = None

            cmds: list[ProposalCommand]
            decision: dict
            target_constraint_override = (
                llm_selected_constraint if llm_selected_constraint is not None else None
            )
            use_frontier_recipe = (
                profile.problem == "p3"
                and phase_after_policy == "frontier_improvement"
                and bool(focus.is_feasible)
            )
            if should_circuit_break:
                cmds = []
                decision = {
                    "batch_id": batch_id,
                    "commands": [],
                    "model_route": "governor_policy/circuit_break",
                    "recipe_mode": "policy_circuit_break",
                    "created_at": _utc_now_iso(),
                }
            else:
                if effective_action == "global_restart":
                    restart_parent_a, restart_parent_b, restart_parent_source = (
                        _resolve_global_restart_parent_pair(
                            run_dir=args.run_dir,
                            batch_id=batch_id,
                            focus=focus,
                            partner=partner,
                            candidates=candidates,
                            bootstrap_parent_a=(
                                run_surgery_parent_a
                                if run_surgery_parent_a is not None
                                else args.bootstrap_parent_a
                            ),
                            bootstrap_parent_b=(
                                run_surgery_parent_b
                                if run_surgery_parent_b is not None
                                else args.bootstrap_parent_b
                            ),
                            run_seed=run_seed,
                        )
                    )
                    restart_t_min, restart_t_max, restart_t_step = (
                        _restart_blend_schedule(
                            run_seed=run_seed,
                            batch_id=batch_id,
                            t_min=float(args.bootstrap_t_min),
                            t_max=float(args.bootstrap_t_max),
                            t_step=float(args.bootstrap_t_step),
                        )
                    )
                    if str(run_surgery_event.get("action")) == "rebootstrap_pair":
                        restart_t_min = max(0.0, float(restart_t_min) - 0.05)
                        restart_t_max = min(1.0, float(restart_t_max) + 0.05)
                        restart_t_step = max(0.002, float(restart_t_step) * 0.75)
                        restart_parent_source = (
                            f"{restart_parent_source}+run_surgery_rebootstrap"
                        )
                    restart_route = "governor_llm/global_restart"
                    restart_cmd = _build_blend_cmd(
                        proposal_script=proposal_script,
                        problem=profile.problem,
                        db=args.db,
                        experiment_id=exp_id,
                        run_dir=args.run_dir,
                        batch_id=batch_id,
                        seed_base=seed_base,
                        parent_a=restart_parent_a,
                        parent_b=restart_parent_b,
                        t_min=restart_t_min,
                        t_max=restart_t_max,
                        t_step=restart_t_step,
                        model_route=restart_route,
                    )
                    cmds = [restart_cmd]
                    decision = {
                        "batch_id": batch_id,
                        "commands": [_cmd_str(restart_cmd)],
                        "model_route": restart_route,
                        "recipe_mode": "llm_global_restart",
                        "restart_parent_source": restart_parent_source,
                        "restart_schedule": {
                            "t_min": restart_t_min,
                            "t_max": restart_t_max,
                            "t_step": restart_t_step,
                        },
                        "created_at": _utc_now_iso(),
                    }
                else:
                    if effective_action == "jump":
                        jump_cmds, jump_diag = _build_jump_recipe_cmds(
                            proposal_script=proposal_script,
                            problem=profile.problem,
                            db=args.db,
                            conn=conn,
                            jump_delta_cap=profile.mutation_budget.jump_delta_cap,
                            experiment_id=exp_id,
                            run_dir=args.run_dir,
                            batch_id=batch_id,
                            seed_base=seed_base,
                            focus=focus,
                            partner=partner_for_recipe,
                            route_prefix=str(route_prefix or "governor_llm"),
                        )
                        cmds = jump_cmds
                        decision = {
                            "batch_id": batch_id,
                            "focus": {
                                "design_hash": focus.design_hash,
                                "candidate_id": focus.candidate_id,
                                "feasibility": focus.feasibility,
                                "aspect": focus.aspect,
                                "lgradb": focus.lgradb,
                            },
                            "partner": None
                            if partner_for_recipe is None
                            else {
                                "design_hash": partner_for_recipe.design_hash,
                                "candidate_id": partner_for_recipe.candidate_id,
                                "aspect": partner_for_recipe.aspect,
                                "lgradb": partner_for_recipe.lgradb,
                            },
                            "commands": [_cmd_str(c) for c in cmds],
                            "model_route": str(route_prefix or "governor_llm/jump"),
                            "recipe_mode": "llm_jump",
                            "jump_policy": jump_diag,
                            "created_at": _utc_now_iso(),
                        }
                    elif args.adaptive:
                        assert parent_group_meta is not None
                        cmds, decision = _select_adaptive_recipe(
                            proposal_script=proposal_script,
                            problem=profile.problem,
                            db=args.db,
                            conn=conn,
                            experiment_id=exp_id,
                            run_dir=args.run_dir,
                            batch_id=batch_id,
                            seed_base=seed_base,
                            focus=focus,
                            partner=partner_for_recipe,
                            history_candidates=candidates,
                            parent_group=str(parent_group_meta["group"]),
                            route_prefix=route_prefix,
                            target_constraint_override=target_constraint_override,
                        )
                    else:
                        if use_frontier_recipe:
                            cmds, decision = _select_frontier_recipe(
                                proposal_script=proposal_script,
                                problem=profile.problem,
                                db=args.db,
                                conn=conn,
                                experiment_id=exp_id,
                                run_dir=args.run_dir,
                                batch_id=batch_id,
                                run_seed=run_seed,
                                seed_base=seed_base,
                                focus=focus,
                                partner=partner_for_recipe,
                                profile=profile,
                                hv_value=hv_value,
                                record_hv=float(args.record_hv),
                                route_prefix=str(
                                    route_prefix
                                    if route_prefix is not None
                                    else "governor_frontier_recipe"
                                ),
                            )
                        else:
                            if route_prefix is None:
                                cmds, decision = _select_static_recipe(
                                    proposal_script=proposal_script,
                                    problem=profile.problem,
                                    db=args.db,
                                    conn=conn,
                                    experiment_id=exp_id,
                                    run_dir=args.run_dir,
                                    batch_id=batch_id,
                                    seed_base=seed_base,
                                    focus=focus,
                                    partner=partner_for_recipe,
                                    target_constraint_override=target_constraint_override,
                                )
                            else:
                                cmds, decision = _select_static_recipe(
                                    proposal_script=proposal_script,
                                    problem=profile.problem,
                                    db=args.db,
                                    conn=conn,
                                    experiment_id=exp_id,
                                    run_dir=args.run_dir,
                                    batch_id=batch_id,
                                    seed_base=seed_base,
                                    focus=focus,
                                    partner=partner_for_recipe,
                                    route_prefix=route_prefix,
                                    target_constraint_override=target_constraint_override,
                                )

            llm_mutation_diagnostics: dict | None = None
            allow_llm_mutations = effective_action in {"repair", "bridge", "jump"}
            if (
                llm_validated_decision is not None
                and llm_validated_decision.decision.mutations
                and route_prefix is not None
                and allow_llm_mutations
            ):
                mutation_payload = [
                    {
                        "parameter_group": mutation.parameter_group,
                        "normalized_delta": mutation.normalized_delta,
                    }
                    for mutation in llm_validated_decision.decision.mutations
                ]
                mutation_cmds, llm_mutation_diagnostics = _build_llm_mutation_cmds(
                    proposal_script=proposal_script,
                    problem=profile.problem,
                    db=args.db,
                    conn=conn,
                    experiment_id=exp_id,
                    run_dir=args.run_dir,
                    batch_id=batch_id,
                    seed_base=seed_base,
                    focus=focus,
                    llm_mutations=mutation_payload,
                    route_prefix=route_prefix,
                )
                remaining_capacity = max(
                    0,
                    int(profile.mutation_budget.max_candidates_per_cycle) - len(cmds),
                )
                if remaining_capacity > 0 and mutation_cmds:
                    cmds.extend(mutation_cmds[:remaining_capacity])
            elif (
                llm_validated_decision is not None
                and llm_validated_decision.decision.mutations
                and not allow_llm_mutations
                and llm_fallback_reason is None
            ):
                llm_fallback_reason = "policy_override:mutations_ignored_for_restart"

            predicted_target_metric: str | None = (
                str(target_constraint_override)
                if target_constraint_override is not None
                else None
            )
            if predicted_target_metric is None:
                focus_payload = decision.get("focus")
                if isinstance(focus_payload, Mapping):
                    worst_constraint = focus_payload.get("worst_constraint")
                    if isinstance(worst_constraint, str) and worst_constraint.strip():
                        predicted_target_metric = str(worst_constraint).strip().lower()
            if predicted_target_metric is None:
                predicted_target_metric = "unknown"

            cap_applied = False
            max_candidates = int(profile.mutation_budget.max_candidates_per_cycle)
            if len(cmds) > max_candidates:
                cap_applied = True
                cmds = cmds[:max_candidates]

            diversity_floor = max(
                1,
                min(
                    int(profile.autonomy_policy.diversity_floor_min_candidates),
                    max_candidates,
                ),
            )
            diversity_floor_triggered = False
            diversity_forced_action: str | None = None
            expected_candidate_count = _expected_candidate_volume(cmds)
            if cmds and expected_candidate_count < diversity_floor:
                diversity_floor_triggered = True
                for forced_action in _diversity_escalation_order(
                    profile=profile,
                    current_action=effective_action,
                    partner_available=partner is not None,
                ):
                    try:
                        forced_cmds, forced_decision, applied_action = (
                            _build_forced_action_plan(
                                action=forced_action,
                                proposal_script=proposal_script,
                                problem=profile.problem,
                                db=args.db,
                                conn=conn,
                                profile=profile,
                                experiment_id=exp_id,
                                run_dir=args.run_dir,
                                batch_id=batch_id,
                                run_seed=run_seed,
                                seed_base=seed_base,
                                focus=focus,
                                partner=partner,
                                candidates=candidates,
                                target_constraint_override=target_constraint_override,
                                bootstrap_parent_a=args.bootstrap_parent_a,
                                bootstrap_parent_b=args.bootstrap_parent_b,
                                bootstrap_t_min=float(args.bootstrap_t_min),
                                bootstrap_t_max=float(args.bootstrap_t_max),
                                bootstrap_t_step=float(args.bootstrap_t_step),
                                route_prefix="governor_policy/diversity_floor",
                            )
                        )
                    except ValueError:
                        continue
                    cmds = forced_cmds
                    decision = forced_decision
                    effective_action = applied_action
                    diversity_forced_action = applied_action
                    expected_candidate_count = _expected_candidate_volume(cmds)
                    if (
                        expected_candidate_count >= diversity_floor
                        or applied_action == "global_restart"
                    ):
                        break
                if len(cmds) > max_candidates:
                    cap_applied = True
                    cmds = cmds[:max_candidates]
                    expected_candidate_count = _expected_candidate_volume(cmds)

            anti_repeat_triggered = False
            anti_repeat_repeat_count = 0
            anti_repeat_forced_action: str | None = None
            knob_signature = _command_knob_signature(cmds) if cmds else ""
            action_for_repeat = (
                str(effective_action)
                if effective_action is not None
                else ("adaptive_recipe" if bool(args.adaptive) else "static_recipe")
            )
            if cmds:
                anti_repeat_repeat_count = _consecutive_no_progress_action_repeats(
                    conn,
                    experiment_id=exp_id,
                    action=action_for_repeat,
                    target_constraint=str(predicted_target_metric),
                    knob_signature=knob_signature,
                )
                if anti_repeat_repeat_count >= int(
                    profile.autonomy_policy.anti_repeat_no_progress_cycles
                ):
                    anti_repeat_triggered = True
                    forced_action = _distinct_action_for_anti_repeat(
                        profile=profile,
                        current_action=effective_action,
                        partner_available=partner is not None,
                    )
                    try:
                        forced_cmds, forced_decision, applied_action = (
                            _build_forced_action_plan(
                                action=forced_action,
                                proposal_script=proposal_script,
                                problem=profile.problem,
                                db=args.db,
                                conn=conn,
                                profile=profile,
                                experiment_id=exp_id,
                                run_dir=args.run_dir,
                                batch_id=batch_id,
                                seed_base=seed_base,
                                focus=focus,
                                partner=partner,
                                candidates=candidates,
                                target_constraint_override=target_constraint_override,
                                bootstrap_parent_a=args.bootstrap_parent_a,
                                bootstrap_parent_b=args.bootstrap_parent_b,
                                bootstrap_t_min=float(args.bootstrap_t_min),
                                bootstrap_t_max=float(args.bootstrap_t_max),
                                bootstrap_t_step=float(args.bootstrap_t_step),
                                route_prefix="governor_policy/anti_repeat",
                            )
                        )
                        cmds = forced_cmds
                        decision = forced_decision
                        effective_action = applied_action
                        anti_repeat_forced_action = applied_action
                        if len(cmds) > max_candidates:
                            cap_applied = True
                            cmds = cmds[:max_candidates]
                        expected_candidate_count = _expected_candidate_volume(cmds)
                        knob_signature = _command_knob_signature(cmds) if cmds else ""
                    except ValueError:
                        anti_repeat_forced_action = None

            if "commands" in decision:
                decision["commands"] = [_cmd_str(cmd) for cmd in cmds]

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
            decision["problem"] = profile.problem
            decision["phase_policy"] = {
                "phase_before": current_phase,
                "phase_after": phase_after_policy,
                "accepted_feasible_last20": accepted_feasible_last20,
                "accepted_feasible_last10": accepted_feasible_last10,
                "dominant_violation_rate_last20": dominant_violation_rate_last20,
            }
            decision["restart_policy"] = {
                "selected": policy_restart_plan,
                "invalid_basin_escape_due": bool(invalid_basin_escape_due),
            }
            decision["run_surgery"] = {
                "action": run_surgery_event.get("action", "none"),
                "reason": run_surgery_event.get("reason"),
                "parent_pair": run_surgery_event.get("parent_pair"),
                "backlog_pruned_count": int(
                    run_surgery_event.get("backlog_pruned_count", 0)
                ),
                "operator_shift_lock_cycles": int(
                    run_surgery_event.get("operator_shift_lock_cycles", 0)
                ),
                "autoscale": run_surgery_event.get("autoscale"),
            }
            decision["llm"] = {
                "enabled": bool(args.llm_enabled),
                "model": str(args.llm_model),
                "session_id": args.llm_session_id,
                "input_source": llm_input_source,
                "selected_action": llm_selected_action,
                "effective_action": effective_action,
                "selected_constraint": llm_selected_constraint,
                "fallback_reason": llm_fallback_reason,
                "decision_file": (
                    str(llm_decision_file) if llm_decision_file is not None else None
                ),
                "decision_command": llm_decision_command,
            }
            if llm_validated_decision is not None:
                decision["llm"]["expected_effect"] = (
                    llm_validated_decision.decision.expected_effect
                )
                decision["llm"]["restart_plan"] = (
                    llm_validated_decision.decision.restart_plan
                )
            if llm_mutation_diagnostics is not None:
                decision["llm"]["mutation_diagnostics"] = llm_mutation_diagnostics
            if llm_observation is not None:
                decision["llm_observation"] = llm_observation
            if llm_intent_payload is not None:
                decision["llm_intent"] = llm_intent_payload
            decision["proposal_script"] = proposal_script
            decision["run_seed"] = run_seed
            decision["seed_base"] = int(seed_base)
            decision["recipe_synthesis"] = {
                "interval": int(recipe_synthesis_interval),
                "rules": recipe_rules,
            }
            decision["command_cap"] = {
                "max_candidates_per_cycle": max_candidates,
                "applied": cap_applied,
                "final_command_count": len(cmds),
                "expected_candidate_count": int(expected_candidate_count),
            }
            decision["autonomy_gates"] = {
                "autonomous_enabled": bool(args.autonomous),
                "autonomous_strategy_hint": autonomous_strategy,
                "autonomous_forced_action": autonomous_forced_action,
                "evals_since_feasibility_improve": int(
                    autonomy_progress.evals_since_feasibility_improve
                ),
                "evals_since_objective_improve": int(
                    autonomy_progress.evals_since_objective_improve
                ),
                "diversity_floor_min_candidates": int(diversity_floor),
                "diversity_floor_triggered": bool(diversity_floor_triggered),
                "diversity_forced_action": diversity_forced_action,
                "anti_repeat_threshold": int(
                    profile.autonomy_policy.anti_repeat_no_progress_cycles
                ),
                "anti_repeat_repeat_count": int(anti_repeat_repeat_count),
                "anti_repeat_triggered": bool(anti_repeat_triggered),
                "anti_repeat_forced_action": anti_repeat_forced_action,
                "zero_yield_streak": int(zero_yield_streak),
                "zero_yield_forced_action": zero_yield_forced_action,
                "knob_signature": knob_signature,
                "invalid_parent_failure_streak": int(invalid_parent_failure_streak),
                "invalid_basin_escape_due": bool(invalid_basin_escape_due),
                "run_surgery_action": run_surgery_event.get("action", "none"),
                "run_surgery_forced_action": run_surgery_forced_action,
                "run_surgery_operator_shift_lock_remaining": int(
                    run_surgery_state.operator_shift_lock_remaining
                ),
                "current_workers": int(current_workers),
                "target_queue": int(target_queue),
            }
            decision["parent_sanitization"] = {
                "focus_source": str(parent_sanitization.get("focus_source", "unknown")),
                "partner_source": str(
                    parent_sanitization.get("partner_source", "unknown")
                ),
                "invalid_parent_detected": bool(
                    parent_sanitization.get("invalid_parent_detected", False)
                ),
                "valid_parent_pool_size": int(
                    parent_sanitization.get("valid_parent_pool_size", 0)
                ),
            }
            model_route = str(decision.get("model_route", "unknown"))
            if bool(args.autonomous):
                action_signature = _action_signature(
                    action=effective_action,
                    model_route=model_route,
                    parent_pair_hash=run_surgery_parent_pair_hash,
                )
                run_surgery_state.previous_action_signature = (
                    run_surgery_state.last_action_signature
                )
                run_surgery_state.last_action_signature = action_signature
            _record_cycle_manifest(
                manifest=resume_manifest,
                path=manifest_path,
                batch_id=batch_id,
                seed_base=seed_base,
                cmds=cmds,
                model_route=model_route,
            )
            _update_manifest_phase(
                manifest=resume_manifest,
                path=manifest_path,
                phase=phase_after_policy,
            )
            reward_event = _compute_model_router_reward_event(
                history_candidates=candidates,
                model_route=model_route,
                window_size=int(_ROUTER_REWARD_WINDOW),
            )
            decision["model_router_reward"] = reward_event
            predicted_direction = _constraint_direction(
                profile,
                constraint=predicted_target_metric,
            )
            predicted_expected_effect = (
                llm_validated_decision.decision.expected_effect
                if llm_validated_decision is not None
                else None
            )

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
                    problem=profile.problem,
                    event=reward_event,
                )
                _log_scratchpad_event(
                    conn,
                    experiment_id=exp_id,
                    cycle=int(batch_id),
                    step=0,
                    planner_intent=llm_intent_payload or {},
                    aso_action=(
                        effective_action
                        if effective_action is not None
                        else (
                            "adaptive_recipe"
                            if bool(args.adaptive)
                            else "static_recipe"
                        )
                    ),
                    intent_agreement=(
                        "accepted"
                        if llm_selected_action is not None
                        and llm_fallback_reason is None
                        else ("fallback" if bool(args.llm_enabled) else "disabled")
                    ),
                    override_reason=llm_fallback_reason,
                    diagnostics={
                        "model_route": model_route,
                        "problem": profile.problem,
                        "action": (
                            effective_action
                            if effective_action is not None
                            else (
                                "adaptive_recipe"
                                if bool(args.adaptive)
                                else "static_recipe"
                            )
                        ),
                        "predicted_target_metric": predicted_target_metric,
                        "predicted_direction": predicted_direction,
                        "predicted_expected_effect": predicted_expected_effect,
                        "knob_signature": knob_signature,
                        "batch_id": int(batch_id),
                        "command_count": len(cmds),
                        "expected_candidate_count": int(expected_candidate_count),
                        "diversity_floor_min_candidates": int(diversity_floor),
                        "diversity_floor_triggered": bool(diversity_floor_triggered),
                        "diversity_forced_action": diversity_forced_action,
                        "anti_repeat_repeat_count": int(anti_repeat_repeat_count),
                        "anti_repeat_triggered": bool(anti_repeat_triggered),
                        "anti_repeat_forced_action": anti_repeat_forced_action,
                        "invalid_parent_failure_streak": int(
                            invalid_parent_failure_streak
                        ),
                        "invalid_basin_escape_due": bool(invalid_basin_escape_due),
                        "parent_sanitization": dict(parent_sanitization),
                        "autonomous_enabled": bool(args.autonomous),
                        "autonomous_strategy_hint": autonomous_strategy,
                        "autonomous_forced_action": autonomous_forced_action,
                        "evals_since_feasibility_improve": int(
                            autonomy_progress.evals_since_feasibility_improve
                        ),
                        "evals_since_objective_improve": int(
                            autonomy_progress.evals_since_objective_improve
                        ),
                        "phase_after_policy": phase_after_policy,
                        "restart_selected": policy_restart_plan,
                        "run_budget_remaining": run_budget_remaining,
                        "run_surgery": dict(decision.get("run_surgery", {})),
                    },
                    outcome={
                        "pending_before": int(pending_n),
                        "target_queue": int(target_queue),
                        "hv_at_decision": float(hv_value),
                        "record_hv": float(args.record_hv),
                        "snapshot_at_decision": dict(snapshot),
                        "current_workers": int(current_workers),
                    },
                )

            for cmd in cmds:
                print(_cmd_str(cmd))
            execution_summary: dict[str, int | bool] | None = None
            if args.execute and cmds:
                execution_summary = _run_cmds(cmds)
                decision["execution"] = {
                    "inserted": int(execution_summary.get("inserted", 0)),
                    "skipped": int(execution_summary.get("skipped", 0)),
                    "parsed": bool(execution_summary.get("parsed", False)),
                }
                _write_json(artifact_path, decision)
                inserted_count = int(execution_summary.get("inserted", 0))
                parsed_summary = bool(execution_summary.get("parsed", False))
                if parsed_summary and inserted_count <= 0:
                    zero_yield_streak += 1
                    if should_circuit_break:
                        force_action_next_cycle = None
                    elif effective_action == "global_restart":
                        force_action_next_cycle = "global_restart"
                    else:
                        force_action_next_cycle = _distinct_action_for_anti_repeat(
                            profile=profile,
                            current_action=effective_action,
                            partner_available=partner is not None,
                        )
                else:
                    zero_yield_streak = 0
                    if (
                        run_surgery_post_restart_forced_action is not None
                        and effective_action == "global_restart"
                    ):
                        force_action_next_cycle = run_surgery_post_restart_forced_action
                    else:
                        force_action_next_cycle = None
            proposal_cycles_emitted += 1

            if should_circuit_break:
                break

            if not args.loop:
                break
            time.sleep(float(args.sleep_sec))
    finally:
        if worker_pool:
            _stop_worker_pool(pool=worker_pool)
        conn.close()


if __name__ == "__main__":
    main()
