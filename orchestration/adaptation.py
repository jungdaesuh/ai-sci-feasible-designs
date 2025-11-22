"""Adaptation utilities for preference logging and PEFT artifacts.

These helpers are production-minded (atomic writes, explicit schemas) and keep
the runner decoupled from any specific trainer implementation. They follow the
artifact conventions described in AI_SCIENTIST_AUTONOMY_PLAN.md and
AI_SCIENTIST_RESEARCH_PRODUCTION_FIX.md Section 5.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


_DEFAULT_ENCODING = "utf-8"
_PREFERENCE_SCHEMA_VERSION = 1


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _serialize(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _write_json_atomic(target: Path, payload: Mapping[str, Any]) -> Path:
    _ensure_dir(target)
    temp = target.with_suffix(target.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding=_DEFAULT_ENCODING)
    temp.replace(target)
    return target


def _append_jsonl(target: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    _ensure_dir(target)
    with target.open("a", encoding=_DEFAULT_ENCODING) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return target


def write_p3_summary(*, base_dir: str | Path, cycle: int, summary: Any) -> Path:
    base = Path(base_dir)
    target = base / f"cycle_{cycle}_p3_summary.json"
    payload = {
        "cycle": cycle,
        "schema_version": _PREFERENCE_SCHEMA_VERSION,
        "summary": _serialize(summary),
        "artifact": "p3_summary",
    }
    return _write_json_atomic(target, payload)


def append_preference_pair(
    *, base_dir: str | Path, cycle: int, pair: Mapping[str, Any]
) -> Path:
    base = Path(base_dir)
    target = base / "preference_pairs.jsonl"
    row = {
        "cycle": cycle,
        "schema_version": _PREFERENCE_SCHEMA_VERSION,
        "artifact": "preference_pair",
        **pair,
    }
    return _append_jsonl(target, [row])


def append_trajectory_entry(
    *, base_dir: str | Path, cycle: int, entry: Mapping[str, Any]
) -> Path:
    base = Path(base_dir)
    target = base / "trajectories.jsonl"
    row = {
        "cycle": cycle,
        "schema_version": _PREFERENCE_SCHEMA_VERSION,
        "artifact": "trajectory_entry",
        **entry,
    }
    return _append_jsonl(target, [row])


def write_metrics_snapshot(
    *, base_dir: str | Path, cycle: int, payload: Mapping[str, Any]
) -> Path:
    base = Path(base_dir)
    target = base / f"cycle_{cycle}_metrics.json"
    snapshot = {
        "cycle": cycle,
        "schema_version": _PREFERENCE_SCHEMA_VERSION,
        "artifact": "metrics_snapshot",
        **payload,
    }
    return _write_json_atomic(target, snapshot)


def append_preference_record(
    *, base_dir: str | Path, record: Mapping[str, Any]
) -> Path:
    base = Path(base_dir)
    target = base / "preference_records.jsonl"
    row = {
        "schema_version": _PREFERENCE_SCHEMA_VERSION,
        "artifact": "preference_record",
        **record,
    }
    return _append_jsonl(target, [row])
