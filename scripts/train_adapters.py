#!/usr/bin/env python3
"""Preference-driven PEFT adapter trainer stub for AI Scientist (RPF loop).

This script:
1) loads preference data from the world-model SQLite DB + adaptation logs,
2) builds lightweight SFT/DPO JSONL datasets per tool/stage,
3) emits a versioned adapter bundle (metadata-only) to reports/adapters/,
4) trims old bundles and appends queue.jsonl entries for observability.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ai_scientist import adapter

LOGGER = logging.getLogger(__name__)
DEFAULT_REPORTS = Path("reports")


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _load_preference_pairs(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                LOGGER.warning("Skipping malformed preference row in %s", path)
    return rows


def _load_statements(db_path: Path) -> list[Mapping[str, Any]]:
    if not db_path.exists():
        return []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT stage, status, text, tool_name, repro_cmd, seed, cycle "
            "FROM statements"
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def _group_by_tool_stage(
    entries: Iterable[Mapping[str, Any]], fallback_tool: str | None, fallback_stage: str | None
) -> dict[tuple[str, str], list[Mapping[str, Any]]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for entry in entries:
        tool = str(entry.get("tool_name") or fallback_tool or "generic")
        stage = str(entry.get("stage") or fallback_stage or "unknown")
        grouped[(tool, stage)].append(entry)
    return grouped


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> int:
    _ensure_dir(path)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def _build_sft_rows(entries: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for entry in entries:
        prompt = entry.get("repro_cmd") or entry.get("reproduction_command") or ""
        text = entry.get("text") or entry.get("quote") or entry.get("status") or ""
        rows.append(
            {
                "prompt": str(prompt),
                "response": str(text),
                "stage": entry.get("stage"),
                "status": entry.get("status"),
                "problem": entry.get("problem"),
            }
        )
    return rows


def _build_dpo_pairs(entries: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    positives = [e for e in entries if str(e.get("status", "")).upper() == "SUPPORTED"]
    negatives = [e for e in entries if str(e.get("status", "")).upper() == "REFUTED"]
    if not positives or not negatives:
        return []
    pairs: list[Mapping[str, Any]] = []
    limit = min(len(positives), len(negatives))
    for idx in range(limit):
        pos = positives[idx]
        neg = negatives[idx]
        prompt = pos.get("repro_cmd") or pos.get("reproduction_command") or ""
        pairs.append(
            {
                "prompt": str(prompt),
                "chosen": str(pos.get("text") or pos.get("quote") or ""),
                "rejected": str(neg.get("text") or neg.get("quote") or ""),
                "stage": pos.get("stage") or neg.get("stage"),
            }
        )
    return pairs


def _save_adapter_bundle(
    tool: str,
    stage: str,
    *,
    out_root: Path,
    sft_path: Path | None,
    dpo_path: Path | None,
    sft_rows: int,
    dpo_rows: int,
    keep: int,
) -> Path:
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    normalized_stage = stage.lower().strip()
    bundle_dir = out_root / tool / normalized_stage
    adapter_path = bundle_dir / "adapter.safetensors"
    versioned_path = bundle_dir / f"adapter_{version}.safetensors"
    _ensure_dir(adapter_path)

    metadata = {
        "version": version,
        "tool": tool,
        "stage": normalized_stage,
        "sft_examples": sft_rows,
        "dpo_pairs": dpo_rows,
        "sft_dataset": sft_path.as_posix() if sft_path else None,
        "dpo_dataset": dpo_path.as_posix() if dpo_path else None,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    versioned_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    adapter_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    metadata_path = bundle_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    versioned_meta = bundle_dir / f"metadata_{version}.json"
    versioned_meta.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    candidates = sorted(bundle_dir.glob("adapter_*.safetensors"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in candidates[keep:]:
        try:
            old.unlink()
        except OSError:
            LOGGER.debug("Unable to remove old adapter bundle %s", old)

    meta_candidates = sorted(bundle_dir.glob("metadata_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in meta_candidates[keep:]:
        try:
            old.unlink()
        except OSError:
            LOGGER.debug("Unable to remove old adapter metadata %s", old)

    adapter.record_adapter_refresh(
        tool,
        stage,
        backend="train_adapters.py",
        status="trained",
        adapter_path=adapter_path,
        version=version,
    )
    return adapter_path


def train_from_preferences(
    *,
    db_path: Path,
    reports_dir: Path,
    out_dir: Path,
    tool_filter: str | None,
    stage_filter: str | None,
    keep: int,
) -> list[Path]:
    preference_pairs = _load_preference_pairs(reports_dir / "adaptation" / "preference_pairs.jsonl")
    statements = _load_statements(db_path)
    combined = preference_pairs + statements
    if not combined:
        LOGGER.info("No preference data found at %s or %s", db_path, reports_dir)
        return []

    grouped = _group_by_tool_stage(combined, tool_filter, stage_filter)
    saved: list[Path] = []
    for (tool, stage), entries in sorted(grouped.items()):
        if tool_filter and tool != tool_filter:
            continue
        if stage_filter and stage.lower().strip() != stage_filter.lower().strip():
            continue
        sft_rows = _build_sft_rows(entries)
        dpo_pairs = _build_dpo_pairs(entries)

        dataset_root = out_dir / "datasets" / tool / stage.lower().strip()
        sft_path = dataset_root / "sft.jsonl"
        dpo_path = dataset_root / "dpo.jsonl"
        sft_count = _write_jsonl(sft_path, sft_rows)
        dpo_count = _write_jsonl(dpo_path, dpo_pairs) if dpo_pairs else 0

        bundle_path = _save_adapter_bundle(
            tool,
            stage,
            out_root=out_dir,
            sft_path=sft_path,
            dpo_path=dpo_path if dpo_pairs else None,
            sft_rows=sft_count,
            dpo_rows=dpo_count,
            keep=keep,
        )
        LOGGER.info(
            "Trained adapter %s:%s (sft=%d dpo=%d) -> %s",
            tool,
            stage,
            sft_count,
            dpo_count,
            bundle_path,
        )
        saved.append(bundle_path)
    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PEFT adapters from preference data.")
    parser.add_argument("--db", type=Path, default=DEFAULT_REPORTS / "ai_scientist.sqlite")
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORTS / "adapters")
    parser.add_argument("--tool", type=str, help="Optional tool filter (e.g., evaluate_p3).")
    parser.add_argument("--stage", type=str, help="Optional stage filter (e.g., screen).")
    parser.add_argument("--keep", type=int, default=3, help="How many historical adapters to keep per tool/stage.")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    saved = train_from_preferences(
        db_path=args.db,
        reports_dir=args.reports_dir,
        out_dir=args.out,
        tool_filter=args.tool,
        stage_filter=args.stage,
        keep=max(1, args.keep),
    )
    if not saved:
        LOGGER.info("No adapters trained (empty preference data).")


if __name__ == "__main__":
    main()
