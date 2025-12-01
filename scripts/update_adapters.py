#!/usr/bin/env python3
"""Build LoRA bundle shims from preference logs so ``adapter.with_peft`` has data to consume."""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ai_scientist import adapter

_LOGGER = logging.getLogger(__name__)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_preference_pairs(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        _LOGGER.info("Preference log %s does not exist, nothing to process.", path)
        return []
    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as exc:
                _LOGGER.warning("Skipping malformed preference entry: %s", exc)
    return entries


def _group_by_tool_stage(
    entries: Iterable[dict[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        tool = entry.get("tool_name", "unknown")
        stage = entry.get("stage", "unknown")
        groups[(tool, stage)].append(entry)
    return groups


def _normalize_stage(stage: str) -> str:
    return stage.lower().strip()


def _write_dataset(entries: list[dict[str, Any]], path: Path) -> int:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            payload = {
                "tool_input_hash": entry.get("tool_input_hash"),
                "stage": entry.get("stage"),
                "status": entry.get("status"),
                "problem": entry.get("problem"),
                "seed": entry.get("seed"),
                "reproduction_command": entry.get("reproduction_command"),
                "metrics": entry.get("metrics"),
            }
            handle.write(json.dumps(payload, default=str) + "\n")
    return len(entries)


def _build_adapter_payload(
    tool: str,
    stage: str,
    dataset_path: Path,
    entries: list[dict[str, Any]],
    sample_limit: int,
) -> dict[str, Any]:
    statuses = sorted({entry.get("status") for entry in entries if entry.get("status")})
    sample_commands = [
        entry.get("reproduction_command")
        for entry in entries
        if entry.get("reproduction_command")
    ][:sample_limit]
    return {
        "tool": tool,
        "stage": stage,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "preference_pair_count": len(entries),
        "dataset_path": dataset_path.as_posix(),
        "statuses": statuses,
        "sample_commands": sample_commands,
    }


def _write_adapter_bundle(path: Path, payload: dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert AI Scientist preference logs into adapter metadata bundles."
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports"),
        help="Root directory containing adaptation/ and adapters/ artifacts.",
    )
    parser.add_argument(
        "--tool",
        type=str,
        help="Optional tool name filter (e.g., evaluate_p3).",
    )
    parser.add_argument(
        "--stage",
        type=str,
        help="Optional stage filter (e.g., screen, p3).",
    )
    parser.add_argument(
        "--sample-commands",
        type=int,
        default=3,
        help="How many reproduction commands to include in the adapter metadata.",
    )
    parser.add_argument(
        "--no-queue",
        action="store_true",
        help="Do not append adapter refresh entries to queue.jsonl.",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    preference_path = args.reports_dir / "adaptation" / "preference_pairs.jsonl"
    entries = _load_preference_pairs(preference_path)
    if not entries:
        _LOGGER.info("No preference entries found, exiting.")
        return

    grouped = _group_by_tool_stage(entries)
    if not grouped:
        _LOGGER.info("No grouped entries after filters, nothing to update.")
        return

    for (tool, stage), group in sorted(grouped.items()):
        if args.tool and tool != args.tool:
            continue
        if args.stage and _normalize_stage(stage) != _normalize_stage(args.stage):
            continue

        normalized_stage = _normalize_stage(stage)
        dataset_path = (
            args.reports_dir
            / "adapters"
            / "datasets"
            / tool
            / normalized_stage
            / "preference_dataset.jsonl"
        )
        count = _write_dataset(group, dataset_path)
        adapter_payload = _build_adapter_payload(
            tool=tool,
            stage=stage,
            dataset_path=dataset_path,
            entries=group,
            sample_limit=args.sample_commands,
        )
        adapter_path = (
            args.reports_dir
            / "adapters"
            / tool
            / normalized_stage
            / "adapter.safetensors"
        )
        _write_adapter_bundle(adapter_path, adapter_payload)

        if not args.no_queue:
            adapter.record_adapter_refresh(
                tool,
                stage,
                backend="update_adapters.py",
                adapter_path=adapter_path,
            )

        _LOGGER.info(
            "Updated adapter %s:%s (entries=%d, dataset=%s)",
            tool,
            stage,
            count,
            dataset_path,
        )


if __name__ == "__main__":
    main()
