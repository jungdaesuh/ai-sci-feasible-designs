"""PEFT/LoRA adapter integration for Wave 7 adaptation hooks (docs/WAVE_7_ADAPTATION.md).

Set ``AI_SCIENTIST_PEFT=1`` to activate adapter loading before tool calls and queue updates
in ``reports/adapters/{tool}/{stage}/adapter.safetensors`` / ``reports/adapters/queue.jsonl``."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

_LOGGER = logging.getLogger(__name__)
_ENV_VAR = "AI_SCIENTIST_PEFT"
_ADAPTERS_ROOT = Path("reports") / "adapters"
_QUEUE_PATH = _ADAPTERS_ROOT / "queue.jsonl"
_PERSIST_DIR_ENV = "AI_SCIENTIST_ADAPTER_PERSIST_DIR"


AdapterLoader = Callable[[Path, str, str], bool]
AdapterPersistHandler = Callable[[Path, str, str], bool]

_REGISTERED_LOADERS: list[tuple[str, AdapterLoader]] = []
_REGISTERED_PERSISTERS: list[tuple[str, AdapterPersistHandler]] = []


def register_adapter_loader(name: str, loader: AdapterLoader) -> None:
    """Register a callable that can load adapters for the runner (HF PEFT, ggml, etc.)."""

    _REGISTERED_LOADERS.append((name, loader))


def register_adapter_persist_handler(name: str, handler: AdapterPersistHandler) -> None:
    """Register a callable that persists adapters produced during a cycle run."""

    _REGISTERED_PERSISTERS.append((name, handler))


def _adapter_bundle_path(tool_name: str, stage: str) -> Path:
    normalized_stage = stage.lower().strip()
    return _ADAPTERS_ROOT / tool_name / normalized_stage / "adapter.safetensors"


def _ensure_adapter_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _queue_entry(
    tool_name: str, stage: str, adapter_path: Path, backend: str | None, status: str
) -> dict[str, Any]:
    return {
        "tool": tool_name,
        "stage": stage,
        "adapter_path": adapter_path.as_posix(),
        "backend": backend,
        "status": status,
        "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }


def _append_to_queue(entry: Mapping[str, Any]) -> None:
    _ensure_adapter_directory(_QUEUE_PATH)
    with _QUEUE_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry))
        handle.write("\n")


def record_adapter_refresh(
    tool_name: str,
    stage: str,
    *,
    backend: str | None = None,
    status: str = "refreshed",
    adapter_path: Path | None = None,
) -> None:
    """Expose queue logging so offline adapters can annotate backend refreshes."""

    bundle_path = adapter_path or _adapter_bundle_path(tool_name, stage)
    entry = _queue_entry(tool_name, stage, bundle_path, backend, status)
    _append_to_queue(entry)


def _try_load_adapter(adapter_path: Path, tool_name: str, stage: str) -> str | None:
    for backend_name, loader in _REGISTERED_LOADERS:
        if loader(adapter_path, tool_name, stage):
            return backend_name
    return None


def _try_persist_adapter(adapter_path: Path, tool_name: str, stage: str) -> str | None:
    for backend_name, handler in _REGISTERED_PERSISTERS:
        if handler(adapter_path, tool_name, stage):
            return backend_name
    return None


def _json_metadata_loader(adapter_path: Path, tool_name: str, stage: str) -> bool:
    """Load JSON-based adapter bundles for inspection."""

    try:
        raw = adapter_path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, OSError) as exc:  # pragma: no cover - upstream guard
        _LOGGER.debug(
            "JSON adapter loader missing %s:%s (%s)",
            tool_name,
            stage,
            exc,
        )
        return False

    if not raw:
        _LOGGER.debug("JSON adapter %s:%s is empty", tool_name, stage)
        return True

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        _LOGGER.warning(
            "JSON adapter %s:%s failed to parse; treating as raw bytes",
            tool_name,
            stage,
        )
        return True

    _LOGGER.info(
        "Loaded JSON adapter %s:%s summary=%s",
        tool_name,
        stage,
        {
            "entry_count": payload.get("preference_pair_count"),
            "dataset_path": payload.get("dataset_path"),
        },
    )
    return True


def _staged_adapter_persist(adapter_path: Path, tool_name: str, stage: str) -> bool:
    """Promote staged adapter bundles when a staging directory is configured."""

    persist_root = os.getenv(_PERSIST_DIR_ENV)
    if not persist_root:
        return False

    normalized_stage = stage.lower().strip()
    staged_path = (
        Path(persist_root) / tool_name / normalized_stage / "adapter.safetensors"
    )
    if not staged_path.exists():
        return False

    _ensure_adapter_directory(adapter_path)
    staged_path.replace(adapter_path)
    _LOGGER.info(
        "Persisted staged adapter %s:%s from %s",
        tool_name,
        stage,
        staged_path,
    )
    return True


class ProblemEvaluator(Protocol):
    """Lightweight protocol that mirrors the evaluator interface used in runner.py."""

    def __call__(
        self,
        boundary_params: Mapping[str, Any],
        *,
        stage: str,
        use_cache: bool = True,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class AdapterState:
    """Tracks which LoRA weights were loaded/applied so Wave 7 can replay the stack."""

    loaded: dict[str, str] = field(default_factory=dict)
    updates: list[str] = field(default_factory=list)

    def load_lora_weights(self, label: str, stage: str) -> None:
        """Record when a LoRA bundle is staged for the current tool/stage."""
        bundle_path = _adapter_bundle_path(label, stage)
        if not bundle_path.exists():
            self.loaded[f"{label}:{stage}"] = "missing"
            _LOGGER.debug(
                "Adapter bundle not found for %s:%s at %s",
                label,
                stage,
                bundle_path,
            )
            return

        backend_name = _try_load_adapter(bundle_path, label, stage)
        status = (
            f"{backend_name}:{bundle_path.as_posix()}"
            if backend_name
            else f"ready:{bundle_path.as_posix()}"
        )
        self.loaded[f"{label}:{stage}"] = status
        _LOGGER.debug(
            "AdapterState.load_lora_weights label=%s stage=%s status=%s",
            label,
            stage,
            status,
        )

    def push_updates(self, label: str, stage: str) -> None:
        """Log when downstream adapters propagated updates."""
        bundle_path = _adapter_bundle_path(label, stage)
        persisted_backend = _try_persist_adapter(bundle_path, label, stage)
        status = f"persisted:{persisted_backend}" if persisted_backend else "queued"

        if not persisted_backend:
            entry = _queue_entry(label, stage, bundle_path, None, status)
            _append_to_queue(entry)

        self.updates.append(f"{label}:{stage}:{status}")
        _LOGGER.debug(
            "AdapterState.push_updates label=%s stage=%s status=%s path=%s",
            label,
            stage,
            status,
            bundle_path,
        )


adapter_state = AdapterState()


def is_peft_enabled() -> bool:
    """Return True when the Wave 7 PEFT toggle is set in the environment."""

    return os.getenv(_ENV_VAR, "0").lower() in {"1", "true", "yes"}


def prepare_peft_hook(tool_name: str, stage: str) -> None:
    """Hook point that loads LoRA weights before a tool call."""

    if not is_peft_enabled():
        return
    _LOGGER.info("Preparing PEFT hook tool=%s stage=%s", tool_name, stage)
    adapter_state.load_lora_weights(tool_name, stage)


def apply_lora_updates(tool_name: str, stage: str) -> None:
    """Hook point that pushes LoRA updates after a tool call."""

    if not is_peft_enabled():
        return
    _LOGGER.info("Applying LoRA updates tool=%s stage=%s", tool_name, stage)
    adapter_state.push_updates(tool_name, stage)


def with_peft(evaluate_fn: ProblemEvaluator, tool_name: str) -> ProblemEvaluator:
    """Return either the evaluator unchanged or a PEFT-wrapped callable."""

    if not is_peft_enabled():
        return evaluate_fn

    def _wrapped(
        boundary_params: Mapping[str, Any],
        *,
        stage: str,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        prepare_peft_hook(tool_name, stage)
        result = evaluate_fn(boundary_params, stage=stage, use_cache=use_cache)
        apply_lora_updates(tool_name, stage)
        return result

    return _wrapped


register_adapter_loader("json_metadata", _json_metadata_loader)
register_adapter_persist_handler("staged_adapter", _staged_adapter_persist)
