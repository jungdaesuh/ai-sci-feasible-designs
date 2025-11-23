from __future__ import annotations

import json
import time
from pathlib import Path

from ai_scientist import adapter


def test_adapter_loader_prefers_latest_version(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AI_SCIENTIST_ADAPTER_ROOT", str(tmp_path / "adapters"))
    monkeypatch.setenv("AI_SCIENTIST_PEFT", "1")
    adapter.adapter_state = adapter.AdapterState()

    tool = "evaluate_p3"
    stage = "screen"
    bundle_dir = tmp_path / "adapters" / tool / stage
    bundle_dir.mkdir(parents=True, exist_ok=True)

    old_version = "20240101T000000Z"
    new_version = "20250101T000000Z"

    old_path = bundle_dir / f"adapter_{old_version}.safetensors"
    old_path.write_text(json.dumps({"version": old_version}), encoding="utf-8")
    (bundle_dir / f"metadata_{old_version}.json").write_text(
        json.dumps({"version": old_version}), encoding="utf-8"
    )
    time.sleep(0.01)
    new_path = bundle_dir / f"adapter_{new_version}.safetensors"
    new_path.write_text(json.dumps({"version": new_version}), encoding="utf-8")
    (bundle_dir / f"metadata_{new_version}.json").write_text(
        json.dumps({"version": new_version}), encoding="utf-8"
    )

    adapter.prepare_peft_hook(tool, stage)

    status = adapter.adapter_state.loaded[f"{tool}:{stage}"]
    version = adapter.current_adapter_version(tool, stage)
    assert new_version in status
    assert version == new_version
