from __future__ import annotations

from pathlib import Path

from scripts.run_codex_native_gateway import _resolve_model_config_path


def test_resolve_model_config_path_defaults_to_none() -> None:
    resolved = _resolve_model_config_path(None)
    assert resolved is None


def test_resolve_model_config_path_preserves_explicit_path() -> None:
    explicit = Path("configs/model.custom.yaml")
    assert _resolve_model_config_path(str(explicit)) == explicit
