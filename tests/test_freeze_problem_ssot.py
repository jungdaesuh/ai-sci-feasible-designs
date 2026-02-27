# ruff: noqa: E402
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import freeze_problem_ssot as ssot


def _load_frozen_spec() -> dict:
    return json.loads(ssot._DEFAULT_SPEC_PATH.read_text(encoding="utf-8"))


def test_frozen_spec_matches_current_problem_contract() -> None:
    frozen = _load_frozen_spec()
    current = ssot._build_current_spec()
    assert frozen == current


def test_validate_spec_raises_on_mismatch(tmp_path: Path) -> None:
    frozen = _load_frozen_spec()
    frozen["problems"]["p1_geometrical"]["constraints"][0]["value"] = 99.0
    candidate = tmp_path / "problem_ssot.json"
    candidate.write_text(json.dumps(frozen), encoding="utf-8")

    with pytest.raises(ValueError, match="out of sync"):
        ssot._validate_spec(candidate)
