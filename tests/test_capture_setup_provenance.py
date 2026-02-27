# ruff: noqa: E402
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import capture_setup_provenance as capture


def _sample_provenance() -> capture.SetupProvenance:
    return capture.SetupProvenance(
        captured_at_utc="2026-02-26T00:00:00+00:00",
        hostname="host",
        platform="platform",
        machine="arm64",
        python_executable="/usr/bin/python3",
        python_version="Python 3.11.8",
        pip_version="pip 24.0",
        uv_version="uv 0.6.0",
        docker_version="Docker version 27.0.0",
        git_sha="abc123",
        git_branch="main",
        constellaration_sha="def456",
        vmecpp_sha="ghi789",
        vmec_mode="docker",
        vmec_mode_reason="auto-selected because docker CLI is available",
        env={key: "" for key in capture._ENV_KEYS},
        capture_command="python scripts/capture_setup_provenance.py",
    )


def test_select_vmec_mode_auto_prefers_docker() -> None:
    mode, reason = capture._select_vmec_mode("auto", docker_available=True)
    assert mode == "docker"
    assert "auto-selected" in reason


def test_select_vmec_mode_auto_falls_back_to_native() -> None:
    mode, reason = capture._select_vmec_mode("auto", docker_available=False)
    assert mode == "native"
    assert "auto-fallback" in reason


def test_write_artifacts_creates_expected_files(tmp_path: Path) -> None:
    provenance = _sample_provenance()
    written = capture._write_artifacts(tmp_path, provenance)
    assert {path.name for path in written} == {
        "versions.txt",
        "commands.md",
        "vmec_path.md",
        "provenance.json",
    }
    versions_text = (tmp_path / "versions.txt").read_text(encoding="utf-8")
    assert "git_sha=abc123" in versions_text
    assert "vmec_mode=docker" in versions_text
    assert "env_OMP_NUM_THREADS=" in versions_text

    commands_text = (tmp_path / "commands.md").read_text(encoding="utf-8")
    assert "python scripts/capture_setup_provenance.py" in commands_text
    assert "docker --version" in commands_text

    vmec_text = (tmp_path / "vmec_path.md").read_text(encoding="utf-8")
    assert "selected_mode" in vmec_text
    assert "docker build -t ai-scientist ." in vmec_text
