from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts import p1_governor, p2_governor


def test_p1_wrapper_sets_problem_default(monkeypatch) -> None:
    captured: list[list[str]] = []

    def _fake_main(argv: list[str] | None = None) -> None:
        captured.append([] if argv is None else list(argv))

    monkeypatch.setattr(p1_governor, "governor_main", _fake_main)
    monkeypatch.setattr(sys, "argv", ["p1_governor.py", "--db", "x.sqlite"])
    p1_governor.main()
    assert captured
    assert captured[0][:2] == ["--problem", "p1"]


def test_p2_wrapper_preserves_explicit_problem(monkeypatch) -> None:
    captured: list[list[str]] = []

    def _fake_main(argv: list[str] | None = None) -> None:
        captured.append([] if argv is None else list(argv))

    monkeypatch.setattr(p2_governor, "governor_main", _fake_main)
    monkeypatch.setattr(
        sys, "argv", ["p2_governor.py", "--problem", "p3", "--db", "x.sqlite"]
    )
    p2_governor.main()
    assert captured
    assert captured[0][:2] == ["--problem", "p3"]


def test_wrappers_support_direct_script_help_execution() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    p1_result = subprocess.run(
        [sys.executable, "scripts/p1_governor.py", "--help"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert p1_result.returncode == 0
    assert "P3 proposal governor." in p1_result.stdout

    p2_result = subprocess.run(
        [sys.executable, "scripts/p2_governor.py", "--help"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert p2_result.returncode == 0
    assert "P3 proposal governor." in p2_result.stdout
