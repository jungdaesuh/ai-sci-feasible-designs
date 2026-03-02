from __future__ import annotations

import py_compile
from pathlib import Path


def test_py_compile_validation_gate_for_governor_contract_files() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    files = [
        repo_root / "ai_scientist" / "llm_controller.py",
        repo_root / "ai_scientist" / "problem_profiles.py",
        repo_root / "scripts" / "governor.py",
        repo_root / "tests" / "test_llm_controller.py",
        repo_root / "tests" / "test_p3_governor_contract.py",
    ]
    for file_path in files:
        py_compile.compile(str(file_path), doraise=True)
