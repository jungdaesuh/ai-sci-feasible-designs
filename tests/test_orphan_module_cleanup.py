import ast
import importlib.util
import warnings
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
ORPHAN_MODULE_PATHS = (
    REPO_ROOT / "ai_scientist" / "model_endpoint.py",
    REPO_ROOT / "ai_scientist" / "guards.py",
    REPO_ROOT / "ai_scientist" / "optim" / "validation.py",
)


def _references_legacy_module(source: str, legacy_import: str) -> bool:
    direct_hit = legacy_import in source
    if direct_hit:
        return True
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source)
    except SyntaxError:
        return False

    parent_module, _, child_name = legacy_import.rpartition(".")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported = alias.name
                if imported == legacy_import or imported.startswith(
                    f"{legacy_import}."
                ):
                    return True
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == legacy_import or module.startswith(f"{legacy_import}."):
                return True
            if parent_module and module == parent_module:
                for alias in node.names:
                    if alias.name == child_name:
                        return True
    return False


@pytest.mark.parametrize("module_path", ORPHAN_MODULE_PATHS)
def test_orphan_module_files_are_removed(module_path: Path) -> None:
    """PR-1 contract: orphan module files are removed from source tree."""
    assert not module_path.exists()


@pytest.mark.parametrize(
    "module_name",
    (
        "ai_scientist.config",
        "ai_scientist.forward_model",
    ),
)
def test_core_modules_still_resolve(module_name: str) -> None:
    """Cleanup must not break core import paths."""
    assert importlib.util.find_spec(module_name) is not None


@pytest.mark.parametrize(
    "legacy_import",
    (
        "ai_scientist.model_endpoint",
        "ai_scientist.guards",
        "ai_scientist.optim.validation",
    ),
)
def test_runtime_code_no_longer_references_orphan_modules(legacy_import: str) -> None:
    """No runtime modules should keep imports pointing to removed files."""
    for py_file in REPO_ROOT.rglob("*.py"):
        if "tests" in py_file.parts:
            continue
        if "docs" in py_file.parts:
            continue
        source = py_file.read_text(encoding="utf-8", errors="ignore")
        assert not _references_legacy_module(source, legacy_import)
