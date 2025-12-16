"""Environment gate for real physics backend.

This test is intentionally opt-in: it only runs when the caller explicitly
requests real-physics validation via environment variables.

Why this exists:
- Importing `vmecpp` can succeed even when the native extension cannot load.
- On macOS, broken dynamic linking often surfaces as `ImportError: dlopen(...):
  Library not loaded: @rpath/libtorch.dylib` (or similar).

Set one of:
- AI_SCIENTIST_PHYSICS_BACKEND=real
- AI_SCIENTIST_REQUIRE_REAL_BACKEND=1
"""

from __future__ import annotations

import os

import pytest


def _require_real_backend() -> bool:
    backend = os.environ.get("AI_SCIENTIST_PHYSICS_BACKEND", "auto").lower()
    require = os.environ.get("AI_SCIENTIST_REQUIRE_REAL_BACKEND", "").strip().lower()
    return backend == "real" or require in {"1", "true", "yes", "y", "on"}


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _require_real_backend(),
        reason="Set AI_SCIENTIST_PHYSICS_BACKEND=real to enforce real-backend gate",
    ),
]


def _require_sitepackages_constellaration() -> bool:
    value = os.environ.get("AI_SCIENTIST_REQUIRE_SITEPACKAGES_CONSTELLARATION", "")
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def test_real_backend_native_extension_loads() -> None:
    import vmecpp.cpp._vmecpp  # noqa: F401


def test_real_backend_is_available() -> None:
    from ai_scientist.backends.real import RealPhysicsBackend

    backend = RealPhysicsBackend()
    assert backend.is_available()


def test_constellaration_imports_from_site_packages_when_requested() -> None:
    """Guard against accidentally using a vendored/editable constellaration checkout.

    This is opt-in because local development may intentionally use editable installs.
    """
    if not _require_sitepackages_constellaration():
        pytest.skip(
            "Set AI_SCIENTIST_REQUIRE_SITEPACKAGES_CONSTELLARATION=1 to enforce this",
        )

    import sys
    from pathlib import Path

    import constellaration.forward_model as fm

    repo_root = Path(__file__).resolve().parents[1]
    fm_path = Path(fm.__file__).resolve()
    prefix_path = Path(sys.prefix).resolve()

    # The vendored checkout would be at repo_root/constellaration/src
    vendored_checkout = repo_root / "constellaration" / "src"

    assert str(fm_path).startswith(str(prefix_path)), (
        "Expected constellaration to import from the active environment "
        f"(sys.prefix={prefix_path}), but got {fm_path}"
    )
    # Check that fm_path is in site-packages AND not in the vendored checkout.
    # Note: .venv may be inside repo_root, so we can't just check startswith(repo_root).
    assert "site-packages" in str(fm_path), (
        f"Expected constellaration to be in site-packages, but got {fm_path}"
    )
    assert not str(fm_path).startswith(str(vendored_checkout)), (
        "Expected constellaration to be the installed package, not the repo checkout "
        f"(vendored_checkout={vendored_checkout}), but got {fm_path}"
    )
