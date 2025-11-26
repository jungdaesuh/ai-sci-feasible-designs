"""Ensure the custom SurfaceRZFourier pytree registration is idempotent."""

from __future__ import annotations

from typing import Any, Callable

from constellaration.geometry import surface_rz_fourier
from constellaration.utils import pytree

_META_FIELDS: list[str] = [
    "n_field_periods",
    "is_stellarator_symmetric",
]

_ORIGINAL_REGISTER = pytree.register_pydantic_data
_IS_INSTALLED = False
_IS_SURFACE_REGISTERED = False


def _guarded_register(cls: type, meta_fields: list[str] | None = None) -> type:
    try:
        return _ORIGINAL_REGISTER(cls, meta_fields=meta_fields)
    except ValueError as exc:
        if "Duplicate custom PyTreeDef type registration" in str(exc):
            return cls
        raise


def _register_surface_once() -> None:
    global _IS_SURFACE_REGISTERED
    if _IS_SURFACE_REGISTERED:
        return
    _guarded_register(
        surface_rz_fourier.SurfaceRZFourier,
        meta_fields=_META_FIELDS,
    )
    _IS_SURFACE_REGISTERED = True


def install() -> None:
    """Patch the pytree helper to swallow duplicate registrations."""

    global _IS_INSTALLED
    if _IS_INSTALLED:
        return
    pytree.register_pydantic_data = _guarded_register  # type: ignore[assignment]
    _IS_INSTALLED = True
    _register_surface_once()


install()
