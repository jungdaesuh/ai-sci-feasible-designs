"""Physics backend abstraction layer.

This package provides a pluggable architecture for physics evaluations,
allowing tests to run with mock backends while production uses the real
constellaration/vmecpp physics stack.

Usage:
    from ai_scientist.backends import get_backend, set_backend, MockPhysicsBackend

    # Auto-select (uses real if available, else mock)
    backend = get_backend()

    # Explicit mock for tests
    set_backend(MockPhysicsBackend())

    # Explicit real for production
    set_backend("real")
"""

from ai_scientist.backends.base import PhysicsBackend
from ai_scientist.backends.mock import MockPhysicsBackend
from ai_scientist.forward_model import get_backend, reset_backend, set_backend

__all__ = [
    "PhysicsBackend",
    "MockPhysicsBackend",
    "get_backend",
    "set_backend",
    "reset_backend",
]
