"""Physics backend abstraction layer.

This package provides a pluggable architecture for physics evaluations,
allowing tests to run with mock backends while production uses the real
constellaration/vmecpp physics stack.

Usage:
    from ai_scientist.backends import MockPhysicsBackend
    from ai_scientist import forward_model

    # Auto-select (uses real if available, else mock)
    backend = forward_model.get_backend()

    # Explicit mock for tests
    forward_model.set_backend(MockPhysicsBackend())

    # Explicit real for production
    forward_model.set_backend("real")
"""

from ai_scientist.backends.base import PhysicsBackend
from ai_scientist.backends.mock import MockPhysicsBackend

__all__ = [
    "PhysicsBackend",
    "MockPhysicsBackend",
]
