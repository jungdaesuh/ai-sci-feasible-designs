import os
import sys
import warnings
from unittest.mock import MagicMock

import pytest


def _is_truthy_env(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


_PHYSICS_BACKEND_ENV = os.environ.get("AI_SCIENTIST_PHYSICS_BACKEND", "auto").lower()
_REQUIRE_REAL_BACKEND = _PHYSICS_BACKEND_ENV == "real" or _is_truthy_env(
    os.environ.get("AI_SCIENTIST_REQUIRE_REAL_BACKEND")
)
_ALLOW_PHYSICS_MOCKS = _PHYSICS_BACKEND_ENV == "mock" or _is_truthy_env(
    os.environ.get("AI_SCIENTIST_ALLOW_PHYSICS_MOCKS")
)

# Check if real constellaration is available BEFORE any optional mocking.
_CONSTELLARATION_AVAILABLE = False
_CONSTELLARATION_IMPORT_ERROR: ImportError | None = None
try:
    # Try importing a key module to check availability.
    import constellaration.forward_model as _check_fm

    _CONSTELLARATION_AVAILABLE = True
    del _check_fm
except ImportError as exc:
    _CONSTELLARATION_IMPORT_ERROR = exc


def _vmecpp_native_available() -> bool:
    """Return True iff the vmecpp C++ extension can be imported."""
    try:
        import vmecpp.cpp._vmecpp  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


if _REQUIRE_REAL_BACKEND:
    # Fail fast with a targeted message when real physics is explicitly required.
    if not _CONSTELLARATION_AVAILABLE:
        msg = (
            "Real physics backend requested but constellaration is not importable.\n"
            "Set AI_SCIENTIST_PHYSICS_BACKEND=mock to run unit tests without physics, "
            "or fix the environment so constellaration/vmecpp import works.\n"
        )
        if _CONSTELLARATION_IMPORT_ERROR is not None:
            msg += f"Original error: {_CONSTELLARATION_IMPORT_ERROR!r}\n"
        pytest.exit(msg, returncode=2)
    if not _vmecpp_native_available():
        pytest.exit(
            "Real physics backend requested but vmecpp native extension cannot be "
            "imported (dlopen/link error). This often presents as a missing "
            "dependency such as @rpath/libtorch.dylib.\n"
            "Fix/reinstall vmecpp in the active environment and retry.\n",
            returncode=2,
        )

# Only mock vmecpp and constellaration if real imports are NOT available
# This is intentionally opt-in: set AI_SCIENTIST_ALLOW_PHYSICS_MOCKS=1 to enable.
if not _CONSTELLARATION_AVAILABLE and _ALLOW_PHYSICS_MOCKS:
    if "vmecpp" not in sys.modules:
        sys.modules["vmecpp"] = MagicMock()
        sys.modules["vmecpp"].__path__ = []
        sys.modules["vmecpp"].__spec__ = MagicMock()
    if "constellaration" not in sys.modules:
        sys.modules["constellaration"] = MagicMock()
        sys.modules["constellaration"].__path__ = []
        sys.modules["constellaration"].__spec__ = MagicMock()
    if "constellaration.forward_model" not in sys.modules:
        sys.modules["constellaration.forward_model"] = MagicMock()
        sys.modules["constellaration.forward_model"].__spec__ = MagicMock()
    if "constellaration.geometry" not in sys.modules:
        sys.modules["constellaration.geometry"] = MagicMock()
        # Configure SurfaceRZFourier to return a mock with r_cos/z_sin that have shape
        mock_surface = MagicMock()
        mock_surface.r_cos = MagicMock()
        mock_surface.r_cos.shape = (2, 5)
        mock_surface.r_cos.__getitem__.return_value = (
            0.2  # For test_make_boundary... assertion
        )
        mock_surface.z_sin = MagicMock()
        mock_surface.z_sin.shape = (2, 5)

        # Mock the submodule surface_rz_fourier
        mock_srf_module = MagicMock()
        mock_srf_module.SurfaceRZFourier.return_value = mock_surface
        sys.modules["constellaration.geometry"].surface_rz_fourier = mock_srf_module

        # Also set it directly on geometry just in case
        sys.modules[
            "constellaration.geometry"
        ].SurfaceRZFourier.return_value = mock_surface
        sys.modules["constellaration.geometry"].__spec__ = MagicMock()
        sys.modules["constellaration.geometry"].SurfaceRZFourier.__spec__ = MagicMock()
    if "constellaration.problems" not in sys.modules:
        sys.modules["constellaration.problems"] = MagicMock()
        sys.modules["constellaration.problems"].__spec__ = MagicMock()
    if "constellaration.boozer" not in sys.modules:
        sys.modules["constellaration.boozer"] = MagicMock()
        sys.modules["constellaration.boozer"].__spec__ = MagicMock()
    if "constellaration.optimization" not in sys.modules:
        sys.modules["constellaration.optimization"] = MagicMock()
        sys.modules["constellaration.optimization"].__path__ = []
        sys.modules["constellaration.optimization"].__spec__ = MagicMock()
    if "constellaration.optimization.augmented_lagrangian" not in sys.modules:
        sys.modules["constellaration.optimization.augmented_lagrangian"] = MagicMock()
        sys.modules[
            "constellaration.optimization.augmented_lagrangian"
        ].__spec__ = MagicMock()
    if "constellaration.mhd" not in sys.modules:
        sys.modules["constellaration.mhd"] = MagicMock()
        sys.modules["constellaration.mhd"].__path__ = []
        sys.modules["constellaration.mhd"].__spec__ = MagicMock()
    if "constellaration.mhd.vmec_utils" not in sys.modules:
        sys.modules["constellaration.mhd.vmec_utils"] = MagicMock()
        sys.modules["constellaration.mhd.vmec_utils"].__spec__ = MagicMock()
    if "constellaration.optimization.settings" not in sys.modules:
        sys.modules["constellaration.optimization.settings"] = MagicMock()
        sys.modules["constellaration.optimization.settings"].__spec__ = MagicMock()
    if "constellaration.utils" not in sys.modules:
        sys.modules["constellaration.utils"] = MagicMock()
        sys.modules["constellaration.utils"].__path__ = []
        sys.modules["constellaration.utils"].__spec__ = MagicMock()
    if "constellaration.utils.pytree" not in sys.modules:
        sys.modules["constellaration.utils.pytree"] = MagicMock()
        sys.modules["constellaration.utils.pytree"].__spec__ = MagicMock()
    if "constellaration.initial_guess" not in sys.modules:
        sys.modules["constellaration.initial_guess"] = MagicMock()
        sys.modules["constellaration.initial_guess"].__spec__ = MagicMock()


# Configure mocks to be JSON serializable (behave like Pydantic models)
class MockPydanticModel(MagicMock):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._mock_data = kwargs

    def model_dump(self, *args, **kwargs):
        return self._mock_data

    def dict(self, *args, **kwargs):
        return self._mock_data

    def model_copy(self, *args, **kwargs):
        return self

    def __iter__(self):
        # Prevent Pydantic from treating this as an iterator
        raise TypeError("MockPydanticModel is not iterable")

    def __repr__(self):
        # Stable repr for caching
        return f"MockPydanticModel({self._mock_data})"

    def __getattr__(self, name):
        try:
            data = object.__getattribute__(self, "_mock_data")
            if name in data:
                return data[name]
        except AttributeError:
            pass
        return super().__getattr__(name)

    @classmethod
    def default_high_fidelity_skip_qi(cls):
        return cls(fidelity="high_skip_qi")

    @classmethod
    def default_high_fidelity(cls):
        # Create nested mock for vmec_preset_settings
        vmec_settings = cls(fidelity="high_fidelity")
        return cls(fidelity="high", vmec_preset_settings=vmec_settings)


@pytest.fixture
def runner_module():
    """
    Fixture that mocks heavy dependencies and imports ai_scientist.runner.
    Restores state after test to avoid polluting global sys.modules.
    """
    import sys
    from unittest.mock import patch

    # Create dummy classes for types used in annotations to satisfy jaxtyping/isinstance checks
    class MockTensor:
        pass

    class MockArray:
        pass

    mock_torch = MagicMock()
    mock_torch.Tensor = MockTensor

    mock_jax = MagicMock()
    mock_jax.Array = MockArray

    mock_jax_numpy = MagicMock()
    mock_jax_numpy.ndarray = MockArray
    mock_jax.numpy = mock_jax_numpy

    mock_modules = {
        "torch": mock_torch,
        "torch.nn": MagicMock(),
        "torch.nn.functional": MagicMock(),
        "torch.optim": MagicMock(),
        "torch.distributions": MagicMock(),
        "torch.utils": MagicMock(),
        "torch.utils.data": MagicMock(),
        "vmecpp": MagicMock(),
        "jax": mock_jax,
        "jaxlib": MagicMock(),
        "jax.numpy": mock_jax_numpy,
        "jax.tree_util": MagicMock(),
        "ai_scientist.coordinator": MagicMock(),
        "ai_scientist.forward_model": MagicMock(),
        "ai_scientist.optim.surrogate_v2": MagicMock(),
        "ai_scientist.tools": MagicMock(),
    }

    with patch.dict(sys.modules, mock_modules):
        # Ensure we get a fresh import of runner using the mocks
        # We must remove it from sys.modules if it exists to force re-import with mocks
        if "ai_scientist.runner" in sys.modules:
            del sys.modules["ai_scientist.runner"]

        import ai_scientist.runner

        yield ai_scientist.runner

        # Cleanup: remove the mocked runner so subsequent tests don't use it
        if "ai_scientist.runner" in sys.modules:
            del sys.modules["ai_scientist.runner"]


# Only configure mock attributes if mocks are active
if not _CONSTELLARATION_AVAILABLE and _ALLOW_PHYSICS_MOCKS:
    sys.modules[
        "constellaration.forward_model"
    ].ConstellarationSettings = MockPydanticModel
    sys.modules[
        "constellaration.forward_model"
    ].ConstellarationMetrics = MockPydanticModel
    sys.modules[
        "constellaration.optimization.settings"
    ].AugmentedLagrangianSettings = MockPydanticModel
    sys.modules[
        "constellaration.optimization.settings"
    ].NevergradSettings = MockPydanticModel
    sys.modules[
        "constellaration.optimization.settings"
    ].OptimizationSettings = MockPydanticModel

    # Configure forward_model to return (metrics, info) tuple
    sys.modules["constellaration.forward_model"].forward_model.return_value = (
        MagicMock(),
        MagicMock(),
    )
    # Make sure metrics has model_dump
    sys.modules["constellaration.forward_model"].forward_model.return_value[
        0
    ].model_dump.return_value = {}
    sys.modules["constellaration.forward_model"].forward_model.return_value[
        0
    ].dict.return_value = {}

    # Link submodules to parent module attributes to ensure consistency
    sys.modules["constellaration"].forward_model = sys.modules[
        "constellaration.forward_model"
    ]
    sys.modules["constellaration"].geometry = sys.modules["constellaration.geometry"]
    sys.modules["constellaration"].optimization = sys.modules[
        "constellaration.optimization"
    ]
    sys.modules["constellaration"].mhd = sys.modules["constellaration.mhd"]
    sys.modules["constellaration"].utils = sys.modules["constellaration.utils"]
    sys.modules["constellaration"].initial_guess = sys.modules[
        "constellaration.initial_guess"
    ]


def pytest_collection_modifyitems(config, items):
    """Skip known-crashy constellaration vmec optimization tests on macOS/arm."""
    for item in items:
        path = str(item.fspath)
        if "constellaration/tests/data_generation/vmec_optimization_test.py" in path:
            item.add_marker(
                pytest.mark.xfail(
                    reason="numpy longdouble crash in forked process pool on this platform",
                    strict=False,
                )
            )


# Silence noisy PendingDeprecationWarning from numpy.matlib inside simsopt dependency.
warnings.filterwarnings(
    "ignore",
    message="Importing from numpy.matlib is deprecated",
    category=PendingDeprecationWarning,
)


@pytest.fixture(autouse=True)
def reset_evaluation_state(request: pytest.FixtureRequest):
    """Reset all cache state and backend before and after each test.

    This fixture addresses test order-dependency issues caused by:
    1. Global _EVALUATION_CACHE and _CACHE_STATS in forward_model.py
    2. Similar global state in tools/evaluation.py
    3. Global _BACKEND state in forward_model.py

    The autouse=True ensures this runs for EVERY test without explicit opt-in.
    """
    # Import here to avoid circular imports and handle mocked modules
    try:
        from ai_scientist import forward_model as fm
        from ai_scientist.tools import evaluation as tools_eval
        from ai_scientist.backends import MockPhysicsBackend

        # Clear before test
        fm.clear_cache()
        # Default to mock backend for unit tests.
        # Integration tests (or explicit real-backend runs) must opt out.
        is_integration = request.node.get_closest_marker("integration") is not None
        if not is_integration and not _REQUIRE_REAL_BACKEND:
            fm.set_backend(MockPhysicsBackend())
        tools_eval._EVALUATION_CACHE.clear()
        tools_eval._CACHE_STATS.clear()
    except (ImportError, AttributeError):
        # Modules may be mocked or not available in all test contexts
        pass

    yield

    # Clear after test for completeness
    try:
        from ai_scientist import forward_model as fm
        from ai_scientist.tools import evaluation as tools_eval

        fm.clear_cache()
        fm.reset_backend()
        tools_eval._EVALUATION_CACHE.clear()
        tools_eval._CACHE_STATS.clear()
    except (ImportError, AttributeError):
        pass


@pytest.fixture
def mock_backend():
    """Fixture to explicitly use mock physics backend.

    Returns the MockPhysicsBackend instance for assertions.

    Example:
        def test_something(mock_backend):
            # ... run code that calls forward_model ...
            assert mock_backend.call_count == expected_calls
    """
    try:
        from ai_scientist import forward_model as fm
        from ai_scientist.backends.mock import MockPhysicsBackend

        backend = MockPhysicsBackend()
        fm.set_backend(backend)
        yield backend
        fm.reset_backend()
    except ImportError:
        yield None


@pytest.fixture
def real_backend():
    """Fixture for tests that require real physics backend.

    Skips the test if constellaration is not available.

    Example:
        @pytest.mark.integration
        def test_physics(real_backend):
            # This test requires real constellaration
            ...
    """
    try:
        from ai_scientist import forward_model as fm
        from ai_scientist.backends.real import RealPhysicsBackend

        backend = RealPhysicsBackend()
        if not backend.is_available():
            if _REQUIRE_REAL_BACKEND:
                pytest.fail(
                    "Real physics backend requested but RealPhysicsBackend.is_available() "
                    "returned False. This usually indicates a broken vmecpp native "
                    "installation (e.g. dlopen/libtorch linking).",
                    pytrace=False,
                )
            pytest.skip("Real physics backend requires constellaration")

        fm.set_backend(backend)
        yield backend
        fm.reset_backend()
    except ImportError:
        if _REQUIRE_REAL_BACKEND:
            pytest.fail(
                "Real physics backend requested but imports failed. Fix the environment "
                "so constellaration/vmecpp can be imported.",
                pytrace=False,
            )
        pytest.skip("Real physics backend requires constellaration")


def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test (may require real physics)",
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow (involves physics calculations)"
    )
