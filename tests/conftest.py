import sys
import warnings
from unittest.mock import MagicMock

import pytest

# Mock vmecpp and constellaration to avoid ImportError due to missing libtorch
# This must be done before any tests are collected or run
if "vmecpp" not in sys.modules:
    sys.modules["vmecpp"] = MagicMock()
    sys.modules["vmecpp"].__path__ = []
    sys.modules["vmecpp"].__spec__ = None
if "constellaration" not in sys.modules:
    sys.modules["constellaration"] = MagicMock()
    sys.modules["constellaration"].__path__ = []
    sys.modules["constellaration"].__spec__ = None
if "constellaration.forward_model" not in sys.modules:
    sys.modules["constellaration.forward_model"] = MagicMock()
    sys.modules["constellaration.forward_model"].__spec__ = None
if "constellaration.geometry" not in sys.modules:
    sys.modules["constellaration.geometry"] = MagicMock()
    sys.modules["constellaration.geometry"].__spec__ = None
if "constellaration.problems" not in sys.modules:
    sys.modules["constellaration.problems"] = MagicMock()
    sys.modules["constellaration.problems"].__spec__ = None
if "constellaration.boozer" not in sys.modules:
    sys.modules["constellaration.boozer"] = MagicMock()
    sys.modules["constellaration.boozer"].__spec__ = None
if "constellaration.optimization" not in sys.modules:
    sys.modules["constellaration.optimization"] = MagicMock()
    sys.modules["constellaration.optimization"].__path__ = []
    sys.modules["constellaration.optimization"].__spec__ = None
if "constellaration.optimization.augmented_lagrangian" not in sys.modules:
    sys.modules["constellaration.optimization.augmented_lagrangian"] = MagicMock()
    sys.modules["constellaration.optimization.augmented_lagrangian"].__spec__ = None
if "constellaration.mhd" not in sys.modules:
    sys.modules["constellaration.mhd"] = MagicMock()
    sys.modules["constellaration.mhd"].__path__ = []
    sys.modules["constellaration.mhd"].__spec__ = None
if "constellaration.mhd.vmec_utils" not in sys.modules:
    sys.modules["constellaration.mhd.vmec_utils"] = MagicMock()
    sys.modules["constellaration.mhd.vmec_utils"].__spec__ = None
if "constellaration.optimization.settings" not in sys.modules:
    sys.modules["constellaration.optimization.settings"] = MagicMock()
    sys.modules["constellaration.optimization.settings"].__spec__ = None
if "constellaration.utils" not in sys.modules:
    sys.modules["constellaration.utils"] = MagicMock()
    sys.modules["constellaration.utils"].__path__ = []
    sys.modules["constellaration.utils"].__spec__ = None
if "constellaration.utils.pytree" not in sys.modules:
    sys.modules["constellaration.utils.pytree"] = MagicMock()
    sys.modules["constellaration.utils.pytree"].__spec__ = None
if "constellaration.initial_guess" not in sys.modules:
    sys.modules["constellaration.initial_guess"] = MagicMock()
    sys.modules["constellaration.initial_guess"].__spec__ = None


# Configure mocks to be JSON serializable (behave like Pydantic models)
class MockPydanticModel(MagicMock):
    def model_dump(self, *args, **kwargs):
        return {}

    def dict(self, *args, **kwargs):
        return {}

    def model_copy(self, *args, **kwargs):
        return self

    @classmethod
    def default_high_fidelity_skip_qi(cls):
        return cls()

    @classmethod
    def default_high_fidelity(cls):
        return cls()


sys.modules["constellaration.forward_model"].ConstellarationSettings = MockPydanticModel
sys.modules["constellaration.forward_model"].ConstellarationMetrics = MockPydanticModel
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
