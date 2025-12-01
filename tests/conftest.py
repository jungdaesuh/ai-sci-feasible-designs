import warnings

import pytest


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
