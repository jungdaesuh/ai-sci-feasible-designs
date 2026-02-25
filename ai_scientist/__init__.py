import os
import warnings
import importlib.util

# Early torch initialization to prevent native library conflicts with vmecpp
# Must happen before any constellaration/vmecpp imports in the package
if importlib.util.find_spec("torch") is not None:
    import torch as _torch

    _ = _torch.zeros(1)
    del _

_suppress = os.getenv("AI_SCIENTIST_SUPPRESS_SIMSOPT_WARN", "1") != "0"
if _suppress:
    # Suppress noisy PendingDeprecationWarning from simsopt importing numpy.matlib.
    warnings.filterwarnings(
        "ignore",
        message="Importing from numpy.matlib is deprecated",
        category=PendingDeprecationWarning,
    )
