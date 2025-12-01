import os
import warnings

_suppress = os.getenv("AI_SCIENTIST_SUPPRESS_SIMSOPT_WARN", "1") != "0"
if _suppress:
    # Suppress noisy PendingDeprecationWarning from simsopt importing numpy.matlib.
    warnings.filterwarnings(
        "ignore",
        message="Importing from numpy.matlib is deprecated",
        category=PendingDeprecationWarning,
    )
