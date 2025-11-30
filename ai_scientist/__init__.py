import os
import warnings

from ai_scientist.experiment_runner import run_experiment
from ai_scientist.forward_model import (
    EvaluationResult,
    ForwardModelSettings,
    forward_model,
)

_suppress = os.getenv("AI_SCIENTIST_SUPPRESS_SIMSOPT_WARN", "1") != "0"
if _suppress:
    # Suppress noisy PendingDeprecationWarning from simsopt importing numpy.matlib.
    warnings.filterwarnings(
        "ignore",
        message="Importing from numpy.matlib is deprecated",
        category=PendingDeprecationWarning,
    )
