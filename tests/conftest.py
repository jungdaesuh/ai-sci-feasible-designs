import warnings

# Silence noisy PendingDeprecationWarning from numpy.matlib inside simsopt dependency.
warnings.filterwarnings(
    "ignore",
    message="Importing from numpy.matlib is deprecated",
    category=PendingDeprecationWarning,
)
