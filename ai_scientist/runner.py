"""
CLI entry point for AI Scientist experiments.
Shim module to maintain backward compatibility after refactoring to experiment_runner.py.
"""

from ai_scientist.experiment_runner import *  # noqa: F403

# Restore symbols expected by tests/external users

if __name__ == "__main__":
    main()  # noqa: F405
