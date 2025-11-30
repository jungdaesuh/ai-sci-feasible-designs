"""
CLI entry point for AI Scientist experiments.
Shim module to maintain backward compatibility after refactoring to experiment_runner.py.
"""
from ai_scientist.experiment_runner import *
from ai_scientist.experiment_runner import run_experiment as run

# Restore symbols expected by tests/external users
from ai_scientist.optim.surrogate import BaseSurrogate, SurrogateBundle
from ai_scientist.cycle_executor import (
    _propose_p3_candidates_for_cycle,
    _surrogate_rank_screen_candidates,
    _surrogate_candidate_pool_size,
)
from ai_scientist.experiment_setup import (
    build_argument_parser as _build_argument_parser,
    load_run_presets as _load_run_presets,
    apply_run_preset as _apply_run_preset,
    validate_runtime_flags as _validate_runtime_flags,
    create_surrogate as _create_surrogate,
    create_generative_model as _create_generative_model,
    resolve_git_sha as _resolve_git_sha,
    RunnerCLIConfig,
)

if __name__ == "__main__":
    main()