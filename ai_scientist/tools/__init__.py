"""Physics tool wrappers for the ConStellaration AI Scientist.

This module has been split into submodules but maintains the original API.
"""

from .design_manipulation import (
    normalized_constraint_distance_sampler,
    structured_flatten,
)
from .evaluation import (
    _DEFAULT_RELATIVE_TOLERANCE,
    FlattenSchema,
    _max_violation,
    _settings_for_stage,
    clear_evaluation_cache,
    compute_constraint_margins,
    design_hash,
    evaluate_p1,
    evaluate_p2,
    evaluate_p3,
    evaluate_p3_set,
    get_cache_stats,
    make_boundary_from_params,
)
from .hypervolume import summarize_p3_candidates

__all__ = [
    "_DEFAULT_RELATIVE_TOLERANCE",
    "FlattenSchema",
    "structured_flatten",
    "evaluate_p1",
    "evaluate_p2",
    "evaluate_p3",
    "evaluate_p3_set",
    "clear_evaluation_cache",
    "design_hash",
    "_settings_for_stage",
    "get_cache_stats",
    "make_boundary_from_params",
    "compute_constraint_margins",
    "_max_violation",
    "normalized_constraint_distance_sampler",
    "summarize_p3_candidates",
]
