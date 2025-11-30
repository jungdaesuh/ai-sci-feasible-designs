"""Physics tool wrappers for the ConStellaration AI Scientist.

This module has been split into submodules but maintains the original API.
"""

from ai_scientist.tools.evaluation import (
    FlattenSchema,
    BoundaryParams,
    _quantize_float,
    _canonicalize_value,
    _hash_params,
    design_hash,
    _ensure_mapping,
    _derive_schema_from_params,
    _coefficient_from_matrix,
    _evaluate_cached_stage,
    _settings_for_stage,
    _normalize_between_bounds,
    _max_violation,
    compute_constraint_margins,
    _log10_or_large,
    _contains_invalid_number,
    _replace_invalid_numbers,
    _penalized_result,
    _safe_evaluate,
    _gradient_score,
    _p2_feasibility,
    _p3_feasibility,
    make_boundary_from_params,
    evaluate_p1,
    evaluate_p2,
    evaluate_p3,
    evaluate_p3_set,
    get_cache_stats,
    clear_evaluation_cache,
    _DEFAULT_RELATIVE_TOLERANCE,
    _CANONICAL_PRECISION,
    _DEFAULT_SCHEMA_VERSION,
    _DEFAULT_ROUNDING,
    _EVALUATION_CACHE,
    _CACHE_STATS,
)

from ai_scientist.tools.hypervolume import (
    P3Summary,
    ParetoEntry,
    _objective_vector,
    _extract_p3_point,
    _dominates,
    _hypervolume_minimization,
    summarize_p3_candidates,
    _P3_REFERENCE_POINT,
)

from ai_scientist.tools.design_manipulation import (
    propose_boundary,
    recombine_designs,
    normalized_constraint_distance_sampler,
    structured_flatten,
    structured_unflatten,
)

from ai_scientist.tools.integration import (
    retrieve_rag,
    write_note,
)
