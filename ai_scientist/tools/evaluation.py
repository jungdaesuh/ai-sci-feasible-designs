"""Physics evaluation wrappers and caching logic."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple
import hashlib
import json
import math
import numpy as np

import ai_scientist.forward_model as centralized_fm
from constellaration import forward_model

from ai_scientist.tools.hypervolume import (
    _hypervolume_minimization,
    _objective_vector,
    _P3_REFERENCE_POINT,
)

_DEFAULT_RELATIVE_TOLERANCE = 1e-2
_DEFAULT_SCHEMA_VERSION = 1
_DEFAULT_ROUNDING = 1e-6
_CANONICAL_PRECISION = 1e-8

# Compatibility placeholders
_EVALUATION_CACHE: Dict[Any, Any] = {}
_CACHE_STATS: Dict[str, Any] = defaultdict(lambda: {"hits": 0, "misses": 0})


@dataclass(frozen=True)
class FlattenSchema:
    """Schema describing the intended Fourier truncation and hash version."""

    mpol: int
    ntor: int
    schema_version: int = _DEFAULT_SCHEMA_VERSION
    rounding: float = _DEFAULT_ROUNDING


@dataclass(frozen=True)
class BoundaryParams:
    """Container for surface parameters that may evolve in future waves."""

    params: Mapping[str, Any]


def _ensure_mapping(params: Mapping[str, Any] | BoundaryParams) -> Mapping[str, Any]:
    if isinstance(params, BoundaryParams):
        return params.params
    return params


def _quantize_float(value: float, *, precision: float = _CANONICAL_PRECISION) -> float:
    if precision <= 0.0:
        return float(value)
    return float(round(value / precision) * precision)


def _canonicalize_value(value: Any, *, precision: float = _CANONICAL_PRECISION) -> Any:
    if isinstance(value, Mapping):
        return {k: _canonicalize_value(v, precision=precision) for k, v in sorted(value.items())}
    if isinstance(value, np.ndarray):
        return _canonicalize_value(value.tolist(), precision=precision)
    if isinstance(value, (list, tuple)):
        return [_canonicalize_value(v, precision=precision) for v in value]
    if isinstance(value, float):
        return _quantize_float(value, precision=precision)
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    return str(value)


def design_hash(
    params: Mapping[str, Any] | BoundaryParams,
    *,
    schema: FlattenSchema | None = None,
    rounding: float | None = None,
) -> str:
    """
    Compute design hash. Delegates to centralized forward model.
    """
    params_map = _ensure_mapping(params)
    return centralized_fm.compute_design_hash(
        params_map, 
        schema_version=schema.schema_version if schema else _DEFAULT_SCHEMA_VERSION,
        rounding=rounding if rounding is not None else _DEFAULT_ROUNDING
    )

def _hash_params(
    params: Mapping[str, Any], *, schema: FlattenSchema | None = None, rounding: float | None = None
) -> str:
    return design_hash(params, schema=schema, rounding=rounding)


def _derive_schema_from_params(
    params: Mapping[str, Any], *, schema_version: int = _DEFAULT_SCHEMA_VERSION, rounding: float = _DEFAULT_ROUNDING
) -> FlattenSchema:
    r_cos = np.asarray(params.get("r_cos", []), dtype=float)
    z_sin = np.asarray(params.get("z_sin", []), dtype=float)
    mpol_candidates = []
    ntor_candidates = []
    if r_cos.size:
        mpol_candidates.append(max(0, r_cos.shape[0] - 1))
        ntor_candidates.append(max(0, (r_cos.shape[1] - 1) // 2))
    if z_sin.size:
        mpol_candidates.append(max(0, z_sin.shape[0] - 1))
        ntor_candidates.append(max(0, (z_sin.shape[1] - 1) // 2))

    mpol = max(mpol_candidates) if mpol_candidates else 0
    ntor = max(ntor_candidates) if ntor_candidates else 0
    return FlattenSchema(mpol=mpol, ntor=ntor, schema_version=schema_version, rounding=rounding)


def _coefficient_from_matrix(matrix: np.ndarray, m: int, n: int, schema_ntor: int) -> float:
    if matrix.ndim != 2 or m < 0 or n < -schema_ntor or n > schema_ntor:
        return 0.0
    if m >= matrix.shape[0]:
        return 0.0

    matrix_ntor = max(0, (matrix.shape[1] - 1) // 2)
    column = n + matrix_ntor
    if column < 0 or column >= matrix.shape[1]:
        return 0.0
    return float(matrix[m, column])


def _log10_or_large(value: float | None) -> float:
    if value is None or value <= 0.0:
        return 10.0
    return float(math.log10(value))


def _contains_invalid_number(node: Any) -> bool:
    if isinstance(node, Mapping):
        return any(_contains_invalid_number(value) for value in node.values())
    if isinstance(node, (list, tuple)):
        return any(_contains_invalid_number(value) for value in node)
    if isinstance(node, np.ndarray):
        return not np.all(np.isfinite(node))
    if isinstance(node, float):
        return not math.isfinite(node)
    return False


def _replace_invalid_numbers(node: Any, replacement: float) -> Any:
    if isinstance(node, Mapping):
        return {key: _replace_invalid_numbers(value, replacement) for key, value in node.items()}
    if isinstance(node, (list, tuple)):
        return [_replace_invalid_numbers(value, replacement) for value in node]
    if isinstance(node, np.ndarray):
        sanitized = np.where(np.isfinite(node), node, replacement)
        return sanitized.tolist()
    if isinstance(node, float) and not math.isfinite(node):
        return float(replacement)
    return node


def _penalized_result(
    *, stage: str, maximize: bool, penalty: float, error: str | None = None
) -> Dict[str, Any]:
    return {
        "stage": stage,
        "objective": penalty,
        "minimize_objective": not maximize,
        "feasibility": float("inf"),
        "score": 0.0,
        "constraint_margins": {},
        "max_violation": float("inf"),
        "metrics": {},
        "error": error,
        "penalized": True,
    }


def _safe_evaluate(
    compute: Callable[[], Dict[str, Any]],
    stage: str,
    *,
    maximize: bool = False,
) -> Dict[str, Any]:
    penalty = -1e9 if maximize else 1e9
    try:
        result = compute()
    except Exception as exc:  # noqa: BLE001
        return _penalized_result(stage=stage, maximize=maximize, penalty=penalty, error=str(exc))

    invalid = _contains_invalid_number(result)
    if invalid:
        sanitized = _replace_invalid_numbers(result, penalty)
        sanitized["objective"] = penalty
        sanitized["feasibility"] = float("inf")
        sanitized["score"] = 0.0
        sanitized["constraint_margins"] = {}
        sanitized["max_violation"] = float("inf")
        sanitized.setdefault("metrics", {})
        sanitized["penalized"] = True
        sanitized["stage"] = stage
        sanitized["minimize_objective"] = not maximize
        return sanitized

    result.setdefault("stage", stage)
    result.setdefault("minimize_objective", not maximize)
    return result


def _evaluate_cached_stage(
    boundary_params: Mapping[str, Any] | BoundaryParams,
    *,
    stage: str,
    compute: Callable[[Mapping[str, Any]], Dict[str, Any]],
    maximize: bool,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """
    Deprecated wrapper. Caching is now handled by forward_model.
    Calls compute directly.
    """
    params_map = _ensure_mapping(boundary_params)
    return _safe_evaluate(lambda: compute(params_map), stage=stage.lower(), maximize=maximize)


# --- Re-exports / Delegates to centralized forward model for compatibility ---

def make_boundary_from_params(params: Mapping[str, Any] | BoundaryParams) -> Any:
    params_map = _ensure_mapping(params)
    return centralized_fm.make_boundary_from_params(params_map)

def compute_constraint_margins(metrics: Any, problem: str) -> Dict[str, float]:
    return centralized_fm.compute_constraint_margins(metrics, problem)

def _max_violation(margins: Mapping[str, float]) -> float:
    return centralized_fm.max_violation(margins)

# -----------------------------------------------------------------------------

def _settings_for_stage(
    stage: str, problem: str, *, skip_qi: bool = False
) -> centralized_fm.ForwardModelSettings:
    stage_lower = stage.lower()
    
    # Determine Constellaration Settings
    if stage_lower == "promote":
        c_settings = forward_model.ConstellarationSettings.default_high_fidelity_skip_qi()
    elif (
        stage_lower.startswith("p2")
        or stage_lower.startswith("p3")
        or stage_lower == "high_fidelity"
    ):
        c_settings = forward_model.ConstellarationSettings.default_high_fidelity()
    else:
        c_settings = forward_model.ConstellarationSettings()

    if skip_qi:
        c_settings = c_settings.model_copy(
            update={
                "boozer_preset_settings": None,
                "qi_settings": None,
            }
        )
        
    return centralized_fm.ForwardModelSettings(
        constellaration_settings=c_settings,
        problem=problem,
        stage=stage_lower,
        fidelity="high" if "high" in stage_lower or "p" in stage_lower else "low"
    )


def _normalize_between_bounds(
    value: float, lower_bound: float, upper_bound: float
) -> float:
    assert lower_bound < upper_bound
    normalized = (value - lower_bound) / (upper_bound - lower_bound)
    return float(np.clip(normalized, 0.0, 1.0))


def _gradient_score(metrics: forward_model.ConstellarationMetrics) -> float:
    gradient = float(metrics.minimum_normalized_magnetic_gradient_scale_length)
    aspect = float(metrics.aspect_ratio)
    return float(gradient / max(1.0, aspect))


def _p2_feasibility(metrics: forward_model.ConstellarationMetrics) -> float:
    margins = compute_constraint_margins(metrics, "p2")
    return _max_violation(margins)


def _p3_feasibility(metrics: forward_model.ConstellarationMetrics) -> float:
    margins = compute_constraint_margins(metrics, "p3")
    return _max_violation(margins)


def evaluate_p1(
    boundary_params: Mapping[str, Any] | BoundaryParams,
    *,
    stage: str = "screen",
    use_cache: bool = True,
) -> Dict[str, Any]:
    """Run a P1-style evaluation using centralized forward model."""
    params_map = _ensure_mapping(boundary_params)
    settings = _settings_for_stage(stage, "p1", skip_qi=True)
    
    try:
        result = centralized_fm.forward_model(params_map, settings, use_cache=use_cache)
    except Exception as e:
        return _penalized_result(stage=stage.lower(), maximize=False, penalty=1e9, error=str(e))

    metrics = result.metrics
    score = 0.0
    if result.feasibility <= _DEFAULT_RELATIVE_TOLERANCE:
        normalized = _normalize_between_bounds(
            value=metrics.max_elongation, lower_bound=1.0, upper_bound=10.0
        )
        score = 1.0 - normalized

    return {
        "stage": stage.lower(),
        "objective": result.objective,
        "minimize_objective": True,
        "feasibility": result.feasibility,
        "score": score,
        "metrics": metrics.model_dump(),
        "settings": settings.constellaration_settings.model_dump(),
        "constraint_margins": result.constraints,
        "max_violation": result.feasibility,
        "cache_hit": result.cache_hit,
    }


def evaluate_p2(
    boundary_params: Mapping[str, Any] | BoundaryParams,
    *,
    stage: str = "p2",
    use_cache: bool = True,
) -> Dict[str, Any]:
    """Run a high-fidelity evaluator for the P2 (QI) problem."""
    params_map = _ensure_mapping(boundary_params)
    settings = _settings_for_stage(stage, "p2")
    
    try:
        result = centralized_fm.forward_model(params_map, settings, use_cache=use_cache)
    except Exception as e:
         return _penalized_result(stage=stage.lower(), maximize=True, penalty=-1e9, error=str(e))

    metrics = result.metrics
    score = _gradient_score(metrics)
    gradient = float(metrics.minimum_normalized_magnetic_gradient_scale_length)

    return {
        "stage": stage.lower(),
        "objective": result.objective,
        "minimize_objective": False, # P2 maximizes gradient
        "feasibility": result.feasibility,
        "score": score,
        "hv": float(max(0.0, gradient - 1.0)),
        "metrics": metrics.model_dump(),
        "settings": settings.constellaration_settings.model_dump(),
        "constraint_margins": result.constraints,
        "max_violation": result.feasibility,
        "cache_hit": result.cache_hit,
    }


def evaluate_p3(
    boundary_params: Mapping[str, Any] | BoundaryParams,
    *,
    stage: str = "p3",
    use_cache: bool = True,
) -> Dict[str, Any]:
    """Run a high-fidelity evaluator for the P3 (multi-objective) problem."""
    params_map = _ensure_mapping(boundary_params)
    settings = _settings_for_stage(stage, "p3")

    try:
        result = centralized_fm.forward_model(params_map, settings, use_cache=use_cache)
    except Exception as e:
         return _penalized_result(stage=stage.lower(), maximize=False, penalty=1e9, error=str(e))

    metrics = result.metrics
    score = _gradient_score(metrics)

    return {
        "stage": stage.lower(),
        "objective": result.objective,
        "minimize_objective": True,
        "feasibility": result.feasibility,
        "score": score,
        "hv": float(
            max(
                0.0, metrics.minimum_normalized_magnetic_gradient_scale_length - 1.0
            )
        ),
        "metrics": metrics.model_dump(),
        "settings": settings.constellaration_settings.model_dump(),
        "constraint_margins": result.constraints,
        "max_violation": result.feasibility,
        "cache_hit": result.cache_hit,
    }


def evaluate_p3_set(
    boundary_specs: Sequence[Mapping[str, Any] | BoundaryParams],
    *,
    stage: str = "p3",
    reference_point: Tuple[float, float] = _P3_REFERENCE_POINT,
) -> Dict[str, Any]:
    """Evaluate a batch of P3 boundaries and compute the set-level hypervolume."""

    stage_lower = stage.lower()
    if not boundary_specs:
        return {
            "stage": stage_lower,
            "objectives": [],
            "feasibilities": [],
            "hv_score": 0.0,
            "metrics_list": [],
        }

    evaluations: list[Dict[str, Any]] = []
    hv_vectors: list[Tuple[float, float]] = []

    for candidate in boundary_specs:
        evaluation = evaluate_p3(candidate, stage=stage)
        # Handle errors gracefully?
        if evaluation.get("error"):
            continue

        metrics = evaluation["metrics"]
        feasibility = float(evaluation["feasibility"])

        if feasibility <= _DEFAULT_RELATIVE_TOLERANCE:
            hv_vectors.append(_objective_vector(metrics))

        evaluations.append(evaluation)

    hv_score = _hypervolume_minimization(hv_vectors, reference_point)
    return {
        "stage": stage_lower,
        "objectives": [
            {
                "aspect_ratio": float(eval_["metrics"]["aspect_ratio"]),
                "gradient": float(
                    eval_["metrics"][
                        "minimum_normalized_magnetic_gradient_scale_length"
                    ]
                ),
                "objective": eval_["objective"],
            }
            for eval_ in evaluations
        ],
        "feasibilities": [float(eval_["feasibility"]) for eval_ in evaluations],
        "hv_score": hv_score,
        "metrics_list": [eval_["metrics"] for eval_ in evaluations],
    }


def get_cache_stats(stage: str) -> Mapping[str, int]:
    """Return hit/miss counts for a given stage."""
    # Forward model aggregates stats, doesn't break down by stage currently.
    # We return the aggregate stats.
    return centralized_fm.get_cache_stats()


def clear_evaluation_cache() -> None:
    """Reset the evaluation cache and stats."""
    centralized_fm.clear_cache()