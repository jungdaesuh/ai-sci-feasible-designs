"""Physics evaluation wrappers and caching logic."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

import numpy as np

from constellaration import forward_model
from constellaration.geometry import surface_rz_fourier

from ai_scientist.tools.hypervolume import (
    _hypervolume_minimization,
    _objective_vector,
    _P3_REFERENCE_POINT,
)

_DEFAULT_RELATIVE_TOLERANCE = 1e-2
_CANONICAL_PRECISION = 1e-8
_DEFAULT_SCHEMA_VERSION = 1
_DEFAULT_ROUNDING = 1e-6
_EVALUATION_CACHE: Dict[Tuple[str, str], Dict[str, Any]] = {}
_CACHE_STATS: Dict[str, Dict[str, int]] = defaultdict(lambda: {"hits": 0, "misses": 0})


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


def _hash_params(
    params: Mapping[str, Any], *, schema: FlattenSchema | None = None, rounding: float | None = None
) -> str:
    return design_hash(params, schema=schema, rounding=rounding)


def design_hash(
    params: Mapping[str, Any] | BoundaryParams,
    *,
    schema: FlattenSchema | None = None,
    rounding: float | None = None,
) -> str:
    params_map = _ensure_mapping(params)
    precision = rounding if rounding is not None else _CANONICAL_PRECISION
    payload: Mapping[str, Any]
    if schema is None:
        payload = params_map
    else:
        payload = {
            "schema_version": schema.schema_version,
            "mpol": schema.mpol,
            "ntor": schema.ntor,
            "rounding": schema.rounding,
            "params": params_map,
        }
        precision = rounding if rounding is not None else schema.rounding

    normalized = _canonicalize_value(payload, precision=precision)
    digest = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(digest.encode("utf-8")).hexdigest()


def _ensure_mapping(params: Mapping[str, Any] | BoundaryParams) -> Mapping[str, Any]:
    if isinstance(params, BoundaryParams):
        return params.params
    return params


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


def _evaluate_cached_stage(
    boundary_params: Mapping[str, Any] | BoundaryParams,
    *,
    stage: str,
    compute: Callable[[Mapping[str, Any]], Dict[str, Any]],
    maximize: bool,
    use_cache: bool = True,
) -> Dict[str, Any]:
    params_map = _ensure_mapping(boundary_params)
    stage_lower = stage.lower()
    schema = _derive_schema_from_params(params_map)
    cache_key = (stage_lower, _hash_params(params_map, schema=schema, rounding=schema.rounding))
    stats = _CACHE_STATS[stage_lower]
    if use_cache:
        cached = _EVALUATION_CACHE.get(cache_key)
        if cached is not None:
            stats["hits"] += 1
            return cached

    stats["misses"] += 1
    result = _safe_evaluate(lambda: compute(params_map), stage=stage_lower, maximize=maximize)
    if use_cache:
        _EVALUATION_CACHE[cache_key] = result
    return result


def _settings_for_stage(
    stage: str, *, skip_qi: bool = False
) -> forward_model.ConstellarationSettings:
    stage_lower = stage.lower()
    if stage_lower == "promote":
        settings = forward_model.ConstellarationSettings.default_high_fidelity_skip_qi()
    elif (
        stage_lower.startswith("p2")
        or stage_lower.startswith("p3")
        or stage_lower == "high_fidelity"
    ):
        settings = forward_model.ConstellarationSettings.default_high_fidelity()
    else:
        settings = forward_model.ConstellarationSettings()

    if skip_qi:
        return settings.model_copy(
            update={
                "boozer_preset_settings": None,
                "qi_settings": None,
            }
        )
    return settings


def _normalize_between_bounds(
    value: float, lower_bound: float, upper_bound: float
) -> float:
    assert lower_bound < upper_bound
    normalized = (value - lower_bound) / (upper_bound - lower_bound)
    return float(np.clip(normalized, 0.0, 1.0))


def _max_violation(margins: Mapping[str, float]) -> float:
    if not margins:
        return float("inf")
    return float(max(0.0, *[max(0.0, value) for value in margins.values()]))


def compute_constraint_margins(
    metrics: Mapping[str, Any] | forward_model.ConstellarationMetrics,
    problem: str,
) -> dict[str, float]:
    metrics_map = metrics.model_dump() if hasattr(metrics, "model_dump") else dict(metrics)
    problem_key = problem.lower()

    def _log10_margin(target: float) -> float:
        return _log10_or_large(metrics_map.get("qi")) - target

    margins: dict[str, float] = {}

    if problem_key.startswith("p1"):
        margins = {
            "aspect_ratio": float(metrics_map.get("aspect_ratio", float("nan"))) - 4.0,
            "average_triangularity": float(metrics_map.get("average_triangularity", float("nan"))) - (-0.5),
            "edge_rotational_transform": 0.3
            - float(metrics_map.get("edge_rotational_transform_over_n_field_periods", float("nan"))),
        }
    elif problem_key.startswith("p2"):
        margins = {
            "aspect_ratio": float(metrics_map.get("aspect_ratio", float("nan"))) - 10.0,
            "edge_rotational_transform": 0.25
            - float(metrics_map.get("edge_rotational_transform_over_n_field_periods", float("nan"))),
            "edge_magnetic_mirror_ratio": float(
                metrics_map.get("edge_magnetic_mirror_ratio", float("nan"))
            )
            - 0.2,
            "max_elongation": float(metrics_map.get("max_elongation", float("nan"))) - 5.0,
            "qi_log10": _log10_margin(-4.0),
        }
    else:
        flux_value = metrics_map.get("flux_compression_in_regions_of_bad_curvature")
        flux_margin = (
            float(flux_value) - 0.9
            if flux_value is not None
            else 0.0
        )
        margins = {
            "edge_rotational_transform": 0.25
            - float(metrics_map.get("edge_rotational_transform_over_n_field_periods", float("nan"))),
            "edge_magnetic_mirror_ratio": float(
                metrics_map.get("edge_magnetic_mirror_ratio", float("nan"))
            )
            - 0.25,
            "vacuum_well": -float(metrics_map.get("vacuum_well", float("nan"))),
            "flux_compression": flux_margin,
            "qi_log10": _log10_margin(-3.5),
        }

    return margins


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


def make_boundary_from_params(
    params: Mapping[str, Any] | BoundaryParams,
) -> surface_rz_fourier.SurfaceRZFourier:
    """Construct a SurfaceRZFourier boundary from a simple parameter dictionary."""

    params_map = _ensure_mapping(params)
    payload: dict[str, Any] = {
        "r_cos": np.asarray(params_map["r_cos"], dtype=float),
        "z_sin": np.asarray(params_map["z_sin"], dtype=float),
        "is_stellarator_symmetric": bool(
            params_map.get("is_stellarator_symmetric", True)
        ),
        "n_field_periods": int(params_map.get("n_field_periods", 1)),
    }

    if "r_sin" in params_map:
        payload["r_sin"] = np.asarray(params_map["r_sin"], dtype=float)
    if "z_cos" in params_map:
        payload["z_cos"] = np.asarray(params_map["z_cos"], dtype=float)
    if "nfp" in params_map:
        payload.setdefault("n_field_periods", int(params_map["nfp"]))

    return surface_rz_fourier.SurfaceRZFourier(**payload)


def evaluate_p1(
    boundary_params: Mapping[str, Any] | BoundaryParams,
    *,
    stage: str = "screen",
    use_cache: bool = True,
) -> Dict[str, Any]:
    """Run a P1-style evaluation and cache results by stage."""

    def compute(params_map: Mapping[str, Any]) -> Dict[str, Any]:
        boundary = make_boundary_from_params(params_map)
        settings = _settings_for_stage(stage, skip_qi=True)
        metrics, _ = forward_model.forward_model(boundary, settings=settings)
        constraint_margins = compute_constraint_margins(metrics, "p1")
        feasibility = _max_violation(constraint_margins)
        score = 0.0
        if feasibility <= _DEFAULT_RELATIVE_TOLERANCE:
            normalized = _normalize_between_bounds(
                value=metrics.max_elongation, lower_bound=1.0, upper_bound=10.0
            )
            score = 1.0 - normalized

        return {
            "stage": stage.lower(),
            "objective": float(metrics.max_elongation),
            "minimize_objective": True,
            "feasibility": feasibility,
            "score": score,
            "metrics": metrics.model_dump(),
            "settings": settings.model_dump(),
            "constraint_margins": constraint_margins,
            "max_violation": feasibility,
        }

    return _evaluate_cached_stage(
        boundary_params,
        stage=stage,
        compute=compute,
        maximize=False,
        use_cache=use_cache,
    )


def evaluate_p2(
    boundary_params: Mapping[str, Any] | BoundaryParams,
    *,
    stage: str = "p2",
    use_cache: bool = True,
) -> Dict[str, Any]:
    """Run a high-fidelity evaluator for the P2 (QI) problem."""

    def compute(params_map: Mapping[str, Any]) -> Dict[str, Any]:
        boundary = make_boundary_from_params(params_map)
        settings = _settings_for_stage(stage)
        metrics, _ = forward_model.forward_model(boundary, settings=settings)
        constraint_margins = compute_constraint_margins(metrics, "p2")
        feasibility = _max_violation(constraint_margins)
        gradient = float(metrics.minimum_normalized_magnetic_gradient_scale_length)
        score = _gradient_score(metrics)

        return {
            "stage": stage.lower(),
            "objective": gradient,
            "minimize_objective": False,
            "feasibility": feasibility,
            "score": score,
            "hv": float(max(0.0, gradient - 1.0)),
            "metrics": metrics.model_dump(),
            "settings": settings.model_dump(),
            "constraint_margins": constraint_margins,
            "max_violation": feasibility,
        }

    return _evaluate_cached_stage(
        boundary_params,
        stage=stage,
        compute=compute,
        maximize=True,
        use_cache=use_cache,
    )


def evaluate_p3(
    boundary_params: Mapping[str, Any] | BoundaryParams,
    *,
    stage: str = "p3",
    use_cache: bool = True,
) -> Dict[str, Any]:
    """Run a high-fidelity evaluator for the P3 (multi-objective) problem."""

    def compute(params_map: Mapping[str, Any]) -> Dict[str, Any]:
        boundary = make_boundary_from_params(params_map)
        settings = _settings_for_stage(stage)
        metrics, _ = forward_model.forward_model(boundary, settings=settings)
        constraint_margins = compute_constraint_margins(metrics, "p3")
        feasibility = _max_violation(constraint_margins)
        score = _gradient_score(metrics)

        return {
            "stage": stage.lower(),
            "objective": float(metrics.aspect_ratio),
            "minimize_objective": True,
            "feasibility": feasibility,
            "score": score,
            "hv": float(
                max(
                    0.0, metrics.minimum_normalized_magnetic_gradient_scale_length - 1.0
                )
            ),
            "metrics": metrics.model_dump(),
            "settings": settings.model_dump(),
            "constraint_margins": constraint_margins,
            "max_violation": feasibility,
        }

    return _evaluate_cached_stage(
        boundary_params,
        stage=stage,
        compute=compute,
        maximize=False,
        use_cache=use_cache,
    )


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

    return _CACHE_STATS[stage.lower()].copy()


def clear_evaluation_cache() -> None:
    """Reset the P1 evaluation cache and stats (useful for tests)."""

    _EVALUATION_CACHE.clear()
    _CACHE_STATS.clear()
