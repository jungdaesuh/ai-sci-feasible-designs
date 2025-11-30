"""Centralized forward model orchestrator for physics evaluations."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
import math
import time
from collections import defaultdict
from typing import Any, Dict, List, Mapping

import numpy as np
import pydantic
from constellaration import forward_model as constellaration_forward
from constellaration.geometry import surface_rz_fourier

logger = logging.getLogger(__name__)

# --- Caching & Constants ---

_EVALUATION_CACHE: Dict[str, "EvaluationResult"] = {}
_CACHE_STATS: Dict[str, int] = defaultdict(int)
_CANONICAL_PRECISION = 1e-8
_DEFAULT_ROUNDING = 1e-6
_DEFAULT_SCHEMA_VERSION = 1


def get_cache_stats() -> Dict[str, int]:
    """Return a copy of the cache statistics."""
    return dict(_CACHE_STATS)


def clear_cache() -> None:
    """Clear the evaluation cache."""
    _EVALUATION_CACHE.clear()
    _CACHE_STATS.clear()


# --- Data Structures ---


class ForwardModelSettings(pydantic.BaseModel):
    """Configuration for the forward model evaluation."""
    
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    constellaration_settings: constellaration_forward.ConstellarationSettings = (
        pydantic.Field(
            default_factory=constellaration_forward.ConstellarationSettings
        )
    )
    problem: str = "p3"  # p1, p2, p3, etc.
    stage: str = "unknown"
    calculate_gradients: bool = False
    fidelity: str = "low"


class EvaluationResult(pydantic.BaseModel):
    """Result of a physics evaluation."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    metrics: constellaration_forward.ConstellarationMetrics
    objective: float
    constraints: Dict[str, float]
    feasibility: float
    is_feasible: bool
    cache_hit: bool = False
    design_hash: str
    evaluation_time_sec: float
    settings: ForwardModelSettings

    # Telemetry & Diagnostics
    fidelity: str = "unknown"
    equilibrium_converged: bool = True
    error_message: str | None = None

    @property
    def constraint_names(self) -> List[str]:
        """List of constraint names (keys)."""
        return list(self.constraints.keys())

    @property
    def constraint_values(self) -> List[float]:
        """List of constraint values."""
        return list(self.constraints.values())

    def to_pareto_point(self) -> tuple[float, float]:
        """
        Return a tuple for Pareto optimization (objective, feasibility).
        """
        return (self.objective, self.feasibility)

    def dominates(self, other: "EvaluationResult") -> bool:
        """
        Check if this result dominates another.
        
        Assumes minimization for feasibility (0 is best).
        For objective, direction depends on problem, so this checks
        feasibility dominance only and equality on objective.
        """
        if self.feasibility > other.feasibility:
            return False
        # This is incomplete without objective direction, 
        # strictly returning False to avoid incorrect optimization.
        return False


# --- Helper Functions (Hashing & Boundary) ---


def _quantize_float(value: float, *, precision: float = _CANONICAL_PRECISION) -> float:
    if precision <= 0.0:
        return float(value)
    return float(round(value / precision) * precision)


def _canonicalize_value(value: Any, *, precision: float = _CANONICAL_PRECISION) -> Any:
    if isinstance(value, Mapping):
        return {
            k: _canonicalize_value(v, precision=precision)
            for k, v in sorted(value.items())
        }
    if isinstance(value, np.ndarray):
        return _canonicalize_value(value.tolist(), precision=precision)
    if isinstance(value, (list, tuple)):
        return [_canonicalize_value(v, precision=precision) for v in value]
    if isinstance(value, float):
        return _quantize_float(value, precision=precision)
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    return str(value)


def compute_design_hash(
    params: Mapping[str, Any],
    *,
    schema_version: int = _DEFAULT_SCHEMA_VERSION,
    rounding: float = _DEFAULT_ROUNDING,
) -> str:
    """Compute a canonical hash for the design parameters."""
    # Simplify params to ensure consistent hashing
    # We focus on the geometric coefficients
    
    r_cos = np.asarray(params.get("r_cos", []), dtype=float)
    z_sin = np.asarray(params.get("z_sin", []), dtype=float)
    
    # Determine dimensions for schema
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

    payload = {
        "schema_version": schema_version,
        "mpol": mpol,
        "ntor": ntor,
        "rounding": rounding,
        "params": params,
    }
    
    normalized = _canonicalize_value(payload, precision=rounding)
    digest = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(digest.encode("utf-8")).hexdigest()


def make_boundary_from_params(
    params: Mapping[str, Any],
) -> surface_rz_fourier.SurfaceRZFourier:
    """Construct a SurfaceRZFourier boundary from a parameter dictionary."""
    payload: dict[str, Any] = {
        "r_cos": np.asarray(params["r_cos"], dtype=float),
        "z_sin": np.asarray(params["z_sin"], dtype=float),
        "is_stellarator_symmetric": bool(
            params.get("is_stellarator_symmetric", True)
        ),
        "n_field_periods": int(params.get("n_field_periods", 1)),
    }

    if "r_sin" in params:
        payload["r_sin"] = np.asarray(params["r_sin"], dtype=float)
    if "z_cos" in params:
        payload["z_cos"] = np.asarray(params["z_cos"], dtype=float)
    if "nfp" in params:
        payload.setdefault("n_field_periods", int(params["nfp"]))

    return surface_rz_fourier.SurfaceRZFourier(**payload)


# --- Feasibility & Objective Logic ---


def _log10_or_large(value: float | None) -> float:
    if value is None or value <= 0.0:
        return 10.0
    return float(math.log10(value))


def compute_constraint_margins(
    metrics: constellaration_forward.ConstellarationMetrics,
    problem: str,
) -> Dict[str, float]:
    """Compute margins for constraints based on the problem definition.
    
    Positive margin indicates violation.
    """
    metrics_map = metrics.model_dump() if hasattr(metrics, "model_dump") else dict(metrics)
    problem_key = problem.lower()
    
    def _log10_margin(target: float) -> float:
        return _log10_or_large(metrics_map.get("qi")) - target

    margins: Dict[str, float] = {}

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
    else: # Default to P3 logic
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


def compute_objective(
    metrics: constellaration_forward.ConstellarationMetrics,
    problem: str,
) -> float:
    """Compute the primary objective function value."""
    problem_key = problem.lower()
    
    if problem_key.startswith("p1"):
        return float(metrics.max_elongation)
    elif problem_key.startswith("p2"):
        return float(metrics.minimum_normalized_magnetic_gradient_scale_length)
    else: # P3
        return float(metrics.aspect_ratio)


def max_violation(margins: Mapping[str, float]) -> float:
    if not margins:
        return float("inf")
    return float(max(0.0, *[max(0.0, value) for value in margins.values()]))


# --- Main Orchestrator ---


def forward_model(
    boundary: Mapping[str, Any],
    settings: ForwardModelSettings,
    *,
    use_cache: bool = True,
) -> EvaluationResult:
    """
    Single entry point for all physics evaluations.

    Handles:
    - Cache lookup
    - Boundary validation
    - Calling constellaration forward_model
    - Result packaging
    """
    start_time = time.time()
    
    # 1. Compute Design Hash
    d_hash = compute_design_hash(boundary)
    
    # 2. Check Cache
    # We key the cache by hash AND problem settings to avoid collisions if settings change
    # But for simplicity and strict adherence to design hash, we might just use design hash if settings are standard.
    # However, different settings produce different metrics.
    # We'll construct a cache key that includes relevant settings.
    
    # For now, we use a simplified key strategy: hash + problem + settings_hash
    settings_dict = settings.model_dump()
    # Canonicalize to handle numpy arrays and floats
    canonical_settings = _canonicalize_value(settings_dict, precision=_DEFAULT_ROUNDING)
    settings_json = json.dumps(canonical_settings, sort_keys=True, separators=(",", ":"))
    settings_hash = hashlib.sha256(settings_json.encode("utf-8")).hexdigest()
    cache_key = f"{d_hash}:{settings_hash}"
    
    if use_cache and cache_key in _EVALUATION_CACHE:
        _CACHE_STATS["hits"] += 1
        result = _EVALUATION_CACHE[cache_key]
        # Update evaluation time to reflect retrieval? No, keep original.
        # Return a copy? Pydantic models are mutable by default but we should treat as immutable.
        # We return a new instance with cache_hit=True
        return result.model_copy(update={"cache_hit": True})

    _CACHE_STATS["misses"] += 1

    # 3. Prepare Boundary
    try:
        surf = make_boundary_from_params(boundary)
    except Exception as e:
        logger.error(f"Failed to create boundary: {e}")
        raise ValueError(f"Invalid boundary parameters: {e}") from e

    # 4. Run Physics Model
    try:
        metrics, _ = constellaration_forward.forward_model(
            boundary=surf,
            settings=settings.constellaration_settings,
        )
    except Exception as e:
        logger.error(f"Physics evaluation failed: {e}")
        # We might want to return a penalized result or re-raise.
        # For robust orchestration, re-raising allows the caller to handle it.
        raise RuntimeError(f"Physics evaluation failed: {e}") from e

    # 5. Compute Derived Values
    objective = compute_objective(metrics, settings.problem)
    constraints = compute_constraint_margins(metrics, settings.problem)
    feasibility = max_violation(constraints)
    is_feasible = feasibility <= 1e-2 # Tolerance

    evaluation_time = time.time() - start_time

    result = EvaluationResult(
        metrics=metrics,
        objective=objective,
        constraints=constraints,
        feasibility=feasibility,
        is_feasible=is_feasible,
        cache_hit=False,
        design_hash=d_hash,
        evaluation_time_sec=evaluation_time,
        settings=settings,
        fidelity=settings.fidelity,
        equilibrium_converged=True,
        error_message=None,
    )

    # 6. Update Cache
    if use_cache:
        _EVALUATION_CACHE[cache_key] = result

    return result


def forward_model_batch(
    boundaries: List[Mapping[str, Any]],
    settings: ForwardModelSettings,
    *,
    n_workers: int = 4,
    use_cache: bool = True,
) -> List[EvaluationResult]:
    """Parallel batch evaluation."""
    results: List[EvaluationResult] = [None] * len(boundaries)
    
    # Identify indices needing computation vs cache
    # Since forward_model handles caching internally, we can just dispatch all.
    # However, we can optimization by checking cache first serially to avoid thread overhead
    # if we wanted. But standard ThreadPoolExecutor is fine.

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        future_to_idx = {
            executor.submit(
                forward_model, 
                boundary=b, 
                settings=settings, 
                use_cache=use_cache
            ): i
            for i, b in enumerate(boundaries)
        }
        
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                logger.error(f"Batch evaluation failed for index {idx}: {exc}")
                # We need to decide how to handle batch failures. 
                # Return a "failed" result or raise?
                # For now, we raise to be safe, or we could insert a dummy failed result.
                # Given the type signature, we must return EvaluationResult.
                # Let's re-raise.
                raise exc
                
    return results
