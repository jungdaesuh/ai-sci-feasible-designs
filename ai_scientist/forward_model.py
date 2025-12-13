"""Centralized forward model orchestrator for physics evaluations."""

from __future__ import annotations

import concurrent.futures
import hashlib
import importlib.util
import json
import logging
import math
import os
from collections import defaultdict
from typing import Any, Dict, List, Mapping

import numpy as np
import pydantic

# Re-export ConstellarationSettings for backwards compatibility.
# Many callers (tools/evaluation.py, cycle_executor.py, etc.) reference
# forward_model.ConstellarationSettings.
try:
    from constellaration.forward_model import ConstellarationSettings
except ImportError:
    # Provide a stub if constellaration is not installed
    # This allows the module to load without constellaration,
    # but callers will get a clear error if they try to use it
    ConstellarationSettings = None  # type: ignore[misc, assignment]

from ai_scientist.backends.base import PhysicsBackend


def _is_constellaration_available() -> bool:
    """Return True if the real constellaration backend is importable.

    This uses `importlib.util.find_spec` to avoid importing constellaration at
    module import time (which can trigger native builds of vmecpp on some
    machines).
    """
    return importlib.util.find_spec("constellaration.forward_model") is not None


logger = logging.getLogger(__name__)

# --- Caching & Constants ---

_EVALUATION_CACHE: Dict[str, "EvaluationResult"] = {}
_CACHE_STATS: Dict[str, int] = defaultdict(int)
_CANONICAL_PRECISION = 1e-8
_DEFAULT_ROUNDING = 1e-6
_DEFAULT_SCHEMA_VERSION = 1

# --- Backend Registry ---

_BACKEND: PhysicsBackend | None = None


def get_backend() -> "PhysicsBackend":
    """Get the current physics backend, auto-selecting if needed.

    Returns:
        The active PhysicsBackend instance.

    Note:
        If no backend is set, auto-selects based on:
        1. AI_SCIENTIST_PHYSICS_BACKEND environment variable
        2. Real backend if constellaration is available
        3. Mock backend as fallback
    """
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = _auto_select_backend()
    return _BACKEND


def set_backend(backend: PhysicsBackend | str) -> None:
    """Set the physics backend explicitly.

    Args:
        backend: Either a PhysicsBackend instance or a string name
                 ("mock", "real", "auto").

    Example:
        >>> from ai_scientist.backends import MockPhysicsBackend
        >>> set_backend(MockPhysicsBackend())
        >>> # or
        >>> set_backend("mock")
    """
    global _BACKEND
    if isinstance(backend, str):
        backend = _create_backend(backend)
    _BACKEND = backend
    logger.info(f"Physics backend set to: {_BACKEND.name}")


def reset_backend() -> None:
    """Reset the backend to trigger auto-selection on next use."""
    global _BACKEND
    _BACKEND = None


def _create_backend(name: str) -> "PhysicsBackend":
    """Create a backend by name."""
    from ai_scientist.backends.mock import MockPhysicsBackend

    name = name.lower()
    if name == "mock":
        return MockPhysicsBackend()
    elif name == "real":
        from ai_scientist.backends.real import RealPhysicsBackend

        return RealPhysicsBackend()
    elif name == "auto":
        return _auto_select_backend()
    else:
        raise ValueError(f"Unknown backend: {name!r}. Use 'mock', 'real', or 'auto'.")


def _auto_select_backend() -> "PhysicsBackend":
    """Auto-select backend based on environment and availability."""
    from ai_scientist.backends.mock import MockPhysicsBackend

    # Check environment variable first
    env_backend = os.environ.get("AI_SCIENTIST_PHYSICS_BACKEND", "auto").lower()

    if env_backend == "mock":
        logger.debug("Using mock backend (AI_SCIENTIST_PHYSICS_BACKEND=mock)")
        return MockPhysicsBackend()
    elif env_backend == "real":
        from ai_scientist.backends.real import RealPhysicsBackend

        backend = RealPhysicsBackend()
        if not backend.is_available():
            logger.warning(
                "AI_SCIENTIST_PHYSICS_BACKEND=real but constellaration not available, "
                "falling back to mock"
            )
            return MockPhysicsBackend()
        logger.debug("Using real backend (AI_SCIENTIST_PHYSICS_BACKEND=real)")
        return backend
    else:  # auto
        # Try real first, fall back to mock
        try:
            from ai_scientist.backends.real import RealPhysicsBackend

            backend = RealPhysicsBackend()
            if _is_constellaration_available() and backend.is_available():
                logger.debug("Auto-selected real physics backend")
                return backend
        except ImportError:
            pass
        logger.debug("Auto-selected mock physics backend")
        return MockPhysicsBackend()


def get_cache_stats() -> Dict[str, int]:
    """Return a copy of the cache statistics."""
    return dict(_CACHE_STATS)


def clear_cache() -> None:
    """Clear the evaluation cache."""
    _EVALUATION_CACHE.clear()
    _CACHE_STATS.clear()


def _process_worker_initializer() -> None:
    """Limit OpenMP threads inside process workers (Phase 5 observability safeguard)."""
    os.environ["OMP_NUM_THREADS"] = "1"


# --- Data Structures ---


class ForwardModelSettings(pydantic.BaseModel):
    """Configuration for the forward model evaluation."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    constellaration_settings: Any = None
    problem: str = "p3"  # p1, p2, p3, etc.
    stage: str = "unknown"
    calculate_gradients: bool = False
    fidelity: str = "low"

    # Pre-relaxation (StellarForge Phase 2)
    prerelax: bool = False
    prerelax_steps: int = 50
    prerelax_lr: float = 1e-2


class EvaluationResult(pydantic.BaseModel):
    """Result of a physics evaluation."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    # Core metrics from constellaration (Any to avoid type issues when module unavailable)
    metrics: Any

    # Optimization-relevant fields
    objective: float
    constraints: List[float]
    constraint_names: List[str]
    feasibility: float  # max(0, max_constraint_violation)
    is_feasible: bool

    # Caching
    cache_hit: bool = False
    design_hash: str

    # Telemetry
    evaluation_time_sec: float
    fidelity: str  # "low", "medium", "high"
    settings: ForwardModelSettings

    # Optional equilibrium data
    equilibrium_converged: bool = True
    error_message: str | None = None

    @property
    def constraints_map(self) -> Dict[str, float]:
        """Map of constraint names to values."""
        return dict(zip(self.constraint_names, self.constraints))

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
) -> Any:
    """Construct a SurfaceRZFourier boundary from a parameter dictionary."""
    from constellaration.geometry import surface_rz_fourier

    payload: dict[str, Any] = {
        "r_cos": np.asarray(params["r_cos"], dtype=float),
        "z_sin": np.asarray(params["z_sin"], dtype=float),
        "is_stellarator_symmetric": bool(params.get("is_stellarator_symmetric", True)),
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
    metrics: Any,
    problem: str,
    *,
    stage: str = "high",
) -> Dict[str, float]:
    """Compute margins for constraints based on the problem definition.

    Positive margin indicates violation.

    Important: these margins are **normalized** to match the benchmark-style
    feasibility tolerance used across the codebase (typically `1e-2`), mirroring
    the normalization in `constellaration.problems.*._normalized_constraint_violations`.

    Note: The returned dict keys use canonical constraint names from
    `ai_scientist.constraints`. These map to full metrics field names via
    `get_metrics_key()`.

    Args:
        metrics: Metrics object or dict
        problem: Problem type (p1, p2, p3)
        stage: Fidelity stage. Low-fidelity stages skip expensive constraints.
               Valid values: "screen", "low", "default" (low fidelity)
                            "promote", "high", "p2", "p3" (high fidelity)
    """
    metrics_map = (
        metrics.model_dump() if hasattr(metrics, "model_dump") else dict(metrics)
    )
    problem_key = problem.lower()
    is_low_fidelity = stage.lower() in ("screen", "low", "default", "promote")

    def _log10_margin(target: float) -> float:
        denom = abs(target) if abs(target) > 0.0 else 1.0
        return (_log10_or_large(metrics_map.get("qi")) - target) / denom

    def _upper_bound_margin(value: float, limit: float) -> float:
        denom = abs(limit) if abs(limit) > 0.0 else 1.0
        return (value - limit) / denom

    def _lower_bound_margin(value: float, limit: float) -> float:
        denom = abs(limit) if abs(limit) > 0.0 else 1.0
        return (limit - value) / denom

    margins: Dict[str, float] = {}

    if problem_key.startswith("p1"):
        ar = float(metrics_map.get("aspect_ratio", float("nan")))
        tri = float(metrics_map.get("average_triangularity", float("nan")))
        iota = float(
            metrics_map.get(
                "edge_rotational_transform_over_n_field_periods", float("nan")
            )
        )
        margins = {
            "aspect_ratio": _upper_bound_margin(ar, 4.0),
            "average_triangularity": _upper_bound_margin(tri, -0.5),
            "edge_rotational_transform": _lower_bound_margin(iota, 0.3),
        }
    elif problem_key.startswith("p2"):
        ar = float(metrics_map.get("aspect_ratio", float("nan")))
        iota = float(
            metrics_map.get(
                "edge_rotational_transform_over_n_field_periods", float("nan")
            )
        )
        mirror = float(metrics_map.get("edge_magnetic_mirror_ratio", float("nan")))
        elong = float(metrics_map.get("max_elongation", float("nan")))
        # Geometric constraints (always required)
        margins = {
            "aspect_ratio": _upper_bound_margin(ar, 10.0),
            "edge_rotational_transform": _lower_bound_margin(iota, 0.25),
            "edge_magnetic_mirror_ratio": _upper_bound_margin(mirror, 0.2),
            "max_elongation": _upper_bound_margin(elong, 5.0),
        }
        # Physics constraints (only at high fidelity)
        if not is_low_fidelity:
            margins["qi_log10"] = _log10_margin(-4.0)
    else:  # Default to P3 logic
        iota = float(
            metrics_map.get(
                "edge_rotational_transform_over_n_field_periods", float("nan")
            )
        )
        mirror = float(metrics_map.get("edge_magnetic_mirror_ratio", float("nan")))
        # Geometric constraints (always required)
        margins = {
            "edge_rotational_transform": _lower_bound_margin(iota, 0.25),
            "edge_magnetic_mirror_ratio": _upper_bound_margin(mirror, 0.25),
        }
        # Physics constraints (only at high fidelity)
        if not is_low_fidelity:
            well = metrics_map.get("vacuum_well")
            well_value = float(well) if well is not None else float("nan")
            # Normalize like constellaration.problems.MHDStableQIStellarator:
            # denom = max(1e-1, vacuum_well_lower_bound) where lower bound is 0.0.
            margins["vacuum_well"] = _lower_bound_margin(well_value, 0.0) / 0.1

            flux_value = metrics_map.get("flux_compression_in_regions_of_bad_curvature")
            flux = float(flux_value) if flux_value is not None else float("nan")
            margins["flux_compression"] = _upper_bound_margin(flux, 0.9)

            margins["qi_log10"] = _log10_margin(-3.5)

    return margins


def compute_objective(
    metrics: Any,
    problem: str,
) -> float:
    """Compute the primary PHYSICS objective function value.

    IMPORTANT: This returns the benchmark-standard physics objective,
    which is NOT the same as:
    - ALM objective (what the optimizer minimizes: 20 - gradient for P2/P3)
    - Ranking score (what surrogate predicts: gradient / aspect for P2/P3)

    See ai_scientist.objective_types for the full vocabulary.

    Args:
        metrics: Metrics object from physics evaluation.
        problem: Problem identifier (p1, p2, p3).

    Returns:
        Physics objective value. Direction varies by problem:
        - P1: max_elongation (minimize)
        - P2: gradient (maximize)
        - P3: aspect_ratio (minimize)
    """
    problem_key = problem.lower()

    if problem_key.startswith("p1"):
        return float(metrics.max_elongation)
    elif problem_key.startswith("p2"):
        return float(metrics.minimum_normalized_magnetic_gradient_scale_length)
    else:  # P3
        return float(metrics.aspect_ratio)


def max_violation(margins: Mapping[str, float]) -> float:
    """Return maximum constraint violation.

    If any margin is non-finite (NaN/inf), returns inf (conservative infeasibility).
    """
    if not margins:
        return float("inf")
    # If ANY margin is non-finite, entire feasibility is undefined → infeasible
    for value in margins.values():
        if not math.isfinite(value):
            return float("inf")
    return float(max(0.0, max(margins.values())))


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
    settings_json = json.dumps(
        canonical_settings, sort_keys=True, separators=(",", ":")
    )
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

    # 3. Delegate to the pluggable backend
    # This allows MockPhysicsBackend to handle evaluation in tests
    backend = get_backend()
    result = backend.evaluate(boundary, settings)

    # 4. Update Cache
    if use_cache:
        _EVALUATION_CACHE[cache_key] = result

    return result


def forward_model_batch(
    boundaries: List[Mapping[str, Any]],
    settings: ForwardModelSettings,
    *,
    n_workers: int = 4,
    pool_type: str = "thread",
    use_cache: bool = True,
) -> List[EvaluationResult]:
    """Parallel batch evaluation."""
    results: List[EvaluationResult | None] = [None] * len(boundaries)

    executor_cls = (
        concurrent.futures.ThreadPoolExecutor
        if pool_type == "thread"
        else concurrent.futures.ProcessPoolExecutor
    )

    executor_kwargs: Dict[str, Any] = {"max_workers": n_workers}
    if executor_cls is concurrent.futures.ProcessPoolExecutor:
        executor_kwargs["initializer"] = _process_worker_initializer

    with executor_cls(**executor_kwargs) as executor:
        future_to_idx = {
            executor.submit(
                forward_model, boundary=b, settings=settings, use_cache=use_cache
            ): i
            for i, b in enumerate(boundaries)
        }

        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                logger.error(f"Batch evaluation failed for index {idx}: {exc}")
                # Return a penalized result instead of raising
                # We need to construct a "failed" EvaluationResult
                # We'll use a helper or manual construction
                # Since we don't have the boundary here easily (it's in boundaries[idx]),
                # we can use a placeholder hash or recompute it if needed.
                # But EvaluationResult requires metrics.

                # Create dummy metrics with penalized values
                from ai_scientist.backends.mock import MockMetrics

                dummy_metrics = MockMetrics(
                    aspect_ratio=float("inf"),
                    max_elongation=float("inf"),
                    minimum_normalized_magnetic_gradient_scale_length=0.0,
                    mirror_ratio=float("inf"),
                    vmec_converged=False,
                )

                # Determine penalty direction based on problem
                # This duplicates logic from _penalized_result in tools/evaluation.py
                # But we are in forward_model.py now.
                maximize = settings.problem.lower().startswith("p2")
                penalty = -1e9 if maximize else 1e9

                results[idx] = EvaluationResult(
                    metrics=dummy_metrics,
                    objective=penalty,
                    constraints=[],
                    constraint_names=[],
                    feasibility=float("inf"),
                    is_feasible=False,
                    cache_hit=False,
                    design_hash="error",  # Placeholder
                    evaluation_time_sec=0.0,
                    settings=settings,
                    fidelity=settings.fidelity,
                    equilibrium_converged=False,
                    error_message=str(exc),
                )

    return results
