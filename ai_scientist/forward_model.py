"""Centralized forward model orchestrator for physics evaluations."""

from __future__ import annotations

import concurrent.futures
import hashlib
import importlib.util
import json
import logging
import math
import os
import threading
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Protocol, runtime_checkable

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
from ai_scientist.constraints import get_constraint_bounds


# --- Type Protocol for Metrics ---
# Fixes Issue #13: Provides type safety for metrics objects from different backends
# (constellaration.Metrics, MockMetrics, or dict fallback)


@runtime_checkable
class MetricsProtocol(Protocol):
    """Protocol defining the interface for physics metrics.

    This protocol allows type-safe handling of metrics from different sources:
    - constellaration.forward_model.Metrics (real physics)
    - ai_scientist.backends.mock.MockMetrics (testing)
    - Dict fallback (when module unavailable)

    Required attributes are those used by compute_objective(), compute_constraint_margins(),
    and related functions.
    """

    @property
    def aspect_ratio(self) -> float:
        """Aspect ratio (major/minor radius). Lower = more compact."""
        ...

    @property
    def max_elongation(self) -> float:
        """Maximum elongation of plasma cross-section."""
        ...

    @property
    def minimum_normalized_magnetic_gradient_scale_length(self) -> float:
        """Gradient scale length. Higher = simpler coils."""
        ...

    @property
    def edge_magnetic_mirror_ratio(self) -> float:
        """Edge magnetic mirror ratio."""
        ...


# Type alias for metrics: accepts Protocol-compliant objects or dict fallback
# This provides type safety while maintaining compatibility with different sources:
# - constellaration.forward_model.Metrics (real physics)
# - ai_scientist.backends.mock.MockMetrics (testing)
# - Dict fallback (when module unavailable or from model_dump())
# Note: dict uses Any because model_dump() returns mixed types (float, bool, etc.)
MetricsLike = MetricsProtocol | Dict[str, Any]


def _metrics_to_dict(metrics: MetricsLike) -> Dict[str, Any]:
    """Convert metrics to dict in a type-safe way.

    Handles both pydantic models (with model_dump()) and plain dicts.
    This helper resolves pyright errors when using Protocol | dict unions.
    """
    if isinstance(metrics, dict):
        return metrics
    if hasattr(metrics, "model_dump"):
        return metrics.model_dump()  # type: ignore[union-attr]
    # Fallback: try to extract known attributes
    result: Dict[str, Any] = {}
    for attr in (
        "aspect_ratio",
        "max_elongation",
        "minimum_normalized_magnetic_gradient_scale_length",
        "edge_magnetic_mirror_ratio",
        "edge_rotational_transform_over_n_field_periods",
        "average_triangularity",
        "vacuum_well",
        "flux_compression_in_regions_of_bad_curvature",
        "qi",
    ):
        if hasattr(metrics, attr):
            result[attr] = getattr(metrics, attr)
    return result


def _is_constellaration_available() -> bool:
    """Return True if the real constellaration backend is importable.

    This uses `importlib.util.find_spec` to avoid importing constellaration at
    module import time (which can trigger native builds of vmecpp on some
    machines).
    """
    return importlib.util.find_spec("constellaration.forward_model") is not None


logger = logging.getLogger(__name__)

# --- Caching & Constants ---

# LRU Cache with configurable max size (fixes unbounded memory growth)
_CACHE_MAX_SIZE_DEFAULT = 10000


def _get_cache_max_size() -> int:
    """Get cache max size from environment or default."""
    return int(
        os.environ.get("AI_SCIENTIST_CACHE_MAX_SIZE", str(_CACHE_MAX_SIZE_DEFAULT))
    )


class _LRUCache:
    """Thread-safe LRU cache with max size eviction policy.

    Uses collections.OrderedDict for zero external dependencies.
    Move-to-end on access provides LRU semantics.
    All operations are protected by a threading.Lock for thread safety.
    """

    def __init__(self, maxsize: int = _CACHE_MAX_SIZE_DEFAULT):
        from collections import OrderedDict

        self._cache: "OrderedDict[str, EvaluationResult]" = OrderedDict()
        self._maxsize = maxsize
        self._lock = threading.Lock()

    def get(self, key: str) -> "EvaluationResult | None":
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                return self._cache[key]
            return None

    def set(self, key: str, value: "EvaluationResult") -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = value
            # Evict oldest entries if over capacity
            while len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._cache

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    @property
    def maxsize(self) -> int:
        return self._maxsize


_EVALUATION_CACHE: _LRUCache = _LRUCache(_get_cache_max_size())
_CACHE_STATS: Dict[str, int] = defaultdict(int)
_CANONICAL_PRECISION = 1e-8
_DEFAULT_ROUNDING = 1e-9
_DEFAULT_SCHEMA_VERSION = 1

# --- Backend Registry ---

_BACKEND: PhysicsBackend | None = None

# --- Batch Executor Singleton (Issue A7 Fix) ---
# Avoids re-creating ProcessPoolExecutor on every batch call, which incurs
# ~100-500ms spawn overhead on macOS. The executor is lazily initialized.

_BATCH_EXECUTOR: concurrent.futures.Executor | None = None
_BATCH_EXECUTOR_LOCK = threading.Lock()
_BATCH_EXECUTOR_CONFIG: Dict[str, Any] = {}  # Stores n_workers, pool_type


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


def _is_cache_disabled() -> bool:
    """Check if cache is disabled via environment variable."""
    return os.environ.get("AI_SCIENTIST_DISABLE_CACHE", "").lower() in (
        "1",
        "true",
        "yes",
    )


def get_cache_stats() -> Dict[str, int]:
    """Return a copy of the cache statistics (hits and misses)."""
    return dict(_CACHE_STATS)


def get_cache_info() -> Dict[str, int | bool]:
    """Return comprehensive cache information for observability.

    Returns:
        Dict containing:
            - size: Current number of cached entries
            - maxsize: Maximum cache capacity
            - hits: Number of cache hits
            - misses: Number of cache misses
            - disabled: Whether cache is disabled via env var
    """
    return {
        "size": len(_EVALUATION_CACHE),
        "maxsize": _EVALUATION_CACHE.maxsize,
        "hits": _CACHE_STATS.get("hits", 0),
        "misses": _CACHE_STATS.get("misses", 0),
        "disabled": _is_cache_disabled(),
    }


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
        """Pareto dominance check (not implemented).

        Raises:
            NotImplementedError: Pareto dominance requires knowing
                objective direction (minimize/maximize), which is
                problem-dependent. Use tools.hypervolume for Pareto
                analysis instead.
        """
        raise NotImplementedError(
            "EvaluationResult.dominates() requires objective direction context. "
            "Use ai_scientist.tools.hypervolume.summarize_p3_candidates() for "
            "Pareto front analysis."
        )


# --- Helper Functions (Hashing & Boundary) ---


def _quantize_float(value: float, *, precision: float = _CANONICAL_PRECISION) -> float:
    if precision <= 0.0:
        return float(value)
    # Hashing must be total: allow NaN/inf to flow through without crashing.
    # This can happen for invalid candidates during optimization.
    if not math.isfinite(value):
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
    exact: bool = False,
) -> str:
    """Compute a canonical hash for the design parameters.

    Args:
        params: Design parameters dictionary with r_cos, z_sin, etc.
        schema_version: Hash schema version for compatibility.
        rounding: Precision for coefficient quantization (default 1e-9).
        exact: If True, use no rounding (exact bytes) for optimization paths
               where even tiny differences must produce different hashes.
    """
    # For exact hashing (e.g., gradient optimization), disable rounding
    if exact:
        rounding = 0.0

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
    metrics: MetricsLike,
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
               Note: "promote" is intended to run high-fidelity VMEC but may
               skip Boozer/QI (see tools.evaluation._settings_for_stage).
    """
    metrics_map = _metrics_to_dict(metrics)
    problem_key = problem.lower()
    stage_key = stage.lower()

    # Stage gating:
    # - "screen"/"low"/"default": fast screening → only geometric constraints.
    # - "promote": high-fidelity VMEC but often configured to skip QI/Boozer.
    # - "p2"/"p3"/"high": full physics constraints.
    include_vmec_constraints = stage_key not in ("screen", "low", "default")
    include_qi_constraint = stage_key not in ("screen", "low", "default", "promote")

    # A5.2 FIX: Warn when 'promote' stage skips QI for problems that require it
    if stage_key == "promote" and problem_key in ("p2", "p3"):
        logger.warning(
            "Stage 'promote' skips QI constraint - feasibility check incomplete for %s. "
            "Use stage 'p2' or 'p3' for full constraint evaluation.",
            problem_key.upper(),
        )

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
        if include_qi_constraint:
            margins["qi"] = _log10_margin(-4.0)
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
        # Physics constraints (only when VMEC metrics are expected to be present)
        if include_vmec_constraints:
            well = metrics_map.get("vacuum_well")
            well_value = float(well) if well is not None else float("nan")
            # M4 FIX: Source normalization from CONSTRAINT_BOUNDS for SSOT
            # Normalize like constellaration.problems.MHDStableQIStellarator:
            # denom = max(1e-1, vacuum_well_lower_bound)
            p3_bounds = get_constraint_bounds("p3")
            well_lower = p3_bounds.get("vacuum_well_lower", 0.0)
            well_denom = max(0.1, abs(well_lower))
            # Constraint violation in natural units: (lower_bound - value)
            # Normalize once by denom = max(1e-1, abs(lower_bound)) to match benchmark.
            margins["vacuum_well"] = (well_lower - well_value) / well_denom

            flux_value = metrics_map.get("flux_compression_in_regions_of_bad_curvature")
            flux = float(flux_value) if flux_value is not None else float("nan")
            flux_upper = p3_bounds.get("flux_compression_upper", 0.9)
            margins["flux_compression"] = _upper_bound_margin(flux, flux_upper)

        if include_qi_constraint:
            margins["qi"] = _log10_margin(-3.5)

    return margins


def compute_objective(
    metrics: MetricsLike,
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
        # B1 NOTE: P3 is intrinsically multi-objective (minimize aspect_ratio AND
        # maximize gradient_scale_length). This returns only the primary objective
        # for backward compatibility. For Pareto analysis, use compute_p3_objectives().
        return float(metrics.aspect_ratio)


def compute_p3_objectives(
    metrics: MetricsLike,
) -> tuple[float, float]:
    """Compute both P3 objectives for multi-objective Pareto analysis (B1 fix).

    P3 is a multi-objective problem:
    - Minimize aspect_ratio (compactness)
    - Maximize gradient_scale_length (buildability)

    Args:
        metrics: Metrics object from physics evaluation.

    Returns:
        (aspect_ratio, gradient_scale_length) tuple for Pareto front analysis.
        First objective should be minimized, second should be maximized.
    """
    metrics_map = _metrics_to_dict(metrics)
    aspect = float(metrics_map.get("aspect_ratio", 10.0))
    gradient = float(
        metrics_map.get("minimum_normalized_magnetic_gradient_scale_length", 0.0)
    )
    return (aspect, gradient)


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
    # L1 FIX: Only include deterministic fields in cache key to avoid cache misses
    # from non-deterministic data in constellaration_settings (which is typed as Any
    # and may contain timestamps, random identifiers, or objects with unstable serialization)
    deterministic_settings = {
        "problem": settings.problem,
        "stage": settings.stage,
        "calculate_gradients": settings.calculate_gradients,
        "fidelity": settings.fidelity,
        "prerelax": settings.prerelax,
        "prerelax_steps": settings.prerelax_steps,
        "prerelax_lr": settings.prerelax_lr,
    }
    # Canonicalize to handle numpy arrays and floats
    canonical_settings = _canonicalize_value(
        deterministic_settings, precision=_DEFAULT_ROUNDING
    )
    settings_json = json.dumps(
        canonical_settings, sort_keys=True, separators=(",", ":")
    )
    settings_hash = hashlib.sha256(settings_json.encode("utf-8")).hexdigest()
    cache_key = f"{d_hash}:{settings_hash}"

    # Respect both use_cache parameter and AI_SCIENTIST_DISABLE_CACHE env var
    cache_enabled = use_cache and not _is_cache_disabled()

    if cache_enabled:
        cached_result = _EVALUATION_CACHE.get(cache_key)
        if cached_result is not None:
            _CACHE_STATS["hits"] += 1
            # Return a copy with cache_hit=True
            return cached_result.model_copy(update={"cache_hit": True})

    _CACHE_STATS["misses"] += 1

    # 3. Delegate to the pluggable backend
    # This allows MockPhysicsBackend to handle evaluation in tests
    backend = get_backend()
    result = backend.evaluate(boundary, settings)

    # 4. Update Cache (LRU eviction handled automatically)
    if cache_enabled:
        _EVALUATION_CACHE.set(cache_key, result)

    return result


def evaluate_boundary(
    boundary: Mapping[str, Any],
    settings: ForwardModelSettings,
    *,
    use_cache: bool = True,
) -> EvaluationResult:
    """Compatibility alias for integration tests and legacy call sites."""
    return forward_model(boundary, settings, use_cache=use_cache)


def _get_batch_executor(n_workers: int, pool_type: str) -> concurrent.futures.Executor:
    """Get or create the batch executor singleton.

    A7 Fix: Reuses executor across batch calls to avoid process spawn overhead.
    If config changes (n_workers/pool_type), shuts down old executor first.
    """
    global _BATCH_EXECUTOR, _BATCH_EXECUTOR_CONFIG

    with _BATCH_EXECUTOR_LOCK:
        current_cfg = {"n_workers": n_workers, "pool_type": pool_type}

        # If config changed, shutdown existing executor
        if _BATCH_EXECUTOR is not None and _BATCH_EXECUTOR_CONFIG != current_cfg:
            logger.info(
                "[batch_executor] Config changed, recreating executor: %s",
                current_cfg,
            )
            _BATCH_EXECUTOR.shutdown(wait=True)
            _BATCH_EXECUTOR = None

        if _BATCH_EXECUTOR is None:
            executor_cls = (
                concurrent.futures.ThreadPoolExecutor
                if pool_type == "thread"
                else concurrent.futures.ProcessPoolExecutor
            )
            executor_kwargs: Dict[str, Any] = {"max_workers": n_workers}
            if executor_cls is concurrent.futures.ProcessPoolExecutor:
                executor_kwargs["initializer"] = _process_worker_initializer

            _BATCH_EXECUTOR = executor_cls(**executor_kwargs)
            _BATCH_EXECUTOR_CONFIG = current_cfg
            logger.info(
                "[batch_executor] Created new %s with %d workers",
                pool_type,
                n_workers,
            )

        return _BATCH_EXECUTOR


def shutdown_batch_executor() -> None:
    """Gracefully shutdown the batch executor singleton.

    Call this during application shutdown or when changing executor config.
    """
    global _BATCH_EXECUTOR
    with _BATCH_EXECUTOR_LOCK:
        if _BATCH_EXECUTOR is not None:
            logger.info("[batch_executor] Shutting down batch executor...")
            _BATCH_EXECUTOR.shutdown(wait=True)
            _BATCH_EXECUTOR = None


def forward_model_batch(
    boundaries: List[Mapping[str, Any]],
    settings: ForwardModelSettings,
    *,
    n_workers: int = 4,
    pool_type: str = "thread",
    use_cache: bool = True,
) -> List[EvaluationResult]:
    """Parallel batch evaluation of multiple boundary configurations.

    A7 Fix: Now uses a persistent executor singleton to avoid per-call
    process spawn overhead (~100-500ms on macOS).

    Args:
        boundaries: List of boundary parameter dictionaries.
        settings: ForwardModelSettings configuration.
        n_workers: Number of parallel workers (default: 4).
        pool_type: Executor type - "thread" or "process" (default: "thread").
        use_cache: Whether to use evaluation cache (default: True).

    Returns:
        List of EvaluationResult objects, one per boundary.

    Note on GIL and Parallelism (Issue #15):
        Python's Global Interpreter Lock (GIL) limits true parallelism for
        CPU-bound Python code in ThreadPoolExecutor. However:

        - If constellaration/VMEC++ releases the GIL (native C++ code does),
          ThreadPoolExecutor achieves true parallelism and is preferred due to
          lower memory overhead and shared cache access.

        - For pure Python compute or if GIL contention is observed, switch to
          pool_type="process" for true parallelism at the cost of:
          - Higher memory usage (separate process per worker)
          - Serialization overhead for data transfer
          - Separate caches per process

        Recommendation: Start with "thread" (default). If profiling shows GIL
        contention (workers waiting despite available CPU), switch to "process".
    """
    results: List[EvaluationResult | None] = [None] * len(boundaries)

    # A7 FIX: Use persistent executor singleton instead of per-call creation
    executor = _get_batch_executor(n_workers, pool_type)
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
