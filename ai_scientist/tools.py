"""Physics tool wrappers for the ConStellaration AI Scientist."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

import numpy as np
from pymoo.indicators import hv as pymoo_hv

from ai_scientist import rag
from constellaration import forward_model
from constellaration.geometry import surface_rz_fourier

_DEFAULT_RELATIVE_TOLERANCE = 1e-2
_CANONICAL_PRECISION = 1e-8
_EVALUATION_CACHE: Dict[Tuple[str, str], Dict[str, Any]] = {}
_CACHE_STATS: Dict[str, Dict[str, int]] = defaultdict(lambda: {"hits": 0, "misses": 0})
_P3_REFERENCE_POINT: Tuple[float, float] = (1.0, 20.0)


@dataclass(frozen=True)
class BoundaryParams:
    """Container for surface parameters that may evolve in future waves."""

    params: Mapping[str, Any]


@dataclass(frozen=True)
class P3Summary:
    """Compact summary of the per-cycle P3 pareto front and hypervolume."""

    hv_score: float
    reference_point: Tuple[float, float]
    feasible_count: int
    archive_size: int
    pareto_entries: Tuple["ParetoEntry", ...]


@dataclass(frozen=True)
class ParetoEntry:
    design_hash: str
    seed: int
    stage: str
    gradient: float
    aspect_ratio: float
    objective: float
    feasibility: float

    def as_mapping(self) -> Mapping[str, float]:
        return {
            "seed": float(self.seed),
            "gradient": self.gradient,
            "aspect_ratio": self.aspect_ratio,
            "objective": self.objective,
            "feasibility": self.feasibility,
        }


def _quantize_float(value: float) -> float:
    return float(round(value / _CANONICAL_PRECISION) * _CANONICAL_PRECISION)


def _canonicalize_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _canonicalize_value(v) for k, v in sorted(value.items())}
    if isinstance(value, np.ndarray):
        return _canonicalize_value(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_canonicalize_value(v) for v in value]
    if isinstance(value, float):
        return _quantize_float(value)
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    return str(value)


def _hash_params(params: Mapping[str, Any]) -> str:
    return design_hash(params)


def design_hash(params: Mapping[str, Any] | BoundaryParams) -> str:
    params_map = _ensure_mapping(params)
    normalized = _canonicalize_value(params_map)
    digest = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(digest.encode("utf-8")).hexdigest()


def _ensure_mapping(params: Mapping[str, Any] | BoundaryParams) -> Mapping[str, Any]:
    if isinstance(params, BoundaryParams):
        return params.params
    return params


def _evaluate_cached_stage(
    boundary_params: Mapping[str, Any] | BoundaryParams,
    *,
    stage: str,
    compute: Callable[[Mapping[str, Any]], Dict[str, Any]],
    use_cache: bool = True,
) -> Dict[str, Any]:
    params_map = _ensure_mapping(boundary_params)
    stage_lower = stage.lower()
    cache_key = (stage_lower, _hash_params(params_map))
    stats = _CACHE_STATS[stage_lower]
    if use_cache:
        cached = _EVALUATION_CACHE.get(cache_key)
        if cached is not None:
            stats["hits"] += 1
            return cached

    stats["misses"] += 1
    result = compute(params_map)
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


def _log10_or_large(value: float | None) -> float:
    if value is None or value <= 0.0:
        return 10.0
    return float(math.log10(value))


def _gradient_score(metrics: forward_model.ConstellarationMetrics) -> float:
    gradient = float(metrics.minimum_normalized_magnetic_gradient_scale_length)
    aspect = float(metrics.aspect_ratio)
    return float(gradient / max(1.0, aspect))


def _p2_feasibility(metrics: forward_model.ConstellarationMetrics) -> float:
    violations = [
        metrics.aspect_ratio - 10.0,
        max(0.0, 0.25 - metrics.edge_rotational_transform_over_n_field_periods),
        metrics.edge_magnetic_mirror_ratio - 0.2,
        max(0.0, metrics.max_elongation - 5.0),
        max(0.0, _log10_or_large(metrics.qi) + 4.0),
    ]
    return float(max(violations + [0.0]))


def _p3_feasibility(metrics: forward_model.ConstellarationMetrics) -> float:
    flux_violation = 0.0
    if metrics.flux_compression_in_regions_of_bad_curvature is not None:
        flux_violation = max(
            0.0, metrics.flux_compression_in_regions_of_bad_curvature - 0.9
        )

    constraints = [
        max(0.0, 0.25 - metrics.edge_rotational_transform_over_n_field_periods),
        max(0.0, metrics.edge_magnetic_mirror_ratio - 0.25),
        max(0.0, -metrics.vacuum_well),
        flux_violation,
        max(0.0, _log10_or_large(metrics.qi) + 3.5),
    ]
    return float(max(constraints + [0.0]))


def _objective_vector(metrics: Mapping[str, Any]) -> Tuple[float, float]:
    gradient = float(metrics["minimum_normalized_magnetic_gradient_scale_length"])
    aspect = float(metrics["aspect_ratio"])
    return -gradient, aspect


def _extract_p3_point(metrics: Mapping[str, Any]) -> Tuple[float, float]:
    vector = _objective_vector(metrics)
    return -vector[0], vector[1]


def _dominates(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    """Return True if objective a Pareto dominates b (higher gradient, lower aspect)."""

    higher_gradient = a[0] >= b[0]
    lower_aspect = a[1] <= b[1]
    strict = a[0] > b[0] or a[1] < b[1]
    return higher_gradient and lower_aspect and strict


def _hypervolume_minimization(
    vectors: Sequence[Tuple[float, float]],
    reference_point: Tuple[float, float],
) -> float:
    if not vectors:
        return 0.0
    indicator = pymoo_hv.Hypervolume(ref_point=np.asarray(reference_point, dtype=float))
    output = indicator(np.asarray(vectors, dtype=float))
    return float(output if output is not None else 0.0)


def summarize_p3_candidates(
    candidates: Sequence[Mapping[str, Any] | dict[str, Any]],
    *,
    reference_point: Tuple[float, float] = _P3_REFERENCE_POINT,
) -> P3Summary:
    """Produce the hypervolume score and all non-dominated seeds for a candidate batch."""

    @dataclass(frozen=True)
    class _P3Entry:
        gradient: float
        aspect: float
        seed: int
        evaluation: Mapping[str, Any]
        feasibility: float
        design_hash: str
        design_hash: str

    entries: list[_P3Entry] = []
    for candidate in candidates:
        design_id = candidate.get("design_hash")
        if design_id is None:
            design_id = design_hash(candidate.get("params", {}))
        design_id = str(design_id)
        eval_metrics = candidate["evaluation"]["metrics"]
        gradient, aspect = _extract_p3_point(eval_metrics)
        seed = int(candidate.get("seed", -1))
        feasibility = float(candidate["evaluation"]["feasibility"])
        entries.append(
            _P3Entry(
                design_hash=design_id,
                gradient=gradient,
                aspect=aspect,
                seed=seed,
                evaluation=candidate["evaluation"],
                feasibility=feasibility,
            )
        )

    hv_vectors: list[Tuple[float, float]] = []
    for entry in entries:
        if entry.feasibility > _DEFAULT_RELATIVE_TOLERANCE:
            continue
        hv_vectors.append((-entry.gradient, entry.aspect))

    pareto_entries: list[ParetoEntry] = []
    for current_index, entry in enumerate(entries):
        if entry.feasibility > _DEFAULT_RELATIVE_TOLERANCE:
            continue
        point = (entry.gradient, entry.aspect)
        dominated = False
        for other_index, other in enumerate(entries):
            if other_index == current_index:
                continue
            if other.feasibility > _DEFAULT_RELATIVE_TOLERANCE:
                continue
            if _dominates((other.gradient, other.aspect), point):
                dominated = True
                break
        if dominated:
            continue
        pareto_entries.append(
            ParetoEntry(
                design_hash=entry.design_hash,
                seed=entry.seed,
                stage=str(entry.evaluation.get("stage", "")),
                gradient=entry.gradient,
                aspect_ratio=entry.aspect,
                objective=float(entry.evaluation["objective"]),
                feasibility=entry.feasibility,
            )
        )

    pareto_entries.sort(key=lambda item: (-item.gradient, item.aspect_ratio))
    return P3Summary(
        hv_score=_hypervolume_minimization(hv_vectors, reference_point),
        reference_point=reference_point,
        feasible_count=sum(
            1 for entry in entries if entry.feasibility <= _DEFAULT_RELATIVE_TOLERANCE
        ),
        archive_size=len(pareto_entries),
        pareto_entries=tuple(pareto_entries),
    )


def retrieve_rag(
    query: str, *, k: int = 3, index_path: Path | str | None = None
) -> list[dict[str, str]]:
    """Expose RAG retrieval via the ai_scientist/rag_index.db index (Phase 3)."""

    index = Path(index_path) if index_path is not None else rag.DEFAULT_INDEX_PATH
    return rag.retrieve(query=query, k=k, index_path=index)


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


def _constraint_metrics(
    metrics: forward_model.ConstellarationMetrics,
) -> Tuple[float, float]:
    targets = np.array([4.0, -0.5, 0.3], dtype=float)
    violations = np.array(
        [
            metrics.aspect_ratio - targets[0],
            metrics.average_triangularity - targets[1],
            targets[2] - metrics.edge_rotational_transform_over_n_field_periods,
        ],
        dtype=float,
    )
    normalized = violations / np.abs(targets)
    feasibility = float(np.max(np.maximum(normalized, 0.0)))
    return feasibility, float(np.max(normalized))


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
        feasibility, _ = _constraint_metrics(metrics)
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
        }

    return _evaluate_cached_stage(
        boundary_params, stage=stage, compute=compute, use_cache=use_cache
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
        feasibility = _p2_feasibility(metrics)
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
        }

    return _evaluate_cached_stage(
        boundary_params, stage=stage, compute=compute, use_cache=use_cache
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
        feasibility = _p3_feasibility(metrics)
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
        }

    return _evaluate_cached_stage(
        boundary_params, stage=stage, compute=compute, use_cache=use_cache
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


def normalized_constraint_distance_sampler(
    base_designs: Sequence[Mapping[str, Sequence[float] | float]],
    *,
    normalized_distances: Sequence[float],
    proposal_count: int,
    jitter_scale: float = 0.01,
    rng: np.random.Generator | None = None,
) -> list[Mapping[str, float | Sequence[float]]]:
    """Constraint-aware sampler for Task X.6 (docs/TASKS_CODEX_MINI.md:233).

    Designs with smaller normalized constraint distances are preferred so the curriculum
    nudges proposals toward near-feasible regions.
    """

    if proposal_count <= 0:
        return []

    if rng is None:
        rng = np.random.default_rng()

    total_candidates = len(base_designs)
    if total_candidates == 0:
        return []

    distances = np.asarray(normalized_distances, dtype=float)
    if distances.shape[0] != total_candidates:
        raise ValueError("normalized_distances must align with base_designs")

    clipped = np.clip(distances, 0.0, 1.0)
    weights = (1.0 - clipped) + 1e-3
    weights_sum = float(np.sum(weights))
    if weights_sum <= 0.0:
        weights = np.ones_like(weights)
        weights_sum = float(weights.size)

    probabilities = (weights / weights_sum).astype(float)
    chosen_indices = rng.choice(total_candidates, size=proposal_count, p=probabilities)
    proposals: list[Mapping[str, float | Sequence[float]]] = []

    for idx in chosen_indices:
        candidate = base_designs[idx]
        perturbed: dict[str, float | Sequence[float]] = {}
        for key, value in candidate.items():
            array = np.asarray(value, dtype=float)
            jitter = rng.normal(scale=jitter_scale, size=array.shape)
            proposal_array = array + jitter
            if proposal_array.shape == ():
                perturbed[key] = float(proposal_array)
            else:
                perturbed[key] = proposal_array.tolist()
        proposals.append(perturbed)

    return proposals


def get_cache_stats(stage: str) -> Mapping[str, int]:
    """Return hit/miss counts for a given stage."""

    return _CACHE_STATS[stage.lower()].copy()


def clear_evaluation_cache() -> None:
    """Reset the P1 evaluation cache and stats (useful for tests)."""

    _EVALUATION_CACHE.clear()
    _CACHE_STATS.clear()
