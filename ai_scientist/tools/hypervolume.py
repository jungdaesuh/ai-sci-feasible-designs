"""Hypervolume and Pareto front utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple

import numpy as np
from pymoo.indicators import hv as pymoo_hv

_P3_REFERENCE_POINT: Tuple[float, float] = (1.0, 20.0)


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


def _objective_vector(metrics: Mapping[str, Any]) -> Tuple[float, float]:
    gradient = float(metrics.get("minimum_normalized_magnetic_gradient_scale_length", 0.0))
    aspect = float(metrics.get("aspect_ratio", 1e9)) # Large value if aspect ratio is missing
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
    
    from ai_scientist.tools.evaluation import design_hash, _DEFAULT_RELATIVE_TOLERANCE

    @dataclass(frozen=True)
    class _P3Entry:
        gradient: float
        aspect: float
        seed: int
        evaluation: Mapping[str, Any]
        feasibility: float
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