"""Hypervolume and Pareto front utilities.

Sign Convention (P3 Multi-Objective):
-------------------------------------
P3 has two objectives:
  1. Minimize aspect_ratio (lower is better - more compact)
  2. Maximize gradient (L_∇B) (higher is better - simpler coils)

For hypervolume calculation with pymoo (which assumes minimization):
  - We convert to minimization form: (-gradient, aspect_ratio)
  - Reference point: (-1.0, 20.0) means worst acceptable is gradient=1, aspect=20
    (ref=-1.0 in minimization form corresponds to natural gradient=1.0)

For Pareto dominance (using natural units):
  - Point A dominates B if: A.gradient >= B.gradient AND A.aspect <= B.aspect
    with at least one strict inequality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple

import numpy as np
from pymoo.indicators import hv as pymoo_hv

# Reference point for hypervolume in MINIMIZATION form: (-gradient, aspect_ratio)
# Since we negate gradient, ref=-1.0 means natural gradient=1.0 is the threshold.
# Points with gradient < 1.0 (minimization form > -1.0) won't contribute hypervolume.
_P3_REFERENCE_POINT: Tuple[float, float] = (-1.0, 20.0)


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


def _to_minimization_form(gradient: float, aspect: float) -> Tuple[float, float]:
    """Convert natural units (max gradient, min aspect) to minimization form for HV.

    Returns (-gradient, aspect) so both objectives are minimized.
    """
    return -gradient, aspect


def _extract_natural_objectives(metrics: Mapping[str, Any]) -> Tuple[float, float]:
    """Extract (gradient, aspect_ratio) in natural units from metrics.

    Returns:
        (gradient, aspect_ratio) where higher gradient is better, lower aspect is better.
    """
    gradient = float(
        metrics.get("minimum_normalized_magnetic_gradient_scale_length", 0.0)
    )
    aspect = float(
        metrics.get("aspect_ratio", 1e9)
    )  # Large value if aspect ratio is missing
    return gradient, aspect


def _objective_vector(metrics: Mapping[str, Any]) -> Tuple[float, float]:
    """Return the P3 objective vector in minimization form for hypervolume.

    This matches `constellaration.problems.MHDStableQIStellarator._score`:
    X = [(-gradient, aspect_ratio), ...]
    """
    gradient, aspect = _extract_natural_objectives(metrics)
    return _to_minimization_form(gradient, aspect)


def _dominates(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    """Return True if point a Pareto dominates b in natural units.

    Args:
        a: (gradient_a, aspect_a) - natural units
        b: (gradient_b, aspect_b) - natural units

    Returns:
        True if a dominates b (higher gradient AND lower aspect, with strict inequality).
    """
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
    """Produce the hypervolume score and all non-dominated seeds for a candidate batch.

    All internal calculations use natural units (gradient, aspect) where:
    - Higher gradient is better
    - Lower aspect is better

    Hypervolume is computed by converting to minimization form (-gradient, aspect).
    """
    from ai_scientist.tools.evaluation import _DEFAULT_RELATIVE_TOLERANCE, design_hash

    @dataclass(frozen=True)
    class _P3Entry:
        gradient: float  # Natural units: higher is better
        aspect: float  # Natural units: lower is better
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
        # Extract in natural units (gradient, aspect)
        gradient, aspect = _extract_natural_objectives(eval_metrics)
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

    # Build HV vectors in minimization form (-gradient, aspect)
    hv_vectors: list[Tuple[float, float]] = []
    for entry in entries:
        if entry.feasibility > _DEFAULT_RELATIVE_TOLERANCE:
            continue
        hv_vectors.append(_to_minimization_form(entry.gradient, entry.aspect))

    # Find non-dominated (Pareto optimal) entries using natural units
    pareto_entries: list[ParetoEntry] = []
    for current_index, entry in enumerate(entries):
        if entry.feasibility > _DEFAULT_RELATIVE_TOLERANCE:
            continue
        current_point = (entry.gradient, entry.aspect)  # Natural units
        dominated = False
        for other_index, other in enumerate(entries):
            if other_index == current_index:
                continue
            if other.feasibility > _DEFAULT_RELATIVE_TOLERANCE:
                continue
            other_point = (other.gradient, other.aspect)  # Natural units
            if _dominates(other_point, current_point):
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

    # Sort by gradient descending (best first), then aspect ascending
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
