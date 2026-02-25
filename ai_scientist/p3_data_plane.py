"""Shared P3 data-plane telemetry helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class DataPlaneSample:
    has_lineage: bool
    novelty_score: float | None
    operator_family: str
    model_route: str


def is_static_route(route: str) -> bool:
    return route == "governor_static_recipe" or route.startswith(
        "governor_static_recipe/"
    )


def is_adaptive_route(route: str) -> bool:
    return (
        route == "governor_adaptive"
        or route.startswith("governor_adaptive/")
        or route == "governor_adaptive_scaffold"
        or route.startswith("governor_adaptive_scaffold/")
    )


def is_fallback_static_delegate_route(route: str) -> bool:
    return route == "governor_adaptive_scaffold/static_delegate" or route.startswith(
        "governor_adaptive_scaffold/static_delegate/"
    )


def summarize_data_plane(
    samples: Iterable[DataPlaneSample], *, novelty_reject_threshold: float
) -> dict:
    sample_list = list(samples)
    operator_counts: dict[str, int] = {}
    route_counts: dict[str, int] = {}
    lineage_count = 0
    novelty_values: list[float] = []
    novelty_reject_count = 0
    static_path_rows = 0
    adaptive_path_rows = 0
    fallback_static_delegate_rows = 0

    for sample in sample_list:
        operator = sample.operator_family or "unknown"
        route = sample.model_route or "unknown"
        operator_counts[operator] = operator_counts.get(operator, 0) + 1
        route_counts[route] = route_counts.get(route, 0) + 1
        if is_static_route(route):
            static_path_rows += 1
        if is_adaptive_route(route):
            adaptive_path_rows += 1
        if is_fallback_static_delegate_route(route):
            fallback_static_delegate_rows += 1
        if sample.has_lineage:
            lineage_count += 1
        novelty_value = sample.novelty_score
        if novelty_value is not None and math.isfinite(novelty_value):
            novelty_values.append(float(novelty_value))
            if novelty_value < float(novelty_reject_threshold):
                novelty_reject_count += 1

    with_novelty = len(novelty_values)
    avg_novelty = (
        float(sum(novelty_values) / len(novelty_values)) if novelty_values else None
    )
    novelty_reject_rate = (
        float(novelty_reject_count) / float(with_novelty)
        if with_novelty > 0
        else None
    )
    return {
        "candidate_rows": len(sample_list),
        "with_lineage": lineage_count,
        "with_novelty": with_novelty,
        "novelty_missing_count": len(sample_list) - with_novelty,
        "avg_novelty": avg_novelty,
        "novelty_reject_threshold": float(novelty_reject_threshold),
        "novelty_reject_count": novelty_reject_count,
        "novelty_reject_rate": novelty_reject_rate,
        "operator_families": dict(
            sorted(operator_counts.items(), key=lambda item: item[1], reverse=True)
        ),
        "model_routes": dict(
            sorted(route_counts.items(), key=lambda item: item[1], reverse=True)
        ),
        "static_path_rows": static_path_rows,
        "adaptive_path_rows": adaptive_path_rows,
        "fallback_static_delegate_rows": fallback_static_delegate_rows,
    }
