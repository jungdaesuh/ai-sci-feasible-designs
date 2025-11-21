"""Lightweight surrogate ranking helpers for candidate screening.

Per docs/MASTER_PLAN_AI_SCIENTIST.md:332 and docs/TASKS_CODEX_MINI.md:151 the
ranker now trains on cached metrics (KRR/MLP-style features) so Phase 4/5
memory data can beat random ordering before promotion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ai_scientist import memory


def _flatten_boundary_params(params: Mapping[str, Any]) -> np.ndarray:
    values: list[np.ndarray] = []
    for key in ("r_cos", "z_sin"):
        payload = params.get(key)
        if payload is None:
            continue
        arr = np.asarray(payload, dtype=float).ravel()
        if arr.size:
            values.append(arr)
    if not values:
        return np.zeros((0,), dtype=float)
    return np.concatenate(values)


def _params_feature_vector(params: Mapping[str, Any]) -> np.ndarray:
    flattened = _flatten_boundary_params(params)
    if flattened.size == 0:
        return np.zeros((2,), dtype=float)
    return np.array([float(np.sum(flattened)), float(flattened.size)], dtype=float)


@dataclass(frozen=True)
class SurrogateRank:
    """Surrogate score + reference for a candidate."""

    score: float
    metrics: Mapping[str, Any]


class SimpleSurrogateRanker:
    """Surrogate ranker that uses a ridge-learned model instead of heuristics."""

    def __init__(self, *, alpha: float = 1e-2) -> None:
        self._alpha = float(alpha)
        self._feature_weights: np.ndarray | None = None
        self._bias: float = 0.0

    def _feature_vector(self, metrics: Mapping[str, Any]) -> np.ndarray:
        params = metrics.get("candidate_params")
        if isinstance(params, Mapping):
            return _params_feature_vector(params)
        gradient = float(
            metrics.get("minimum_normalized_magnetic_gradient_scale_length", 0.0)
        )
        aspect = float(metrics.get("aspect_ratio", 0.0))
        hv = float(metrics.get("hv", gradient - aspect))
        return np.array([gradient, aspect, hv, gradient - aspect], dtype=float)

    def fit(
        self,
        metrics_list: Sequence[Mapping[str, Any]],
        target_values: Sequence[float],
    ) -> None:
        if not metrics_list:
            raise ValueError("training data cannot be empty")
        if len(metrics_list) != len(target_values):
            raise ValueError("metrics and targets must be the same length")
        features = np.vstack([self._feature_vector(m) for m in metrics_list])
        targets = np.asarray(target_values, dtype=float)
        intercept = np.ones((features.shape[0], 1), dtype=float)
        design = np.hstack([features, intercept])
        reg = np.eye(design.shape[1], dtype=float)
        reg[-1, -1] = 0.0
        xtx = design.T @ design + self._alpha * reg
        xty = design.T @ targets
        weights = np.linalg.solve(xtx, xty)
        self._feature_weights = weights[:-1]
        self._bias = float(weights[-1])

    def fit_from_world_model(
        self,
        world_model: memory.WorldModel,
        *,
        target_column: str = "hv",
        problem: str | None = None,
    ) -> None:
        history = world_model.surrogate_training_data(
            target=target_column, problem=problem
        )
        if not history:
            raise ValueError("insufficient history for surrogate training")
        metrics_list, targets = zip(*history)
        self.fit(metrics_list, targets)

    def _ensure_trained(self) -> None:
        if self._feature_weights is None:
            raise RuntimeError("surrogate ranker has not been trained")

    def rank(self, metrics_list: Sequence[Mapping[str, Any]]) -> list[SurrogateRank]:
        self._ensure_trained()
        assert self._feature_weights is not None
        ranked: list[SurrogateRank] = []
        weights = self._feature_weights
        for metrics in metrics_list:
            features = self._feature_vector(metrics)
            score = float(np.dot(features, weights) + self._bias)
            ranked.append(SurrogateRank(score=score, metrics=metrics))
        return sorted(ranked, key=lambda item: item.score, reverse=True)
