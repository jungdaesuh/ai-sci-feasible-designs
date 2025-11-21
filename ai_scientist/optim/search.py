"""Wave 7 search wrappers for P3 candidate generation and ranking.

These helpers follow the gating notes in docs/MASTER_PLAN_AI_SCIENTIST.md:205 and
docs/TASKS_CODEX_MINI.md:145, where the Wave 7 DoD requires a structured
``search wrapper`` kernel (Nelder–Mead or CMA-ES) that still calls
`tools.evaluate_p3_set` for HV-aware scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import cycle
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from ai_scientist import tools

_Method = Literal["nelder_mead", "cma_es"]


@dataclass(frozen=True)
class BatchSummary:
    """Minimal record of a candidate batch tied to P3 evaluations."""

    stage: str
    hv_score: float
    objectives: Sequence[Mapping[str, Any]]


@dataclass(frozen=True)
class _ParamSlot:
    path: tuple[str, ...]
    shape: tuple[int, ...]
    length: int


class _ParameterVectorizer:
    """Simple flatten/unflatten helper for the active parameter dict."""

    def __init__(self, prototype: Mapping[str, Any]) -> None:
        self._slots: list[_ParamSlot] = []
        self._dim = 0
        self._collect_slots(prototype, ())
        self._reference = self.flatten(prototype)

    def _collect_slots(self, node: Any, path: tuple[str, ...]) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                self._collect_slots(value, path + (key,))
            return
        array = np.asarray(node, dtype=float)
        shape = () if array.ndim == 0 else array.shape
        length = int(array.size)
        self._slots.append(_ParamSlot(path, shape, length))
        self._dim += length

    def _get_by_path(self, params: Mapping[str, Any], path: tuple[str, ...]) -> Any:
        node: Any = params
        for key in path:
            node = node[key]
        return node

    def _set_by_path(
        self, target: dict[str, Any], path: tuple[str, ...], value: Any
    ) -> None:
        node = target
        for key in path[:-1]:
            if key not in node or not isinstance(node[key], dict):
                node[key] = {}
            node = node[key]
        node[path[-1]] = value

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def reference(self) -> np.ndarray:
        return self._reference.copy()

    def flatten(self, params: Mapping[str, Any]) -> np.ndarray:
        if self._dim == 0:
            return np.zeros(0, dtype=float)
        vector = np.empty(self._dim, dtype=float)
        offset = 0
        for slot in self._slots:
            raw_value = self._get_by_path(params, slot.path)
            array = np.asarray(raw_value, dtype=float)
            if slot.shape:
                array = array.reshape(slot.shape)
            else:
                array = array.reshape(())
            length = slot.length
            vector[offset : offset + length] = array.reshape(-1)
            offset += length
        return vector

    def unflatten(self, vector: Sequence[float] | np.ndarray) -> dict[str, Any]:
        if self._dim == 0:
            return {}
        params: dict[str, Any] = {}
        offset = 0
        arr = np.asarray(vector, dtype=float)
        for slot in self._slots:
            segment = arr[offset : offset + slot.length]
            offset += slot.length
            value: Any
            if slot.shape:
                value = segment.reshape(slot.shape).tolist()
            else:
                value = float(segment[0])
            self._set_by_path(params, slot.path, value)
        return params


class P3SearchWrapper:
    """Generate and score P3 proposals using the HV-aware set evaluator.

    The wrapper now follows the Wave 7 optimizer kernel mandate in
    docs/MASTER_PLAN_AI_SCIENTIST.md:331 and docs/TASKS_CODEX_MINI.md:145 by
    flattening the parameter dictionary and emitting either a Nelder–Mead
    simplex or CMA-ES-style batch before the HV surrogate ranking step.
    """

    def __init__(
        self,
        base_params: Mapping[str, Any],
        *,
        perturbation_scale: float = 0.05,
        method: _Method = "nelder_mead",
    ) -> None:
        self._vectorizer = _ParameterVectorizer(base_params)
        self._method = method
        self._scale = float(perturbation_scale)
        self._dim = self._vectorizer.dim
        self._mean = self._vectorizer.flatten(base_params)
        self._simplex = self._build_simplex(self._mean)
        self._sigma = max(self._scale, 1e-4)
        self._best_score = float("-inf")

    def _build_simplex(self, center: np.ndarray) -> list[np.ndarray]:
        if self._dim == 0:
            return [center.copy()]
        simplex: list[np.ndarray] = [center.copy()]
        for axis in range(self._dim):
            direction = np.zeros(self._dim, dtype=float)
            direction[axis] = self._scale
            simplex.append(center + direction)
        return simplex

    def propose_candidates(
        self, batch_size: int, seed: int | None = None
    ) -> list[Mapping[str, Any]]:
        """Generate a structured batch of proposals anchored on the current mean."""

        if batch_size <= 0:
            return []
        rng = np.random.default_rng(seed)
        proposals: list[Mapping[str, Any]] = []
        if self._dim == 0:
            base = self._vectorizer.unflatten(self._mean)
            return [base for _ in range(batch_size)]

        if self._method == "cma_es":
            for _ in range(batch_size):
                delta = rng.normal(scale=self._sigma, size=self._dim)
                proposals.append(self._vectorizer.unflatten(self._mean + delta))
            return proposals

        simplex_iter = cycle(self._simplex)
        while len(proposals) < batch_size:
            vertex = next(simplex_iter)
            if len(proposals) < len(self._simplex):
                proposals.append(self._vectorizer.unflatten(vertex))
            else:
                jitter = rng.uniform(-self._scale, self._scale, size=self._dim)
                proposals.append(self._vectorizer.unflatten(self._mean + jitter))
        return proposals[:batch_size]

    def evaluate_batch(self, candidates: Sequence[Mapping[str, Any]]) -> BatchSummary:
        """Score the proposals using the P3 HV-aware set evaluator."""

        evaluation = tools.evaluate_p3_set(candidates)
        return BatchSummary(
            stage=evaluation["stage"],
            hv_score=float(evaluation["hv_score"]),
            objectives=tuple(evaluation["objectives"]),
        )

    def _score_metrics(self, metrics: Mapping[str, Any]) -> float:
        gradient = float(metrics["minimum_normalized_magnetic_gradient_scale_length"])
        aspect = float(metrics["aspect_ratio"])
        return gradient - aspect

    def _update_state(
        self, best_vector: np.ndarray, hv_score: float, improved: bool
    ) -> None:
        if self._dim == 0:
            return
        if self._method == "nelder_mead":
            self._mean = best_vector
            self._simplex = self._build_simplex(best_vector)
            return
        self._best_score = max(self._best_score, hv_score)
        if improved:
            self._mean = best_vector
            self._sigma = max(self._sigma * 0.9, 1e-6)
        else:
            self._sigma = min(self._sigma * 1.1, 1.0)

    def rank_candidates(
        self, candidates: Sequence[Mapping[str, Any]]
    ) -> list[tuple[Mapping[str, Any], float]]:
        """Return proposals ordered by the HV proxy (gradient minus aspect)."""

        evaluation = tools.evaluate_p3_set(candidates)
        metrics_seq = evaluation.get("metrics_list", [])
        if not metrics_seq:
            return []
        scored: list[tuple[Mapping[str, Any], float]] = []
        for candidate, metrics in zip(candidates, metrics_seq):
            proxy_score = self._score_metrics(metrics)
            scored.append((candidate, proxy_score))
        best_idx = max(range(len(scored)), key=lambda idx: scored[idx][1])
        best_vector = self._vectorizer.flatten(candidates[best_idx])
        improved = evaluation["hv_score"] >= self._best_score
        self._update_state(best_vector, float(evaluation["hv_score"]), improved)
        return sorted(scored, key=lambda item: item[1], reverse=True)
