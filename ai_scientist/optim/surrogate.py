"""Surrogate helpers for candidate screening and feasibility-aware ranking.

Per the unified roadmap (docs/AI_SCIENTIST_UNIFIED_ROADMAP.md) the production
surrogate is a bundled vectorizer + scaler + RF classifier/regressor that
estimates feasibility and objective jointly. The legacy linear ranker remains
for backward compatibility, but SurrogateBundle is the default path going
forward.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from ai_scientist import memory, tools


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


@dataclass(frozen=True)
class SurrogatePrediction:
    """Bundle of surrogate scores for a single candidate."""

    expected_value: float
    prob_feasible: float
    predicted_objective: float
    minimize_objective: bool
    metadata: Mapping[str, Any]


class SurrogateBundle:
    """Feasibility-first surrogate bundle with RF heads and structured features.

    - Vectorizes boundary params with tools.structured_flatten using a persisted
      schema (mpol, ntor, schema_version, rounding).
    - Standard-scales the flattened vector.
    - Fits a RF classifier for feasibility probability and a RF regressor for
      objective/HV. PyTorch ensemble is intentionally kept behind the
      `enable_torch` flag but not initialized here.
    - Training policy: fit only when >= `min_samples`; otherwise emit a cold
      start log and keep caller order. The regressor trains on feasible-only
      points when there are >= `min_feasible_for_regressor` feasible rows.
    - Ranking uses E[value] = P(feasible) * corrected_objective where the sign
      is inverted for minimize problems so that higher is always better.
    - Fit/predict run inside a timeout guard to avoid surprises in CI.
    - Retrain cadence: either `points_cadence` new rows or `cycle_cadence`
      elapsed cycles since the last fit.
    """

    def __init__(
        self,
        *,
        min_samples: int = 8,
        min_feasible_for_regressor: int = 4,
        feasibility_tolerance: float = tools._DEFAULT_RELATIVE_TOLERANCE,
        timeout_seconds: float = 1.0,
        points_cadence: int = 16,
        cycle_cadence: int = 1,
        enable_torch: bool = False,
    ) -> None:
        self._min_samples = int(min_samples)
        self._min_feasible_for_regressor = int(min_feasible_for_regressor)
        self._feasibility_tolerance = float(feasibility_tolerance)
        self._timeout_seconds = float(timeout_seconds)
        self._points_cadence = int(points_cadence)
        self._cycle_cadence = int(cycle_cadence)
        self._enable_torch = bool(enable_torch)

        self._scaler: StandardScaler | None = None
        self._classifier: RandomForestClassifier | None = None
        self._regressor: RandomForestRegressor | None = None
        self._schema: tools.FlattenSchema | None = None
        self._trained = False
        self._last_fit_count = 0
        self._last_fit_cycle = 0

    def _with_timeout(self, func):
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(func)
            return future.result(timeout=self._timeout_seconds)

    def _vectorize(self, params: Mapping[str, Any]) -> np.ndarray:
        vector, schema = tools.structured_flatten(
            params, schema=self._schema
        )
        if self._schema is None:
            self._schema = schema
        return vector

    def _feature_matrix(self, metrics_list: Sequence[Mapping[str, Any]]) -> np.ndarray:
        vectors: list[np.ndarray] = []
        for metrics in metrics_list:
            params = metrics.get("candidate_params") or metrics.get("params")
            if not isinstance(params, Mapping):
                params = {}
            vectors.append(self._vectorize(params))
        if not vectors:
            return np.zeros((0, 0), dtype=float)
        return np.vstack(vectors)

    def _label_feasibility(self, metrics_list: Sequence[Mapping[str, Any]]) -> np.ndarray:
        labels: list[int] = []
        for metrics in metrics_list:
            feasibility = metrics.get("feasibility")
            if feasibility is None:
                feasibility = metrics.get("max_violation", float("inf"))
            labels.append(int(float(feasibility) <= self._feasibility_tolerance))
        return np.asarray(labels, dtype=int)

    def fit(
        self,
        metrics_list: Sequence[Mapping[str, Any]],
        target_values: Sequence[float],
        *,
        minimize_objective: bool,
        cycle: int | None = None,
    ) -> None:
        sample_count = len(metrics_list)
        self._last_fit_count = sample_count
        if cycle is not None:
            self._last_fit_cycle = int(cycle)

        if sample_count < self._min_samples:
            logging.info("[surrogate] cold start: %d samples (< %d)", sample_count, self._min_samples)
            self._trained = False
            return

        features = self._feature_matrix(metrics_list)
        if features.size == 0:
            self._trained = False
            return

        feasibility_labels = self._label_feasibility(metrics_list)
        regression_targets = np.asarray(target_values, dtype=float)

        feasible_mask = feasibility_labels == 1
        if np.count_nonzero(feasible_mask) >= self._min_feasible_for_regressor:
            features_reg = features[feasible_mask]
            regression_targets = regression_targets[feasible_mask]
        else:
            features_reg = features

        # Align classifier features to all rows; regressor uses features_reg.
        def _train_bundle() -> None:
            scaler = StandardScaler()
            scaled_class = scaler.fit_transform(features)
            clf = RandomForestClassifier(
                n_estimators=12,
                max_depth=6,
                random_state=0,
                n_jobs=1,
            )
            clf.fit(scaled_class, feasibility_labels)

            scaled_reg = scaler.transform(features_reg)
            reg = RandomForestRegressor(
                n_estimators=12,
                max_depth=6,
                random_state=0,
                n_jobs=1,
            )
            reg.fit(scaled_reg, regression_targets)

            self._scaler = scaler
            self._classifier = clf
            self._regressor = reg
            self._trained = True

        self._with_timeout(_train_bundle)

    def should_retrain(self, sample_count: int, cycle: int | None = None) -> bool:
        if not self._trained:
            return True
        delta_points = sample_count - self._last_fit_count
        if delta_points >= self._points_cadence:
            return True
        if cycle is None:
            return False
        return (cycle - self._last_fit_cycle) >= self._cycle_cadence

    def _predict_batch(self, feature_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert self._classifier is not None
        assert self._regressor is not None
        assert self._scaler is not None

        def _predict() -> tuple[np.ndarray, np.ndarray]:
            scaled = self._scaler.transform(feature_matrix)
            prob = self._classifier.predict_proba(scaled)[:, 1]
            preds = self._regressor.predict(scaled)
            return prob, preds

        return self._with_timeout(_predict)

    def _expected_value(
        self, prob_feasible: float, predicted_objective: float, minimize_objective: bool
    ) -> float:
        oriented = -float(predicted_objective) if minimize_objective else float(predicted_objective)
        return float(prob_feasible) * oriented

    def rank_candidates(
        self,
        candidates: Sequence[Mapping[str, Any]],
        *,
        minimize_objective: bool,
        exploration_ratio: float = 0.0,
    ) -> list[SurrogatePrediction]:
        if not candidates:
            return []

        exploration_weight = max(0.0, float(exploration_ratio)) * 0.1

        if not self._trained:
            logging.info("[surrogate] cold start ranking; using heuristic features")
            cold_ranks: list[SurrogatePrediction] = []
            for candidate in candidates:
                params = candidate.get("candidate_params") or candidate.get("params", {})
                features = _params_feature_vector(params)
                base_score = float(features[0]) if features.size else 0.0
                score = -base_score if minimize_objective else base_score
                cold_ranks.append(
                    SurrogatePrediction(
                        expected_value=score,
                        prob_feasible=0.0,
                        predicted_objective=base_score,
                        minimize_objective=minimize_objective,
                        metadata=candidate,
                    )
                )
            return sorted(cold_ranks, key=lambda item: item.expected_value, reverse=True)

        metrics_list: list[Mapping[str, Any]] = []
        for candidate in candidates:
            params = candidate.get("candidate_params")
            if params is None:
                params = candidate.get("params", {})
            metrics_list.append({"candidate_params": params})
        feature_matrix = self._feature_matrix(metrics_list)
        prob, preds = self._predict_batch(feature_matrix)

        ranked: list[SurrogatePrediction] = []
        for candidate, pf, obj in zip(candidates, prob, preds):
            constraint_distance = float(candidate.get("constraint_distance", 0.0))
            constraint_distance = max(0.0, constraint_distance)
            uncertainty = float(pf * (1.0 - pf))
            base_score = self._expected_value(pf, obj, minimize_objective)
            score = (float(pf) - constraint_distance) + exploration_weight * uncertainty
            # Preserve expected_value semantics for downstream consumers.
            score = score if self._trained else base_score
            ranked.append(
                SurrogatePrediction(
                    expected_value=score,
                    prob_feasible=float(pf),
                    predicted_objective=float(obj),
                    minimize_objective=minimize_objective,
                    metadata=candidate,
                )
            )

        return sorted(ranked, key=lambda item: item.expected_value, reverse=True)
