"""Surrogate helpers for candidate screening and feasibility-aware ranking.

Per the unified roadmap (docs/AI_SCIENTIST_UNIFIED_ROADMAP.md) the production
surrogate is a bundled vectorizer + scaler + RF classifier/regressor that
estimates feasibility and objective jointly. The legacy linear ranker remains
for backward compatibility, but SurrogateBundle is the default path going
forward.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from ai_scientist import memory, tools

# Define locally to avoid circular import (tools → evaluation → forward_model → optim → surrogate → tools)
_DEFAULT_RELATIVE_TOLERANCE = 1e-2


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
    nfp = float(params.get("n_field_periods") or params.get("nfp") or 1.0)
    if flattened.size == 0:
        vec = np.array([0.0, 0.0, nfp], dtype=float)
    else:
        vec = np.array(
            [float(np.sum(flattened)), float(flattened.size), nfp], dtype=float
        )
    return vec


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
        aspect = float(metrics.get("aspect_ratio", 1.0))
        # H2 FIX: Use gradient_proxy (with hv fallback for legacy data)
        gradient_proxy = float(
            metrics.get("gradient_proxy", metrics.get("hv", gradient - aspect))
        )
        return np.array(
            [gradient, aspect, gradient_proxy, gradient - aspect], dtype=float
        )

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
        problem: str,
        target_column: str = "hv",
        cycle: int = 0,
        experiment_id: int | None = None,
    ) -> None:
        """Fetch history from WorldModel and fit the ranker."""
        history = world_model.surrogate_training_data(
            target=target_column, problem=problem, experiment_id=experiment_id
        )
        if not history:
            return

        metrics_list, target_values = zip(*history)
        minimize_obj = problem.lower().startswith("p1")
        self.fit(
            metrics_list,
            target_values,
            minimize_objective=minimize_obj,
            cycle=cycle,
        )

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
    predicted_mhd: float | None = None
    predicted_qi: float | None = None
    predicted_elongation: float | None = None


class BaseSurrogate(ABC):
    """Abstract base class for all surrogate models."""

    @abstractmethod
    def fit(
        self,
        metrics_list: Sequence[Mapping[str, Any]],
        target_values: Sequence[float],
        *,
        minimize_objective: bool,
        cycle: int | None = None,
    ) -> None:
        """Train the surrogate on historical data."""
        pass

    @abstractmethod
    def should_retrain(self, sample_count: int, cycle: int | None = None) -> bool:
        """Determine if retraining is necessary."""
        pass

    @abstractmethod
    def rank_candidates(
        self,
        candidates: Sequence[Mapping[str, Any]],
        *,
        minimize_objective: bool,
        exploration_ratio: float = 0.0,
        problem: str | None = None,
    ) -> list[SurrogatePrediction]:
        """Rank candidates using the surrogate model."""
        pass


class SurrogateBundle(BaseSurrogate):
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
        feasibility_tolerance: float = _DEFAULT_RELATIVE_TOLERANCE,
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
        self._regressors: dict[str, RandomForestRegressor] = {}
        self._schema: tools.FlattenSchema | None = None
        self._trained = False
        self._last_fit_count = 0
        self._last_fit_cycle = 0
        self._single_class_fallback: int | None = (
            None  # Tracks degenerate training state
        )
        # Explicit class encoding: 1 = feasible, 0 = infeasible
        # Used for defensive single-class fallback logic
        self._positive_class_label: int = 1

    def _with_timeout(self, func):
        """Execute func with a timeout, ensuring non-blocking cleanup.

        NOTE: We use shutdown(wait=False, cancel_futures=True) to prevent
        deadlocks during test module reloading. This may leave orphaned threads
        if the worker is truly stuck, but they will be cleaned up when the
        process exits. This is an acceptable tradeoff for test suite stability.
        See: https://github.com/python/cpython/issues/87423
        """
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            print(f"DEBUG: submitting {func}")
            future = executor.submit(func)
            print("DEBUG: waiting for result")
            res = future.result(timeout=self._timeout_seconds)
            print("DEBUG: result received")
            return res
        finally:
            # Non-blocking shutdown: prevents deadlock if worker is stuck
            # cancel_futures=True prevents queued tasks from starting (Python 3.9+)
            executor.shutdown(wait=False, cancel_futures=True)

    def _vectorize(self, params: Mapping[str, Any]) -> np.ndarray:
        vector, schema = tools.structured_flatten(params, schema=self._schema)
        if self._schema is None:
            self._schema = schema
            # Canonicalization Warning: Schema derived from first candidate
            logging.warning(
                "[surrogate] Schema derived from first candidate: mpol=%d, ntor=%d. "
                "Mixing shapes may cause silent truncation/zero-padding.",
                schema.mpol,
                schema.ntor,
            )
        nfp = float(params.get("n_field_periods") or params.get("nfp") or 1.0)
        return np.append(vector, nfp)

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

    def _label_feasibility(
        self, metrics_list: Sequence[Mapping[str, Any]]
    ) -> np.ndarray:
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
            logging.info(
                "[surrogate] cold start: %d samples (< %d)",
                sample_count,
                self._min_samples,
            )
            self._trained = False
            return

        features = self._feature_matrix(metrics_list)
        if features.size == 0:
            self._trained = False
            return

        feasibility_labels = self._label_feasibility(metrics_list)

        # Prepare targets for regression
        primary_targets = np.asarray(target_values, dtype=float)

        # Auxiliary targets
        aux_targets: dict[str, list[float]] = {"mhd": [], "qi": [], "elongation": []}

        for metrics in metrics_list:
            # Handle nested metrics structure if present, or direct keys
            m_payload = metrics.get("metrics", metrics)

            # MHD: vacuum_well
            vacuum_well = m_payload.get("vacuum_well")
            aux_targets["mhd"].append(
                float(vacuum_well) if vacuum_well is not None else -1.0
            )

            # QI: qi
            qi_val = m_payload.get("qi")
            # A4.2 FIX: Use log10 scale for QI
            if qi_val is not None:
                val = float(qi_val)
                aux_targets["qi"].append(math.log10(max(val, 1e-12)))
            else:
                aux_targets["qi"].append(0.0)

            # Elongation: max_elongation
            elongation = m_payload.get("max_elongation")
            aux_targets["elongation"].append(
                float(elongation) if elongation is not None else 10.0
            )

        aux_target_arrays = {
            k: np.asarray(v, dtype=float) for k, v in aux_targets.items()
        }

        feasible_mask = feasibility_labels == 1

        # We must define the training function here to capture local variables
        def _train_bundle() -> None:
            scaler = StandardScaler()
            scaled_class = scaler.fit_transform(features)

            clf = RandomForestClassifier(
                n_estimators=100, max_depth=12, random_state=0, n_jobs=1
            )

            # Validate training data
            unique_classes = np.unique(feasibility_labels)
            if len(unique_classes) < 2:
                logging.warning(
                    f"[surrogate] Single-class training data (class={unique_classes[0]}) "
                    "- skipping classifier training, using constant fallback"
                )
                self._single_class_fallback = int(unique_classes[0])
                clf = None
            else:
                self._single_class_fallback = None
                clf.fit(scaled_class, feasibility_labels)

            regressors = {}

            # Primary Regressor
            primary_reg = RandomForestRegressor(
                n_estimators=100, max_depth=12, random_state=0, n_jobs=1
            )
            if np.count_nonzero(feasible_mask) >= self._min_feasible_for_regressor:
                primary_reg.fit(
                    scaler.transform(features[feasible_mask]),
                    primary_targets[feasible_mask],
                )
            else:
                primary_reg.fit(scaled_class, primary_targets)
            regressors["objective"] = primary_reg

            # Aux Regressors
            for name, targets in aux_target_arrays.items():
                reg = RandomForestRegressor(
                    n_estimators=100, max_depth=12, random_state=0, n_jobs=1
                )
                reg.fit(scaled_class, targets)
                regressors[name] = reg

            self._scaler = scaler
            self._classifier = clf
            self._regressors = regressors
            self._trained = True

        try:
            self._with_timeout(_train_bundle)
        except TimeoutError:
            logging.warning(
                "[surrogate] fit timed out; downgrading to untrained status."
            )
            self._trained = False

    def should_retrain(self, sample_count: int, cycle: int | None = None) -> bool:
        if not self._trained:
            return True
        delta_points = sample_count - self._last_fit_count
        if delta_points >= self._points_cadence:
            return True
        if cycle is None:
            return False
        return (cycle - self._last_fit_cycle) >= self._cycle_cadence

    def _predict_batch(
        self, feature_matrix: np.ndarray
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        # Classifier can be None if single-class fallback is active
        assert self._classifier is not None or self._single_class_fallback is not None
        assert self._regressors
        assert self._scaler is not None

        def _predict() -> tuple[np.ndarray, dict[str, np.ndarray]]:
            scaled = self._scaler.transform(feature_matrix)

            # Use tracked single-class fallback instead of runtime detection
            if self._single_class_fallback is not None:
                # Degenerate classifier: return constant probability based on known class
                # Use explicit _positive_class_label for defensive coding
                if self._single_class_fallback == self._positive_class_label:
                    prob = np.ones(scaled.shape[0])  # All "feasible"
                else:
                    prob = np.zeros(scaled.shape[0])  # All "infeasible"
            else:
                proba = self._classifier.predict_proba(scaled)
                prob = proba[:, 1]

            preds = {}
            for name, reg in self._regressors.items():
                preds[name] = reg.predict(scaled)
            return prob, preds

        return self._with_timeout(_predict)

    def _expected_value(
        self, prob_feasible: float, predicted_objective: float, minimize_objective: bool
    ) -> float:
        oriented = (
            -float(predicted_objective)
            if minimize_objective
            else float(predicted_objective)
        )
        return float(prob_feasible) * oriented

    def _heuristic_rank(
        self, candidates: Sequence[Mapping[str, Any]], minimize_objective: bool
    ) -> list[SurrogatePrediction]:
        logging.info("[surrogate] using heuristic ranking (fallback/cold start)")
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

    def rank_candidates(
        self,
        candidates: Sequence[Mapping[str, Any]],
        *,
        minimize_objective: bool,
        exploration_ratio: float = 0.0,
        problem: str | None = None,
    ) -> list[SurrogatePrediction]:
        if not candidates:
            return []

        exploration_weight = max(0.0, float(exploration_ratio)) * 0.1

        if not self._trained:
            return self._heuristic_rank(candidates, minimize_objective)

        metrics_list: list[Mapping[str, Any]] = []
        for candidate in candidates:
            params = candidate.get("candidate_params")
            if params is None:
                params = candidate.get("params", {})
            metrics_list.append({"candidate_params": params})
        feature_matrix = self._feature_matrix(metrics_list)

        try:
            prob, preds_dict = self._predict_batch(feature_matrix)
        except TimeoutError:
            logging.warning(
                "[surrogate] prediction timed out; using heuristic fallback"
            )
            return self._heuristic_rank(candidates, minimize_objective)

        # Default to objective predictions
        objs = preds_dict.get("objective", np.zeros(len(candidates)))
        mhds = preds_dict.get("mhd", np.zeros(len(candidates)))
        qis = preds_dict.get("qi", np.zeros(len(candidates)))
        # A4.2 FIX: Denormalize QI from log10 to linear scale
        qis = 10.0**qis
        elongations = preds_dict.get("elongation", np.zeros(len(candidates)))

        ranked: list[SurrogatePrediction] = []

        # Gate objective weight on actual regressor existence
        # self._regressors is dict[str, RandomForestRegressor]
        objective_regressor_trained = (
            self._trained and self._regressors.get("objective") is not None
        )

        if objective_regressor_trained:
            training_size = self._last_fit_count
            MIN_SAMPLES_FOR_OBJ = 32
            obj_weight = min(1.0, training_size / (MIN_SAMPLES_FOR_OBJ * 2))
        else:
            obj_weight = 0.0

        for i, candidate in enumerate(candidates):
            pf = prob[i]
            obj = objs[i]

            constraint_distance = float(candidate.get("constraint_distance", 0.0))
            constraint_distance = max(0.0, constraint_distance)

            # Classifier entropy (binary classification uncertainty proxy)
            uncertainty = float(pf * (1.0 - pf))

            # Expected value: feasibility-weighted objective
            base_score = self._expected_value(pf, obj, minimize_objective)

            # Composite score with ramped objective contribution
            score = (
                obj_weight * base_score
                + (1.0 - obj_weight) * float(pf)
                - constraint_distance
                + exploration_weight * uncertainty
            )
            ranked.append(
                SurrogatePrediction(
                    expected_value=score,
                    prob_feasible=float(pf),
                    predicted_objective=float(obj),
                    minimize_objective=minimize_objective,
                    metadata=candidate,
                    predicted_mhd=float(mhds[i]),
                    predicted_qi=float(qis[i]),
                    predicted_elongation=float(elongations[i]),
                )
            )

        return sorted(ranked, key=lambda item: item.expected_value, reverse=True)
