from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier


class FeasibilityPrefilter:
    """
    Quick classifier to reject obviously infeasible candidates before expensive physics evaluation.
    """

    def __init__(self):
        self.model: Optional[RandomForestClassifier] = None
        self.is_trained = False

    def train(self, X: np.ndarray, y_feasible: np.ndarray):
        """
        Train the prefilter on historical evaluations.

        Args:
            X: Array of feature vectors (N, D).
            y_feasible: Boolean array of feasibility labels (N,).
        """
        if len(X) == 0:
            return

        # Handle case where all are feasible or all are infeasible
        if len(np.unique(y_feasible)) < 2:
            # Cannot train a classifier with only one class
            # We mark as not trained so we default to passing everything
            self.is_trained = False
            return

        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X, y_feasible)
        self.is_trained = True

    def predict_feasible(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Return boolean mask of likely-feasible candidates.

        Args:
            X: Array of feature vectors (N, D).
            threshold: Probability threshold for feasibility.

        Returns:
            Boolean array (N,) where True means likely feasible.
        """
        if not self.is_trained or self.model is None:
            return np.ones(len(X), dtype=bool)  # Pass all if not trained

        # Predict probability of class 1 (feasible)
        probs = self.model.predict_proba(X)[:, 1]
        return probs >= threshold

    def filter_candidates(
        self, candidates: List[Dict[str, Any]], threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Filter a list of candidate dictionaries, keeping only likely-feasible ones.
        Assumes candidates have 'params' that can be flattened into features.

        Args:
            candidates: List of candidate dicts.
            threshold: Probability threshold.

        Returns:
            Filtered list of candidates.
        """
        if not self.is_trained or not candidates:
            return candidates

        # Extract features from candidates
        # We assume a simple flattening strategy for now, or use specific keys
        # This needs to match how 'X' was constructed during training
        # For now, let's assume we extract 'r_cos' and 'z_sin' and flatten them
        X = self._extract_features(candidates)

        mask = self.predict_feasible(X, threshold)

        # Return only candidates where mask is True
        return [c for c, m in zip(candidates, mask) if m]

    def _extract_features(self, candidates: List[Dict[str, Any]]) -> np.ndarray:
        """
        Helper to extract feature matrix X from candidates.
        """
        features = []
        for cand in candidates:
            # Try to get params from 'params' key or the dict itself
            params = cand.get("params", cand)

            # Extract R and Z coefficients
            # We flatten them into a single vector
            # Handle both list and numpy array inputs
            r_cos = np.array(params.get("r_cos", [])).flatten()
            z_sin = np.array(params.get("z_sin", [])).flatten()

            # Concatenate
            feat = np.concatenate([r_cos, z_sin])
            features.append(feat)

        return np.array(features)
