from typing import List, Dict, Optional, Any
import numpy as np
from sklearn.ensemble import RandomForestClassifier  # type: ignore


class FeasibilityPrefilter:
    """Quick classifier to reject obviously infeasible candidates."""

    def __init__(self):
        self.model: Optional[RandomForestClassifier] = None
        self.feature_keys = [
            "aspect_ratio",
            "max_elongation",
        ]

    @property
    def is_trained(self) -> bool:
        """Check if the model has been trained."""
        return self.model is not None

    def train(self, X: np.ndarray, y_feasible: np.ndarray):
        """Train on historical evaluations.

        Args:
            X: Feature matrix (n_samples, n_features)
            y_feasible: Boolean array of feasibility labels
        """
        if len(X) < 10 or len(np.unique(y_feasible)) < 2:
            # Not enough data or only one class
            return

        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X, y_feasible)

    def predict_feasible(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return boolean mask of likely-feasible candidates."""
        if self.model is None:
            return np.ones(len(X), dtype=bool)  # Pass all if not trained

        try:
            probs = self.model.predict_proba(X)[:, 1]
            return probs >= threshold
        except Exception as e:
            print(f"[Prefilter] Prediction failed: {e}. Passing all.")
            return np.ones(len(X), dtype=bool)

    def filter_candidates(
        self, candidates: List[Dict[str, Any]], threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Filter candidates, keeping only likely-feasible ones.

        Args:
            candidates: List of candidate dictionaries. Must contain feature keys.
            threshold: Probability threshold for keeping a candidate.

        Returns:
            Filtered list of candidates.
        """
        if self.model is None or not candidates:
            return candidates

        # Extract features
        # We need to handle missing keys gracefully or skip filtering
        try:
            X = []
            valid_indices = []
            for i, cand in enumerate(candidates):
                # Check if we have the features (e.g. from Geometer or previous steps)
                # If features are missing (e.g. pure params), we can't filter yet.
                # Assuming candidates here might have some metrics if they passed Geometer?
                # Or if this is used after a cheap evaluation step.
                # If candidates are just params, we can't use this unless we compute features.
                # For now, let's assume candidates have these keys (e.g. from a cheap proxy).
                # If not, we skip filtering for that candidate (or all).

                # Actually, Geometer computes some of these.
                row = []
                has_all = True
                for key in self.feature_keys:
                    if key not in cand:
                        has_all = False
                        break
                    row.append(cand[key])

                if has_all:
                    X.append(row)
                    valid_indices.append(i)

            if not X:
                return candidates

            X_arr = np.array(X)
            mask = self.predict_feasible(X_arr, threshold)

            # Reconstruct list
            kept_candidates = []
            # Candidates that didn't have features are kept (conservative)
            # Candidates that had features are kept if mask is True

            mask_idx = 0
            for i in range(len(candidates)):
                if i in valid_indices:
                    if mask[mask_idx]:
                        kept_candidates.append(candidates[i])
                    mask_idx += 1
                else:
                    kept_candidates.append(candidates[i])

            return kept_candidates

        except Exception as e:
            print(f"[Prefilter] Filtering failed: {e}. Returning original list.")
            return candidates
