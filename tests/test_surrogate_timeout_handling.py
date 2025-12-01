import unittest

from ai_scientist.optim.surrogate import SurrogateBundle


class TestSurrogateTimeoutHandling(unittest.TestCase):
    def test_fit_timeout_handling(self):
        # Set extremely short timeout to force timeout during fit
        bundle = SurrogateBundle(timeout_seconds=1e-9, min_samples=1)

        metrics = [
            {
                "metrics": {"vacuum_well": 0.1, "qi": 1.0, "max_elongation": 2.0},
                "feasibility": 1.0,
                "candidate_params": {"r_cos": [[1.0]], "z_sin": [[1.0]]},
            }
        ]
        targets = [1.0]

        # This should not raise TimeoutError, but log a warning and set _trained=False
        bundle.fit(metrics, targets, minimize_objective=False)

        self.assertFalse(bundle._trained, "Surrogate should be untrained after timeout")

    def test_rank_timeout_handling(self):
        bundle = SurrogateBundle(timeout_seconds=1e-6, min_samples=1)
        # Manually set trained=True to bypass cold start check
        bundle._trained = True

        # But we need _scaler, _classifier etc to be set for _predict_batch to run
        # So let's train it properly first with a long timeout
        bundle._timeout_seconds = 5.0
        metrics = [
            {
                "metrics": {"vacuum_well": 0.1, "qi": 1.0, "max_elongation": 2.0},
                "feasibility": 1.0,
                "candidate_params": {"r_cos": [[1.0]], "z_sin": [[1.0]]},
            }
        ] * 5 + [
            {
                "metrics": {"vacuum_well": -0.1, "qi": 2.0, "max_elongation": 3.0},
                "feasibility": 0.0,
                "candidate_params": {"r_cos": [[1.0]], "z_sin": [[1.0]]},
            }
        ] * 5
        targets = [1.0] * 5 + [0.0] * 5
        bundle.fit(metrics, targets, minimize_objective=False)
        self.assertTrue(bundle._trained)

        # Now set timeout to near zero
        bundle._timeout_seconds = 1e-9

        candidates = [{"candidate_params": {"r_cos": [[1.0]], "z_sin": [[1.0]]}}]

        # This should catch TimeoutError and return heuristic ranks
        ranks = bundle.rank_candidates(candidates, minimize_objective=False)

        self.assertTrue(len(ranks) > 0)


if __name__ == "__main__":
    unittest.main()
