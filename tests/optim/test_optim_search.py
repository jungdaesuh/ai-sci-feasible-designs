from ai_scientist.optim import search, surrogate


def test_p3_search_wrapper_ranks_with_highest_proxy(monkeypatch):
    expected_objectives = [
        {"aspect_ratio": 2.0, "gradient": 1.3, "objective": 2.0},
        {"aspect_ratio": 4.0, "gradient": 2.0, "objective": 4.0},
    ]
    metrics_list = [
        {
            "minimum_normalized_magnetic_gradient_scale_length": 1.3,
            "aspect_ratio": 2.0,
        },
        {
            "minimum_normalized_magnetic_gradient_scale_length": 2.0,
            "aspect_ratio": 4.0,
        },
    ]

    def fake_evaluate(candidates):
        assert len(candidates) == len(expected_objectives)
        return {
            "stage": "p3",
            "hv_score": 1.5,
            "objectives": expected_objectives,
            "metrics_list": metrics_list,
            "feasibilities": [0.0, 0.0],
        }

    monkeypatch.setattr("ai_scientist.tools.evaluate_p3_set", fake_evaluate)

    wrapper = search.P3SearchWrapper(base_params={"foo": [[1.0]], "bar": 1})
    candidates = wrapper.propose_candidates(batch_size=2, seed=1)
    ranked = wrapper.rank_candidates(candidates)

    assert ranked[0][1] > ranked[1][1]
    assert len(ranked) == 2


def test_simple_surrogate_ranker_learns_from_training():
    ranker = surrogate.SimpleSurrogateRanker(alpha=1e-3)
    training_metrics = [
        {
            "minimum_normalized_magnetic_gradient_scale_length": 3.0,
            "aspect_ratio": 1.0,
            "hv": 0.75,
        },
        {
            "minimum_normalized_magnetic_gradient_scale_length": 1.0,
            "aspect_ratio": 3.0,
            "hv": 0.05,
        },
    ]
    ranker.fit(training_metrics, target_values=[0.75, 0.05])

    candidates = [training_metrics[1], training_metrics[0]]
    ranked = ranker.rank(candidates)

    assert ranked[0].score > ranked[1].score
    assert ranked[0].metrics["aspect_ratio"] == 1.0
