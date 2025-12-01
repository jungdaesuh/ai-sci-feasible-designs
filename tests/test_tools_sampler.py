from typing import Mapping, Sequence

import numpy as np

from ai_scientist import tools


def test_normalized_constraint_distance_sampler_improves_feasible_rate() -> None:
    base_values = np.linspace(-1.0, 1.0, 11, dtype=float)
    base_designs = [{"offset": float(value)} for value in base_values]
    normalized_distances = np.abs(base_values).tolist()
    proposals = tools.normalized_constraint_distance_sampler(
        base_designs,
        normalized_distances=normalized_distances,
        proposal_count=4096,
        jitter_scale=0.01,
        rng=np.random.default_rng(42),
    )

    uniform_rng = np.random.default_rng(1337)
    uniform_proposals = [
        {"offset": float(uniform_rng.uniform(-1.0, 1.0))} for _ in range(4096)
    ]

    def feasible_rate(
        samples: Sequence[Mapping[str, float | Sequence[float]]],
    ) -> float:
        return sum(
            1
            for sample in samples
            if abs(np.asarray(sample["offset"], dtype=float).item()) <= 0.25
        ) / len(samples)

    sampler_rate = feasible_rate(proposals)
    uniform_rate = feasible_rate(uniform_proposals)

    assert sampler_rate > uniform_rate + 0.15
