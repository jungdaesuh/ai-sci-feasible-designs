from __future__ import annotations

import numpy as np
import pytest

from ai_scientist.restart_seed_selector import RestartSeedCandidate, select_restart_seed


def test_select_restart_seed_prefers_lower_objective_for_p1() -> None:
    candidates = [
        RestartSeedCandidate(
            label="a",
            x=np.array([0.0, 0.0], dtype=float),
            objective=3.0,
            feasibility=0.0,
        ),
        RestartSeedCandidate(
            label="b",
            x=np.array([1.0, 0.0], dtype=float),
            objective=2.0,
            feasibility=0.0,
        ),
    ]

    selected, _ = select_restart_seed(
        candidates,
        problem="p1",
        selection_counts={},
        reference_x=np.array([0.0, 0.0], dtype=float),
        diversity_weight=0.0,
    )
    assert selected.label == "b"


def test_select_restart_seed_prefers_higher_objective_for_p2() -> None:
    candidates = [
        RestartSeedCandidate(
            label="a",
            x=np.array([0.0, 0.0], dtype=float),
            objective=5.0,
            feasibility=0.0,
        ),
        RestartSeedCandidate(
            label="b",
            x=np.array([1.0, 0.0], dtype=float),
            objective=8.0,
            feasibility=0.0,
        ),
    ]

    selected, _ = select_restart_seed(
        candidates,
        problem="p2",
        selection_counts={},
        reference_x=np.array([0.0, 0.0], dtype=float),
        diversity_weight=0.0,
    )
    assert selected.label == "b"


def test_select_restart_seed_applies_saturation_penalty() -> None:
    candidates = [
        RestartSeedCandidate(
            label="dominant",
            x=np.array([0.0, 0.0], dtype=float),
            objective=2.0,
            feasibility=0.0,
        ),
        RestartSeedCandidate(
            label="alternate",
            x=np.array([0.1, 0.0], dtype=float),
            objective=2.1,
            feasibility=0.0,
        ),
    ]

    _, baseline = select_restart_seed(
        candidates,
        problem="p1",
        selection_counts={},
        reference_x=np.array([0.0, 0.0], dtype=float),
        diversity_weight=0.0,
    )
    selected, _ = select_restart_seed(
        candidates,
        problem="p1",
        selection_counts={str(baseline["selected_identity"]): 10},
        reference_x=np.array([0.0, 0.0], dtype=float),
        diversity_weight=0.0,
        saturation_penalty=0.2,
    )
    assert selected.label == "alternate"


def test_select_restart_seed_ignores_non_finite_objective() -> None:
    candidates = [
        RestartSeedCandidate(
            label="bad",
            x=np.array([0.0, 0.0], dtype=float),
            objective=float("nan"),
            feasibility=0.0,
        ),
        RestartSeedCandidate(
            label="good",
            x=np.array([0.1, 0.0], dtype=float),
            objective=2.0,
            feasibility=0.0,
        ),
    ]

    selected, decision = select_restart_seed(
        candidates,
        problem="p1",
        selection_counts={},
        reference_x=np.array([0.0, 0.0], dtype=float),
        diversity_weight=0.0,
    )
    assert selected.label == "good"
    bad_score = next(row for row in decision["scores"] if row["label"] == "bad")
    assert bad_score["objective_score"] == 0.0


def test_select_restart_seed_ignores_non_finite_feasibility() -> None:
    candidates = [
        RestartSeedCandidate(
            label="bad",
            x=np.array([0.0, 0.0], dtype=float),
            objective=2.0,
            feasibility=float("inf"),
        ),
        RestartSeedCandidate(
            label="good",
            x=np.array([0.1, 0.0], dtype=float),
            objective=2.0,
            feasibility=0.0,
        ),
    ]

    selected, decision = select_restart_seed(
        candidates,
        problem="p1",
        selection_counts={},
        reference_x=np.array([0.0, 0.0], dtype=float),
        objective_weight=0.0,
        diversity_weight=0.0,
    )
    assert selected.label == "good"
    bad_score = next(row for row in decision["scores"] if row["label"] == "bad")
    assert bad_score["feasibility_score"] == 0.0


def test_select_restart_seed_rejects_empty_candidate_list() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        select_restart_seed(
            [],
            problem="p1",
            selection_counts={},
            reference_x=np.array([0.0], dtype=float),
        )


def test_select_restart_seed_penalizes_same_identity_across_labels() -> None:
    shared = np.array([0.0, 0.0], dtype=float)
    candidates = [
        RestartSeedCandidate(
            label="state",
            x=shared,
            objective=2.0,
            feasibility=0.0,
        ),
        RestartSeedCandidate(
            label="best_low",
            x=shared,
            objective=2.0,
            feasibility=0.0,
        ),
        RestartSeedCandidate(
            label="alternate",
            x=np.array([0.1, 0.0], dtype=float),
            objective=2.0,
            feasibility=0.0,
        ),
    ]

    _, first_decision = select_restart_seed(
        candidates,
        problem="p1",
        selection_counts={},
        reference_x=np.array([0.0, 0.0], dtype=float),
        objective_weight=0.0,
        diversity_weight=0.0,
    )
    selected_identity = str(first_decision["selected_identity"])
    selected, _ = select_restart_seed(
        candidates,
        problem="p1",
        selection_counts={selected_identity: 10},
        reference_x=np.array([0.0, 0.0], dtype=float),
        objective_weight=0.0,
        diversity_weight=0.0,
        saturation_penalty=0.2,
    )
    assert selected.label == "alternate"


def test_select_restart_seed_all_non_finite_totals_stay_finite() -> None:
    candidates = [
        RestartSeedCandidate(
            label="nan_1",
            x=np.array([np.nan], dtype=float),
            objective=float("nan"),
            feasibility=float("nan"),
        ),
        RestartSeedCandidate(
            label="nan_2",
            x=np.array([np.nan], dtype=float),
            objective=float("nan"),
            feasibility=float("nan"),
        ),
    ]

    selected, decision = select_restart_seed(
        candidates,
        problem="p1",
        selection_counts={},
        reference_x=np.array([0.0], dtype=float),
    )
    assert selected.label in {"nan_1", "nan_2"}
    assert all(np.isfinite(score["total"]) for score in decision["scores"])


def test_select_restart_seed_treats_signed_zero_as_same_identity() -> None:
    candidates = [
        RestartSeedCandidate(
            label="plus_zero",
            x=np.array([0.0], dtype=float),
            objective=2.0,
            feasibility=0.0,
        ),
        RestartSeedCandidate(
            label="minus_zero",
            x=np.array([-0.0], dtype=float),
            objective=2.0,
            feasibility=0.0,
        ),
        RestartSeedCandidate(
            label="alternate",
            x=np.array([0.1], dtype=float),
            objective=2.0,
            feasibility=0.0,
        ),
    ]

    _, baseline = select_restart_seed(
        candidates,
        problem="p1",
        selection_counts={},
        reference_x=np.array([0.0], dtype=float),
        objective_weight=0.0,
        diversity_weight=0.0,
    )
    selected_identity = str(baseline["selected_identity"])
    selected, decision = select_restart_seed(
        candidates,
        problem="p1",
        selection_counts={selected_identity: 10},
        reference_x=np.array([0.0], dtype=float),
        objective_weight=0.0,
        diversity_weight=0.0,
        saturation_penalty=0.2,
    )

    assert selected.label == "alternate"
    identities = {
        row["label"]: row["identity"]
        for row in decision["scores"]
        if row["label"] in {"plus_zero", "minus_zero"}
    }
    assert identities["plus_zero"] == identities["minus_zero"]
