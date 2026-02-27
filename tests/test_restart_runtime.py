from __future__ import annotations

import json

import numpy as np
import pytest

from ai_scientist.restart_runtime import (
    append_restart_history,
    select_adaptive_restart_runtime,
)


def test_select_adaptive_restart_runtime_updates_identity_counts() -> None:
    x_state = np.array([0.0, 0.0], dtype=float)
    x_alt = np.array([0.1, 0.0], dtype=float)

    x_center, selected_label, selected_identity, decision, counts = (
        select_adaptive_restart_runtime(
            problem="p1",
            state_x=x_state,
            state_objective=2.0,
            state_feasibility=0.0,
            best_violation_x=x_state,
            best_violation_objective=2.0,
            best_violation_feasibility=0.0,
            best_low_x=x_alt,
            best_low_objective=1.8,
            best_low_feasibility=0.0,
            best_high_x=None,
            best_high_objective=None,
            best_high_feasibility=None,
            selection_counts={},
            feasibility_weight=0.55,
            objective_weight=0.35,
            diversity_weight=0.10,
            saturation_penalty=0.15,
        )
    )

    assert selected_label in {"state", "best_violation", "best_low"}
    assert isinstance(selected_identity, str) and selected_identity
    assert np.allclose(x_center, x_alt)
    assert counts[selected_identity] == 1
    assert decision["selected_identity"] == selected_identity


def test_append_restart_history_writes_jsonl(tmp_path) -> None:
    path = tmp_path / "restart_history.jsonl"
    append_restart_history(
        path,
        outer=3,
        selected_seed="best_low",
        selected_seed_identity="x:test",
        counts={"x:test": 2},
        decision={"selected_label": "best_low"},
    )

    lines = path.read_text().strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["outer"] == 3
    assert payload["selected_seed"] == "best_low"
    assert payload["selected_seed_identity"] == "x:test"
    assert payload["counts"]["x:test"] == 2


def test_select_adaptive_restart_runtime_p2_ignores_failure_sentinel() -> None:
    x_state = np.array([0.0, 0.0], dtype=float)
    x_alt = np.array([0.1, 0.0], dtype=float)

    x_center, selected_label, _selected_identity, _decision, _counts = (
        select_adaptive_restart_runtime(
            problem="p2",
            state_x=x_state,
            state_objective=1e9,
            state_feasibility=1e9,
            best_violation_x=x_alt,
            best_violation_objective=4.0,
            best_violation_feasibility=1e9,
            best_low_x=x_alt,
            best_low_objective=5.0,
            best_low_feasibility=1e9,
            best_high_x=None,
            best_high_objective=None,
            best_high_feasibility=None,
            selection_counts={},
            feasibility_weight=0.0,
            objective_weight=1.0,
            diversity_weight=0.0,
            saturation_penalty=0.0,
        )
    )

    assert selected_label in {"best_violation", "best_low"}
    assert np.allclose(x_center, x_alt)


def test_select_adaptive_restart_runtime_p2_uses_maximize_space_from_caller() -> None:
    x_state = np.array([0.0], dtype=float)
    x_alt = np.array([0.1], dtype=float)

    x_center, selected_label, _selected_identity, _decision, _counts = (
        select_adaptive_restart_runtime(
            problem="p2",
            state_x=x_state,
            state_objective=8.0,
            state_feasibility=0.0,
            best_violation_x=x_alt,
            best_violation_objective=6.0,
            best_violation_feasibility=0.0,
            best_low_x=x_alt,
            best_low_objective=7.0,
            best_low_feasibility=0.0,
            best_high_x=None,
            best_high_objective=None,
            best_high_feasibility=None,
            selection_counts={},
            feasibility_weight=0.0,
            objective_weight=1.0,
            diversity_weight=0.0,
            saturation_penalty=0.0,
        )
    )

    assert selected_label == "state"
    assert np.allclose(x_center, x_state)


def test_select_adaptive_restart_runtime_p2_demotes_non_finite_objective() -> None:
    x_state = np.array([0.0], dtype=float)
    x_alt = np.array([0.1], dtype=float)

    x_center, selected_label, _selected_identity, _decision, _counts = (
        select_adaptive_restart_runtime(
            problem="p2",
            state_x=x_state,
            state_objective=5.0,
            state_feasibility=0.0,
            best_violation_x=x_alt,
            best_violation_objective=float("-inf"),
            best_violation_feasibility=0.0,
            best_low_x=x_alt,
            best_low_objective=float("-inf"),
            best_low_feasibility=0.0,
            best_high_x=None,
            best_high_objective=None,
            best_high_feasibility=None,
            selection_counts={},
            feasibility_weight=0.0,
            objective_weight=1.0,
            diversity_weight=0.0,
            saturation_penalty=0.0,
        )
    )

    assert selected_label == "state"
    assert np.allclose(x_center, x_state)


def test_select_adaptive_restart_runtime_includes_finite_best_high() -> None:
    x_center, selected_label, _selected_identity, decision, _counts = (
        select_adaptive_restart_runtime(
            problem="p1",
            state_x=np.array([0.0], dtype=float),
            state_objective=3.0,
            state_feasibility=0.0,
            best_violation_x=np.array([0.0], dtype=float),
            best_violation_objective=3.0,
            best_violation_feasibility=0.0,
            best_low_x=np.array([0.2], dtype=float),
            best_low_objective=2.5,
            best_low_feasibility=0.0,
            best_high_x=np.array([0.3], dtype=float),
            best_high_objective=1.0,
            best_high_feasibility=0.0,
            selection_counts={},
            feasibility_weight=0.0,
            objective_weight=1.0,
            diversity_weight=0.0,
            saturation_penalty=0.0,
        )
    )
    assert selected_label == "best_high"
    assert np.allclose(x_center, np.array([0.3], dtype=float))
    labels = {row["label"] for row in decision["scores"]}
    assert "best_high" in labels


def test_select_adaptive_restart_runtime_skips_non_finite_best_high() -> None:
    x_center, selected_label, _selected_identity, decision, _counts = (
        select_adaptive_restart_runtime(
            problem="p1",
            state_x=np.array([0.0], dtype=float),
            state_objective=3.0,
            state_feasibility=0.0,
            best_violation_x=np.array([0.0], dtype=float),
            best_violation_objective=3.0,
            best_violation_feasibility=0.0,
            best_low_x=np.array([0.2], dtype=float),
            best_low_objective=2.5,
            best_low_feasibility=0.0,
            best_high_x=np.array([0.3], dtype=float),
            best_high_objective=float("nan"),
            best_high_feasibility=0.0,
            selection_counts={},
            feasibility_weight=0.0,
            objective_weight=1.0,
            diversity_weight=0.0,
            saturation_penalty=0.0,
        )
    )
    assert selected_label == "best_low"
    assert np.allclose(x_center, np.array([0.2], dtype=float))
    labels = {row["label"] for row in decision["scores"]}
    assert "best_high" not in labels


def test_select_adaptive_restart_runtime_applies_novelty_distance_gate() -> None:
    x_state = np.array([0.0, 0.0], dtype=float)
    x_far = np.array([0.2, 0.0], dtype=float)

    x_center, selected_label, _selected_identity, decision, _counts = (
        select_adaptive_restart_runtime(
            problem="p1",
            state_x=x_state,
            state_objective=2.0,
            state_feasibility=0.0,
            best_violation_x=x_state,
            best_violation_objective=2.0,
            best_violation_feasibility=0.0,
            best_low_x=x_far,
            best_low_objective=2.5,
            best_low_feasibility=0.0,
            best_high_x=None,
            best_high_objective=None,
            best_high_feasibility=None,
            selection_counts={},
            feasibility_weight=0.0,
            objective_weight=1.0,
            diversity_weight=0.0,
            saturation_penalty=0.0,
            novelty_min_distance=0.1,
            novelty_feasibility_max=float("inf"),
        )
    )

    assert selected_label == "best_low"
    assert np.allclose(x_center, x_far)
    novelty_gate = decision["novelty_gate"]
    assert novelty_gate["enabled"] is True
    assert novelty_gate["fallback_to_ungated"] is False
    assert novelty_gate["kept_count"] == 1


def test_select_adaptive_restart_runtime_novelty_gate_falls_back_when_empty() -> None:
    x_state = np.array([0.0, 0.0], dtype=float)

    x_center, selected_label, _selected_identity, decision, _counts = (
        select_adaptive_restart_runtime(
            problem="p1",
            state_x=x_state,
            state_objective=2.0,
            state_feasibility=0.0,
            best_violation_x=x_state,
            best_violation_objective=3.0,
            best_violation_feasibility=0.0,
            best_low_x=x_state,
            best_low_objective=4.0,
            best_low_feasibility=0.0,
            best_high_x=None,
            best_high_objective=None,
            best_high_feasibility=None,
            selection_counts={},
            feasibility_weight=0.0,
            objective_weight=1.0,
            diversity_weight=0.0,
            saturation_penalty=0.0,
            novelty_min_distance=0.1,
            novelty_feasibility_max=float("inf"),
        )
    )

    assert selected_label == "state"
    assert np.allclose(x_center, x_state)
    novelty_gate = decision["novelty_gate"]
    assert novelty_gate["enabled"] is True
    assert novelty_gate["fallback_to_ungated"] is True


def test_select_adaptive_restart_runtime_applies_novelty_feasibility_gate() -> None:
    x_state = np.array([0.0, 0.0], dtype=float)
    x_alt = np.array([0.2, 0.0], dtype=float)
    x_rejected = np.array([0.3, 0.0], dtype=float)

    _x_center, selected_label, _selected_identity, decision, _counts = (
        select_adaptive_restart_runtime(
            problem="p1",
            state_x=x_state,
            state_objective=2.0,
            state_feasibility=0.0,
            best_violation_x=x_alt,
            best_violation_objective=3.0,
            best_violation_feasibility=0.0,
            best_low_x=x_rejected,
            best_low_objective=1.0,
            best_low_feasibility=0.5,
            best_high_x=None,
            best_high_objective=None,
            best_high_feasibility=None,
            selection_counts={},
            feasibility_weight=0.0,
            objective_weight=1.0,
            diversity_weight=0.0,
            saturation_penalty=0.0,
            novelty_min_distance=0.0,
            novelty_feasibility_max=0.1,
        )
    )

    assert selected_label == "state"
    novelty_gate = decision["novelty_gate"]
    assert novelty_gate["enabled"] is True
    assert novelty_gate["fallback_to_ungated"] is False
    assert novelty_gate["kept_count"] == 2
    rejected_row = next(
        row for row in novelty_gate["rows"] if row["label"] == "best_low"
    )
    assert rejected_row["accepted"] is False
    assert rejected_row["pass_feasibility"] is False


def test_select_adaptive_restart_runtime_rejects_negative_novelty_distance() -> None:
    with pytest.raises(ValueError, match="novelty_min_distance"):
        select_adaptive_restart_runtime(
            problem="p1",
            state_x=np.array([0.0], dtype=float),
            state_objective=2.0,
            state_feasibility=0.0,
            best_violation_x=np.array([0.0], dtype=float),
            best_violation_objective=2.0,
            best_violation_feasibility=0.0,
            best_low_x=np.array([0.1], dtype=float),
            best_low_objective=1.9,
            best_low_feasibility=0.0,
            best_high_x=None,
            best_high_objective=None,
            best_high_feasibility=None,
            selection_counts={},
            feasibility_weight=0.55,
            objective_weight=0.35,
            diversity_weight=0.10,
            saturation_penalty=0.15,
            novelty_min_distance=-0.01,
            novelty_feasibility_max=float("inf"),
        )


def test_select_adaptive_restart_runtime_rejects_invalid_judge_mode() -> None:
    with pytest.raises(ValueError, match="novelty_judge_mode"):
        select_adaptive_restart_runtime(
            problem="p1",
            state_x=np.array([0.0], dtype=float),
            state_objective=2.0,
            state_feasibility=0.0,
            best_violation_x=np.array([0.0], dtype=float),
            best_violation_objective=2.0,
            best_violation_feasibility=0.0,
            best_low_x=np.array([0.1], dtype=float),
            best_low_objective=1.9,
            best_low_feasibility=0.0,
            best_high_x=None,
            best_high_objective=None,
            best_high_feasibility=None,
            selection_counts={},
            feasibility_weight=0.55,
            objective_weight=0.35,
            diversity_weight=0.10,
            saturation_penalty=0.15,
            novelty_min_distance=0.05,
            novelty_feasibility_max=float("inf"),
            novelty_judge_mode="not-a-mode",
        )
