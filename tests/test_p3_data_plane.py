from __future__ import annotations

import math

from ai_scientist.p3_data_plane import DataPlaneSample, summarize_data_plane


def test_summarize_data_plane_counts_routes_and_fallback() -> None:
    summary = summarize_data_plane(
        [
            DataPlaneSample(
                has_lineage=True,
                novelty_score=0.2,
                operator_family="blend",
                model_route="governor_static_recipe/mirror",
            ),
            DataPlaneSample(
                has_lineage=False,
                novelty_score=0.01,
                operator_family="scale_groups",
                model_route="governor_static_recipe",
            ),
            DataPlaneSample(
                has_lineage=False,
                novelty_score=None,
                operator_family="blend",
                model_route="governor_adaptive_scaffold/static_delegate/mirror",
            ),
        ],
        novelty_reject_threshold=0.05,
    )

    assert summary["candidate_rows"] == 3
    assert summary["with_lineage"] == 1
    assert summary["with_novelty"] == 2
    assert summary["novelty_missing_count"] == 1
    assert summary["novelty_reject_count"] == 1
    assert summary["novelty_reject_rate"] == 0.5
    assert summary["static_path_rows"] == 2
    assert summary["adaptive_path_rows"] == 1
    assert summary["fallback_static_delegate_rows"] == 1


def test_summarize_data_plane_ignores_non_finite_novelty() -> None:
    summary = summarize_data_plane(
        [
            DataPlaneSample(
                has_lineage=False,
                novelty_score=math.nan,
                operator_family="blend",
                model_route="governor_static_recipe",
            ),
            DataPlaneSample(
                has_lineage=False,
                novelty_score=math.inf,
                operator_family="blend",
                model_route="governor_static_recipe",
            ),
            DataPlaneSample(
                has_lineage=False,
                novelty_score=None,
                operator_family="blend",
                model_route="governor_static_recipe",
            ),
        ],
        novelty_reject_threshold=0.05,
    )

    assert summary["with_novelty"] == 0
    assert summary["novelty_missing_count"] == 3
    assert summary["novelty_reject_count"] == 0
    assert summary["novelty_reject_rate"] is None
