from __future__ import annotations

from ai_scientist.reporting import _format_property_graph_section


def test_format_property_graph_section_includes_p3_telemetry_fields() -> None:
    summary = {
        "node_count": 5,
        "edge_count": 7,
        "citation_count": 0,
        "p3_data_plane": {
            "candidate_rows": 12,
            "with_lineage": 8,
            "with_novelty": 10,
            "novelty_missing_count": 2,
            "avg_novelty": 0.11,
            "novelty_reject_threshold": 0.05,
            "novelty_reject_count": 4,
            "novelty_reject_rate": 0.4,
            "static_path_rows": 9,
            "adaptive_path_rows": 3,
            "fallback_static_delegate_rows": 3,
            "operator_families": {"blend": 7},
            "model_routes": {"governor_static_recipe/mirror": 5},
        },
    }

    lines = _format_property_graph_section(summary)
    rendered = "\n".join(lines)
    assert "Missing novelty metadata: 2" in rendered
    assert "Novelty reject threshold: 0.05" in rendered
    assert "Novelty reject count: 4" in rendered
    assert "Novelty reject rate: 0.4" in rendered
    assert "Static path usage: 9" in rendered
    assert "Adaptive path usage: 3" in rendered
    assert "Fallback static delegate usage: 3" in rendered
