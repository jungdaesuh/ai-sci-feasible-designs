# ruff: noqa: E402
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

from ai_scientist.memory.schema import init_db

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import p3_governor as governor


def _boundary() -> dict:
    return {
        "r_cos": [[1.0, 0.0, 0.0], [0.0, 0.01, 0.0]],
        "z_sin": [[0.0, 0.0, 0.0], [0.0, 0.01, 0.0]],
        "n_field_periods": 3,
        "is_stellarator_symmetric": True,
    }


def _prepare_db(db_path: Path, *, experiment_id: int, design_hash: str) -> None:
    init_db(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            INSERT INTO experiments
            (id, started_at, config_json, git_sha, constellaration_sha, notes)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_id,
                "2026-02-25T00:00:00+00:00",
                "{}",
                "deadbeef",
                "const-sha",
                None,
            ),
        )
        conn.execute(
            """
            INSERT INTO candidates
            (experiment_id, problem, params_json, seed, status, design_hash, lineage_parent_hashes_json, novelty_score, operator_family, model_route)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_id,
                "p3",
                json.dumps(_boundary()),
                1,
                "done",
                design_hash,
                "[]",
                0.2,
                "blend",
                "governor_static_recipe/mirror",
            ),
        )
        conn.commit()
    finally:
        conn.close()


def test_adaptive_scaffold_uses_static_delegate_route(tmp_path: Path) -> None:
    db_path = tmp_path / "wm.sqlite"
    run_dir = tmp_path / "run"
    (run_dir / "candidates").mkdir(parents=True, exist_ok=True)
    experiment_id = 11
    design_hash = "focus-hash"
    _prepare_db(db_path, experiment_id=experiment_id, design_hash=design_hash)

    focus = governor.CandidateRow(
        candidate_id=1,
        design_hash=design_hash,
        seed=1,
        feasibility=0.2,
        is_feasible=False,
        lgradb=1.2,
        aspect=8.0,
        violations={"mirror": 0.1},
        metrics={"aspect_ratio": 8.0},
        meta={},
        lineage_parent_hashes=[],
        novelty_score=0.2,
        operator_family="blend",
        model_route="governor_static_recipe/mirror",
    )

    cmds, decision = governor._select_adaptive_recipe_scaffold(
        db=db_path,
        experiment_id=experiment_id,
        run_dir=run_dir,
        batch_id=1,
        seed_base=1000,
        focus=focus,
        partner=None,
    )

    assert decision["recipe_mode"] == "adaptive_scaffold"
    assert decision["adaptive_policy"]["strategy"] == "static_delegate"
    route = str(decision["model_route"])
    assert route.startswith("governor_adaptive_scaffold/static_delegate/")
    for cmd in cmds:
        assert "--model-route" in cmd.argv
        idx = cmd.argv.index("--model-route")
        assert cmd.argv[idx + 1].startswith("governor_adaptive_scaffold/static_delegate/")


def test_recent_data_plane_summary_tracks_fallback_delegate_usage() -> None:
    summary = governor._recent_data_plane_summary(
        [
            governor.CandidateRow(
                candidate_id=1,
                design_hash="a",
                seed=1,
                feasibility=0.2,
                is_feasible=False,
                lgradb=1.0,
                aspect=8.0,
                violations={},
                metrics={},
                meta={},
                lineage_parent_hashes=[],
                novelty_score=0.2,
                operator_family="blend",
                model_route="governor_adaptive_scaffold/static_delegate/mirror",
            ),
            governor.CandidateRow(
                candidate_id=2,
                design_hash="b",
                seed=2,
                feasibility=0.1,
                is_feasible=False,
                lgradb=1.1,
                aspect=8.5,
                violations={},
                metrics={},
                meta={},
                lineage_parent_hashes=[],
                novelty_score=0.1,
                operator_family="scale_groups",
                model_route="governor_static_recipe/mirror",
            ),
        ],
        novelty_reject_threshold=0.05,
    )

    assert summary["adaptive_path_rows"] == 1
    assert summary["static_path_rows"] == 1
    assert summary["fallback_static_delegate_rows"] == 1
