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


def _candidate_row(
    *,
    candidate_id: int,
    design_hash: str,
    feasibility: float,
    is_feasible: bool,
    lgradb: float,
    aspect: float,
) -> governor.CandidateRow:
    return governor.CandidateRow(
        candidate_id=candidate_id,
        design_hash=design_hash,
        seed=candidate_id,
        feasibility=feasibility,
        is_feasible=is_feasible,
        lgradb=lgradb,
        aspect=aspect,
        violations={"mirror": 0.1},
        metrics={"aspect_ratio": aspect},
        meta={},
        lineage_parent_hashes=[],
        novelty_score=0.2,
        operator_family="blend",
        model_route="governor_static_recipe/mirror",
    )


def test_select_parent_group_does_not_collapse_to_broad() -> None:
    near = _candidate_row(
        candidate_id=1,
        design_hash="near",
        feasibility=0.2,
        is_feasible=False,
        lgradb=1.5,
        aspect=8.0,
    )
    feasible = _candidate_row(
        candidate_id=2,
        design_hash="feasible",
        feasibility=0.0,
        is_feasible=True,
        lgradb=1.0,
        aspect=9.0,
    )

    selected = governor._select_parent_group(
        [near, feasible],
        max_feasibility=0.5,
    )

    assert selected is not None
    assert selected.group in {"near_feasible", "feasible"}
    assert selected.group != "broad"


def test_select_parent_group_uses_broad_as_fallback() -> None:
    broad_only = _candidate_row(
        candidate_id=3,
        design_hash="broad",
        feasibility=0.005,
        is_feasible=False,
        lgradb=1.2,
        aspect=8.5,
    )

    selected = governor._select_parent_group(
        [broad_only],
        max_feasibility=0.5,
    )

    assert selected is not None
    assert selected.group == "broad"
    assert selected.candidate_count == 1


def test_adaptive_recipe_emits_adaptive_route_and_policy(tmp_path: Path) -> None:
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

    cmds, decision = governor._select_adaptive_recipe(
        db=db_path,
        experiment_id=experiment_id,
        run_dir=run_dir,
        batch_id=1,
        seed_base=1000,
        focus=focus,
        partner=None,
        history_candidates=[focus],
        parent_group="near_feasible",
    )

    assert decision["recipe_mode"] == "adaptive"
    assert (
        decision["adaptive_policy"]["strategy"]
        == "parent_group_operator_bandit_novelty_gate"
    )
    route = str(decision["model_route"])
    assert route.startswith("governor_adaptive/near_feasible/")
    for cmd in cmds:
        assert "--model-route" in cmd.argv
        idx = cmd.argv.index("--model-route")
        assert cmd.argv[idx + 1].startswith("governor_adaptive/near_feasible/")


def test_adaptive_recipe_falls_back_when_novelty_gate_rejects_all(tmp_path: Path) -> None:
    db_path = tmp_path / "wm.sqlite"
    run_dir = tmp_path / "run"
    (run_dir / "candidates").mkdir(parents=True, exist_ok=True)
    experiment_id = 12
    design_hash = "focus-hash-2"
    _prepare_db(db_path, experiment_id=experiment_id, design_hash=design_hash)

    focus = governor.CandidateRow(
        candidate_id=2,
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

    original_gate = governor._ADAPTIVE_NOVELTY_GATE
    try:
        governor._ADAPTIVE_NOVELTY_GATE = 0.5
        cmds, decision = governor._select_adaptive_recipe(
            db=db_path,
            experiment_id=experiment_id,
            run_dir=run_dir,
            batch_id=1,
            seed_base=2000,
            focus=focus,
            partner=None,
            history_candidates=[focus],
            parent_group="near_feasible",
        )
    finally:
        governor._ADAPTIVE_NOVELTY_GATE = original_gate

    assert cmds
    assert decision["recipe_mode"] == "adaptive"
    assert decision["adaptive_policy"]["strategy"] == "static_delegate_fallback"
    assert str(decision["model_route"]).startswith(
        "governor_adaptive_scaffold/static_delegate/"
    )


def test_adaptive_recipe_handles_feasible_focus_without_violations(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "wm.sqlite"
    run_dir = tmp_path / "run"
    (run_dir / "candidates").mkdir(parents=True, exist_ok=True)
    experiment_id = 13
    design_hash = "focus-feasible"
    _prepare_db(db_path, experiment_id=experiment_id, design_hash=design_hash)

    focus = governor.CandidateRow(
        candidate_id=3,
        design_hash=design_hash,
        seed=1,
        feasibility=0.0,
        is_feasible=True,
        lgradb=1.5,
        aspect=7.5,
        violations={},
        metrics={"aspect_ratio": 7.5},
        meta={},
        lineage_parent_hashes=[],
        novelty_score=0.2,
        operator_family="scale_groups",
        model_route="governor_static_recipe/objective",
    )

    cmds, decision = governor._select_adaptive_recipe(
        db=db_path,
        experiment_id=experiment_id,
        run_dir=run_dir,
        batch_id=1,
        seed_base=3000,
        focus=focus,
        partner=None,
        history_candidates=[focus],
        parent_group="feasible",
    )

    assert cmds
    assert decision["recipe_mode"] == "adaptive"
    assert (
        decision["adaptive_policy"]["strategy"]
        == "parent_group_operator_bandit_novelty_gate"
    )
    assert str(decision["model_route"]).startswith("governor_adaptive/feasible/unknown")


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
            governor.CandidateRow(
                candidate_id=3,
                design_hash="c",
                seed=3,
                feasibility=0.1,
                is_feasible=False,
                lgradb=1.1,
                aspect=8.5,
                violations={},
                metrics={},
                meta={},
                lineage_parent_hashes=[],
                novelty_score=0.2,
                operator_family="blend",
                model_route="governor_adaptive/near_feasible/mirror",
            ),
        ],
        novelty_reject_threshold=0.05,
    )

    assert summary["adaptive_path_rows"] == 2
    assert summary["static_path_rows"] == 1
    assert summary["fallback_static_delegate_rows"] == 1
