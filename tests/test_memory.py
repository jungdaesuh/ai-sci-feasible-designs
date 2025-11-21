import json
import sqlite3

import pytest

from ai_scientist import memory


def test_world_model_records_cycle(tmp_path):
    db_path = tmp_path / "world.db"
    payload = {"problem": "p1", "cycles": 1}
    with memory.WorldModel(db_path) as wm:
        experiment_id = wm.start_experiment(payload, "deadbeef")
        wm.record_cycle(
            experiment_id=experiment_id,
            cycle_number=1,
            screen_evals=2,
            promoted_evals=1,
            high_fidelity_evals=1,
            wall_seconds=0.5,
            best_params={"foo": 1},
            best_evaluation={
                "stage": "promote",
                "objective": 2.0,
                "feasibility": 0.0,
                "score": 0.6,
                "metrics": {"example": 1},
                "design_hash": "cyclehash-best",
            },
            seed=7,
            problem="p1",
        )
        usage = wm.budget_usage(experiment_id)
        assert usage.screen_evals == 2
        assert usage.high_fidelity_evals == 1
        assert wm.cycles_completed(experiment_id) == 1
    conn = sqlite3.connect(db_path)
    assert conn.execute("SELECT COUNT(*) FROM budgets").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0] == 1


def test_record_cycle_can_skip_candidate_logging(tmp_path):
    db_path = tmp_path / "world.db"
    payload = {"problem": "p1", "cycles": 1}
    with memory.WorldModel(db_path) as wm:
        experiment_id = wm.start_experiment(payload, "deadbeef")
        wm.record_cycle(
            experiment_id=experiment_id,
            cycle_number=1,
            screen_evals=3,
            promoted_evals=0,
            high_fidelity_evals=0,
            wall_seconds=0.2,
            best_params={"foo": 2},
            best_evaluation={
                "stage": "screen",
                "objective": 1.5,
                "feasibility": 0.1,
                "score": 0.4,
                "metrics": {"example": 2},
                "design_hash": "skiphash",
            },
            seed=8,
            problem="p1",
            log_best_candidate=False,
        )
        usage = wm.budget_usage(experiment_id)
        assert usage.screen_evals == 3
        assert usage.promoted_evals == 0
        assert wm.cycles_completed(experiment_id) == 1
        assert wm._conn.execute("SELECT COUNT(*) FROM budgets").fetchone()[0] == 1
        assert wm._conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0] == 0


def test_memory_helpers_graph_and_metrics(tmp_path):
    db_path = tmp_path / "world.db"
    with memory.WorldModel(db_path) as wm:
        experiment_id = wm.start_experiment({"problem": "p1"}, "deadbeef")
        candidate_id, metrics_id = wm.log_candidate(
            experiment_id=experiment_id,
            problem="p1",
            params={"foo": 1},
            seed=42,
            status="promote",
            evaluation={
                "stage": "promote",
                "objective": 1.2,
                "feasibility": 0.0,
                "score": 0.5,
                "metrics": {"x": 1},
            },
            design_hash="graph-hash",
        )
        additional_metrics_id = wm.log_metrics(
            candidate_id,
            {"x": 2, "feasibility": 0.0, "objective": 1.1, "hv": 0.01},
        )
        assert additional_metrics_id is not None
        metrics_count = wm._conn.execute(
            "SELECT COUNT(*) FROM metrics WHERE candidate_id = ?", (candidate_id,)
        ).fetchone()[0]
        assert metrics_count == 2

        assert metrics_id is not None
        wm.upsert_pareto(experiment_id, candidate_id)
        wm._conn.execute(
            "INSERT INTO citations (experiment_id, source_path, anchor, quote) VALUES (?, ?, ?, ?)",
            (experiment_id, "ref.md", "Anchor One", "quote details"),
        )
        wm._conn.execute(
            "INSERT INTO artifacts (experiment_id, path, kind) VALUES (?, ?, ?)",
            (experiment_id, "artifact.png", "figure"),
        )
        wm._conn.commit()

        graph = wm.to_networkx(experiment_id)
        exp_node = f"experiment:{experiment_id}"
        cand_node = f"candidate:{candidate_id}"
        assert exp_node in graph.nodes
        assert cand_node in graph.nodes
        assert any(attrs.get("type") == "metrics" for attrs in graph.nodes.values())
        assert any(edge[2].get("relation") == "pareto_member" for edge in graph.edges)
        assert any(edge[2].get("relation") == "cites" for edge in graph.edges)
        assert any(edge[2].get("relation") == "produces" for edge in graph.edges)


def test_world_model_surrogate_training_data(tmp_path):
    db_path = tmp_path / "world.db"
    with memory.WorldModel(db_path) as wm:
        experiment_id = wm.start_experiment({"problem": "p3"}, "deadbeef")
        candidate_id, metrics_id = wm.log_candidate(
            experiment_id=experiment_id,
            problem="p3",
            params={"foo": 1},
            seed=11,
            status="p3",
            evaluation={
                "stage": "p3",
                "objective": 2.1,
                "feasibility": 0.0,
                "score": 1.0,
                "metrics": {
                    "aspect_ratio": 2.0,
                    "minimum_normalized_magnetic_gradient_scale_length": 3.0,
                },
            },
            design_hash="surrogate-hash",
        )
        wm.log_metrics(
            candidate_id,
            {"aspect_ratio": 2.0},
            hv=0.3,
            objective=2.1,
        )
        wm.log_metrics(
            candidate_id,
            {"aspect_ratio": 1.0},
            hv=0.1,
            objective=1.0,
        )
        history = wm.surrogate_training_data(target="hv", problem="p3")
        assert len(history) == 2
        assert history[0][1] == 0.3
        objective_history = wm.surrogate_training_data(target="objective", problem="p3")
        assert all(value >= 1.0 for _, value in objective_history)


def test_record_cycle_hv_and_pareto_archive(tmp_path):
    db_path = tmp_path / "world.db"
    with memory.WorldModel(db_path) as wm:
        experiment_id = wm.start_experiment({"problem": "p3"}, "feedface")
        wm.record_cycle_hv(
            experiment_id=experiment_id,
            cycle_number=1,
            hv_score=0.75,
            reference_point=(1.0, 20.0),
            pareto_entries=(
                {
                    "seed": 1.0,
                    "gradient": 1.4,
                    "aspect_ratio": 2.1,
                    "objective": 1.0,
                    "feasibility": 0.0,
                },
            ),
            n_feasible=2,
            n_archive=1,
        )
        wm.record_pareto_archive(
            experiment_id=experiment_id,
            cycle_number=1,
            entries=[
                {
                    "design_hash": "archive-hash",
                    "fidelity": "promote",
                    "gradient": 1.4,
                    "aspect": 2.1,
                    "metrics_id": None,
                    "git_sha": "deadbeef",
                    "constellaration_sha": "cafebabe",
                    "settings_json": "{}",
                    "seed": 99,
                }
            ],
        )
        hv_row = tuple(
            wm._conn.execute(
                "SELECT hv_value, n_feasible, n_archive FROM cycle_hv WHERE experiment_id = ?",
                (experiment_id,),
            ).fetchone()
        )
        assert hv_row == (0.75, 2, 1)
        archive_count = wm._conn.execute(
            "SELECT COUNT(*) FROM pareto_archive WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone()[0]
        assert archive_count == 1


def test_transaction_rolls_back_on_failure(tmp_path):
    db_path = tmp_path / "world.db"
    with memory.WorldModel(db_path) as wm:
        experiment_id = wm.start_experiment({"problem": "p1"}, "feedbead")
        with pytest.raises(RuntimeError):
            with wm.transaction():
                wm.record_cycle(
                    experiment_id=experiment_id,
                    cycle_number=1,
                    screen_evals=1,
                    promoted_evals=0,
                    high_fidelity_evals=0,
                    wall_seconds=0.1,
                    best_params={"foo": 1},
                    best_evaluation={
                        "stage": "screen",
                        "objective": 1.0,
                        "feasibility": 0.0,
                        "score": 1.0,
                        "metrics": {"foo": 1},
                        "design_hash": "rollback-hash",
                    },
                    seed=1,
                    problem="p1",
                    commit=False,
                )
                raise RuntimeError("boom")
        budgets = wm._conn.execute("SELECT COUNT(*) FROM budgets").fetchone()[0]
        assert budgets == 0


def test_log_statement_and_stage_history(tmp_path):
    db_path = tmp_path / "world.db"
    with memory.WorldModel(db_path) as wm:
        experiment_id = wm.start_experiment({"problem": "p1"}, "cafe123")
        wm.record_stage_history(experiment_id, cycle=1, stage="s1")
        statement_id = wm.log_statement(
            experiment_id=experiment_id,
            cycle=1,
            stage="s1",
            text="Replayed metrics claim for verification.",
            status="SUPPORTED",
            tool_name="evaluate_p1",
            tool_input={"params": {"foo": 1.0}, "stage": "screen"},
            seed=7,
            git_sha="deadbeef",
            repro_cmd="python -m ai_scientist.runner",
        )
        assert statement_id is not None
        statements = wm.statements_for_cycle(experiment_id, 1)
        assert len(statements) == 1
        assert statements[0].status == "SUPPORTED"
        stage_row = wm._conn.execute(
            "SELECT stage FROM stage_history WHERE experiment_id = ? AND cycle = ?",
            (experiment_id, 1),
        ).fetchone()
        assert stage_row["stage"] == "s1"


def test_stage_history_method_returns_all_cycles(tmp_path):
    db_path = tmp_path / "world.db"
    with memory.WorldModel(db_path) as wm:
        experiment_id = wm.start_experiment({"problem": "p1"}, "cafe123")
        wm.record_stage_history(experiment_id, cycle=1, stage="s1")
        wm.record_stage_history(experiment_id, cycle=2, stage="s2")
        entries = wm.stage_history(experiment_id)
        assert [entry.stage for entry in entries] == ["s1", "s2"]
        assert [entry.cycle for entry in entries] == [1, 2]
        assert all(entry.selected_at for entry in entries)


def _make_pareto_entry() -> dict[str, float]:
    return {
        "gradient": 1.0,
        "aspect_ratio": 2.0,
    }


def test_previous_best_hv_returns_max_before_cycle(tmp_path):
    db_path = tmp_path / "world.db"
    with memory.WorldModel(db_path) as wm:
        experiment_id = wm.start_experiment({"problem": "p1"}, "cafe123")
        wm.record_cycle_hv(
            experiment_id=experiment_id,
            cycle_number=1,
            hv_score=0.12,
            reference_point=(1.0, 20.0),
            pareto_entries=[_make_pareto_entry()],
            n_feasible=1,
            n_archive=1,
        )
        wm.record_cycle_hv(
            experiment_id=experiment_id,
            cycle_number=2,
            hv_score=0.08,
            reference_point=(1.0, 20.0),
            pareto_entries=[_make_pareto_entry()],
            n_feasible=1,
            n_archive=1,
        )
        assert wm.previous_best_hv(experiment_id, cycle_number=2) == pytest.approx(0.12)
        assert wm.previous_best_hv(experiment_id, cycle_number=3) == pytest.approx(0.12)


def test_record_deterministic_snapshot(tmp_path):
    db_path = tmp_path / "world.db"
    snapshot_payload = {"config": "deterministic", "random_seed": 17}
    with memory.WorldModel(db_path) as wm:
        experiment_id = wm.start_experiment({"problem": "p3"}, "c0ffeebe")
        wm.record_deterministic_snapshot(
            experiment_id=experiment_id,
            cycle_number=1,
            snapshot=snapshot_payload,
            constellaration_sha="deadcafe",
            seed=17,
        )
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT snapshot_json, constellaration_sha, seed FROM deterministic_snapshots WHERE experiment_id = ?",
        (experiment_id,),
    ).fetchone()
    assert row is not None
    assert json.loads(row[0]) == snapshot_payload
    assert row[1] == "deadcafe"
    assert row[2] == 17
