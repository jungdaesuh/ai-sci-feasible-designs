import json
import sqlite3

import pytest

from ai_scientist import memory
from ai_scientist.staged_governor import build_staged_seed_plan_from_snapshots


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


def test_log_candidate_persists_p3_data_plane_metadata(tmp_path):
    db_path = tmp_path / "world.db"
    with memory.WorldModel(db_path) as wm:
        experiment_id = wm.start_experiment({"problem": "p3"}, "deadbeef")
        wm.log_candidate(
            experiment_id=experiment_id,
            problem="p3",
            params={"r_cos": [[1.0]], "z_sin": [[0.1]]},
            seed=5,
            status="pending",
            evaluation={
                "stage": "pending",
                "objective": 1.0,
                "feasibility": 0.5,
                "metrics": {"aspect_ratio": 9.0},
            },
            design_hash="metadata-hash",
            lineage_parent_hashes=["parent-a", "parent-b"],
            novelty_score=0.12,
            operator_family="blend",
            model_route="governor_static_recipe/mirror",
        )
        row = wm._conn.execute(
            """
            SELECT lineage_parent_hashes_json, novelty_score, operator_family, model_route
            FROM candidates
            WHERE experiment_id = ?
            """,
            (experiment_id,),
        ).fetchone()
        assert row is not None
        assert json.loads(row["lineage_parent_hashes_json"]) == ["parent-a", "parent-b"]
        assert row["novelty_score"] == pytest.approx(0.12)
        assert row["operator_family"] == "blend"
        assert row["model_route"] == "governor_static_recipe/mirror"


def test_candidate_data_plane_summary_counts_metadata(tmp_path):
    db_path = tmp_path / "world.db"
    with memory.WorldModel(db_path) as wm:
        experiment_id = wm.start_experiment({"problem": "p3"}, "deadbeef")
        wm.log_candidate(
            experiment_id=experiment_id,
            problem="p3",
            params={"r_cos": [[1.0]], "z_sin": [[0.1]]},
            seed=1,
            status="pending",
            evaluation={
                "stage": "pending",
                "objective": 0.9,
                "feasibility": 0.2,
                "metrics": {"aspect_ratio": 8.0},
            },
            design_hash="p3-hash-a",
            lineage_parent_hashes=["p0"],
            novelty_score=0.2,
            operator_family="blend",
            model_route="governor_static_recipe/mirror",
        )
        wm.log_candidate(
            experiment_id=experiment_id,
            problem="p3",
            params={"r_cos": [[1.1]], "z_sin": [[0.2]]},
            seed=2,
            status="pending",
            evaluation={
                "stage": "pending",
                "objective": 1.1,
                "feasibility": 0.3,
                "metrics": {"aspect_ratio": 10.0},
            },
            design_hash="p3-hash-b",
            lineage_parent_hashes=[],
            novelty_score=0.01,
            operator_family="scale_groups",
            model_route="governor_static_recipe",
        )
        wm.log_candidate(
            experiment_id=experiment_id,
            problem="p3",
            params={"r_cos": [[1.2]], "z_sin": [[0.3]]},
            seed=3,
            status="pending",
            evaluation={
                "stage": "pending",
                "objective": 1.2,
                "feasibility": 0.4,
                "metrics": {"aspect_ratio": 9.0},
            },
            design_hash="p3-hash-c",
            lineage_parent_hashes=[],
            novelty_score=None,
            operator_family="blend",
            model_route="governor_adaptive_scaffold/static_delegate",
        )
        summary = wm.candidate_data_plane_summary(experiment_id, problem="p3")
        assert summary["candidate_rows"] == 3
        assert summary["with_lineage"] == 1
        assert summary["with_novelty"] == 2
        assert summary["novelty_missing_count"] == 1
        assert summary["avg_novelty"] == pytest.approx(0.105)
        assert summary["novelty_reject_threshold"] == pytest.approx(0.05)
        assert summary["novelty_reject_count"] == 1
        assert summary["novelty_reject_rate"] == pytest.approx(0.5)
        assert summary["operator_families"]["blend"] == 2
        assert summary["operator_families"]["scale_groups"] == 1
        assert summary["model_routes"]["governor_static_recipe/mirror"] == 1
        assert summary["model_routes"]["governor_static_recipe"] == 1
        assert (
            summary["model_routes"]["governor_adaptive_scaffold/static_delegate"] == 1
        )
        assert summary["static_path_rows"] == 2
        assert summary["adaptive_path_rows"] == 1
        assert summary["fallback_static_delegate_rows"] == 1
        strict_summary = wm.candidate_data_plane_summary(
            experiment_id,
            problem="p3",
            novelty_reject_threshold=0.25,
        )
        assert strict_summary["novelty_reject_threshold"] == pytest.approx(0.25)
        assert strict_summary["novelty_reject_count"] == 2
        assert strict_summary["novelty_reject_rate"] == pytest.approx(1.0)


def test_model_router_reward_event_persistence_and_summary(tmp_path):
    db_path = tmp_path / "world.db"
    with memory.WorldModel(db_path) as wm:
        experiment_id = wm.start_experiment({"problem": "p3"}, "deadbeef")
        wm.log_model_router_reward_event(
            experiment_id=experiment_id,
            problem="p3",
            model_route="governor_adaptive/near_feasible/mirror",
            window_size=20,
            previous_feasible_yield=10.0,
            current_feasible_yield=15.0,
            previous_hv=1.0,
            current_hv=1.5,
            reward=0.3,
            reward_components={"reward": 0.3, "relative_hv": 0.5},
        )
        summary = wm.model_router_reward_summary(experiment_id, problem="p3")
        assert summary["event_count"] == 1
        assert summary["sampled_event_count"] == 1
        assert summary["avg_reward"] == pytest.approx(0.3)
        assert summary["last_reward"] == pytest.approx(0.3)
        assert summary["model_routes"]["governor_adaptive/near_feasible/mirror"] == 1
        plane = wm.candidate_data_plane_summary(experiment_id, problem="p3")
        reward_summary = plane["model_router_reward"]
        assert reward_summary["event_count"] == 1
        assert reward_summary["sampled_event_count"] == 1
        assert reward_summary["last_reward"] == pytest.approx(0.3)


def test_model_router_reward_summary_reports_total_count_beyond_limit(tmp_path):
    db_path = tmp_path / "world_limit.db"
    with memory.WorldModel(db_path) as wm:
        experiment_id = wm.start_experiment({"problem": "p3"}, "deadbeef")
        for idx, reward in enumerate([0.1, 0.2, 0.3]):
            wm.log_model_router_reward_event(
                experiment_id=experiment_id,
                problem="p3",
                model_route="governor_adaptive/near_feasible/mirror",
                window_size=20,
                previous_feasible_yield=10.0 + idx,
                current_feasible_yield=11.0 + idx,
                previous_hv=1.0 + idx,
                current_hv=1.1 + idx,
                reward=reward,
                reward_components={"reward": reward},
            )

        summary = wm.model_router_reward_summary(experiment_id, problem="p3", limit=2)
        assert summary["event_count"] == 3
        assert summary["sampled_event_count"] == 2


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


def test_surrogate_training_data_experiment_isolation(tmp_path):
    """A3 FIX: Verify experiment_id filters training data correctly.

    This test ensures that surrogate training data is isolated per experiment
    for experimental hygiene, preventing transfer learning across experiments
    unless explicitly desired.
    """
    db_path = tmp_path / "world.db"
    with memory.WorldModel(db_path) as wm:
        # Create two experiments
        exp1 = wm.start_experiment({"problem": "p3"}, "sha_exp1")
        exp2 = wm.start_experiment({"problem": "p3"}, "sha_exp2")

        # Log candidates to experiment 1 with distinct hv values
        wm.log_candidate(
            experiment_id=exp1,
            problem="p3",
            params={"r_cos": [[1.0]], "z_sin": [[0.1]]},
            seed=1,
            status="promote",
            evaluation={
                "stage": "promote",
                "objective": 1.0,
                "feasibility": 0.0,
                "metrics": {"aspect_ratio": 8.0},
                "design_hash": "exp1_hash",
            },
            design_hash="exp1_hash",
        )
        wm.log_metrics(1, {"aspect_ratio": 8.0}, hv=0.5, objective=1.0)

        # Log candidates to experiment 2 with different hv values
        wm.log_candidate(
            experiment_id=exp2,
            problem="p3",
            params={"r_cos": [[2.0]], "z_sin": [[0.2]]},
            seed=2,
            status="screen",
            evaluation={
                "stage": "screen",
                "objective": 2.0,
                "feasibility": 0.01,
                "metrics": {"aspect_ratio": 10.0},
                "design_hash": "exp2_hash",
            },
            design_hash="exp2_hash",
        )
        wm.log_metrics(2, {"aspect_ratio": 10.0}, hv=0.8, objective=2.0)

        # Test isolation: filter by exp1
        data_exp1 = wm.surrogate_training_data(problem="p3", experiment_id=exp1)
        assert len(data_exp1) == 1, f"Expected 1 entry for exp1, got {len(data_exp1)}"
        assert data_exp1[0][1] == 0.5, "Exp1 should have hv=0.5"

        # Test isolation: filter by exp2
        data_exp2 = wm.surrogate_training_data(problem="p3", experiment_id=exp2)
        assert len(data_exp2) == 1, f"Expected 1 entry for exp2, got {len(data_exp2)}"
        assert data_exp2[0][1] == 0.8, "Exp2 should have hv=0.8"

        # Test no filter: returns all experiments
        data_all = wm.surrogate_training_data(problem="p3")
        assert len(data_all) == 2, f"Expected 2 entries total, got {len(data_all)}"

        # Verify _stage injection (A4.3 helper fix)
        for metrics, _ in data_all:
            assert "_stage" in metrics, "Should inject _stage key for fidelity"


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


def test_world_model_v2_schema(tmp_path):
    """Verify that new tables for V2 upgrade (Phase 1.3) are created and usable."""
    db_path = tmp_path / "test_v2.db"

    # 1. Initialize DB (should create new tables)
    memory.init_db(db_path)

    with memory.WorldModel(db_path) as wm:
        # Start an experiment
        exp_id = wm.start_experiment(
            config_payload={"foo": "bar"},
            git_sha="abc1234",
        )

        # 2. Test record_optimization_state
        cycle = 1
        multipliers = {"constraint_1": 1.5, "constraint_2": 0.0}
        penalty = 100.0
        opt_state = {"momentum": 0.9, "step": 50}

        wm.record_optimization_state(
            experiment_id=exp_id,
            cycle=cycle,
            alm_multipliers=multipliers,
            penalty_parameter=penalty,
            optimizer_state=opt_state,
        )

        # Verify directly in DB
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM optimization_state WHERE experiment_id = ? AND cycle = ?",
            (exp_id, cycle),
        ).fetchone()

        assert row is not None
        assert json.loads(row["alm_multipliers_json"]) == multipliers
        assert float(row["penalty_parameter"]) == penalty
        assert json.loads(row["optimizer_state_json"]) == opt_state

        # 3. Test record_surrogate_checkpoint
        backend = "neural_operator"
        filepath = "/tmp/model_checkpoint.pt"
        metrics = {"val_loss": 0.05, "r2": 0.98}

        ckpt_id = wm.record_surrogate_checkpoint(
            experiment_id=exp_id,
            cycle=cycle,
            backend=backend,
            filepath=filepath,
            metrics=metrics,
        )

        row = conn.execute(
            "SELECT * FROM surrogate_checkpoints WHERE id = ?", (ckpt_id,)
        ).fetchone()

        assert row is not None
        assert row["backend"] == backend
        assert row["filepath"] == filepath
        assert json.loads(row["metrics_json"]) == metrics
        assert row["experiment_id"] == exp_id
        assert row["cycle"] == cycle

        conn.close()


def test_record_optimization_state_minimal(tmp_path):
    """Test recording optimization state without optional fields."""
    db_path = tmp_path / "test_v2_min.db"
    memory.init_db(db_path)

    with memory.WorldModel(db_path) as wm:
        exp_id = wm.start_experiment({}, "sha")

        wm.record_optimization_state(
            experiment_id=exp_id,
            cycle=1,
            alm_multipliers={"c1": 1.0},
            penalty_parameter=10.0,
        )

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM optimization_state").fetchone()
        assert row is not None
        assert row["optimizer_state_json"] is None
        conn.close()


def test_record_surrogate_checkpoint_rolls_back_on_failure(tmp_path):
    db_path = tmp_path / "world.db"
    with memory.WorldModel(db_path) as wm:
        experiment_id = wm.start_experiment({"problem": "p1"}, "rollbacksha")
        with pytest.raises(RuntimeError):
            with wm.transaction():
                wm.record_surrogate_checkpoint(
                    experiment_id=experiment_id,
                    cycle=1,
                    backend="test_backend",
                    filepath="/tmp/test.pt",
                    metrics={"some_metric": 1.0},
                    commit=False,  # Ensure it doesn't commit by itself
                )
                raise RuntimeError("Simulated failure after checkpoint record")

        # Verify that no surrogate checkpoint was recorded
        checkpoints_count = wm._conn.execute(
            "SELECT COUNT(*) FROM surrogate_checkpoints"
        ).fetchone()[0]
        assert checkpoints_count == 0


def test_record_surrogate_checkpoint_minimal(tmp_path):
    """Test recording surrogate checkpoint without optional fields."""
    db_path = tmp_path / "test_v2_surr_min.db"
    memory.init_db(db_path)

    with memory.WorldModel(db_path) as wm:
        exp_id = wm.start_experiment({}, "sha")

        wm.record_surrogate_checkpoint(
            experiment_id=exp_id,
            cycle=1,
            backend="rf",
            filepath="model.pkl",
        )

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM surrogate_checkpoints").fetchone()
        assert row is not None
        assert row["metrics_json"] is None
        conn.close()


def test_recent_failures_include_error_taxonomy_fields(tmp_path):
    db_path = tmp_path / "world.db"
    with memory.WorldModel(db_path) as wm:
        experiment_id = wm.start_experiment({"problem": "p2"}, "deadbeef")
        wm.log_candidate(
            experiment_id=experiment_id,
            problem="p2",
            params={"r_cos": [[1.0]], "z_sin": [[0.1]]},
            seed=11,
            status="promote",
            evaluation={
                "stage": "p2",
                "objective": -1e9,
                "feasibility": float("inf"),
                "metrics": {"aspect_ratio": 9.0},
                "constraint_margins": {"qi": 0.2},
                "error": "JACOBIAN IS BAD, 75 TIMES BAD",
                "failure_label": "vmec_jacobian_bad",
                "failure_source": "vmec",
                "failure_signature": "vmec_jacobian_bad:abc12345",
                "vmec_status": "exception",
            },
            design_hash="failure-taxonomy-hash",
        )
        failures = wm.recent_failures(experiment_id, "p2", limit=5)
        assert len(failures) == 1
        assert failures[0]["failure_label"] == "vmec_jacobian_bad"
        assert failures[0]["failure_source"] == "vmec"
        assert failures[0]["failure_signature"] == "vmec_jacobian_bad:abc12345"
        assert failures[0]["vmec_status"] == "exception"
        assert failures[0]["error"] == "JACOBIAN IS BAD, 75 TIMES BAD"


def test_recent_experience_pack_balances_success_near_failure(tmp_path):
    db_path = tmp_path / "world.db"
    with memory.WorldModel(db_path) as wm:
        experiment_id = wm.start_experiment({"problem": "p3"}, "deadbeef")

        wm.log_candidate(
            experiment_id=experiment_id,
            problem="p3",
            params={"r_cos": [[1.0]], "z_sin": [[0.0]]},
            seed=1,
            status="screen",
            evaluation={
                "stage": "p3",
                "objective": 2.0,
                "feasibility": 0.35,
                "metrics": {"aspect_ratio": 8.0},
                "constraint_margins": {"mhd": 0.3, "qi": 0.05},
            },
            design_hash="failure-case",
            operator_family="blend",
            model_route="route-failure",
        )
        wm.log_candidate(
            experiment_id=experiment_id,
            problem="p3",
            params={"r_cos": [[1.1]], "z_sin": [[0.0]]},
            seed=2,
            status="screen",
            evaluation={
                "stage": "p3",
                "objective": 1.5,
                "feasibility": 0.05,
                "metrics": {"aspect_ratio": 7.8},
                "constraint_margins": {"qi": 0.05},
            },
            design_hash="near-case",
            operator_family="mutate",
            model_route="route-near",
        )
        wm.log_candidate(
            experiment_id=experiment_id,
            problem="p3",
            params={"r_cos": [[1.2]], "z_sin": [[0.0]]},
            seed=3,
            status="promote",
            evaluation={
                "stage": "p3",
                "objective": 1.0,
                "feasibility": 0.0,
                "metrics": {"aspect_ratio": 7.5},
                "constraint_margins": {},
            },
            design_hash="success-case",
            operator_family="exploit",
            model_route="route-success",
        )

        pack = wm.recent_experience_pack(
            experiment_id=experiment_id,
            problem="p3",
            limit_per_bucket=2,
            near_feasibility_threshold=0.1,
        )

        assert len(pack["recent_successes"]) == 1
        assert len(pack["recent_near_successes"]) == 1
        assert len(pack["recent_failures"]) == 1
        failure = pack["recent_failures"][0]
        assert failure["worst_constraint"] == "mhd"
        assert (
            pytest.approx(sum(failure["normalized_violations"].values()), rel=1e-8)
            == 1.0
        )
        adapter = pack["feedback_adapter"]
        assert "worst_constraint_trend" in adapter
        assert "recent_effective_deltas" in adapter
        assert len(adapter["recent_effective_deltas"]) >= 1


def test_recent_candidate_snapshots_support_flat_metrics_for_staged_governor(tmp_path):
    db_path = tmp_path / "world.db"
    with memory.WorldModel(db_path) as wm:
        experiment_id = wm.start_experiment({"problem": "p3"}, "deadbeef")

        def _log_snapshot_candidate(
            *,
            design_hash: str,
            seed: int,
            r00: float,
            objective: float,
            feasibility: float,
            mirror: float,
            log10_qi: float,
            constraint_margins: dict[str, float],
        ) -> None:
            wm.log_candidate(
                experiment_id=experiment_id,
                problem="p3",
                params={"r_cos": [[r00]], "z_sin": [[0.0]]},
                seed=seed,
                status="screen",
                evaluation={
                    "stage": "p3",
                    "objective": objective,
                    "feasibility": feasibility,
                    "metrics": {"mirror": mirror, "log10_qi": log10_qi},
                    "constraint_margins": constraint_margins,
                },
                design_hash=design_hash,
            )

        _log_snapshot_candidate(
            design_hash="focus",
            seed=1,
            r00=1.0,
            objective=5.0,
            feasibility=0.09,
            mirror=0.30,
            log10_qi=-3.2,
            constraint_margins={"mirror": 0.12, "log10_qi": 0.02},
        )
        _log_snapshot_candidate(
            design_hash="metric-partner",
            seed=2,
            r00=0.98,
            objective=1.0,
            feasibility=0.0,
            mirror=0.24,
            log10_qi=-3.7,
            constraint_margins={},
        )
        _log_snapshot_candidate(
            design_hash="objective-fallback",
            seed=3,
            r00=1.1,
            objective=9.0,
            feasibility=0.0,
            mirror=0.60,
            log10_qi=-3.9,
            constraint_margins={},
        )

        snapshots = wm.recent_candidate_snapshots(experiment_id, "p3", limit=8)
        plan = build_staged_seed_plan_from_snapshots(
            snapshots=snapshots,
            problem="p3",
            near_feasibility_threshold=0.25,
            max_repair_candidates=3,
            bridge_blend_t=0.86,
        )

        assert plan is not None
        assert plan.focus_hash == "focus"
        assert plan.worst_constraint == "mirror"
        assert plan.partner_hash == "metric-partner"


def test_recent_candidate_snapshots_ignore_non_numeric_constraint_margins(tmp_path):
    db_path = tmp_path / "world.db"
    with memory.WorldModel(db_path) as wm:
        experiment_id = wm.start_experiment({"problem": "p3"}, "deadbeef")

        wm.log_candidate(
            experiment_id=experiment_id,
            problem="p3",
            params={"r_cos": [[1.0]], "z_sin": [[0.0]]},
            seed=1,
            status="screen",
            evaluation={
                "stage": "p3",
                "objective": 2.0,
                "feasibility": 0.2,
                "metrics": {"mirror": 0.3},
                "constraint_margins": {
                    "mirror": None,
                    "qi": "bad",
                    "flux": 0.04,
                    "flag": True,
                    "negative": -0.2,
                },
            },
            design_hash="mixed-margins",
        )

        snapshots = wm.recent_candidate_snapshots(experiment_id, "p3", limit=5)
        assert len(snapshots) == 1
        assert snapshots[0]["constraint_margins"] == {"flux": 0.04}


def test_ancestor_chains_collects_parent_lineage(tmp_path):
    db_path = tmp_path / "world.db"
    with memory.WorldModel(db_path) as wm:
        experiment_id = wm.start_experiment({"problem": "p3"}, "deadbeef")

        def _log_chain(
            *,
            design_hash: str,
            parent_hashes: list[str],
            feasibility: float,
            objective: float,
            worst_name: str | None = None,
            worst_value: float = 0.0,
        ) -> None:
            margins = {}
            if worst_name is not None:
                margins[worst_name] = worst_value
            wm.log_candidate(
                experiment_id=experiment_id,
                problem="p3",
                params={"r_cos": [[1.0]], "z_sin": [[0.0]]},
                seed=1,
                status="screen",
                evaluation={
                    "stage": "screen",
                    "objective": objective,
                    "feasibility": feasibility,
                    "metrics": {"aspect_ratio": 8.0},
                    "constraint_margins": margins,
                },
                design_hash=design_hash,
                lineage_parent_hashes=parent_hashes,
            )

        _log_chain(
            design_hash="grandparent",
            parent_hashes=[],
            feasibility=0.4,
            objective=2.0,
            worst_name="mirror",
            worst_value=0.2,
        )
        _log_chain(
            design_hash="parent",
            parent_hashes=["grandparent"],
            feasibility=0.2,
            objective=3.0,
            worst_name="log10_qi",
            worst_value=0.1,
        )
        _log_chain(
            design_hash="child",
            parent_hashes=["parent"],
            feasibility=0.05,
            objective=4.0,
            worst_name="mirror",
            worst_value=0.03,
        )

        chains = wm.ancestor_chains(
            experiment_id=experiment_id,
            problem="p3",
            design_hashes=["child"],
            max_depth=3,
        )

        assert len(chains) == 1
        assert chains[0]["target_design_hash"] == "child"
        ancestors = chains[0]["ancestors"]
        assert ancestors[0]["design_hash"] == "parent"
        assert ancestors[0]["depth"] == 1
        assert ancestors[1]["design_hash"] == "grandparent"
        assert ancestors[1]["depth"] == 2


def test_ancestor_chains_ignores_non_numeric_constraint_margins(tmp_path):
    db_path = tmp_path / "world.db"
    with memory.WorldModel(db_path) as wm:
        experiment_id = wm.start_experiment({"problem": "p3"}, "deadbeef")

        wm.log_candidate(
            experiment_id=experiment_id,
            problem="p3",
            params={"r_cos": [[1.0]], "z_sin": [[0.0]]},
            seed=1,
            status="screen",
            evaluation={
                "stage": "screen",
                "objective": 3.0,
                "feasibility": 0.1,
                "metrics": {"aspect_ratio": 8.0},
                "constraint_margins": {
                    "bool_margin": True,
                    "string_margin": "0.5",
                    "valid_margin": 0.03,
                },
            },
            design_hash="parent",
            lineage_parent_hashes=[],
        )
        wm.log_candidate(
            experiment_id=experiment_id,
            problem="p3",
            params={"r_cos": [[1.0]], "z_sin": [[0.0]]},
            seed=2,
            status="screen",
            evaluation={
                "stage": "screen",
                "objective": 3.5,
                "feasibility": 0.05,
                "metrics": {"aspect_ratio": 8.1},
                "constraint_margins": {},
            },
            design_hash="child",
            lineage_parent_hashes=["parent"],
        )

        chains = wm.ancestor_chains(
            experiment_id=experiment_id,
            problem="p3",
            design_hashes=["child"],
            max_depth=1,
        )

        assert len(chains) == 1
        assert chains[0]["target_design_hash"] == "child"
        assert len(chains[0]["ancestors"]) == 1
        assert chains[0]["ancestors"][0]["worst_constraint"] == "valid_margin"
        assert chains[0]["ancestors"][0]["worst_constraint_violation"] == 0.03


def test_to_networkx_materializes_lineage_parent_edges(tmp_path):
    db_path = tmp_path / "world.db"
    with memory.WorldModel(db_path) as wm:
        experiment_id = wm.start_experiment({"problem": "p3"}, "deadbeef")

        wm.log_candidate(
            experiment_id=experiment_id,
            problem="p3",
            params={"r_cos": [[1.0]], "z_sin": [[0.0]]},
            seed=1,
            status="screen",
            evaluation={
                "stage": "screen",
                "objective": 2.0,
                "feasibility": 0.2,
                "metrics": {},
                "constraint_margins": {},
            },
            design_hash="p0",
        )
        wm.log_candidate(
            experiment_id=experiment_id,
            problem="p3",
            params={"r_cos": [[1.0]], "z_sin": [[0.0]]},
            seed=2,
            status="screen",
            evaluation={
                "stage": "screen",
                "objective": 3.0,
                "feasibility": 0.1,
                "metrics": {},
                "constraint_margins": {},
            },
            design_hash="c0",
            lineage_parent_hashes=["p0"],
        )
        wm._conn.execute(
            """
            UPDATE candidates
            SET lineage_parent_hashes_json = ?
            WHERE experiment_id = ? AND design_hash = ?
            """,
            ('[null, "", "null", "p0"]', experiment_id, "c0"),
        )
        wm._conn.commit()

        graph = wm.to_networkx(experiment_id)
        lineage_edges = [
            (src, dst, attrs)
            for src, dst, attrs in graph.edges
            if attrs.get("relation") == "lineage_parent_of"
        ]
        assert len(lineage_edges) == 1
        assert lineage_edges[0][0].startswith("candidate:")
        assert lineage_edges[0][1].startswith("candidate:")

        chains = wm.ancestor_chains(
            experiment_id=experiment_id,
            problem="p3",
            design_hashes=["c0"],
            max_depth=2,
        )
        assert len(chains[0]["ancestors"]) == 1
        assert chains[0]["ancestors"][0]["design_hash"] == "p0"


def test_scratchpad_event_persistence_and_summary(tmp_path):
    db_path = tmp_path / "world.db"
    with memory.WorldModel(db_path) as wm:
        experiment_id = wm.start_experiment({"problem": "p3"}, "deadbeef")
        wm.log_scratchpad_event(
            experiment_id=experiment_id,
            cycle=2,
            step=1,
            planner_intent={"penalty_focus_indices": [0]},
            aso_action="ADJUST",
            intent_agreement="aligned",
            override_reason=None,
            diagnostics={"status": "STAGNATION", "max_violation": 0.2},
            outcome={"objective_delta": -0.1, "violation_delta": -0.05},
        )
        wm.log_scratchpad_event(
            experiment_id=experiment_id,
            cycle=2,
            step=2,
            planner_intent={"penalty_focus_indices": [1]},
            aso_action="RESTART",
            intent_agreement="overridden",
            override_reason="restart requested by diagnostics",
            diagnostics={"status": "DIVERGING", "max_violation": 0.4},
            outcome={"objective_delta": 0.0, "violation_delta": 0.2},
        )

        summary = wm.scratchpad_cycle_summary(experiment_id=experiment_id, cycle=2)

        assert summary["event_count"] == 2
        assert summary["action_counts"]["ADJUST"] == 1
        assert summary["action_counts"]["RESTART"] == 1
        assert summary["events"][0]["step"] == 1
        assert summary["events"][1]["intent_agreement"] == "overridden"


def test_select_parent_group_performance_novelty(tmp_path):
    db_path = tmp_path / "world.db"
    with memory.WorldModel(db_path) as wm:
        experiment_id = wm.start_experiment({"problem": "p1"}, "deadbeef")

        wm.log_candidate(
            experiment_id=experiment_id,
            problem="p1",
            params={"r_cos": [[1.0]], "z_sin": [[0.0]]},
            seed=1,
            status="screen",
            evaluation={
                "stage": "screen",
                "objective": 1.0,
                "feasibility": 0.0,
                "metrics": {"aspect_ratio": 8.0},
                "constraint_margins": {},
            },
            design_hash="feasible-a",
            novelty_score=0.20,
        )
        wm.log_candidate(
            experiment_id=experiment_id,
            problem="p1",
            params={"r_cos": [[0.95]], "z_sin": [[0.02]]},
            seed=2,
            status="screen",
            evaluation={
                "stage": "screen",
                "objective": 1.4,
                "feasibility": 0.08,
                "metrics": {"aspect_ratio": 8.5},
                "constraint_margins": {"qi": 0.08},
            },
            design_hash="near-b",
            novelty_score=0.12,
        )
        wm.log_candidate(
            experiment_id=experiment_id,
            problem="p1",
            params={"r_cos": [[0.80]], "z_sin": [[0.05]]},
            seed=3,
            status="screen",
            evaluation={
                "stage": "screen",
                "objective": 4.0,
                "feasibility": 0.9,
                "metrics": {"aspect_ratio": 12.0},
                "constraint_margins": {"qi": 0.9},
            },
            design_hash="bad-c",
            novelty_score=0.01,
        )

        selected = wm.select_parent_group_performance_novelty(
            experiment_id=experiment_id,
            problem="p1",
            group_size=2,
            worst_constraint="qi",
            focus_constraint_margin=0.2,
            leverage_weight=0.5,
        )

        assert len(selected) == 2
        hashes = {item["design_hash"] for item in selected}
        assert "feasible-a" in hashes
        assert "near-b" in hashes
        assert all("parent_selection_score" in item for item in selected)
        assert all("constraint_leverage_score" in item for item in selected)


def test_nearest_case_deltas_returns_feasible_and_near_buckets(tmp_path):
    db_path = tmp_path / "world.db"
    with memory.WorldModel(db_path) as wm:
        experiment_id = wm.start_experiment({"problem": "p2"}, "deadbeef")

        wm.log_candidate(
            experiment_id=experiment_id,
            problem="p2",
            params={"r_cos": [[1.0, 0.0]], "z_sin": [[0.0, 0.1]]},
            seed=1,
            status="screen",
            evaluation={
                "stage": "screen",
                "objective": 6.0,
                "feasibility": 0.0,
                "metrics": {"aspect_ratio": 9.0},
                "constraint_margins": {},
            },
            design_hash="good-a",
        )
        wm.log_candidate(
            experiment_id=experiment_id,
            problem="p2",
            params={"r_cos": [[1.02, 0.0]], "z_sin": [[0.0, 0.08]]},
            seed=2,
            status="screen",
            evaluation={
                "stage": "screen",
                "objective": 5.5,
                "feasibility": 0.1,
                "metrics": {"aspect_ratio": 8.7},
                "constraint_margins": {"vacuum_well": 0.1},
            },
            design_hash="near-b",
        )

        nearest = wm.nearest_case_deltas(
            experiment_id=experiment_id,
            problem="p2",
            seed_params={"r_cos": [[1.01, 0.0]], "z_sin": [[0.0, 0.09]]},
            limit=2,
            near_feasibility_threshold=0.2,
            include_recipes=True,
        )

        assert len(nearest) == 2
        buckets = {item["bucket"] for item in nearest}
        assert "feasible" in buckets
        assert "near_feasible" in buckets
        assert all("delta_summary" in item for item in nearest)
        assert all("delta_recipe" in item for item in nearest)
        assert all(
            item["delta_recipe"]["type"] == "sparse_additive_delta" for item in nearest
        )


def test_model_router_reward_summary_can_filter_reward_eligible_only(tmp_path):
    db_path = tmp_path / "world_reward_eligible.db"
    with memory.WorldModel(db_path) as wm:
        experiment_id = wm.start_experiment({"problem": "p3"}, "deadbeef")
        wm.log_model_router_reward_event(
            experiment_id=experiment_id,
            problem="p3",
            model_route="route-a",
            window_size=10,
            previous_feasible_yield=10.0,
            current_feasible_yield=11.0,
            previous_hv=1.0,
            current_hv=1.1,
            reward=0.1,
            reward_components={"reward_eligible": False},
        )
        wm.log_model_router_reward_event(
            experiment_id=experiment_id,
            problem="p3",
            model_route="route-a",
            window_size=10,
            previous_feasible_yield=11.0,
            current_feasible_yield=12.0,
            previous_hv=1.1,
            current_hv=1.2,
            reward=0.2,
            reward_components={"reward_eligible": True},
        )
        summary = wm.model_router_reward_summary(
            experiment_id,
            problem="p3",
            reward_eligible_only=True,
        )
        assert summary["event_count"] == 1
        assert summary["sampled_event_count"] == 1
        assert summary["last_reward"] == pytest.approx(0.2)


def test_model_router_reward_eligible_history_filters_rows(tmp_path):
    db_path = tmp_path / "world_reward_history.db"
    with memory.WorldModel(db_path) as wm:
        experiment_id = wm.start_experiment({"problem": "p3"}, "deadbeef")
        wm.log_model_router_reward_event(
            experiment_id=experiment_id,
            problem="p3",
            model_route="route-a",
            window_size=10,
            previous_feasible_yield=10.0,
            current_feasible_yield=11.0,
            previous_hv=1.0,
            current_hv=1.1,
            reward=0.1,
            reward_components={"reward_eligible": False},
        )
        wm.log_model_router_reward_event(
            experiment_id=experiment_id,
            problem="p3",
            model_route="route-b",
            window_size=10,
            previous_feasible_yield=11.0,
            current_feasible_yield=13.0,
            previous_hv=1.1,
            current_hv=1.4,
            reward=0.3,
            reward_components={"reward_eligible": True},
        )
        history = wm.model_router_reward_eligible_history(experiment_id, problem="p3")
        assert len(history) == 1
        assert history[0]["model_route"] == "route-b"
        assert history[0]["reward"] == pytest.approx(0.3)
