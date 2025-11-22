from pathlib import Path

import pytest

from ai_scientist.world_model import BoundaryWorldModel, hash_boundary


def test_roundtrip_boundary_eval_cycle(tmp_path: Path) -> None:
    db_path = tmp_path / "wm.sqlite"
    r_cos = [[1.0, 0.1], [0.0, -0.2]]
    z_sin = [[0.0, 0.05], [0.1, 0.2]]

    with BoundaryWorldModel(db_path) as wm:
        design_hash = wm.add_boundary(
            r_cos=r_cos,
            z_sin=z_sin,
            schema_version="v1",
            p=3,
            nfp=3,
            source="seed",
            parent_id=None,
        )
        eval_id = wm.add_evaluation(
            boundary_hash=design_hash,
            stage="screen",
            vmec_status="ok",
            runtime_sec=1.2,
            metrics={"aspect_ratio": 8.0},
            margins={"aspect_ratio": 0.5, "qi_log10": -1.0},
            is_feasible=False,
            schema_version="v1",
        )
        wm.log_cycle(
            cycle_idx=0,
            phase="screen",
            p=3,
            new_evals=1,
            new_feasible=0,
            cumulative_feasible=0,
            hv=0.0,
            notes="init",
        )

    with BoundaryWorldModel(db_path) as resumed:
        stats = resumed.cache_stats()
        assert stats["boundaries"] == 1
        assert stats["evaluations"] == 1
        assert stats["cycles"] == 1

        near = resumed.get_near_feasible(max_l2_margin=1.0)
        assert len(near) == 1
        assert near[0]["id"] == eval_id
        assert near[0]["boundary"]["hash"] == design_hash
        assert near[0]["boundary"]["r_cos"] == r_cos
        assert near[0]["boundary"]["z_sin"] == z_sin

        archive = resumed.latest_archive()
        assert archive == []  # no feasible rows yet


def test_near_feasible_filters_by_margin(tmp_path: Path) -> None:
    db_path = tmp_path / "wm.sqlite"
    with BoundaryWorldModel(db_path) as wm:
        h = wm.add_boundary(
            r_cos=[[1.0]],
            z_sin=[[0.0]],
            schema_version="v1",
            p=3,
            nfp=3,
            source="seed",
            parent_id=None,
        )
        wm.add_evaluation(
            boundary_hash=h,
            stage="screen",
            vmec_status="ok",
            runtime_sec=0.2,
            metrics={"aspect_ratio": 9.0},
            margins={"aspect_ratio": 0.1},
            is_feasible=False,
            schema_version="v1",
        )
        wm.add_evaluation(
            boundary_hash=h,
            stage="screen",
            vmec_status="ok",
            runtime_sec=0.2,
            metrics={"aspect_ratio": 9.0},
            margins={"aspect_ratio": 1.5},
            is_feasible=False,
            schema_version="v1",
        )

    with BoundaryWorldModel(db_path) as resumed:
        near = resumed.get_near_feasible(max_l2_margin=0.5)
        assert len(near) == 1
        assert near[0]["margins"]["aspect_ratio"] == 0.1


def test_resume_keeps_hv_and_archive(tmp_path: Path) -> None:
    db_path = tmp_path / "wm.sqlite"
    r_cos = [[1.0]]
    z_sin = [[0.0]]
    with BoundaryWorldModel(db_path) as wm:
        h = wm.add_boundary(
            r_cos=r_cos,
            z_sin=z_sin,
            schema_version="v1",
            p=3,
            nfp=3,
            source="seed",
            parent_id=None,
        )
        wm.add_evaluation(
            boundary_hash=h,
            stage="final",
            vmec_status="ok",
            runtime_sec=0.5,
            metrics={"aspect_ratio": 6.0},
            margins={"aspect_ratio": -0.2},
            is_feasible=True,
            schema_version="v1",
        )
        wm.log_cycle(
            cycle_idx=1,
            phase="final",
            p=3,
            new_evals=1,
            new_feasible=1,
            cumulative_feasible=1,
            hv=5.5,
            notes="after resume",
        )

    with BoundaryWorldModel(db_path) as resumed:
        archive = resumed.latest_archive()
        assert len(archive) == 1
        assert archive[0]["boundary"]["hash"] == h
        assert archive[0]["metrics"]["aspect_ratio"] == 6.0

        hv_row = resumed._conn.execute("SELECT hv FROM cycles WHERE cycle_idx = 1").fetchone()
        assert hv_row is not None
        assert hv_row["hv"] == pytest.approx(5.5)

        # Hash should be stable with canonical rounding
        stable_hash = hash_boundary(r_cos, z_sin, "v1")
        assert stable_hash == h
