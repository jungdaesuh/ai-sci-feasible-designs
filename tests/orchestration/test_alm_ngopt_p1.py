from __future__ import annotations

from pathlib import Path
import sys
from typing import Any
import json

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

from orchestration.alm_ngopt_p1 import (
    BoundaryVectorizer,
    OptimizerRuntime,
    RunConfig,
    RunnerState,
    boundary_hash,
    run_stage,
    compute_bounds,
    load_config,
    persist_state,
    resume_run,
)
from orchestration.problem_adapters import get_problem_adapter
from orchestration.run_paths import JSONLWriter, RunPaths, create_run_paths, write_json
from constellaration.geometry import surface_rz_fourier
from constellaration.initial_guess import generate_rotating_ellipse
from orchestration import evaluation


class DummyParam:
    def __init__(self, length: int, fill: float) -> None:
        self._value = np.full(length, fill, dtype=float)

    @property
    def value(self) -> np.ndarray:
        return self._value

    @value.setter
    def value(self, new_value: np.ndarray) -> None:
        self._value = np.asarray(new_value, dtype=float)

    def spawn_child(self) -> "DummyParam":
        return self


class DummyOptimizer:
    def __init__(self, length: int) -> None:
        self._length = length
        self.ask_calls = 0
        self.tell_records: list[tuple[np.ndarray, float]] = []

    def ask(self) -> DummyParam:
        param = DummyParam(self._length, float(self.ask_calls))
        self.ask_calls += 1
        return param

    def tell(self, param: DummyParam, loss_value: float) -> None:
        self.tell_records.append((param.value.copy(), float(loss_value)))


class DummyFuture:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def result(self, timeout: float | None = None) -> dict[str, Any]:  # pragma: no cover
        return self._payload


class DummyExecutor:
    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self._payloads = iter(payloads)

    def __enter__(self) -> "DummyExecutor":  # pragma: no cover
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover
        return None

    def submit(self, fn, *args, **kwargs) -> DummyFuture:  # pragma: no cover
        try:
            payload = next(self._payloads)
        except StopIteration as exc:
            raise RuntimeError("DummyExecutor exhausted") from exc
        return DummyFuture(payload)


def make_surface() -> surface_rz_fourier.SurfaceRZFourier:
    surface = generate_rotating_ellipse(
        aspect_ratio=4.0,
        elongation=1.6,
        rotational_transform=1.0,
        n_field_periods=3,
    )
    return surface_rz_fourier.set_max_mode_numbers(
        surface=surface,
        max_poloidal_mode=2,
        max_toroidal_mode=1,
    )


def test_boundary_vectorizer_roundtrip_freezes_dc() -> None:
    surface = make_surface()
    vectorizer = BoundaryVectorizer.build(surface)

    flat = vectorizer.to_vector(surface)
    assert flat.shape[0] == int(vectorizer.mask_r_cos.sum() + vectorizer.mask_z_sin.sum())

    toroidal_modes = vectorizer.toroidal_modes
    assert not vectorizer.mask_r_cos[0, toroidal_modes[0] < 0].any()
    assert not vectorizer.mask_z_sin[0, toroidal_modes[0] <= 0].any()

    r_indices = np.argwhere(vectorizer.mask_r_cos)
    z_indices = np.argwhere(vectorizer.mask_z_sin)
    r_part = flat[: len(r_indices)].copy()
    z_part = flat[len(r_indices) :].copy()

    toroidal_modes = surface.toroidal_modes
    for idx, (m_idx, n_idx) in enumerate(r_indices):
        if m_idx == 0 and toroidal_modes[m_idx, n_idx] < 0:
            continue  # respect stellarator symmetry restriction
        r_part[idx] += 0.01

    for idx, (m_idx, n_idx) in enumerate(z_indices):
        if m_idx == 0 and toroidal_modes[m_idx, n_idx] <= 0:
            continue
        z_part[idx] += 0.01

    perturbed = np.concatenate([r_part, z_part])
    reconstructed = vectorizer.from_vector(perturbed)

    # DC coefficients stay fixed
    assert reconstructed.r_cos[0, 0] == pytest.approx(surface.r_cos[0, 0])
    assert reconstructed.z_sin[0, 0] == pytest.approx(surface.z_sin[0, 0])
    # At least one trainable coefficient moves
    trainable_index = tuple(np.argwhere(vectorizer.mask_r_cos)[0])
    assert reconstructed.r_cos[trainable_index] != pytest.approx(surface.r_cos[trainable_index])


def test_boundary_hash_precision_control() -> None:
    base = make_surface()
    shifted = make_surface()
    shifted.r_cos[0, 1] += 4e-4  # shifts below 1e-3 threshold

    coarse_hash = boundary_hash(base, decimals=3)
    assert coarse_hash == boundary_hash(shifted, decimals=3)

    fine_hash = boundary_hash(base, decimals=4)
    assert fine_hash != boundary_hash(shifted, decimals=4)


def test_serialize_and_resume_roundtrip(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    logs_dir = run_root / "logs"
    artifacts_dir = run_root / "artifacts"
    vmec_inputs_dir = artifacts_dir / "vmec_inputs"
    wout_dir = artifacts_dir / "wout"
    for directory in (run_root, logs_dir, artifacts_dir, vmec_inputs_dir, wout_dir):
        directory.mkdir(parents=True, exist_ok=True)

    config = RunConfig(
        tag="test",
        resume_path=run_root,
        seed=7,
        workers=1,
        ladder=["low_fidelity"],
        budget_initial=5,
        budget_increment=5,
        budget_max=5,
        topk=3,
        bound_scale=5.0,
        sigma_floor_frac=0.05,
       sigma_floor_alpha=1.0,
        bound_contract_factor=0.5,
        bound_contract_loops=2,
        promotion_distance_threshold=0.05,
        triangularity_unlock_patience=3,
        triangularity_unlock_threshold=0.05,
        triangularity_unlock_scale=1.5,
        triangularity_unlock_sigma_scale=1.5,
        init_A=4.0,
        init_E=1.5,
        init_iota=1.2,
        init_nfp=3,
        oracle="ngopt",
        eval_timeout_sec=120.0,
        max_consecutive_failures=4,
        backoff_sec=0.0,
        checkpoint_every=1,
        hash_decimals=5,
        problem_type="P1",
    )

    run_paths = RunPaths(
        root=run_root,
        logs_dir=logs_dir,
        artifacts_dir=artifacts_dir,
        vmec_inputs_dir=vmec_inputs_dir,
        wout_dir=wout_dir,
        evaluations_path=run_root / "evaluations.jsonl",
        config_path=run_root / "config.json",
        best_low_path=run_root / "best_low_fidelity.json",
        best_high_path=run_root / "best_high_fidelity.json",
    )

    surface = make_surface()
    vectorizer = BoundaryVectorizer.build(surface)
    bounds_low, bounds_high, sigma_floor = compute_bounds(
        vectorizer,
        surface,
        config.bound_scale,
        config.sigma_floor_frac,
        config.sigma_floor_alpha,
    )
    optimizer = OptimizerRuntime(
        vectorizer=vectorizer,
        bounds_low=bounds_low,
        bounds_high=bounds_high,
        lambda_vec=np.array([0.1, 0.0, 0.2]),
        rho=2.5,
        sigma_floor=sigma_floor,
    )

    rng = np.random.default_rng(config.seed)
    adapter = get_problem_adapter(config.problem_type)
    state = RunnerState(
        config=config,
        run_paths=run_paths,
        optimizer=optimizer,
        adapter=adapter,
        rng=rng,
        evaluation_id=2,
        outer_iter=1,
        stage_index=0,
        budget_total=5,
        budget_used=2,
        best_low=None,
        best_high=None,
        promotions_done={"abc"},
        promotion_history=[],
        triangularity_counter=0,
        triangularity_unlocked=False,
    )

    boundary_dict = evaluation.surface_to_dict(surface)
    hash_value = boundary_hash(surface, decimals=config.hash_decimals)
    feasible_records = {
        hash_value: {
            "evaluation_id": 0,
            "outer_iter": 0,
            "boundary_hash": hash_value,
            "boundary": boundary_dict,
            "success": True,
            "is_feasible": True,
            "objective": 3.0,
            "minimize": True,
            "metrics": {
                "aspect_ratio": 3.0,
                "average_triangularity": -0.6,
                "edge_rotational_transform_over_n_field_periods": 0.34,
            },
            "fidelity": "low_fidelity",
            "duration_sec": 1.0,
            "alm_loss": 3.1,
            "lambda": [0.1, 0.0, 0.2],
            "rho": 2.5,
            "feasibility_norm": 0.05,
        }
    }

    config_payload = {
        "version": 1,
        "problem": "P1",
        "git_commit": "test",
        "args": {
            "tag": config.tag,
            "seed": config.seed,
            "workers": config.workers,
            "ladder": config.ladder,
            "budget_initial": config.budget_initial,
            "budget_increment": config.budget_increment,
            "budget_max": config.budget_max,
            "topk": config.topk,
            "bound_scale": config.bound_scale,
            "sigma_floor_frac": config.sigma_floor_frac,
            "sigma_floor_alpha": config.sigma_floor_alpha,
            "bound_contract_factor": config.bound_contract_factor,
            "bound_contract_loops": config.bound_contract_loops,
            "promotion_distance_threshold": config.promotion_distance_threshold,
            "triangularity_unlock_patience": config.triangularity_unlock_patience,
            "triangularity_unlock_threshold": config.triangularity_unlock_threshold,
            "triangularity_unlock_scale": config.triangularity_unlock_scale,
            "triangularity_unlock_sigma_scale": config.triangularity_unlock_sigma_scale,
            "init_A": config.init_A,
            "init_E": config.init_E,
            "init_iota": config.init_iota,
            "init_nfp": config.init_nfp,
            "oracle": config.oracle,
            "eval_timeout_sec": config.eval_timeout_sec,
            "max_consecutive_failures": config.max_consecutive_failures,
            "backoff_sec": config.backoff_sec,
            "checkpoint_every": config.checkpoint_every,
            "hash_decimals": config.hash_decimals,
            "problem_type": config.problem_type,
        },
    }
    write_json(run_paths.config_path, config_payload)

    persist_state(state, feasible_records)

    stored_config = load_config(run_paths.config_path)
    resumed_state, resumed_records = resume_run(stored_config)

    assert np.allclose(resumed_state.optimizer.lambda_vec, optimizer.lambda_vec)
    assert resumed_state.optimizer.rho == pytest.approx(optimizer.rho)
    assert resumed_state.budget_used == state.budget_used
    assert stored_config.hash_decimals == config.hash_decimals
    assert hash_value in resumed_records
    assert resumed_records[hash_value]["boundary_hash"] == hash_value
    assert resumed_state.adapter.name == "P1"
    assert np.all(resumed_state.optimizer.sigma_floor >= 0.0)
    assert resumed_state.triangularity_counter == 0
    assert isinstance(resumed_state.promotion_history, list)


def test_run_stage_calls_tell_per_candidate(monkeypatch, tmp_path: Path) -> None:
    surface = make_surface()
    vectorizer = BoundaryVectorizer.build(surface)
    bounds_low, bounds_high, sigma_floor = compute_bounds(
        vectorizer,
        surface,
        scale=4.0,
        sigma_floor_frac=0.05,
        sigma_floor_alpha=1.0,
    )
    optimizer_runtime = OptimizerRuntime(
        vectorizer=vectorizer,
        bounds_low=bounds_low,
        bounds_high=bounds_high,
        lambda_vec=np.zeros(3),
        rho=1.0,
        sigma_floor=sigma_floor,
    )

    run_paths = create_run_paths(str(tmp_path), "test_run_stage")
    config = RunConfig(
        tag="test_run_stage",
        resume_path=tmp_path,
        seed=0,
        workers=2,
        ladder=["very_low_fidelity"],
        budget_initial=4,
        budget_increment=0,
        budget_max=4,
        topk=2,
        bound_scale=4.0,
        sigma_floor_frac=0.05,
        sigma_floor_alpha=1.0,
        bound_contract_factor=0.5,
        bound_contract_loops=3,
        promotion_distance_threshold=0.05,
        triangularity_unlock_patience=3,
        triangularity_unlock_threshold=0.05,
        triangularity_unlock_scale=1.5,
        triangularity_unlock_sigma_scale=1.5,
        init_A=4.0,
        init_E=1.5,
        init_iota=1.2,
        init_nfp=3,
        oracle="ngopt",
        eval_timeout_sec=180.0,
        max_consecutive_failures=5,
        backoff_sec=0.0,
        checkpoint_every=2,
        hash_decimals=6,
        problem_type="P1",
    )
    state = RunnerState(
        config=config,
        run_paths=run_paths,
        optimizer=optimizer_runtime,
        adapter=get_problem_adapter("P1"),
        rng=np.random.default_rng(config.seed),
        evaluation_id=0,
        outer_iter=0,
        stage_index=0,
        budget_total=config.budget_initial,
        budget_used=0,
        best_low=None,
        best_high=None,
        promotions_done=set(),
        promotion_history=[],
        triangularity_counter=0,
        triangularity_unlocked=False,
    )
    writer = JSONLWriter(run_paths.evaluations_path)

    payload_template = {
        "success": True,
        "metrics": {
            "aspect_ratio": 3.5,
            "average_triangularity": -0.55,
            "edge_rotational_transform_over_n_field_periods": 0.35,
        },
        "objective": 2.0,
        "minimize": True,
        "is_feasible": True,
        "feasibility_norm": 0.01,
        "score": 0.0,
        "fidelity": "very_low_fidelity",
        "duration_sec": 1.0,
    }
    results = [dict(payload_template, objective=float(i)) for i in range(config.budget_initial)]

    dummy_optimizer = DummyOptimizer(len(bounds_low))
    monkeypatch.setattr(
        "orchestration.alm_ngopt_p1.create_ng_optimizer",
        lambda *args, **kwargs: dummy_optimizer,
    )
    monkeypatch.setattr(
        "orchestration.alm_ngopt_p1.evaluate_high_fidelity",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "orchestration.alm_ngopt_p1.persist_state",
        lambda *args, **kwargs: None,
    )

    feasible_records: dict[str, dict[str, Any]] = {}
    run_stage(state, feasible_records, writer, DummyExecutor(results.copy()))

    assert len(dummy_optimizer.tell_records) == config.budget_initial
    assert len(feasible_records) == config.budget_initial


def test_summarize_run_uses_adapter(monkeypatch, tmp_path: Path, capsys) -> None:
    run_dir = tmp_path / "summary"
    run_dir.mkdir()

    config_payload = {
        "version": 1,
        "problem": "P1",
        "args": {
            "tag": "summary",
            "problem_type": "P1",
        },
    }
    (run_dir / "config.json").write_text(json.dumps(config_payload), encoding="utf-8")

    evaluations = [
        {
            "evaluation_id": 0,
            "outer_iter": 0,
            "boundary_hash": "hash0",
            "boundary": {
                "n_field_periods": 3,
                "is_stellarator_symmetric": True,
                "r_cos": [1.0],
                "z_sin": [0.1],
            },
            "success": True,
            "is_feasible": True,
            "objective": 3.5,
            "minimize": True,
            "feasibility_norm": 0.05,
            "adapter_feas_norm": 0.02,
            "metrics": {
                "aspect_ratio": 3.5,
                "average_triangularity": -0.55,
                "edge_rotational_transform_over_n_field_periods": 0.35,
            },
            "fidelity": "very_low_fidelity",
            "duration_sec": 1.0,
        },
        {
            "evaluation_id": 1,
            "outer_iter": 0,
            "boundary_hash": "hash1",
            "boundary": {
                "n_field_periods": 3,
                "is_stellarator_symmetric": True,
                "r_cos": [1.0],
                "z_sin": [0.1],
            },
            "success": True,
            "is_feasible": True,
            "objective": 3.2,
            "minimize": True,
            "feasibility_norm": 0.04,
            "adapter_feas_norm": 0.01,
            "metrics": {
                "aspect_ratio": 3.4,
                "average_triangularity": -0.6,
                "edge_rotational_transform_over_n_field_periods": 0.36,
            },
            "promotion_source": 0,
            "fidelity": "high_fidelity",
            "duration_sec": 1.5,
        },
    ]

    with (run_dir / "evaluations.jsonl").open("w", encoding="utf-8") as handle:
        for record in evaluations:
            handle.write(json.dumps(record) + "\n")

    args = [
        "summarize_run.py",
        "--run",
        str(run_dir),
        "--topk",
        "1",
        "--csv",
        str(run_dir / "summary.csv"),
        "--export-best",
        str(run_dir / "top1.json"),
    ]
    monkeypatch.setattr(sys, "argv", args)

    from orchestration import summarize_run

    summarize_run.main()
    output = capsys.readouterr().out

    assert "adapter_feas" in output
    assert (run_dir / "top1.json").exists()
    csv_text = (run_dir / "summary.csv").read_text(encoding="utf-8")
    assert "adapter_feas_norm" in csv_text.splitlines()[0]
