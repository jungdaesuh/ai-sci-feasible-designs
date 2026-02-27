from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np


class _DummySurface:
    def __init__(self) -> None:
        self.r_cos = np.array([[1.0]], dtype=float)
        self.z_sin = np.array([[0.0]], dtype=float)
        self.n_field_periods = 3
        self.is_stellarator_symmetric = True
        self.poloidal_modes = np.array([[0]], dtype=int)
        self.toroidal_modes = np.array([[0]], dtype=int)
        self.max_poloidal_mode = 1
        self.max_toroidal_mode = 1

    def model_copy(self, update: dict | None = None) -> "_DummySurface":
        if update:
            for key, value in update.items():
                setattr(self, key, value)
        return self


class _DummyALState:
    def __init__(
        self,
        *,
        x: np.ndarray,
        multipliers: np.ndarray,
        penalty_parameters: np.ndarray,
        objective: float,
        constraints: np.ndarray,
        bounds: np.ndarray,
    ) -> None:
        self.x = np.asarray(x, dtype=float)
        self.multipliers = np.asarray(multipliers, dtype=float)
        self.penalty_parameters = np.asarray(penalty_parameters, dtype=float)
        self.objective = float(objective)
        self.constraints = np.asarray(constraints, dtype=float)
        self.bounds = np.asarray(bounds, dtype=float)

    def model_copy(self, update: dict | None = None) -> "_DummyALState":
        if update:
            for key, value in update.items():
                setattr(self, key, value)
        return self


class _DummyALSettings:
    def __init__(self, **_: object) -> None:
        pass


def _dummy_mask() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        r_cos=np.array([[True]], dtype=bool),
        z_sin=np.array([[True]], dtype=bool),
    )


def _install_import_stubs(monkeypatch) -> None:
    jax_mod = types.ModuleType("jax")
    jax_mod.numpy = np
    monkeypatch.setitem(sys.modules, "jax", jax_mod)
    monkeypatch.setitem(sys.modules, "jax.numpy", np)

    ng_mod = types.ModuleType("nevergrad")
    ng_mod.p = types.SimpleNamespace(Array=lambda init, lower, upper: np.asarray(init))

    class _DummyCandidate:
        def __init__(self, value: np.ndarray) -> None:
            self.value = value

    class _DummyNGOpt:
        def __init__(
            self, *, parametrization: np.ndarray, budget: int, num_workers: int
        ):
            self._value = np.asarray(parametrization, dtype=float)
            self._cand = _DummyCandidate(self._value)

        def ask(self) -> _DummyCandidate:
            return self._cand

        def tell(self, cand: _DummyCandidate, loss: float) -> None:
            _ = (cand, loss)

        def provide_recommendation(self) -> _DummyCandidate:
            return self._cand

    ng_mod.optimizers = types.SimpleNamespace(NGOpt=_DummyNGOpt)
    monkeypatch.setitem(sys.modules, "nevergrad", ng_mod)

    const_mod = types.ModuleType("constellaration")
    monkeypatch.setitem(sys.modules, "constellaration", const_mod)

    forward_model_mod = types.ModuleType("constellaration.forward_model")

    class _ConstellarationSettings:
        def __init__(self, **_: object) -> None:
            pass

    forward_model_mod.ConstellarationSettings = _ConstellarationSettings
    monkeypatch.setitem(sys.modules, "constellaration.forward_model", forward_model_mod)

    geometry_mod = types.ModuleType("constellaration.geometry")
    surface_mod = types.ModuleType("constellaration.geometry.surface_rz_fourier")
    surface_mod.SurfaceRZFourier = type("SurfaceRZFourier", (), {})
    geometry_mod.surface_rz_fourier = surface_mod
    monkeypatch.setitem(sys.modules, "constellaration.geometry", geometry_mod)
    monkeypatch.setitem(
        sys.modules, "constellaration.geometry.surface_rz_fourier", surface_mod
    )

    initial_guess_mod = types.ModuleType("constellaration.initial_guess")
    initial_guess_mod.generate_rotating_ellipse = lambda **_: _DummySurface()
    monkeypatch.setitem(sys.modules, "constellaration.initial_guess", initial_guess_mod)

    mhd_mod = types.ModuleType("constellaration.mhd")
    vmec_settings_mod = types.ModuleType("constellaration.mhd.vmec_settings")
    vmec_settings_mod.VmecPresetSettings = type("VmecPresetSettings", (), {})
    mhd_mod.vmec_settings = vmec_settings_mod
    monkeypatch.setitem(sys.modules, "constellaration.mhd", mhd_mod)
    monkeypatch.setitem(
        sys.modules, "constellaration.mhd.vmec_settings", vmec_settings_mod
    )

    optim_mod = types.ModuleType("constellaration.optimization")
    al_mod = types.ModuleType("constellaration.optimization.augmented_lagrangian")
    optim_mod.augmented_lagrangian = al_mod
    monkeypatch.setitem(sys.modules, "constellaration.optimization", optim_mod)
    monkeypatch.setitem(
        sys.modules, "constellaration.optimization.augmented_lagrangian", al_mod
    )

    utils_mod = types.ModuleType("constellaration.utils")
    pytree_mod = types.ModuleType("constellaration.utils.pytree")
    utils_mod.pytree = pytree_mod
    monkeypatch.setitem(sys.modules, "constellaration.utils", utils_mod)
    monkeypatch.setitem(sys.modules, "constellaration.utils.pytree", pytree_mod)


def _patch_runtime_dependencies(
    monkeypatch,
    module,
    *,
    result: types.SimpleNamespace,
    dummy_surface: _DummySurface,
) -> None:
    mask = _dummy_mask()

    monkeypatch.setattr(
        module, "generate_rotating_ellipse", lambda **_: _DummySurface()
    )
    monkeypatch.setattr(module, "_make_settings", lambda vmec_fidelity: object())
    monkeypatch.setattr(
        module.surface_rz_fourier,
        "set_max_mode_numbers",
        lambda surface, **_: surface,
        raising=False,
    )
    monkeypatch.setattr(
        module.surface_rz_fourier,
        "build_mask",
        lambda *_, **__: mask,
        raising=False,
    )
    monkeypatch.setattr(
        module.surface_rz_fourier,
        "compute_infinity_norm_spectrum_scaling_fun",
        lambda **_: np.array([1.0], dtype=float),
        raising=False,
    )
    monkeypatch.setattr(
        module.pytree,
        "register_pydantic_data",
        lambda *_, **__: None,
        raising=False,
    )
    monkeypatch.setattr(
        module.pytree,
        "mask_and_ravel",
        lambda *_, **__: (
            np.array([0.0, 0.0], dtype=float),
            lambda _x: dummy_surface,
        ),
        raising=False,
    )
    monkeypatch.setattr(module, "forward_model_batch", lambda *_, **__: [result])

    monkeypatch.setattr(
        module.al,
        "AugmentedLagrangianState",
        _DummyALState,
        raising=False,
    )
    monkeypatch.setattr(
        module.al,
        "AugmentedLagrangianSettings",
        _DummyALSettings,
        raising=False,
    )
    monkeypatch.setattr(
        module.al,
        "augmented_lagrangian_function",
        lambda objective, constraints, state: float(objective),
        raising=False,
    )
    monkeypatch.setattr(
        module.al,
        "update_augmented_lagrangian_state",
        lambda *, x, objective, constraints, state, settings: _DummyALState(
            x=np.asarray(x, dtype=float),
            multipliers=np.asarray(state.multipliers, dtype=float),
            penalty_parameters=np.asarray(state.penalty_parameters, dtype=float),
            objective=float(objective),
            constraints=np.asarray(constraints, dtype=float),
            bounds=np.asarray(state.bounds, dtype=float),
        ),
        raising=False,
    )


def _install_selector_stub(
    monkeypatch,
    module,
    *,
    selected_identity: str,
) -> list[dict]:
    selector_calls: list[dict] = []

    def _fake_select(**kwargs):
        selector_calls.append(kwargs)
        return (
            np.asarray(kwargs["state_x"], dtype=float),
            "state",
            selected_identity,
            {
                "selected_label": "state",
                "selected_identity": selected_identity,
                "scores": [],
            },
            {selected_identity: 1},
        )

    monkeypatch.setattr(module, "select_adaptive_restart_runtime", _fake_select)
    return selector_calls


def _run_adaptive_once(
    monkeypatch, module, *, script_name: str, output_dir: Path
) -> None:
    argv = [
        script_name,
        "--adaptive-restart",
        "--outer-iters",
        "1",
        "--budget-initial",
        "1",
        "--budget-increment",
        "0",
        "--budget-max",
        "1",
        "--promote-every",
        "0",
        "--output-dir",
        str(output_dir),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    module.main()


def _read_restart_history(output_dir: Path) -> dict:
    restart_history = output_dir / "restart_history.jsonl"
    assert restart_history.exists()
    return json.loads(restart_history.read_text(encoding="utf-8").splitlines()[0])


def test_p1_main_adaptive_restart_runtime_path(monkeypatch, tmp_path: Path) -> None:
    _install_import_stubs(monkeypatch)
    module = importlib.import_module("scripts.p1_alm_ngopt_multifidelity")

    metrics = types.SimpleNamespace(
        aspect_ratio=4.0,
        average_triangularity=-0.5,
        edge_rotational_transform_over_n_field_periods=0.3,
        max_elongation=2.0,
    )
    result = types.SimpleNamespace(
        objective=2.0,
        feasibility=0.0,
        metrics=metrics,
        error_message=None,
    )

    _patch_runtime_dependencies(
        monkeypatch,
        module,
        result=result,
        dummy_surface=_DummySurface(),
    )
    selector_calls = _install_selector_stub(
        monkeypatch,
        module,
        selected_identity="x:runtime-test",
    )

    output_dir = tmp_path / "out"
    _run_adaptive_once(
        monkeypatch,
        module,
        script_name="p1_alm_ngopt_multifidelity.py",
        output_dir=output_dir,
    )

    assert len(selector_calls) == 1
    assert selector_calls[0]["problem"] == "p1"
    assert float(selector_calls[0]["novelty_min_distance"]) == 0.05
    assert float(selector_calls[0]["novelty_near_duplicate_distance"]) == 0.08
    assert np.isinf(float(selector_calls[0]["novelty_feasibility_max"]))
    assert str(selector_calls[0]["novelty_judge_mode"]) == "heuristic"

    payload = _read_restart_history(output_dir)
    assert payload["selected_seed"] == "state"
    assert payload["selected_seed_identity"] == "x:runtime-test"


def test_p2_main_adaptive_restart_runtime_path(monkeypatch, tmp_path: Path) -> None:
    _install_import_stubs(monkeypatch)
    module = importlib.import_module("experiments.p1_p2.p2_alm_ngopt_multifidelity")

    metrics = types.SimpleNamespace(
        aspect_ratio=8.0,
        edge_rotational_transform_over_n_field_periods=0.3,
        qi=1e-6,
        edge_magnetic_mirror_ratio=0.1,
        max_elongation=3.0,
        minimum_normalized_magnetic_gradient_scale_length=6.0,
    )
    result = types.SimpleNamespace(
        objective=-6.0,
        feasibility=0.0,
        equilibrium_converged=True,
        metrics=metrics,
        error_message=None,
    )

    _patch_runtime_dependencies(
        monkeypatch,
        module,
        result=result,
        dummy_surface=_DummySurface(),
    )
    selector_calls = _install_selector_stub(
        monkeypatch,
        module,
        selected_identity="x:p2-runtime-test",
    )

    output_dir = tmp_path / "out-p2"
    _run_adaptive_once(
        monkeypatch,
        module,
        script_name="p2_alm_ngopt_multifidelity.py",
        output_dir=output_dir,
    )

    assert len(selector_calls) == 1
    assert selector_calls[0]["problem"] == "p2"
    assert float(selector_calls[0]["state_objective"]) == 6.0
    assert float(selector_calls[0]["feasibility_weight"]) == 0.45
    assert float(selector_calls[0]["objective_weight"]) == 0.45
    assert float(selector_calls[0]["novelty_min_distance"]) == 0.0
    assert float(selector_calls[0]["novelty_near_duplicate_distance"]) == 0.08
    assert np.isinf(float(selector_calls[0]["novelty_feasibility_max"]))
    assert str(selector_calls[0]["novelty_judge_mode"]) == "heuristic"

    payload = _read_restart_history(output_dir)
    assert payload["selected_seed"] == "state"
    assert payload["selected_seed_identity"] == "x:p2-runtime-test"


def test_p2_main_logs_null_lgradb_when_missing(monkeypatch, tmp_path: Path) -> None:
    _install_import_stubs(monkeypatch)
    module = importlib.import_module("experiments.p1_p2.p2_alm_ngopt_multifidelity")

    metrics = types.SimpleNamespace(
        aspect_ratio=8.0,
        edge_rotational_transform_over_n_field_periods=0.3,
        qi=1e-6,
        edge_magnetic_mirror_ratio=0.1,
        max_elongation=3.0,
    )
    result = types.SimpleNamespace(
        objective=-6.0,
        feasibility=0.0,
        equilibrium_converged=True,
        metrics=metrics,
        error_message=None,
    )
    _patch_runtime_dependencies(
        monkeypatch,
        module,
        result=result,
        dummy_surface=_DummySurface(),
    )
    _install_selector_stub(
        monkeypatch,
        module,
        selected_identity="x:p2-runtime-null-lgradb",
    )

    output_dir = tmp_path / "out-p2-null-lgradb"
    _run_adaptive_once(
        monkeypatch,
        module,
        script_name="p2_alm_ngopt_multifidelity.py",
        output_dir=output_dir,
    )

    history_path = output_dir / "history.jsonl"
    assert history_path.exists()
    payload = json.loads(history_path.read_text(encoding="utf-8").splitlines()[0])
    assert payload["lgradb"] is None
