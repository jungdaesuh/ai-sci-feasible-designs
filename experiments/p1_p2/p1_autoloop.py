#!/usr/bin/env python
"""Autonomous P1 search loop: generate → screen → promote → learn.

Runs a systematic, non-LLM optimization loop to find P1 high-fidelity feasible designs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[2]
VMEC_BUILD = ROOT / "vmecpp" / "build"
skip_paths = os.environ.get("SKBUILD_EDITABLE_SKIP", "").split(os.pathsep)
if str(VMEC_BUILD) not in skip_paths:
    skip_paths = [p for p in skip_paths if p]
    skip_paths.append(str(VMEC_BUILD))
    os.environ["SKBUILD_EDITABLE_SKIP"] = os.pathsep.join(skip_paths)
os.environ.setdefault("SKBUILD_EDITABLE_VERBOSE", "0")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_scientist.forward_model import forward_model_batch, get_backend
from ai_scientist.tools.evaluation import (
    _DEFAULT_RELATIVE_TOLERANCE,
    _settings_for_stage,
)


@dataclass(frozen=True)
class MetricsSummary:
    objective: float
    feasibility: float
    aspect_ratio: float
    average_triangularity: float
    edge_rotational_transform: float
    max_elongation: float


def _fourier_to_real_space(
    r_cos: torch.Tensor,
    z_sin: torch.Tensor,
    *,
    nfp: int,
    n_theta: int = 64,
    n_zeta: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, mpol_plus_1, two_ntor_plus_1 = r_cos.shape
    mpol = mpol_plus_1 - 1
    ntor = (two_ntor_plus_1 - 1) // 2
    device = r_cos.device

    theta = torch.linspace(0, 2 * torch.pi, n_theta + 1, device=device)[:-1]
    zeta = torch.linspace(0, 2 * torch.pi / nfp, n_zeta + 1, device=device)[:-1]

    m_idx = torch.arange(mpol + 1, dtype=r_cos.dtype, device=device)
    n_idx = torch.arange(2 * ntor + 1, dtype=r_cos.dtype, device=device)
    n_vals = n_idx - ntor

    m_theta = m_idx[:, None] * theta[None, :]
    cos_m_theta = torch.cos(m_theta)
    sin_m_theta = torch.sin(m_theta)

    n_nfp_zeta = (n_vals * nfp)[:, None] * zeta[None, :]
    cos_n_zeta = torch.cos(n_nfp_zeta)
    sin_n_zeta = torch.sin(n_nfp_zeta)

    A_rc = torch.einsum("bmn,nz->bmz", r_cos, cos_n_zeta)
    B_rc = torch.einsum("bmn,nz->bmz", r_cos, sin_n_zeta)
    A_zs = torch.einsum("bmn,nz->bmz", z_sin, cos_n_zeta)
    B_zs = torch.einsum("bmn,nz->bmz", z_sin, sin_n_zeta)

    R = torch.einsum("mt,bmz->btz", cos_m_theta, A_rc) + torch.einsum(
        "mt,bmz->btz", sin_m_theta, B_rc
    )
    Z = torch.einsum("mt,bmz->btz", sin_m_theta, A_zs) - torch.einsum(
        "mt,bmz->btz", cos_m_theta, B_zs
    )

    return R, Z


def _aspect_ratio(R: torch.Tensor) -> torch.Tensor:
    R_center_per_zeta = torch.mean(R, dim=1)
    R_major = torch.mean(R_center_per_zeta, dim=1)

    R_max = torch.max(R, dim=1).values
    R_min = torch.min(R, dim=1).values
    minor_radius = (R_max - R_min) / 2.0
    mean_minor = torch.mean(minor_radius, dim=1).clamp(min=1e-6)

    return R_major / mean_minor


def _elongation(R: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
    R_mean = torch.mean(R, dim=1, keepdim=True)
    Z_mean = torch.mean(Z, dim=1, keepdim=True)
    Rc = R - R_mean
    Zc = Z - Z_mean

    var_R = torch.mean(Rc**2, dim=1)
    var_Z = torch.mean(Zc**2, dim=1)
    cov_RZ = torch.mean(Rc * Zc, dim=1)

    tr = var_R + var_Z
    det = var_R * var_Z - cov_RZ**2
    disc = torch.clamp(tr**2 - 4 * det, min=0.0)
    sqrt_disc = torch.sqrt(disc)

    l1 = (tr + sqrt_disc) / 2.0
    l1_safe = torch.clamp(l1, min=1e-10)
    l2 = det / l1_safe
    l2_safe = torch.clamp(l2, min=1e-10)
    elo_slice = torch.sqrt(l1_safe / l2_safe)

    return torch.max(elo_slice, dim=1).values


def _triangularity(R: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
    _, _, n_zeta = R.shape
    mid_idx = n_zeta // 2

    def _tri_from_plane(Rp: torch.Tensor, Zp: torch.Tensor) -> torch.Tensor:
        R0 = torch.mean(Rp, dim=1)
        R_max = torch.max(Rp, dim=1).values
        R_min = torch.min(Rp, dim=1).values
        minor = (R_max - R_min) / 2.0
        idx = torch.argmax(Zp, dim=1)
        R_at_zmax = Rp.gather(1, idx[:, None]).squeeze(1)
        return (R0 - R_at_zmax) / (minor + 1e-8)

    tri0 = _tri_from_plane(R[:, :, 0], Z[:, :, 0])
    tri1 = _tri_from_plane(R[:, :, mid_idx], Z[:, :, mid_idx])
    return (tri0 + tri1) / 2.0


def _flatten_params(r_cos: np.ndarray, z_sin: np.ndarray, nfp: int) -> np.ndarray:
    return np.concatenate([r_cos.reshape(-1), z_sin.reshape(-1), np.array([nfp])])


def _resize_coeffs(coeffs: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(target_shape, dtype=float)
    if coeffs.size == 0:
        return out
    src_h, src_w = coeffs.shape
    dst_h, dst_w = target_shape
    copy_h = min(src_h, dst_h)
    src_center = src_w // 2
    dst_center = dst_w // 2
    src_start = max(0, src_center - dst_center)
    dst_start = max(0, dst_center - src_center)
    copy_w = min(src_w - src_start, dst_w - dst_start)
    out[:copy_h, dst_start : dst_start + copy_w] = coeffs[
        :copy_h, src_start : src_start + copy_w
    ]
    return out


def _load_seed_boundaries(paths: list[Path]) -> list[dict]:
    seeds: list[dict] = []
    for path in paths:
        payload = json.loads(path.read_text())
        if isinstance(payload, dict) and "r_cos" in payload and "z_sin" in payload:
            seeds.append(payload)
    return seeds


def _iter_dataset(split: str):
    return load_dataset("proxima-fusion/constellaration", split=split, streaming=True)


def _fit_iota_regressor(
    *,
    samples: int,
    split: str,
    nfp_filter: int,
    seed: int,
    history_path: Path,
) -> tuple[RandomForestRegressor, tuple[int, int]]:
    rng = np.random.default_rng(seed)
    X: list[np.ndarray] = []
    y: list[float] = []

    expected_shape: tuple[int, int] | None = None

    dataset = _iter_dataset(split)
    for example in dataset:
        if len(X) >= samples:
            break
        raw_nfp = example.get("boundary.n_field_periods")
        if raw_nfp is None:
            continue
        nfp_val = int(raw_nfp)
        if nfp_filter and nfp_val != nfp_filter:
            continue
        iota = example.get("metrics.edge_rotational_transform_over_n_field_periods")
        if iota is None:
            continue
        r_cos = np.asarray(example["boundary.r_cos"], dtype=float)
        z_sin = np.asarray(example["boundary.z_sin"], dtype=float)
        if r_cos.ndim != 2 or z_sin.ndim != 2:
            continue
        if expected_shape is None:
            expected_shape = r_cos.shape
        if r_cos.shape != expected_shape or z_sin.shape != expected_shape:
            continue
        X.append(_flatten_params(r_cos, z_sin, nfp_val))
        y.append(float(iota))

    if history_path.exists():
        for line in history_path.read_text().splitlines():
            record = json.loads(line)
            r_cos = np.asarray(record["r_cos"], dtype=float)
            z_sin = np.asarray(record["z_sin"], dtype=float)
            nfp_val = int(record["n_field_periods"])
            if expected_shape is None:
                expected_shape = r_cos.shape
            if r_cos.shape != expected_shape or z_sin.shape != expected_shape:
                continue
            X.append(_flatten_params(r_cos, z_sin, nfp_val))
            y.append(float(record["edge_rotational_transform_over_n_field_periods"]))

    if expected_shape is None:
        raise RuntimeError("No training samples found for iota regressor.")

    X_arr = np.vstack(X)
    y_arr = np.asarray(y, dtype=float)

    reg = RandomForestRegressor(
        n_estimators=120,
        max_depth=16,
        random_state=seed,
        n_jobs=1,
    )
    reg.fit(X_arr, y_arr)
    return reg, expected_shape


def _propose_candidate(
    rng: np.random.Generator,
    base_pool: list[dict],
    *,
    perturb_scale: float,
    recombine_ratio: float,
    target_shape: tuple[int, int],
    nfp_default: int,
) -> dict:
    pick = rng.random()
    if len(base_pool) >= 2 and pick < recombine_ratio:
        a = base_pool[int(rng.integers(0, len(base_pool)))]
        b = base_pool[int(rng.integers(0, len(base_pool)))]
        r_a = np.asarray(a["r_cos"], dtype=float)
        z_a = np.asarray(a["z_sin"], dtype=float)
        r_b = np.asarray(b["r_cos"], dtype=float)
        z_b = np.asarray(b["z_sin"], dtype=float)
        r_a = _resize_coeffs(r_a, target_shape)
        z_a = _resize_coeffs(z_a, target_shape)
        r_b = _resize_coeffs(r_b, target_shape)
        z_b = _resize_coeffs(z_b, target_shape)
        alpha = rng.random()
        r_cos = alpha * r_a + (1 - alpha) * r_b
        z_sin = alpha * z_a + (1 - alpha) * z_b
        base_nfp = int(a.get("n_field_periods", nfp_default))
    else:
        base = base_pool[int(rng.integers(0, len(base_pool)))]
        r_cos = _resize_coeffs(np.asarray(base["r_cos"], dtype=float), target_shape)
        z_sin = _resize_coeffs(np.asarray(base["z_sin"], dtype=float), target_shape)
        r_cos = r_cos + rng.normal(scale=perturb_scale, size=r_cos.shape)
        z_sin = z_sin + rng.normal(scale=perturb_scale, size=z_sin.shape)
        base_nfp = int(base.get("n_field_periods", nfp_default))

    center = target_shape[1] // 2
    if center > 0:
        r_cos[0, :center] = 0.0
        z_sin[0, : center + 1] = 0.0

    return {
        "r_cos": r_cos.tolist(),
        "z_sin": z_sin.tolist(),
        "n_field_periods": base_nfp,
        "n_periodicity": 1,
        "is_stellarator_symmetric": True,
    }


def _summarize_result(result) -> MetricsSummary:
    metrics = result.metrics
    return MetricsSummary(
        objective=float(result.objective),
        feasibility=float(result.feasibility),
        aspect_ratio=float(metrics.aspect_ratio),
        average_triangularity=float(metrics.average_triangularity),
        edge_rotational_transform=float(
            metrics.edge_rotational_transform_over_n_field_periods
        ),
        max_elongation=float(metrics.max_elongation),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="P1 high-fidelity autoloop.")
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--screen-per-cycle", type=int, default=128)
    parser.add_argument("--promote-k", type=int, default=16)
    parser.add_argument("--pool-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--nfp", type=int, default=3)
    parser.add_argument("--ar-max", type=float, default=4.0)
    parser.add_argument("--tri-max", type=float, default=-0.5)
    parser.add_argument("--iota-min", type=float, default=0.3)
    parser.add_argument("--perturb-scale", type=float, default=0.02)
    parser.add_argument("--recombine-ratio", type=float, default=0.3)
    parser.add_argument("--target-count", type=int, default=5)
    parser.add_argument("--train-samples", type=int, default=5000)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--train-nfp", type=int, default=0)
    parser.add_argument("--retrain-every", type=int, default=1)
    parser.add_argument("--screen-workers", type=int, default=1)
    parser.add_argument("--promote-workers", type=int, default=1)
    parser.add_argument("--pool-type", default="process")
    parser.add_argument(
        "--seed-dir",
        type=Path,
        default=Path("artifacts/p1/low_feasible/generated"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/p1/autoloop")
    )
    args = parser.parse_args()

    backend = get_backend()
    if backend.name not in ("real", "constellaration"):
        print(
            "Real physics backend not active. Set AI_SCIENTIST_PHYSICS_BACKEND=real "
            "and ensure vmecpp loads correctly."
        )
        sys.exit(1)

    seed_paths = sorted(args.seed_dir.glob("*.json"))
    if not seed_paths:
        print(f"No seeds found in {args.seed_dir}.")
        sys.exit(1)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "iota_history.jsonl"

    rng = np.random.default_rng(args.seed)
    pool = _load_seed_boundaries(seed_paths)

    print("Training iota regressor...")
    iota_regressor, target_shape = _fit_iota_regressor(
        samples=args.train_samples,
        split=args.train_split,
        nfp_filter=args.train_nfp,
        seed=args.seed,
        history_path=history_path,
    )
    print(f"Iota regressor ready. Target shape={target_shape}.")

    high_fid_passed: list[dict] = []

    for cycle in range(1, args.cycles + 1):
        print(f"[cycle {cycle}] generating {args.screen_per_cycle} candidates")
        candidates: list[dict] = []
        for _ in range(args.screen_per_cycle):
            cand = _propose_candidate(
                rng,
                pool,
                perturb_scale=args.perturb_scale,
                recombine_ratio=args.recombine_ratio,
                target_shape=target_shape,
                nfp_default=args.nfp,
            )
            candidates.append(cand)

        r_batch = np.stack(
            [
                _resize_coeffs(np.asarray(c["r_cos"], dtype=float), target_shape)
                for c in candidates
            ]
        )
        z_batch = np.stack(
            [
                _resize_coeffs(np.asarray(c["z_sin"], dtype=float), target_shape)
                for c in candidates
            ]
        )
        r_tensor = torch.tensor(r_batch, dtype=torch.float32)
        z_tensor = torch.tensor(z_batch, dtype=torch.float32)

        R, Z = _fourier_to_real_space(r_tensor, z_tensor, nfp=args.nfp)
        aspect_ratio = _aspect_ratio(R).cpu().numpy()
        triangularity = _triangularity(R, Z).cpu().numpy()
        elongation = _elongation(R, Z).cpu().numpy()

        geom_mask = (aspect_ratio <= args.ar_max) & (triangularity <= args.tri_max)
        if not np.any(geom_mask):
            print(f"[cycle {cycle}] no geometry-feasible candidates")
            continue

        geom_indices = np.where(geom_mask)[0]
        feat_list = [
            _flatten_params(r_batch[idx], z_batch[idx], args.nfp)
            for idx in geom_indices
        ]
        iota_pred = iota_regressor.predict(np.vstack(feat_list))
        iota_mask = iota_pred >= args.iota_min
        screened_candidates = [
            candidates[int(geom_indices[i])]
            for i in range(len(iota_mask))
            if iota_mask[i]
        ]

        if not screened_candidates:
            print(f"[cycle {cycle}] no iota-feasible candidates")
            continue

        screen_settings = _settings_for_stage("screen", "p1", skip_qi=True)
        screen_results = forward_model_batch(
            screened_candidates,
            screen_settings,
            n_workers=args.screen_workers,
            pool_type=args.pool_type,
        )

        screen_summaries = [_summarize_result(r) for r in screen_results]
        feasible_indices = [
            i
            for i, s in enumerate(screen_summaries)
            if s.feasibility <= _DEFAULT_RELATIVE_TOLERANCE
        ]

        promoted = []
        if feasible_indices:
            feasible_sorted = sorted(
                feasible_indices, key=lambda idx: screen_summaries[idx].max_elongation
            )
            promoted = [
                screened_candidates[idx] for idx in feasible_sorted[: args.promote_k]
            ]
        print(
            f"[cycle {cycle}] screened={len(screened_candidates)} "
            f"screen_feasible={len(feasible_indices)} promoted={len(promoted)}"
        )

        promote_results = []
        promote_summaries = []
        if promoted:
            promote_settings = _settings_for_stage("promote", "p1", skip_qi=True)
            promote_results = forward_model_batch(
                promoted,
                promote_settings,
                n_workers=args.promote_workers,
                pool_type=args.pool_type,
            )
            promote_summaries = [_summarize_result(r) for r in promote_results]

        for cand, summary in zip(promoted, promote_summaries):
            if summary.feasibility <= _DEFAULT_RELATIVE_TOLERANCE:
                record = dict(cand)
                record["metrics"] = summary.__dict__
                high_fid_passed.append(record)

        cycle_report = {
            "cycle": cycle,
            "screened": len(screened_candidates),
            "screen_feasible": len(feasible_indices),
            "promoted": len(promoted),
            "high_fid_feasible": sum(
                1
                for s in promote_summaries
                if s.feasibility <= _DEFAULT_RELATIVE_TOLERANCE
            ),
            "best_high_fid": min(
                [s.max_elongation for s in promote_summaries],
                default=None,
            ),
        }
        (output_dir / f"cycle_{cycle:03}.json").write_text(
            json.dumps(cycle_report, indent=2)
        )

        if screened_candidates:
            with history_path.open("a", encoding="utf-8") as handle:
                for cand, summary in zip(screened_candidates, screen_summaries):
                    record = {
                        "r_cos": cand["r_cos"],
                        "z_sin": cand["z_sin"],
                        "n_field_periods": cand.get("n_field_periods", args.nfp),
                        "edge_rotational_transform_over_n_field_periods": summary.edge_rotational_transform,
                    }
                    handle.write(json.dumps(record) + "\n")

        if promote_summaries:
            best_idx = int(np.argmin([s.max_elongation for s in promote_summaries]))
            best_params = promoted[best_idx]
            pool.append(best_params)
        else:
            if feasible_indices:
                best_idx = feasible_indices[0]
                pool.append(screened_candidates[best_idx])

        pool = pool[-args.pool_size :]

        if cycle % args.retrain_every == 0:
            print(f"[cycle {cycle}] retraining iota regressor")
            iota_regressor, target_shape = _fit_iota_regressor(
                samples=args.train_samples,
                split=args.train_split,
                nfp_filter=args.train_nfp,
                seed=int(rng.integers(0, 1_000_000_000)),
                history_path=history_path,
            )

        if len(high_fid_passed) >= args.target_count:
            break

    (output_dir / "high_fidelity_passed.json").write_text(
        json.dumps(high_fid_passed, indent=2)
    )


if __name__ == "__main__":
    main()
