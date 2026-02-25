#!/usr/bin/env python
"""Generate low-fidelity P1 candidates using a lightweight iota surrogate.

Workflow:
1) Stream HF dataset to train a regressor for edge rotational transform (iota).
2) Sample Fourier coefficients that satisfy geometry constraints (AR, triangularity).
3) Filter by surrogate-predicted iota and keep lowest-elongation candidates.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from constellaration.geometry import surface_rz_fourier
from constellaration.initial_guess import generate_rotating_ellipse
from datasets import load_dataset
from sklearn.ensemble import RandomForestRegressor


@dataclass(frozen=True)
class Candidate:
    max_elongation: float
    aspect_ratio: float
    average_triangularity: float
    predicted_iota: float
    n_field_periods: int
    r_cos: np.ndarray
    z_sin: np.ndarray
    seed: int


def _fourier_to_real_space(
    r_cos: torch.Tensor,
    z_sin: torch.Tensor,
    *,
    nfp: int,
    n_theta: int = 64,
    n_zeta: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute R, Z grids from Fourier coefficients (batched)."""
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
    """Approximate aspect ratio using mean major radius and mean minor radius."""
    R_center_per_zeta = torch.mean(R, dim=1)
    R_major = torch.mean(R_center_per_zeta, dim=1)

    R_max = torch.max(R, dim=1).values
    R_min = torch.min(R, dim=1).values
    minor_radius = (R_max - R_min) / 2.0
    mean_minor = torch.mean(minor_radius, dim=1).clamp(min=1e-6)

    return R_major / mean_minor


def _elongation(R: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
    """Compute max elongation using covariance eigenvalues per zeta slice."""
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


def _triangularity(R: torch.Tensor, Z: torch.Tensor, *, nfp: int) -> torch.Tensor:
    """Compute average triangularity at zeta=0 and zeta=pi/nfp planes."""
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


def _get_metric(example: dict, key: str) -> float | None:
    val = example.get(f"metrics.{key}")
    if val is not None:
        return float(val)
    metrics = example.get("metrics")
    if isinstance(metrics, dict):
        inner = metrics.get(key)
        return float(inner) if inner is not None else None
    return None


def _iter_dataset(split: str) -> np.ndarray:
    return load_dataset("proxima-fusion/constellaration", split=split, streaming=True)


def _flatten_params(r_cos: np.ndarray, z_sin: np.ndarray, nfp: int) -> np.ndarray:
    return np.concatenate([r_cos.reshape(-1), z_sin.reshape(-1), np.array([nfp])])


def _fit_iota_regressor(
    *,
    samples: int,
    split: str,
    nfp_filter: int,
    seed: int,
) -> tuple[RandomForestRegressor, tuple[int, int], list[tuple[np.ndarray, np.ndarray]]]:
    rng = np.random.default_rng(seed)
    X: list[np.ndarray] = []
    y: list[float] = []
    bases: list[tuple[np.ndarray, np.ndarray]] = []

    dataset = _iter_dataset(split)
    expected_shape: tuple[int, int] | None = None

    for example in dataset:
        if len(X) >= samples:
            break

        raw_nfp = example.get("boundary.n_field_periods")
        if raw_nfp is None:
            continue
        nfp_val = int(raw_nfp)
        if nfp_filter and nfp_val != nfp_filter:
            continue

        iota = _get_metric(example, "edge_rotational_transform_over_n_field_periods")
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

        if len(bases) < 200:
            bases.append((r_cos.copy(), z_sin.copy()))

        X.append(_flatten_params(r_cos, z_sin, nfp_val))
        y.append(float(iota))

    if expected_shape is None:
        raise RuntimeError("No training samples found for iota regressor")

    X_arr = np.vstack(X)
    y_arr = np.asarray(y, dtype=float)

    reg = RandomForestRegressor(
        n_estimators=120,
        max_depth=16,
        random_state=seed,
        n_jobs=1,
    )
    reg.fit(X_arr, y_arr)

    return reg, expected_shape, bases


def _sample_coefficients(
    *,
    rng: np.random.Generator,
    shape: tuple[int, int],
    nfp: int,
    base_mode: str,
    base_surfaces: list[tuple[np.ndarray, np.ndarray]] | None = None,
    use_base_prob: float = 0.7,
) -> tuple[np.ndarray, np.ndarray]:
    mpol_plus_1, two_ntor_plus_1 = shape
    ntor = (two_ntor_plus_1 - 1) // 2
    center = ntor

    if base_mode == "rotating_ellipse":
        aspect_ratio = rng.uniform(3.0, 4.4)
        elongation = rng.uniform(1.2, 2.0)
        rotational_transform = rng.uniform(0.6, 1.4)
        base_surface = generate_rotating_ellipse(
            aspect_ratio=float(aspect_ratio),
            elongation=float(elongation),
            rotational_transform=float(rotational_transform),
            n_field_periods=nfp,
        )
        base_surface = surface_rz_fourier.set_max_mode_numbers(
            base_surface,
            max_poloidal_mode=mpol_plus_1 - 1,
            max_toroidal_mode=ntor,
        )
        r_cos = np.asarray(base_surface.r_cos, dtype=float)
        z_sin = np.asarray(base_surface.z_sin, dtype=float)
        minor_radius = float(r_cos[1, center])
    else:
        if base_surfaces and rng.random() < use_base_prob:
            base_r, base_z = base_surfaces[rng.integers(0, len(base_surfaces))]
            r_cos = base_r.copy()
            z_sin = base_z.copy()
        else:
            r_cos = np.zeros(shape, dtype=float)
            z_sin = np.zeros(shape, dtype=float)

        major_radius = rng.uniform(0.95, 1.05)
        minor_radius = rng.uniform(0.22, 0.28)
        kappa = rng.uniform(0.9, 1.1)

        base_minor = max(abs(r_cos[1, center]), abs(z_sin[1, center]), 1e-6)
        scale = minor_radius / base_minor
        r_cos = r_cos * scale
        z_sin = z_sin * scale

        r_cos[0, center] = major_radius
        r_cos[1, center] = minor_radius
        z_sin[1, center] = minor_radius * kappa

    tri_scale = rng.uniform(0.5, 0.8)
    if mpol_plus_1 > 2:
        r_cos[2, center] = -tri_scale * minor_radius

    helical_scale = (
        rng.uniform(0.01, 0.05)
        if base_mode == "rotating_ellipse"
        else rng.uniform(0.02, 0.08)
    )
    for m in range(1, min(4, mpol_plus_1)):
        if center + 1 < two_ntor_plus_1:
            amp_r = rng.normal(scale=helical_scale)
            amp_z = rng.normal(scale=helical_scale)
            r_cos[m, center + 1] = amp_r
            z_sin[m, center + 1] = amp_z
            r_cos[m, center - 1] = amp_r
            z_sin[m, center - 1] = amp_z
        if center + 2 < two_ntor_plus_1 and m >= 2:
            amp_r = rng.normal(scale=helical_scale * 0.6)
            amp_z = rng.normal(scale=helical_scale * 0.6)
            r_cos[m, center + 2] = amp_r
            z_sin[m, center + 2] = amp_z
            r_cos[m, center - 2] = amp_r
            z_sin[m, center - 2] = amp_z

    return r_cos, z_sin


def _write_boundary(path: Path, candidate: Candidate) -> None:
    payload = {
        "r_cos": candidate.r_cos.tolist(),
        "z_sin": candidate.z_sin.tolist(),
        "n_field_periods": candidate.n_field_periods,
        "n_periodicity": 1,
        "is_stellarator_symmetric": True,
    }
    path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate low-fidelity P1 candidates using an iota surrogate."
    )
    parser.add_argument("--train-samples", type=int, default=20000)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--train-nfp", type=int, default=0)
    parser.add_argument("--generate-samples", type=int, default=200000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--nfp", type=int, default=3)
    parser.add_argument(
        "--base-mode",
        choices=["dataset", "rotating_ellipse"],
        default="rotating_ellipse",
    )
    parser.add_argument("--use-base-prob", type=float, default=0.7)
    parser.add_argument("--ar-max", type=float, default=4.0)
    parser.add_argument("--tri-max", type=float, default=-0.5)
    parser.add_argument("--iota-min", type=float, default=0.3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/p1/low_feasible/generated"),
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print(
        f"Training iota regressor on {args.train_samples} samples "
        f"(nfp filter={args.train_nfp or 'any'})..."
    )
    regressor, shape, base_surfaces = _fit_iota_regressor(
        samples=args.train_samples,
        split=args.train_split,
        nfp_filter=args.train_nfp,
        seed=args.seed,
    )
    print(f"Training complete. Shape={shape}.")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates: list[Candidate] = []
    total_generated = 0
    geom_pass = 0
    iota_pass = 0
    total = args.generate_samples
    batch = args.batch_size
    n_batches = (total + batch - 1) // batch

    for batch_idx in range(n_batches):
        current = min(batch, total - batch_idx * batch)
        r_batch = np.zeros((current, *shape), dtype=float)
        z_batch = np.zeros((current, *shape), dtype=float)
        seeds = []

        for i in range(current):
            seed_val = int(rng.integers(0, 1_000_000_000))
            seeds.append(seed_val)
            local_rng = np.random.default_rng(seed_val)
            r_cos, z_sin = _sample_coefficients(
                rng=local_rng,
                shape=shape,
                nfp=args.nfp,
                base_mode=args.base_mode,
                base_surfaces=base_surfaces,
                use_base_prob=args.use_base_prob,
            )
            r_batch[i] = r_cos
            z_batch[i] = z_sin

        total_generated += current
        r_tensor = torch.tensor(r_batch, dtype=torch.float32)
        z_tensor = torch.tensor(z_batch, dtype=torch.float32)

        R, Z = _fourier_to_real_space(
            r_tensor, z_tensor, nfp=args.nfp, n_theta=64, n_zeta=64
        )
        aspect_ratio = _aspect_ratio(R).cpu().numpy()
        triangularity = _triangularity(R, Z, nfp=args.nfp).cpu().numpy()
        elongation = _elongation(R, Z).cpu().numpy()

        geom_mask = (aspect_ratio <= args.ar_max) & (triangularity <= args.tri_max)
        geom_pass += int(np.count_nonzero(geom_mask))
        if not np.any(geom_mask):
            continue

        geom_indices = np.where(geom_mask)[0]
        feat_list = []
        for idx in geom_indices:
            feat_list.append(_flatten_params(r_batch[idx], z_batch[idx], args.nfp))
        feat_arr = np.vstack(feat_list)
        iota_pred = regressor.predict(feat_arr)
        iota_mask = iota_pred >= args.iota_min
        iota_pass += int(np.count_nonzero(iota_mask))

        for rel_idx, ok in enumerate(iota_mask):
            if not ok:
                continue
            idx = int(geom_indices[rel_idx])
            cand = Candidate(
                max_elongation=float(elongation[idx]),
                aspect_ratio=float(aspect_ratio[idx]),
                average_triangularity=float(triangularity[idx]),
                predicted_iota=float(iota_pred[rel_idx]),
                n_field_periods=int(args.nfp),
                r_cos=r_batch[idx],
                z_sin=z_batch[idx],
                seed=int(seeds[idx]),
            )
            candidates.append(cand)

    if not candidates:
        print(
            "No candidates found. "
            f"Generated={total_generated}, geom_pass={geom_pass}, iota_pass={iota_pass}."
        )
        return

    candidates.sort(key=lambda item: item.max_elongation)
    selected = candidates[: args.top_k]

    manifest = []
    for rank, cand in enumerate(selected, start=1):
        fname = f"p1_low_fidelity_{rank:03}.json"
        path = output_dir / fname
        _write_boundary(path, cand)
        manifest.append(
            {
                "rank": rank,
                "seed": cand.seed,
                "max_elongation": cand.max_elongation,
                "aspect_ratio": cand.aspect_ratio,
                "average_triangularity": cand.average_triangularity,
                "predicted_iota": cand.predicted_iota,
                "n_field_periods": cand.n_field_periods,
                "file": str(path),
            }
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(
        f"Wrote {len(selected)} candidates to {output_dir} "
        f"(generated={total_generated}, geom_pass={geom_pass}, iota_pass={iota_pass})."
    )


if __name__ == "__main__":
    main()
