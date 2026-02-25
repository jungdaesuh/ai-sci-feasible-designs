#!/usr/bin/env python
"""Refine P1 feasible designs using low-dimensional surrogate-guided search."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from ai_scientist.forward_model import forward_model_batch
from ai_scientist.tools.evaluation import _settings_for_stage


@dataclass(frozen=True)
class Bounds:
    aspect_ratio: tuple[float, float]
    elongation: tuple[float, float]
    rotational_transform: tuple[float, float]
    tri_scale: tuple[float, float]

    def clip(self, params: dict[str, float]) -> dict[str, float]:
        return {
            "aspect_ratio": float(np.clip(params["aspect_ratio"], *self.aspect_ratio)),
            "elongation": float(np.clip(params["elongation"], *self.elongation)),
            "rotational_transform": float(
                np.clip(params["rotational_transform"], *self.rotational_transform)
            ),
            "tri_scale": float(np.clip(params["tri_scale"], *self.tri_scale)),
        }


def _params_to_vec(params: dict[str, float]) -> np.ndarray:
    return np.array(
        [
            params["aspect_ratio"],
            params["elongation"],
            params["rotational_transform"],
            params["tri_scale"],
        ],
        dtype=float,
    )


def _predict_margins(
    pred: dict[str, float],
    *,
    ar_max: float,
    tri_max: float,
    iota_min: float,
) -> dict[str, float]:
    ar = pred["aspect_ratio"]
    tri = pred["average_triangularity"]
    iota = pred["edge_rotational_transform_over_n_field_periods"]
    ar_denom = abs(ar_max) if abs(ar_max) > 0.0 else 1.0
    tri_denom = abs(tri_max) if abs(tri_max) > 0.0 else 1.0
    iota_denom = abs(iota_min) if abs(iota_min) > 0.0 else 1.0
    return {
        "aspect_ratio": (ar - ar_max) / ar_denom,
        "average_triangularity": (tri - tri_max) / tri_denom,
        "edge_rotational_transform": (iota_min - iota) / iota_denom,
    }


def _max_violation(margins: dict[str, float]) -> float:
    values = list(margins.values())
    if not values:
        return float("inf")
    if not np.all(np.isfinite(values)):
        return float("inf")
    return float(max(0.0, max(values)))


def _is_finite_record(record: dict) -> bool:
    metrics = record["metrics"]
    values = [
        metrics["aspect_ratio"],
        metrics["average_triangularity"],
        metrics["edge_rotational_transform_over_n_field_periods"],
        metrics["max_elongation"],
    ]
    return bool(np.all(np.isfinite(values)))


def _make_boundary(params: dict[str, float], template: dict) -> dict:
    r_cos = np.asarray(template["r_cos"], dtype=float)
    z_sin = np.asarray(template["z_sin"], dtype=float)
    center = r_cos.shape[1] // 2
    minor = float(r_cos[1, center])

    # Replace low-dim parameters on the template surface
    r_cos[0, center] = 1.0
    r_cos[1, center] = minor
    z_sin[1, center] = minor * params["elongation"]
    if r_cos.shape[0] > 2:
        r_cos[2, center] = -params["tri_scale"] * minor

    if center > 0:
        r_cos[0, :center] = 0.0
        z_sin[0, : center + 1] = 0.0

    return {
        "r_cos": r_cos.tolist(),
        "z_sin": z_sin.tolist(),
        "n_field_periods": template["n_field_periods"],
        "n_periodicity": template.get("n_periodicity", 1),
        "is_stellarator_symmetric": True,
    }


def _load_seed(path: Path) -> dict:
    data = json.loads(path.read_text())
    params = data.get("params")
    if params is None:
        raise ValueError(f"Seed missing params: {path}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refine P1 feasible designs with surrogate-guided local search."
    )
    parser.add_argument(
        "--seed-dir",
        type=Path,
        default=Path("artifacts/p1/low_feasible/lowdim_search_refined"),
    )
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--proposals-per-seed", type=int, default=30)
    parser.add_argument("--eval-per-seed", type=int, default=5)
    parser.add_argument("--min-train", type=int, default=12)
    parser.add_argument("--radius", type=float, default=0.15)
    parser.add_argument("--radius-min", type=float, default=0.03)
    parser.add_argument("--radius-max", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/p1/lowdim_refine"),
    )
    parser.add_argument("--nfp", type=int, default=3)
    parser.add_argument("--max-poloidal", type=int, default=3)
    parser.add_argument("--max-toroidal", type=int, default=3)
    parser.add_argument("--strict-ar-max", type=float, default=4.0)
    parser.add_argument("--strict-tri-max", type=float, default=-0.5)
    parser.add_argument("--strict-iota-min", type=float, default=0.3)
    parser.add_argument("--strict-feasibility-target", type=float, default=1e-3)
    parser.add_argument(
        "--stage",
        type=str,
        default="screen",
        help="Evaluation stage: screen (low fidelity) or promote (high fidelity).",
    )
    parser.add_argument("--prerelax", action="store_true")
    parser.add_argument("--prerelax-steps", type=int, default=50)
    parser.add_argument("--prerelax-lr", type=float, default=1e-2)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.jsonl"

    seed_paths = sorted(args.seed_dir.glob("p1_real_feasible_*.json"))
    if not seed_paths:
        raise RuntimeError(f"No seeds found in {args.seed_dir}")

    seeds = [_load_seed(path) for path in seed_paths]
    anchors = [seed["params"] for seed in seeds]
    templates = [
        {
            "r_cos": seed["r_cos"],
            "z_sin": seed["z_sin"],
            "n_field_periods": seed.get("n_field_periods", args.nfp),
            "n_periodicity": seed.get("n_periodicity", 1),
        }
        for seed in seeds
    ]

    bounds = Bounds(
        aspect_ratio=(3.0, 4.0),
        elongation=(1.1, 2.4),
        rotational_transform=(1.2, 2.4),
        tri_scale=(0.5, 0.9),
    )

    records: list[dict] = []
    for seed in seeds:
        seed_feas = seed.get("feasibility")
        if seed_feas is None and isinstance(seed.get("metrics"), dict):
            seed_feas = seed["metrics"].get("feasibility")
        if seed_feas is None:
            seed_feas = 0.0
        record = {
            "params": seed["params"],
            "metrics": seed["metrics"],
            "feasibility": float(seed_feas),
        }
        records.append(record)

    best_by_seed = [seed for seed in seeds]

    def _fit_surrogates() -> dict[str, RandomForestRegressor]:
        train_records = [r for r in records if _is_finite_record(r)]
        if not train_records:
            raise RuntimeError("No finite records available for surrogate training.")
        X = np.vstack([_params_to_vec(r["params"]) for r in train_records])
        targets = {
            "aspect_ratio": np.array(
                [r["metrics"]["aspect_ratio"] for r in train_records], dtype=float
            ),
            "average_triangularity": np.array(
                [r["metrics"]["average_triangularity"] for r in train_records],
                dtype=float,
            ),
            "edge_rotational_transform_over_n_field_periods": np.array(
                [
                    r["metrics"]["edge_rotational_transform_over_n_field_periods"]
                    for r in train_records
                ],
                dtype=float,
            ),
            "max_elongation": np.array(
                [r["metrics"]["max_elongation"] for r in train_records], dtype=float
            ),
        }
        models: dict[str, RandomForestRegressor] = {}
        for key, y in targets.items():
            reg = RandomForestRegressor(
                n_estimators=120,
                max_depth=10,
                random_state=args.seed,
                n_jobs=1,
            )
            reg.fit(X, y)
            models[key] = reg
        return models

    for cycle in range(1, args.cycles + 1):
        # Re-center proposals around the current best per seed.
        anchors = [best["params"] for best in best_by_seed]
        models = None
        if len(records) >= args.min_train:
            models = _fit_surrogates()

        batch_params: list[dict[str, float]] = []
        batch_boundaries: list[dict] = []
        batch_seed_idx: list[int] = []

        for seed_idx, anchor in enumerate(anchors):
            proposals = []
            for _ in range(args.proposals_per_seed):
                noise = rng.normal(scale=args.radius, size=4)
                candidate = {
                    "aspect_ratio": anchor["aspect_ratio"] * (1 + noise[0]),
                    "elongation": anchor["elongation"] * (1 + noise[1]),
                    "rotational_transform": anchor["rotational_transform"]
                    * (1 + noise[2]),
                    "tri_scale": anchor["tri_scale"] * (1 + noise[3]),
                }
                proposals.append(bounds.clip(candidate))

            if models is None:
                chosen = proposals[: args.eval_per_seed]
            else:
                scores = []
                Xp = np.vstack([_params_to_vec(p) for p in proposals])
                pred = {key: model.predict(Xp) for key, model in models.items()}
                for idx in range(len(proposals)):
                    pred_metrics = {
                        "aspect_ratio": float(pred["aspect_ratio"][idx]),
                        "average_triangularity": float(
                            pred["average_triangularity"][idx]
                        ),
                        "edge_rotational_transform_over_n_field_periods": float(
                            pred["edge_rotational_transform_over_n_field_periods"][idx]
                        ),
                        "max_elongation": float(pred["max_elongation"][idx]),
                    }
                    margins = _predict_margins(
                        pred_metrics,
                        ar_max=args.strict_ar_max,
                        tri_max=args.strict_tri_max,
                        iota_min=args.strict_iota_min,
                    )
                    violation = _max_violation(margins)
                    scores.append((violation, pred_metrics["max_elongation"], idx))
                scores.sort(key=lambda item: (item[0], item[1]))
                chosen = [proposals[item[2]] for item in scores[: args.eval_per_seed]]

            for cand in chosen:
                boundary = _make_boundary(cand, templates[seed_idx])
                batch_params.append(cand)
                batch_boundaries.append(boundary)
                batch_seed_idx.append(seed_idx)

        settings = _settings_for_stage(args.stage, "p1", skip_qi=True)
        if args.prerelax:
            settings = settings.model_copy(
                update={
                    "prerelax": True,
                    "prerelax_steps": args.prerelax_steps,
                    "prerelax_lr": args.prerelax_lr,
                }
            )
        results = forward_model_batch(
            batch_boundaries, settings, n_workers=1, pool_type="thread"
        )

        improved = False
        for params, seed_idx, result in zip(batch_params, batch_seed_idx, results):
            metrics = {
                "aspect_ratio": float(result.metrics.aspect_ratio),
                "average_triangularity": float(result.metrics.average_triangularity),
                "edge_rotational_transform_over_n_field_periods": float(
                    result.metrics.edge_rotational_transform_over_n_field_periods
                ),
                "max_elongation": float(result.metrics.max_elongation),
            }
            strict_margins = _predict_margins(
                metrics,
                ar_max=args.strict_ar_max,
                tri_max=args.strict_tri_max,
                iota_min=args.strict_iota_min,
            )
            strict_feasibility = _max_violation(strict_margins)
            record = {
                "params": params,
                "metrics": metrics,
                "feasibility": float(result.feasibility),
                "strict_feasibility": strict_feasibility,
                "seed_index": seed_idx,
            }
            records.append(record)
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")

            current = best_by_seed[seed_idx]
            current_strict = float(current.get("strict_feasibility", float("inf")))
            current_elong = float(current["metrics"]["max_elongation"])
            better_feas = strict_feasibility < current_strict - 1e-6
            similar_feas = abs(strict_feasibility - current_strict) <= 1e-6
            better_elong = metrics["max_elongation"] < current_elong

            if better_feas or (similar_feas and better_elong):
                best_by_seed[seed_idx] = {
                    "params": params,
                    "metrics": metrics,
                    "feasibility": record["feasibility"],
                    "strict_feasibility": strict_feasibility,
                }
                improved = True

        if improved:
            args.radius = max(args.radius_min, args.radius * 0.9)
        else:
            args.radius = min(args.radius_max, args.radius * 1.05)

        summary = {
            "cycle": cycle,
            "radius": args.radius,
            "best_by_seed": best_by_seed,
        }
        (output_dir / f"cycle_{cycle:03}.json").write_text(
            json.dumps(summary, indent=2)
        )

    (output_dir / "best_by_seed.json").write_text(json.dumps(best_by_seed, indent=2))

    best_dir = output_dir / "best_boundaries"
    best_dir.mkdir(parents=True, exist_ok=True)
    for idx, best in enumerate(best_by_seed):
        boundary = _make_boundary(best["params"], templates[idx])
        record = dict(boundary)
        record["params"] = best["params"]
        record["metrics"] = best["metrics"]
        record["feasibility"] = float(best.get("feasibility", float("nan")))
        record["strict_feasibility"] = float(
            best.get("strict_feasibility", float("nan"))
        )
        (best_dir / f"p1_best_{idx:03}.json").write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
