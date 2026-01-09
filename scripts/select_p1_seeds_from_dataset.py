#!/usr/bin/env python
"""Select low-elongation P1-feasible seeds directly from the HF dataset."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset


@dataclass(frozen=True)
class Candidate:
    max_elongation: float
    aspect_ratio: float
    average_triangularity: float
    edge_rotational_transform: float
    n_field_periods: int
    r_cos: list[list[float]]
    z_sin: list[list[float]]


def _get(example: dict, key: str) -> float | None:
    value = example.get(f"metrics.{key}")
    if value is not None:
        return float(value)
    metrics = example.get("metrics")
    if isinstance(metrics, dict):
        inner = metrics.get(key)
        if inner is not None:
            return float(inner)
    return None


def _is_truthy(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pick P1-feasible seeds from the constellaration dataset."
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--nfp", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=100000)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/p1/low_feasible/generated"),
    )
    args = parser.parse_args()

    dataset = load_dataset(
        "proxima-fusion/constellaration", split=args.split, streaming=True
    )

    candidates: list[Candidate] = []
    inspected = 0

    for example in dataset:
        inspected += 1
        if inspected > args.max_samples:
            break

        nfp_val = example.get("boundary.n_field_periods")
        if nfp_val is None:
            continue
        if args.nfp and int(nfp_val) != args.nfp:
            continue

        if _is_truthy(example.get("misc.has_neurips_2025_forward_model_error")):
            continue

        aspect_ratio = _get(example, "aspect_ratio")
        avg_tri = _get(example, "average_triangularity")
        edge_iota = _get(example, "edge_rotational_transform_over_n_field_periods")
        max_elong = _get(example, "max_elongation")
        if (
            aspect_ratio is None
            or avg_tri is None
            or edge_iota is None
            or max_elong is None
        ):
            continue

        if not (aspect_ratio <= 4.0 and avg_tri <= -0.5 and edge_iota >= 0.3):
            continue

        r_cos = example.get("boundary.r_cos")
        z_sin = example.get("boundary.z_sin")
        if r_cos is None or z_sin is None:
            continue

        candidates.append(
            Candidate(
                max_elongation=float(max_elong),
                aspect_ratio=float(aspect_ratio),
                average_triangularity=float(avg_tri),
                edge_rotational_transform=float(edge_iota),
                n_field_periods=int(nfp_val),
                r_cos=r_cos,
                z_sin=z_sin,
            )
        )

    if not candidates:
        print("No dataset candidates satisfied P1 constraints.")
        return

    candidates.sort(key=lambda item: item.max_elongation)
    selected = candidates[: args.top_k]

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for rank, cand in enumerate(selected, start=1):
        payload = {
            "r_cos": cand.r_cos,
            "z_sin": cand.z_sin,
            "n_field_periods": cand.n_field_periods,
            "n_periodicity": 1,
            "is_stellarator_symmetric": True,
        }
        fname = f"p1_low_fidelity_{rank:03}.json"
        path = output_dir / fname
        path.write_text(json.dumps(payload, indent=2))
        manifest.append(
            {
                "rank": rank,
                "max_elongation": cand.max_elongation,
                "aspect_ratio": cand.aspect_ratio,
                "average_triangularity": cand.average_triangularity,
                "edge_rotational_transform": cand.edge_rotational_transform,
                "n_field_periods": cand.n_field_periods,
                "file": str(path),
            }
        )

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {len(selected)} seeds to {output_dir}.")


if __name__ == "__main__":
    main()
