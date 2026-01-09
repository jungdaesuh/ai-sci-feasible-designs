#!/usr/bin/env python
"""Find low-elongation feasible P1 designs from the HF dataset.

This script scans the proxima-fusion/constellaration dataset (streaming by default),
filters by P1 feasibility constraints, then writes the lowest-elongation boundaries
as JSON files plus a manifest.
"""

from __future__ import annotations

import argparse
import heapq
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from datasets import load_dataset


@dataclass(frozen=True)
class Candidate:
    idx: int
    max_elongation: float
    aspect_ratio: float
    average_triangularity: float
    edge_rotational_transform: float
    boundary: dict


def _get_metric(example: dict, key: str) -> float | None:
    val = example.get(f"metrics.{key}")
    if val is not None:
        return float(val)
    metrics = example.get("metrics")
    if isinstance(metrics, dict):
        inner = metrics.get(key)
        return float(inner) if inner is not None else None
    return None


def _iter_dataset(split: str, streaming: bool) -> Iterable[dict]:
    return load_dataset(
        "proxima-fusion/constellaration", split=split, streaming=streaming
    )


def _boundary_from_example(example: dict) -> dict:
    return {
        "r_cos": example["boundary.r_cos"],
        "z_sin": example["boundary.z_sin"],
        "n_field_periods": int(example["boundary.n_field_periods"]),
        "n_periodicity": int(example.get("boundary.n_periodicity", 1)),
        "is_stellarator_symmetric": bool(example["boundary.is_stellarator_symmetric"]),
        "r_sin": example.get("boundary.r_sin"),
        "z_cos": example.get("boundary.z_cos"),
    }


def _passes_constraints(
    ar: float,
    tri: float,
    iota: float,
    *,
    ar_max: float,
    tri_max: float,
    iota_min: float,
) -> bool:
    return ar <= ar_max and tri <= tri_max and iota >= iota_min


def _maybe_write_boundary(path: Path, boundary: dict) -> None:
    payload = {
        "r_cos": boundary["r_cos"],
        "z_sin": boundary["z_sin"],
        "n_field_periods": boundary["n_field_periods"],
        "n_periodicity": boundary.get("n_periodicity", 1),
        "is_stellarator_symmetric": boundary["is_stellarator_symmetric"],
    }
    if boundary.get("r_sin") is not None:
        payload["r_sin"] = boundary["r_sin"]
    if boundary.get("z_cos") is not None:
        payload["z_cos"] = boundary["z_cos"]
    path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract low-elongation feasible P1 designs from the HF dataset."
    )
    parser.add_argument("--split", default="train", help="HF split (train/test)")
    parser.add_argument(
        "--top-k", type=int, default=20, help="Number of designs to keep"
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=200_000,
        help="Max examples to scan (streaming).",
    )
    parser.add_argument(
        "--nfp",
        type=int,
        default=3,
        help="Filter by n_field_periods (set to 0 to disable)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/p1/low_feasible"),
        help="Output directory for boundary JSON files",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Disable streaming (loads full dataset in memory).",
    )
    parser.add_argument("--ar-max", type=float, default=4.0)
    parser.add_argument("--tri-max", type=float, default=-0.5)
    parser.add_argument("--iota-min", type=float, default=0.3)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    streaming = not args.no_streaming
    dataset = _iter_dataset(args.split, streaming)

    heap: list[tuple[float, int, Candidate]] = []  # (-max_elongation, idx, candidate)

    for idx, example in enumerate(dataset):
        if args.max_examples and idx >= args.max_examples:
            break

        if args.nfp:
            raw_nfp = example.get("boundary.n_field_periods")
            if raw_nfp is None:
                continue
            nfp_val = int(raw_nfp)
            if nfp_val != args.nfp:
                continue

        ar = _get_metric(example, "aspect_ratio")
        tri = _get_metric(example, "average_triangularity")
        iota = _get_metric(example, "edge_rotational_transform_over_n_field_periods")
        elong = _get_metric(example, "max_elongation")

        if ar is None or tri is None or iota is None or elong is None:
            continue

        if not _passes_constraints(
            ar,
            tri,
            iota,
            ar_max=args.ar_max,
            tri_max=args.tri_max,
            iota_min=args.iota_min,
        ):
            continue

        boundary = _boundary_from_example(example)
        candidate = Candidate(
            idx=idx,
            max_elongation=float(elong),
            aspect_ratio=float(ar),
            average_triangularity=float(tri),
            edge_rotational_transform=float(iota),
            boundary=boundary,
        )

        score = -candidate.max_elongation
        if len(heap) < args.top_k:
            heapq.heappush(heap, (score, idx, candidate))
        else:
            worst = heap[0]
            if score > worst[0]:
                heapq.heapreplace(heap, (score, idx, candidate))

    results = [item[2] for item in heap]
    results.sort(key=lambda item: item.max_elongation)

    manifest = []
    for rank, cand in enumerate(results, start=1):
        fname = f"p1_low_feasible_{rank:03}.json"
        path = output_dir / fname
        _maybe_write_boundary(path, cand.boundary)
        manifest.append(
            {
                "rank": rank,
                "dataset_index": cand.idx,
                "max_elongation": cand.max_elongation,
                "aspect_ratio": cand.aspect_ratio,
                "average_triangularity": cand.average_triangularity,
                "edge_rotational_transform_over_n_field_periods": cand.edge_rotational_transform,
                "file": str(path),
                "n_field_periods": cand.boundary["n_field_periods"],
            }
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"Wrote {len(results)} designs to {output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
