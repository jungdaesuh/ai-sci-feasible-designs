#!/usr/bin/env python
"""Score P1 candidate boundaries using the official geometrical evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from constellaration import problems
from constellaration.geometry import surface_rz_fourier


def _iter_paths(inputs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        if item.is_dir():
            paths.extend(sorted(item.glob("*.json")))
        else:
            paths.append(item)
    return paths


def _extract_boundary(data: dict) -> dict | None:
    if "r_cos" not in data or "z_sin" not in data:
        return None
    boundary = {
        "r_cos": data["r_cos"],
        "z_sin": data["z_sin"],
        "n_field_periods": data.get("n_field_periods", 3),
        "is_stellarator_symmetric": data.get("is_stellarator_symmetric", True),
    }
    if data.get("r_sin") is not None:
        boundary["r_sin"] = data["r_sin"]
    if data.get("z_cos") is not None:
        boundary["z_cos"] = data["z_cos"]
    return boundary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score P1 candidates with constellaration GeometricalProblem."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Boundary JSON files or directories containing JSON candidates.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    problem = problems.GeometricalProblem()
    results: list[dict] = []
    skipped = 0

    for path in _iter_paths(args.inputs):
        data = json.loads(path.read_text())
        boundary = _extract_boundary(data)
        if boundary is None:
            skipped += 1
            continue
        surface = surface_rz_fourier.SurfaceRZFourier.model_validate(boundary)
        evaluation = problem.evaluate(surface)
        record = {
            "path": str(path),
            "objective": float(evaluation.objective),
            "feasibility": float(evaluation.feasibility),
            "score": float(evaluation.score),
        }
        results.append(record)

    results.sort(key=lambda item: (-item["score"], item["objective"]))

    print(f"evaluated={len(results)} skipped={skipped}")
    for record in results[: args.top_k]:
        print(
            f"score={record['score']:.6f} "
            f"feas={record['feasibility']:.6f} "
            f"obj={record['objective']:.6f} "
            f"path={record['path']}"
        )

    if args.output is not None:
        args.output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
