#!/usr/bin/env python
"""Score candidate boundary JSONs with the official constellaration problems.

This is the high-fidelity validation step: it runs the reference evaluator(s),
which call VMEC++ (and Boozer/QI where needed).
"""

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


def _extract_boundary(data: object) -> dict:
    if not isinstance(data, dict):
        raise TypeError("Boundary JSON must be an object.")
    if "r_cos" not in data or "z_sin" not in data:
        raise KeyError("Boundary JSON missing required keys: r_cos, z_sin.")
    boundary: dict = {
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


def _load_json(path: Path) -> object:
    payload = json.loads(path.read_text())
    if isinstance(payload, str):
        return json.loads(payload)
    return payload


def _score_p1_p2(problem: object, paths: list[Path]) -> list[dict]:
    results: list[dict] = []
    for path in paths:
        data = _load_json(path)
        boundary = _extract_boundary(data)
        surface = surface_rz_fourier.SurfaceRZFourier.model_validate(boundary)
        evaluation = problem.evaluate(surface)
        results.append(
            {
                "path": str(path),
                "objective": float(evaluation.objective),
                "feasibility": float(evaluation.feasibility),
                "score": float(evaluation.score),
            }
        )
    results.sort(key=lambda item: (-item["score"], item["objective"]))
    return results


def _score_p3(problem: object, submission_path: Path) -> dict:
    raw = _load_json(submission_path)
    if not isinstance(raw, list):
        raise TypeError(
            "P3 expects a JSON list of boundaries (or JSON-encoded strings)."
        )

    surfaces: list[surface_rz_fourier.SurfaceRZFourier] = []
    for item in raw:
        item_obj = json.loads(item) if isinstance(item, str) else item
        boundary = _extract_boundary(item_obj)
        surfaces.append(surface_rz_fourier.SurfaceRZFourier.model_validate(boundary))

    evaluation = problem.evaluate(surfaces)
    return {
        "path": str(submission_path),
        "score": float(evaluation.score),
        "feasibility": [float(v) for v in evaluation.feasibility],
        "objectives": [
            [(float(val), bool(minimize)) for val, minimize in obj]
            for obj in evaluation.objectives
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score candidates with constellaration."
    )
    parser.add_argument(
        "--problem",
        required=True,
        choices=["p1", "p2", "p3"],
        help="Problem type to evaluate.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="For p1/p2: JSON file(s) or directories. For p3: exactly one JSON file.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    if args.problem == "p1":
        problem = problems.GeometricalProblem()
    elif args.problem == "p2":
        problem = problems.SimpleToBuildQIStellarator()
    else:
        problem = problems.MHDStableQIStellarator()

    if args.problem in {"p1", "p2"}:
        paths = _iter_paths(args.inputs)
        results = _score_p1_p2(problem, paths)
        print(f"evaluated={len(results)}")
        for record in results[: args.top_k]:
            print(
                f"score={record['score']:.6f} "
                f"feas={record['feasibility']:.6f} "
                f"obj={record['objective']:.6f} "
                f"path={record['path']}"
            )
        if args.output is not None:
            args.output.write_text(json.dumps(results, indent=2))
        return

    if len(args.inputs) != 1:
        raise SystemExit("P3 expects exactly one input file (a list of boundaries).")
    result = _score_p3(problem, args.inputs[0])
    print(f"score={result['score']:.6f} path={result['path']}")
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
