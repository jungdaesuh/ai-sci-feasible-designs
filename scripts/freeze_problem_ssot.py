#!/usr/bin/env python
"""Freeze/validate ConStellaration problem SSOT constraints and objectives."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PROBLEMS_PATH = _REPO_ROOT / "constellaration" / "src" / "constellaration" / "problems.py"
_DEFAULT_SPEC_PATH = _REPO_ROOT / "constraints" / "constellaration_problem_ssot.json"
_SOURCE_PATH = "constellaration/src/constellaration/problems.py"


def _constraint(metric: str, operator: str, value: float) -> dict[str, Any]:
    return {"metric": metric, "operator": operator, "value": value}


def _load_module_ast(path: Path) -> ast.Module:
    source = path.read_text(encoding="utf-8")
    return ast.parse(source, filename=str(path))


def _find_class(module: ast.Module, class_name: str) -> ast.ClassDef:
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise ValueError(f"Missing class in {_PROBLEMS_PATH}: {class_name}")


def _numeric_literal(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        operand = node.operand
        if isinstance(operand, ast.Constant) and isinstance(operand.value, (int, float)):
            return float(-operand.value)
    raise ValueError(f"Expected numeric literal, got {ast.dump(node, include_attributes=False)}")


def _class_numeric_attr(class_def: ast.ClassDef, attr_name: str) -> float:
    for node in class_def.body:
        if isinstance(node, ast.AnnAssign):
            target = node.target
            if isinstance(target, ast.Name) and target.id == attr_name:
                value = node.value
                if value is None:
                    raise ValueError(f"Missing value for {class_def.name}.{attr_name}")
                return _numeric_literal(value)
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1:
                continue
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == attr_name:
                return _numeric_literal(node.value)
    raise ValueError(f"Missing class attribute {class_def.name}.{attr_name}")


def _find_method(class_def: ast.ClassDef, method_name: str) -> ast.FunctionDef:
    for node in class_def.body:
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return node
    raise ValueError(f"Missing method {class_def.name}.{method_name}")


def _parse_objective_tuple(
    objective_node: ast.Tuple, *, class_name: str, method_name: str
) -> tuple[str, str]:
    if len(objective_node.elts) != 2:
        raise ValueError(f"Unexpected objective tuple shape in {class_name}.{method_name}")

    metric_node = objective_node.elts[0]
    direction_node = objective_node.elts[1]

    if not (
        isinstance(metric_node, ast.Attribute)
        and isinstance(metric_node.value, ast.Name)
        and metric_node.value.id == "metrics"
    ):
        raise ValueError(f"Unexpected objective metric expression in {class_name}.{method_name}")
    if not (isinstance(direction_node, ast.Constant) and isinstance(direction_node.value, bool)):
        raise ValueError(f"Unexpected objective direction in {class_name}.{method_name}")

    direction = "minimize" if direction_node.value else "maximize"
    return metric_node.attr, direction


def _parse_single_objective(class_def: ast.ClassDef) -> tuple[str, str]:
    method_name = "get_objective"
    method = _find_method(class_def, method_name)
    for stmt in method.body:
        if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Tuple):
            return _parse_objective_tuple(
                stmt.value, class_name=class_def.name, method_name=method_name
            )
    raise ValueError(f"Missing return tuple in {class_def.name}.{method_name}")


def _parse_multi_objectives(class_def: ast.ClassDef) -> list[dict[str, str]]:
    method_name = "get_objectives"
    method = _find_method(class_def, method_name)
    for stmt in method.body:
        if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.List):
            objectives: list[dict[str, str]] = []
            for item in stmt.value.elts:
                if not isinstance(item, ast.Tuple) or len(item.elts) != 2:
                    raise ValueError(f"Unexpected objective tuple in {class_def.name}.{method_name}")
                metric, direction = _parse_objective_tuple(
                    item, class_name=class_def.name, method_name=method_name
                )
                objectives.append({"metric": metric, "direction": direction})
            return objectives
    raise ValueError(f"Missing objective list in {class_def.name}.{method_name}")


def _constraints_from_attrs(
    class_def: ast.ClassDef, definitions: list[tuple[str, str, str]]
) -> list[dict[str, Any]]:
    return [
        _constraint(metric, operator, _class_numeric_attr(class_def, attr_name))
        for metric, operator, attr_name in definitions
    ]


def _build_current_spec() -> dict[str, Any]:
    module = _load_module_ast(_PROBLEMS_PATH)
    p1 = _find_class(module, "GeometricalProblem")
    p2 = _find_class(module, "SimpleToBuildQIStellarator")
    p3 = _find_class(module, "MHDStableQIStellarator")

    p1_metric, p1_direction = _parse_single_objective(p1)
    p2_metric, p2_direction = _parse_single_objective(p2)
    p3_objectives = _parse_multi_objectives(p3)

    p1_constraints = _constraints_from_attrs(
        p1,
        [
            ("aspect_ratio", "<=", "_aspect_ratio_upper_bound"),
            ("average_triangularity", "<=", "_average_triangularity_upper_bound"),
            (
                "edge_rotational_transform_over_n_field_periods",
                ">=",
                "_edge_rotational_transform_over_n_field_periods_lower_bound",
            ),
        ],
    )
    p2_constraints = _constraints_from_attrs(
        p2,
        [
            ("aspect_ratio", "<=", "_aspect_ratio_upper_bound"),
            (
                "edge_rotational_transform_over_n_field_periods",
                ">=",
                "_edge_rotational_transform_over_n_field_periods_lower_bound",
            ),
            ("log10_qi", "<=", "_log10_qi_upper_bound"),
            ("edge_magnetic_mirror_ratio", "<=", "_edge_magnetic_mirror_ratio_upper_bound"),
            ("max_elongation", "<=", "_max_elongation_upper_bound"),
        ],
    )
    p3_constraints = _constraints_from_attrs(
        p3,
        [
            (
                "edge_rotational_transform_over_n_field_periods",
                ">=",
                "_edge_rotational_transform_over_n_field_periods_lower_bound",
            ),
            ("log10_qi", "<=", "_log10_qi_upper_bound"),
            ("edge_magnetic_mirror_ratio", "<=", "_edge_magnetic_mirror_ratio_upper_bound"),
            (
                "flux_compression_in_regions_of_bad_curvature",
                "<=",
                "_flux_compression_in_regions_of_bad_curvature_upper_bound",
            ),
            ("vacuum_well", ">=", "_vacuum_well_lower_bound"),
        ],
    )

    return {
        "version": "2026-02-26",
        "source": _SOURCE_PATH,
        "problems": {
            "p1_geometrical": {
                "type": "single_objective",
                "objective": {
                    "metric": p1_metric,
                    "direction": p1_direction,
                },
                "constraints": p1_constraints,
            },
            "p2_simple_to_build_qi": {
                "type": "single_objective",
                "objective": {
                    "metric": p2_metric,
                    "direction": p2_direction,
                },
                "constraints": p2_constraints,
            },
            "p3_mhd_stable_qi": {
                "type": "multi_objective",
                "objectives": p3_objectives,
                "constraints": p3_constraints,
            },
        },
    }


def _load_spec(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"Missing SSOT spec file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in SSOT spec: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"SSOT spec must be a JSON object: {path}")
    return payload


def _write_spec(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def _validate_spec(path: Path) -> None:
    current = _build_current_spec()
    frozen = _load_spec(path)
    if frozen != current:
        raise ValueError(
            "Frozen SSOT spec is out of sync with current problems.py. "
            f"Run with --write to refresh: {path}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze/validate ConStellaration constraints/objectives SSOT."
    )
    parser.add_argument(
        "--spec-path",
        type=Path,
        default=_DEFAULT_SPEC_PATH,
        help=f"Path to SSOT JSON file (default: {_DEFAULT_SPEC_PATH})",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write current constraints/objectives into the SSOT file.",
    )
    args = parser.parse_args()

    if args.write:
        _write_spec(args.spec_path, _build_current_spec())
        print(f"Wrote SSOT spec: {args.spec_path}")
        return 0

    _validate_spec(args.spec_path)
    print(f"SSOT spec is up to date: {args.spec_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
