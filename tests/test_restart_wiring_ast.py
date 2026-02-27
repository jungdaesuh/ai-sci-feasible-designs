from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P1_SCRIPT = ROOT / "scripts" / "p1_alm_ngopt_multifidelity.py"
P2_SCRIPT = ROOT / "experiments" / "p1_p2" / "p2_alm_ngopt_multifidelity.py"


class ScriptAst:
    def __init__(self, path: Path):
        self.path = path
        self.source = path.read_text(encoding="utf-8")
        self.tree = ast.parse(self.source, filename=str(path))

    def calls(self, name: str) -> list[ast.Call]:
        return [
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call) and _call_name(node) == name
        ]

    def function(self, name: str) -> ast.FunctionDef:
        for node in self.tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return node
        raise AssertionError(f"missing function: {name}")


def _call_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _argument_flags(tree: ast.AST) -> set[str]:
    flags: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _call_name(node) != "add_argument":
            continue
        if not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            flags.add(first.value)
    return flags


def _dict_keys(tree: ast.AST) -> set[str]:
    keys: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key in node.keys:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                keys.add(key.value)
    return keys


def _keywords(call: ast.Call) -> set[str]:
    return {kw.arg for kw in call.keywords if kw.arg is not None}


def _find_adaptive_if_nodes(tree: ast.AST) -> list[ast.If]:
    adaptive_ifs: list[ast.If] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        try:
            text = ast.unparse(node.test)
        except Exception:
            continue
        if "args.adaptive_restart" in text:
            adaptive_ifs.append(node)
    return adaptive_ifs


def _block_calls(if_node: ast.If) -> set[str]:
    calls: set[str] = set()
    for node in ast.walk(if_node):
        if not isinstance(node, ast.Call):
            continue
        call_name = _call_name(node)
        if call_name is not None:
            calls.add(call_name)
    return calls


def _assert_restart_plumbing(
    script: ScriptAst,
    *,
    expected_problem_literal: str,
    expected_state_objective_expr_contains: str | None = None,
) -> None:
    flags = _argument_flags(script.tree)
    assert "--adaptive-restart" in flags
    assert "--restart-feasibility-weight" in flags
    assert "--restart-objective-weight" in flags
    assert "--restart-diversity-weight" in flags
    assert "--restart-saturation-penalty" in flags
    assert "--restart-novelty-min-distance" in flags
    assert "--restart-novelty-feasibility-max" in flags
    assert "--restart-novelty-near-duplicate-distance" in flags
    assert "--restart-novelty-judge-mode" in flags

    select_calls = script.calls("select_adaptive_restart_runtime")
    assert len(select_calls) == 1
    select_call = select_calls[0]
    select_keywords = _keywords(select_call)
    assert {
        "problem",
        "state_x",
        "state_objective",
        "state_feasibility",
        "best_violation_x",
        "best_violation_objective",
        "best_violation_feasibility",
        "best_low_x",
        "best_low_objective",
        "best_low_feasibility",
        "best_high_x",
        "best_high_objective",
        "best_high_feasibility",
        "selection_counts",
        "feasibility_weight",
        "objective_weight",
        "diversity_weight",
        "saturation_penalty",
        "novelty_min_distance",
        "novelty_feasibility_max",
        "novelty_near_duplicate_distance",
        "novelty_judge_mode",
    }.issubset(select_keywords)

    problem_kw = next(kw for kw in select_call.keywords if kw.arg == "problem")
    assert isinstance(problem_kw.value, ast.Constant)
    assert problem_kw.value.value == expected_problem_literal
    if expected_state_objective_expr_contains is not None:
        state_objective_kw = next(
            kw for kw in select_call.keywords if kw.arg == "state_objective"
        )
        assert expected_state_objective_expr_contains in ast.unparse(
            state_objective_kw.value
        )

    append_calls = script.calls("append_restart_history")
    assert len(append_calls) == 1
    append_keywords = _keywords(append_calls[0])
    assert {
        "outer",
        "selected_seed",
        "selected_seed_identity",
        "counts",
        "decision",
    }.issubset(append_keywords)

    assert "restart_seed" in _dict_keys(script.tree)

    adaptive_if_nodes = _find_adaptive_if_nodes(script.tree)
    assert adaptive_if_nodes
    for if_node in adaptive_if_nodes:
        block_names = _block_calls(if_node)
        if {
            "select_adaptive_restart_runtime",
            "append_restart_history",
        }.issubset(block_names):
            break
    else:
        raise AssertionError(
            "adaptive restart guard does not contain selector + history calls"
        )


def test_p1_adaptive_restart_ast_wiring() -> None:
    _assert_restart_plumbing(
        ScriptAst(P1_SCRIPT),
        expected_problem_literal="p1",
        expected_state_objective_expr_contains="float(state.objective)",
    )


def test_p2_adaptive_restart_ast_wiring() -> None:
    _assert_restart_plumbing(
        ScriptAst(P2_SCRIPT),
        expected_problem_literal="p2",
        expected_state_objective_expr_contains="-float(state.objective)",
    )


def test_p2_restart_lgradb_defaults_to_negative_infinity() -> None:
    script = ScriptAst(P2_SCRIPT)
    fn = script.function("_restart_lgradb")
    fn_source = ast.unparse(fn)
    assert "record.get('lgradb', float('-inf'))" in fn_source
    assert "_restart_lgradb(" in script.source


def test_p2_telemetry_lgradb_uses_null_for_missing_or_non_finite() -> None:
    script = ScriptAst(P2_SCRIPT)
    fn = script.function("_telemetry_lgradb")
    fn_source = ast.unparse(fn)
    assert "if 'lgradb' not in record" in fn_source
    assert "return None" in fn_source
    assert "_telemetry_lgradb(record)" in script.source
