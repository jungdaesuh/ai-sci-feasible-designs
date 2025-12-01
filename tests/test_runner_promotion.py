from typing import Any, Mapping

from ai_scientist import config as ai_config
from ai_scientist import cycle_executor
from ai_scientist.fidelity_controller import FidelityController


def _make_controller() -> FidelityController:
    cfg = ai_config.load_experiment_config()
    return FidelityController(cfg)


def _entry(
    design_hash: str, gradient: float, aspect: float, feasibility: float
) -> dict:
    metrics = {
        "minimum_normalized_magnetic_gradient_scale_length": gradient,
        "aspect_ratio": aspect,
    }
    return {
        "design_hash": design_hash,
        "params": {"r_cos": [[1.0]], "z_sin": [[0.0]]},
        "seed": int(abs(hash(design_hash))) % 1000,
        "evaluation": {
            "metrics": metrics,
            "feasibility": feasibility,
            "objective": aspect,
            "minimize_objective": True,
            "stage": "screen",
        },
    }


def test_rank_candidates_prioritizes_nondominated_designs():
    entries = {
        e["design_hash"]: e
        for e in [
            _entry("hash-a", gradient=1.4, aspect=2.0, feasibility=0.0),
            _entry(
                "hash-b",
                gradient=1.2,
                aspect=2.5,
                feasibility=cycle_executor.FEASIBILITY_CUTOFF * 2,
            ),
            _entry("hash-c", gradient=1.6, aspect=1.8, feasibility=0.0),
        ]
    }
    controller = _make_controller()
    ranked = controller.get_promotion_candidates(
        entries, promote_limit=2, reference_point=cycle_executor.P3_REFERENCE_POINT
    )
    assert [entry["design_hash"] for entry in ranked] == ["hash-c", "hash-a"]


def test_rank_candidates_fills_from_infeasible_when_needed():
    entries = {
        e["design_hash"]: e
        for e in [
            _entry(
                "hash-a",
                gradient=1.4,
                aspect=2.0,
                feasibility=cycle_executor.FEASIBILITY_CUTOFF * 4,
            ),
            _entry(
                "hash-b",
                gradient=1.2,
                aspect=2.5,
                feasibility=cycle_executor.FEASIBILITY_CUTOFF * 3,
            ),
        ]
    }
    controller = _make_controller()
    ranked = controller.get_promotion_candidates(
        entries, promote_limit=2, reference_point=cycle_executor.P3_REFERENCE_POINT
    )
    assert len(ranked) == 2
    assert {entry["design_hash"] for entry in ranked} == {"hash-a", "hash-b"}


def test_verify_best_claim_replays_without_cache():
    class _WorldModelStub:
        def __init__(self) -> None:
            self.statements: list[dict[str, Any]] = []

        def log_statement(
            self,
            experiment_id: int,
            cycle: int,
            stage: str,
            text: str,
            status: str,
            tool_name: str,
            tool_input: Mapping[str, Any],
            *,
            metrics_id: int | None = None,
            seed: int | None = None,
            git_sha: str,
            repro_cmd: str,
            created_at: str | None = None,
            commit: bool = True,
        ) -> int:
            self.statements.append(
                {
                    "cycle": cycle,
                    "status": status,
                    "tool_name": tool_name,
                    "tool_input_hash": "stub",
                }
            )
            return len(self.statements)

    class _EvalStub:
        def __init__(self) -> None:
            self.calls: list[bool] = []

        def __call__(
            self,
            boundary_params: Mapping[str, Any],
            *,
            stage: str,
            use_cache: bool = True,
        ) -> dict[str, Any]:
            self.calls.append(use_cache)
            return {
                "objective": 1.0,
                "feasibility": 0.0,
                "metrics": {},
            }

    world = _WorldModelStub()
    evaluator = _EvalStub()
    best_entry = {"params": {"r_cos": [[1.0]], "z_sin": [[0.0]]}, "design_hash": "abcd"}
    best_eval = {"objective": 1.0, "feasibility": 0.0}

    status = cycle_executor._verify_best_claim(
        world_model=world,
        experiment_id=1,
        cycle_number=1,
        best_entry=best_entry,
        best_eval=best_eval,
        evaluation_fn=evaluator,
        tool_name="evaluate_p1",
        best_seed=7,
        git_sha="deadbeef",
        reproduction_command="python -m ai_scientist.runner",
        stage="screen",
        metrics_id=42,
    )

    assert status == "SUPPORTED"
    assert evaluator.calls == [False]
    assert world.statements and world.statements[0]["status"] == "SUPPORTED"
