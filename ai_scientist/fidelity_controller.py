"""Fidelity management and stage transition logic."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Protocol, Tuple

from ai_scientist import adapter, memory, tools
from ai_scientist import config as ai_config
from ai_scientist.forward_model import forward_model_batch

FEASIBILITY_CUTOFF = getattr(tools, "_DEFAULT_RELATIVE_TOLERANCE", 1e-2)
P3_REFERENCE_POINT = getattr(tools, "_P3_REFERENCE_POINT", (1.0, 20.0))


class ProblemEvaluator(Protocol):
    def __call__(
        self,
        boundary_params: Mapping[str, Any],
        *,
        stage: str,
        use_cache: bool = True,
    ) -> dict[str, Any]: ...


@dataclass
class CycleSummary:
    cycle: int
    objective: float | None
    feasibility: float | None
    hv: float | None
    stage: str


def _time_exceeded(start: float, limit_minutes: float) -> bool:
    elapsed = time.perf_counter() - start
    return elapsed >= limit_minutes * 60


def _extract_objectives(
    entry: Mapping[str, Any], problem_type: str = "p3"
) -> tuple[float, float]:
    metrics = entry["evaluation"]["metrics"]

    if problem_type == "p1":
        # P1: minimize max_elongation.
        # We treat "gradient" slot as -max_elongation (higher is better)
        elong = float(metrics.get("max_elongation", 0.0))
        return -elong, 0.0

    # P3/P2 Check
    if "minimum_normalized_magnetic_gradient_scale_length" in metrics:
        gradient = float(metrics["minimum_normalized_magnetic_gradient_scale_length"])
        aspect = float(metrics.get("aspect_ratio", 0.0))
        return gradient, aspect

    # Fallback if metrics missing for non-P1 (or legacy behavior)
    elong = float(metrics.get("max_elongation", 0.0))
    return -elong, 0.0


def _objective_proxy(entry: Mapping[str, Any], problem_type: str = "p3") -> float:
    gradient, aspect = _extract_objectives(entry, problem_type=problem_type)
    return gradient - aspect


def _crowding_distance(
    entries_by_design: Mapping[str, Mapping[str, Any]],
    problem_type: str = "p3",
) -> dict[str, float]:
    if not entries_by_design:
        return {}
    values: list[tuple[str, float, float]] = []
    for design_hash, entry in entries_by_design.items():
        gradient, aspect = _extract_objectives(entry, problem_type=problem_type)
        values.append((design_hash, float(gradient), float(aspect)))
    distances = {design_hash: 0.0 for design_hash in entries_by_design}
    if len(values) <= 2:
        for design_hash in distances:
            distances[design_hash] = float("inf")
        return distances
    sorted_grad = sorted(values, key=lambda item: item[1], reverse=True)
    grad_values = [item[1] for item in sorted_grad]
    grad_span = max(max(grad_values) - min(grad_values), 1e-9)
    distances[sorted_grad[0][0]] = float("inf")
    distances[sorted_grad[-1][0]] = float("inf")
    for pos in range(1, len(sorted_grad) - 1):
        prev_val = sorted_grad[pos - 1][1]
        next_val = sorted_grad[pos + 1][1]
        distances[sorted_grad[pos][0]] += abs(next_val - prev_val) / grad_span

    sorted_aspect = sorted(values, key=lambda item: item[2], reverse=False)
    aspect_values = [item[2] for item in sorted_aspect]
    aspect_span = max(max(aspect_values) - min(aspect_values), 1e-9)
    distances[sorted_aspect[0][0]] = float("inf")
    distances[sorted_aspect[-1][0]] = float("inf")
    for pos in range(1, len(sorted_aspect) - 1):
        prev_val = sorted_aspect[pos - 1][2]
        next_val = sorted_aspect[pos + 1][2]
        distances[sorted_aspect[pos][0]] += abs(next_val - prev_val) / aspect_span
    return distances


def _relative_objective_improvement(
    history: list[CycleSummary], lookback: int
) -> float:
    if len(history) <= lookback:
        return 0.0
    earlier = history[-lookback - 1].objective
    latest = history[-1].objective
    if earlier is None or latest is None:
        return 0.0
    diff = earlier - latest
    denom = abs(earlier) if abs(earlier) > 1e-6 else 1.0
    return float(diff / denom)


def _feasibility_value(entry: Mapping[str, Any]) -> float:
    return float(entry["evaluation"].get("feasibility", float("inf")))


class FidelityController:
    """Manages candidate evaluation, fidelity ladders, and promotion gates."""

    def __init__(self, config: ai_config.ExperimentConfig):
        self.config = config

    def evaluate_stage(
        self,
        candidates: Iterable[Mapping[str, Any]],
        stage: str,
        budgets: ai_config.BudgetConfig,
        cycle_start: float,
        evaluate_fn: ProblemEvaluator
        | None = None,  # Deprecated/Unused if using forward_model
        *,
        sleep_per_eval: float = 0.0,
        tool_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """Evaluate candidates at a given fidelity, respecting wall-clock budget."""

        # Convert iterable to list
        candidate_list = list(candidates)
        if not candidate_list:
            return []

        # Check wall clock before starting batch
        wall_limit = budgets.wall_clock_minutes
        if _time_exceeded(cycle_start, wall_limit):
            return []

        # Prepare settings
        # We use tools._settings_for_stage to get the correct config
        # This assumes tools is available and has this helper.
        # If not, we should replicate logic or import from where it lives.
        # For now, we rely on tools.
        fm_settings = tools._settings_for_stage(stage, self.config.problem)

        # Prepare PEFT hook
        if tool_name:
            adapter.prepare_peft_hook(tool_name, stage)

        # Extract boundaries
        boundaries = [c["params"] for c in candidate_list]

        # Run Batch Evaluation
        # We use the budgets.n_workers for parallelism
        batch_results = forward_model_batch(
            boundaries,
            fm_settings,
            n_workers=budgets.n_workers,
            pool_type=budgets.pool_type,
            use_cache=True,
        )

        # Apply PEFT updates
        if tool_name:
            adapter.apply_lora_updates(tool_name, stage)

        # Process Results
        results: list[dict[str, Any]] = []

        for i, result in enumerate(batch_results):
            candidate = candidate_list[i]
            design_id = candidate.get("design_hash") or result.design_hash

            # Convert EvaluationResult to dict format expected by runner
            # We need to reconstruct the dict structure:
            # {
            #     "stage": stage,
            #     "objective": ...,
            #     "feasibility": ...,
            #     "metrics": ...,
            #     "constraint_margins": ...,
            #     ...
            # }

            # Helper to calculate score (gradient - aspect for P3/P2)
            # This logic was in tools.evaluate_p3 etc.
            # We can reuse _objective_proxy or similar, but we need 'metrics' dict first.

            metrics_dict = result.metrics.model_dump()

            # Determine score/hv based on problem
            # This duplicates logic from tools.evaluate_p*, but that's inevitable if we decouple.
            # Or we can use the result.objective directly if it matches?
            # forward_model.compute_objective returns:
            # P1: max_elongation
            # P2: min_grad_scale
            # P3: aspect_ratio

            # But runner expects:
            # P1: score = 1.0 - normalized(max_elongation)
            # P2: score = gradient / aspect
            # P3: score = gradient / aspect

            # And 'objective' field in runner dict:
            # P1: result.objective (minimize=True)
            # P2: result.objective (minimize=False)
            # P3: result.objective (minimize=True)

            # Let's map carefully.

            eval_dict = {
                "stage": stage,
                "objective": result.objective,
                "feasibility": result.feasibility,
                "is_feasible": result.is_feasible,
                "metrics": metrics_dict,
                "constraint_margins": result.constraints_map,  # Use map for compatibility
                "max_violation": result.feasibility,
                "cache_hit": result.cache_hit,
                "vmec_status": "ok" if not result.error_message else "exception",
                "settings": (
                    result.settings.constellaration_settings.model_dump()
                    if result.settings.constellaration_settings is not None
                    else {}
                ),
            }

            # Add derived fields
            if self.config.problem == "p1":
                eval_dict["minimize_objective"] = True
                # Score logic from tools.evaluate_p1
                # score = 1.0 - normalized(max_elongation)
                # We can re-implement or just leave it 0.0 if not used for ranking here?
                # FidelityController uses _objective_proxy for ranking P3/P2.
                # For P1 it uses _extract_objectives.
                pass
            elif self.config.problem in ["p2", "p3"]:
                eval_dict["minimize_objective"] = self.config.problem == "p3"
                # Use canonical ranking score for consistency across codebase
                # See ai_scientist.objective_types for semantics documentation
                from ai_scientist.objective_types import compute_ranking_score

                score = compute_ranking_score(metrics_dict, self.config.problem)
                eval_dict["score"] = score

                if self.config.problem == "p2":
                    grad = float(
                        metrics_dict.get(
                            "minimum_normalized_magnetic_gradient_scale_length", 0.0
                        )
                    )
                    # H2 FIX: Renamed from "hv" to "gradient_proxy" - this is NOT hypervolume
                    # This is a per-candidate proxy score = max(0, gradient - 1)
                    # True hypervolume is computed at cycle level from the Pareto front
                    eval_dict["gradient_proxy"] = max(0.0, grad - 1.0)
                else:  # p3
                    grad = float(
                        metrics_dict.get(
                            "minimum_normalized_magnetic_gradient_scale_length", 0.0
                        )
                    )
                    # H2 FIX: See comment above
                    eval_dict["gradient_proxy"] = max(0.0, grad - 1.0)

            if result.error_message:
                eval_dict["error"] = result.error_message
                # Penalize if error?
                eval_dict["feasibility"] = float("inf")
                eval_dict["max_violation"] = float("inf")
                eval_dict["objective"] = (
                    1e9 if eval_dict.get("minimize_objective", True) else -1e9
                )

            results.append(
                {
                    "params": candidate["params"],
                    "evaluation": eval_dict,
                    "seed": int(candidate["seed"]),
                    "design_hash": design_id,
                }
            )

            if sleep_per_eval > 0:
                time.sleep(sleep_per_eval)

        return results

    def get_promotion_candidates(
        self,
        entries_by_design: Mapping[str, Mapping[str, Any]],
        promote_limit: int,
        reference_point: Tuple[float, float] = P3_REFERENCE_POINT,
    ) -> list[Mapping[str, Any]]:
        """Select top candidates for promotion based on feasibility and objective."""
        if not entries_by_design:
            return []

        summary = tools.summarize_p3_candidates(
            list(entries_by_design.values()), reference_point=reference_point
        )
        # Phase 5 HV/Objective: rank promotions by feasibility first, then objective proxy.
        ordered_hashes = [entry.design_hash for entry in summary.pareto_entries]
        ranked: list[Mapping[str, Any]] = sorted(
            (entries_by_design[h] for h in ordered_hashes if h in entries_by_design),
            key=lambda entry: (
                0.0 if _feasibility_value(entry) <= FEASIBILITY_CUTOFF else 1.0,
                _feasibility_value(entry),
                -_objective_proxy(entry, problem_type=self.config.problem),
            ),
        )
        if len(ranked) >= promote_limit:
            return ranked[:promote_limit]

        remaining = {
            design_hash: entry
            for design_hash, entry in entries_by_design.items()
            if design_hash
            not in {entry.design_hash for entry in summary.pareto_entries}
        }
        crowding = _crowding_distance(remaining, problem_type=self.config.problem)

        def _sort_key(design_hash: str) -> tuple[float, float, float, float]:
            entry = remaining[design_hash]
            feas = _feasibility_value(entry)
            feasible_flag = 0.0 if feas <= FEASIBILITY_CUTOFF else 1.0
            return (
                feasible_flag,
                -crowding.get(design_hash, 0.0),
                feas,
                -_objective_proxy(entry, problem_type=self.config.problem),
            )

        for design_hash in sorted(remaining, key=_sort_key):
            ranked.append(remaining[design_hash])
            if len(ranked) >= promote_limit:
                break
        return ranked

    def should_transition_s1_to_s2(
        self,
        history: list[CycleSummary],
    ) -> bool:
        """Check if conditions for S1 -> S2 transition are met."""
        gate_cfg = self.config.stage_gates
        if not history:
            return False
        last = history[-1]
        if last.feasibility is not None:
            triggered = last.feasibility <= gate_cfg.s1_to_s2_feasibility_margin
            print(
                f"[fidelity][stage-gate] S1→S2 feasibility check: margin={last.feasibility:.5f} "
                f"<= {gate_cfg.s1_to_s2_feasibility_margin:.5f} -> {triggered}"
            )
            if triggered:
                return True
        improvement = _relative_objective_improvement(
            history, gate_cfg.s1_to_s2_lookback_cycles
        )
        triggered_improvement = improvement >= gate_cfg.s1_to_s2_objective_improvement
        print(
            f"[fidelity][stage-gate] S1→S2 objective improvement check: "
            f"{improvement:.4f} >= {gate_cfg.s1_to_s2_objective_improvement:.4f} -> {triggered_improvement}"
        )
        return triggered_improvement

    def should_transition_s2_to_s3(
        self,
        history: list[CycleSummary],
        world_model: memory.WorldModel,
        experiment_id: int,
        current_cycle: int,
    ) -> bool:
        """Check if conditions for S2 -> S3 transition are met."""
        gate_cfg = self.config.stage_gates
        governance_cfg = self.config.governance

        avg_delta = world_model.average_recent_hv_delta(
            experiment_id, governance_cfg.hv_lookback
        )
        if avg_delta is not None:
            triggered_delta = avg_delta <= gate_cfg.s2_to_s3_hv_delta
            print(
                f"[fidelity][stage-gate] S2→S3 average HV delta over "
                f"{governance_cfg.hv_lookback} cycles: {avg_delta:.4f} <= {gate_cfg.s2_to_s3_hv_delta:.4f} -> {triggered_delta}"
            )
            if triggered_delta:
                return True
        else:
            print(
                f"[fidelity][stage-gate] insufficient HV delta history ({len(history)} cycles) "
                f"to evaluate lookback={governance_cfg.hv_lookback}; deferring promotion"
            )
        exhausted = current_cycle >= self.config.cycles
        print(
            f"[fidelity][stage-gate] S2→S3 budget check: cycle={current_cycle} >= total={self.config.cycles} -> {exhausted}"
        )
        return exhausted
