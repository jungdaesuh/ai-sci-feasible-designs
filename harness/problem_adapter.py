"""Wraps ProblemProfile with frontier/target logic for P1/P2/P3.

Translates between the SSOT problem_profiles and the harness cycle
loop's needs: direction-normalized objective extraction, frontier
delta computation, target checks, and binding constraint analysis.
"""

from __future__ import annotations

from ai_scientist.problem_profiles import ProblemProfile

from harness.types import CycleSnapshot

# Baseline targets from the challenge (ALM-NGOpt scores).
_TARGETS: dict[str, float] = {
    "p1": 2.10,  # max_elongation (minimize) — beat means < 2.10
    "p2": 8.61,  # lgradB (maximize) — beat means > 8.61
    "p3": 0.0,  # multi-objective HV — no single scalar target
}


class ProblemAdapter:
    """Wraps a ProblemProfile with frontier/target logic.

    Constructed once per run; all methods operate on the
    encapsulated profile so callers never pass it explicitly.
    """

    __slots__ = ("_profile",)

    def __init__(self, profile: ProblemProfile) -> None:
        self._profile = profile

    @property
    def problem(self) -> str:
        return self._profile.problem

    @property
    def profile(self) -> ProblemProfile:
        return self._profile

    def objective_value(self, metrics_row: dict) -> float:
        """Extract direction-normalized objective from a metrics dict.

        Returns the value such that *higher is always better* regardless
        of the underlying problem direction. This simplifies all
        downstream comparisons (frontier delta, target check).
        """
        metric_name = self._profile.frontier_recipe.objective_metric
        raw = float(metrics_row[metric_name])
        if self._is_maximize():
            return raw
        # For minimization, negate so higher = better.
        return -raw

    def frontier_delta(
        self,
        prev: CycleSnapshot,
        now: CycleSnapshot,
    ) -> float:
        """Compute improvement between two snapshots.

        For P1/P2 frontier_value is direction-normalized (higher = better).
        For P3 frontier_value stores HV directly.
        Both cases: delta = now - prev; positive when improving.
        """
        if prev.frontier_value is None or now.frontier_value is None:
            return 0.0
        return now.frontier_value - prev.frontier_value

    def target_reached(self, snapshot: CycleSnapshot) -> bool:
        """Has the problem's baseline target been met?"""
        if snapshot.frontier_value is None:
            return False
        target = _TARGETS.get(self._profile.problem)
        if target is None or target == 0.0:
            # P3 has no single scalar target.
            return False
        # frontier_value is direction-normalized (higher = better).
        # For P1: target 2.10 means elongation < 2.10 -> normalized = -2.10.
        #         frontier_value > -2.10 means we beat the target.
        # For P2: target 8.61 means lgradB > 8.61 -> normalized = 8.61.
        #         frontier_value > 8.61 means we beat the target.
        if self._is_maximize():
            return snapshot.frontier_value > target
        return snapshot.frontier_value > -target

    def binding_constraints(
        self,
        candidates: list[dict],
    ) -> tuple[str, ...]:
        """Identify which constraints are closest to their thresholds.

        Examines near-feasible candidates and returns constraint names
        sorted by average proximity to the constraint boundary (closest
        first). Uses the 'constraint_margins' dict if present in raw_json.
        """
        if not candidates:
            return ()

        margin_sums: dict[str, float] = {}
        margin_counts: dict[str, int] = {}

        for cand in candidates:
            margins = cand.get("constraint_margins", {})
            for spec in self._profile.constraints:
                margin = margins.get(spec.name)
                if margin is None:
                    continue
                abs_margin = abs(float(margin))
                margin_sums[spec.name] = margin_sums.get(spec.name, 0.0) + abs_margin
                margin_counts[spec.name] = margin_counts.get(spec.name, 0) + 1

        if not margin_sums:
            return ()

        avg_margins = {
            name: margin_sums[name] / margin_counts[name] for name in margin_sums
        }
        return tuple(sorted(avg_margins, key=lambda n: avg_margins[n]))

    def _is_maximize(self) -> bool:
        return self._profile.frontier_recipe.objective_direction == "maximize"
