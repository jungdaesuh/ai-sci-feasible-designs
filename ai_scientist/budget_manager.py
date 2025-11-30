"""
Budget management logic for the AI Scientist.
Extracted from runner.py (Task 1.1).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping

from ai_scientist import config as ai_config
from ai_scientist import tools


@dataclass(frozen=True)
class BudgetSnapshot:
    screen_evals_per_cycle: int
    promote_top_k: int
    max_high_fidelity_evals_per_cycle: int
    remaining_budget: int


@dataclass(frozen=True)
class CycleBudgetFeedback:
    hv_delta: float | None
    feasibility_rate: float | None
    cache_hit_rate: float | None


class BudgetController:
    STATE_VERSION = 1
    _STATE_FIELDS = {"_last_feedback", "_cache_stats", "_total_consumed"}

    def __init__(self, config: ai_config.ExperimentConfig) -> None:
        self._base = config.budgets
        self._adaptive_cfg = config.adaptive_budgets
        self._last_feedback: CycleBudgetFeedback | None = None
        self._cache_stats: Dict[str, Dict[str, int]] = {}
        self._total_consumed: int = 0

    def to_dict(self) -> dict[str, Any]:
        unknown_state = {
            key
            for key in self.__dict__
            if key.startswith("_")
            and key not in {"_base", "_adaptive_cfg"}
            and key not in self._STATE_FIELDS
        }
        if unknown_state:
            raise ValueError(
                f"BudgetController state fields {unknown_state} are not serialized; "
                "update _STATE_FIELDS/to_dict/restore for deterministic resume."
            )
        feedback = (
            {
                "hv_delta": self._last_feedback.hv_delta,
                "feasibility_rate": self._last_feedback.feasibility_rate,
                "cache_hit_rate": self._last_feedback.cache_hit_rate,
            }
            if self._last_feedback
            else None
        )
        return {
            "state_version": self.STATE_VERSION,
            "base_budgets": asdict(self._base),
            "adaptive_cfg": asdict(self._adaptive_cfg),
            "adaptive_enabled": self._adaptive_cfg.enabled,
            "last_feedback": feedback,
            "cache_stats": self._cache_stats,
            "total_consumed": self._total_consumed,
        }

    def restore(self, payload: Mapping[str, Any]) -> None:
        version = payload.get("state_version", 0)
        if version != self.STATE_VERSION:
            print(
                f"[budget] warning: checkpoint state_version={version} "
                f"!= expected {self.STATE_VERSION}; attempting best-effort restore."
            )
        feedback = payload.get("last_feedback")
        if feedback:
            self._last_feedback = CycleBudgetFeedback(
                hv_delta=feedback.get("hv_delta"),
                feasibility_rate=feedback.get("feasibility_rate"),
                cache_hit_rate=feedback.get("cache_hit_rate"),
            )
        cache_stats = payload.get("cache_stats")
        if isinstance(cache_stats, dict):
            self._cache_stats = cache_stats
        
        self._total_consumed = int(payload.get("total_consumed", 0))

    def consume(self, n_evals: int) -> None:
        """Track budget usage."""
        self._total_consumed += n_evals

    def adjust_for_cycle(
        self,
        hv_delta: float | None,
        feasibility_rate: float | None,
        cache_hit_rate: float | None = None,
    ) -> None:
        """Adaptive adjustment based on cycle feedback."""
        self.record_feedback(
            CycleBudgetFeedback(
                hv_delta=hv_delta,
                feasibility_rate=feasibility_rate,
                cache_hit_rate=cache_hit_rate,
            )
        )

    def snapshot(self) -> BudgetSnapshot:
        """Returns current budget state."""
        if not self._adaptive_cfg.enabled or self._last_feedback is None:
            return BudgetSnapshot(
                screen_evals_per_cycle=self._base.screen_evals_per_cycle,
                promote_top_k=self._base.promote_top_k,
                max_high_fidelity_evals_per_cycle=self._base.max_high_fidelity_evals_per_cycle,
                remaining_budget=self._base.screen_evals_per_cycle,
            )
        
        screen_evals = self._blend_budget(
            self._base.screen_evals_per_cycle,
            self._adaptive_cfg.screen_bounds,
            self._screen_score(self._last_feedback),
        )
        return BudgetSnapshot(
            screen_evals_per_cycle=screen_evals,
            promote_top_k=self._blend_budget(
                self._base.promote_top_k,
                self._adaptive_cfg.promote_top_k_bounds,
                self._promote_score(self._last_feedback),
            ),
            max_high_fidelity_evals_per_cycle=self._blend_budget(
                self._base.max_high_fidelity_evals_per_cycle,
                self._adaptive_cfg.high_fidelity_bounds,
                self._high_fidelity_score(self._last_feedback),
            ),
            remaining_budget=screen_evals,
        )

    def capture_cache_hit_rate(
        self,
        stage: str,
        stats: Mapping[str, int] | None = None,
    ) -> float:
        stats = stats or tools.get_cache_stats(stage)
        previous = self._cache_stats.get(stage)
        delta_hits = stats.get("hits", 0)
        delta_misses = stats.get("misses", 0)
        if previous:
            delta_hits -= previous.get("hits", 0)
            delta_misses -= previous.get("misses", 0)
        self._cache_stats[stage] = {
            "hits": stats.get("hits", 0),
            "misses": stats.get("misses", 0),
        }
        total = max(0, delta_hits + delta_misses)
        if total <= 0:
            return 0.0
        return float(max(0.0, min(1.0, delta_hits / total)))

    def record_feedback(self, feedback: CycleBudgetFeedback) -> None:
        self._last_feedback = feedback

    def _blend_budget(
        self,
        base_value: int,
        bounds: ai_config.BudgetRangeConfig,
        score: float,
    ) -> int:
        constrained_base = max(bounds.min, min(bounds.max, base_value))
        if bounds.min >= bounds.max:
            return constrained_base
        clamped = max(0.0, min(1.0, score))
        mid = 0.5
        if clamped == mid:
            return constrained_base
        if clamped > mid:
            ratio = (clamped - mid) * 2.0
            increment = bounds.max - constrained_base
            return min(bounds.max, constrained_base + int(round(increment * ratio)))
        ratio = (mid - clamped) * 2.0
        decrement = constrained_base - bounds.min
        return max(bounds.min, constrained_base - int(round(decrement * ratio)))

    def _normalize(self, value: float | None, target: float) -> float:
        if value is None or target <= 0.0:
            return 0.0
        ratio = float(value) / float(target)
        return max(0.0, min(1.0, ratio))

    def _screen_score(self, feedback: CycleBudgetFeedback) -> float:
        progress = max(0.0, feedback.hv_delta or 0.0)
        normalized = self._normalize(progress, self._adaptive_cfg.hv_slope_reference)
        feasibility = self._normalize(
            feedback.feasibility_rate, self._adaptive_cfg.feasibility_target
        )
        return 1.0 - (0.6 * normalized + 0.4 * feasibility)

    def _promote_score(self, feedback: CycleBudgetFeedback) -> float:
        progress = max(0.0, feedback.hv_delta or 0.0)
        normalized = self._normalize(progress, self._adaptive_cfg.hv_slope_reference)
        feasibility = self._normalize(
            feedback.feasibility_rate, self._adaptive_cfg.feasibility_target
        )
        return 0.6 * normalized + 0.4 * feasibility

    def _high_fidelity_score(self, feedback: CycleBudgetFeedback) -> float:
        progress = max(0.0, feedback.hv_delta or 0.0)
        normalized = self._normalize(progress, self._adaptive_cfg.hv_slope_reference)
        cache = self._normalize(
            feedback.cache_hit_rate, self._adaptive_cfg.cache_hit_target
        )
        return 0.5 * normalized + 0.5 * cache
