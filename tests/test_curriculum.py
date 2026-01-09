"""Tests for curriculum learning scheduler (Issue #12).

Verifies that curriculum learning:
1. Tracks stage progression P1 → P2 → P3
2. Respects min_cycles_per_stage before advancing
3. Advances when feasibility threshold is met
4. Stays at P3 (final stage)
5. Is correctly wired to config
"""

import pytest

from ai_scientist.curriculum import (
    CurriculumConfig,
    CurriculumScheduler,
    CurriculumStage,
)


class TestCurriculumScheduler:
    """Tests for CurriculumScheduler class."""

    def test_initial_stage_default(self):
        """Verify scheduler starts at P1 by default."""
        config = CurriculumConfig(enabled=True)
        scheduler = CurriculumScheduler(config)
        assert scheduler.current_stage == CurriculumStage.P1

    def test_initial_stage_custom(self):
        """Verify scheduler can start at different stages."""
        config = CurriculumConfig(enabled=True, initial_stage="p2")
        scheduler = CurriculumScheduler(config)
        assert scheduler.current_stage == CurriculumStage.P2

    def test_tick_increments_cycles(self):
        """Verify tick() increments cycle counter."""
        config = CurriculumConfig(enabled=True)
        scheduler = CurriculumScheduler(config)
        assert scheduler.cycles_in_stage == 0
        scheduler.tick()
        assert scheduler.cycles_in_stage == 1
        scheduler.tick()
        scheduler.tick()
        assert scheduler.cycles_in_stage == 3

    def test_should_advance_respects_min_cycles(self):
        """Verify should_advance returns False before min_cycles_per_stage."""
        config = CurriculumConfig(
            enabled=True, min_cycles_per_stage=5, advancement_threshold=0.3
        )
        scheduler = CurriculumScheduler(config)

        # High feasibility but not enough cycles
        for _ in range(4):
            scheduler.tick()
        assert not scheduler.should_advance(feasibility_rate=0.5)

        # After 5 cycles, should be able to advance
        scheduler.tick()
        assert scheduler.should_advance(feasibility_rate=0.5)

    def test_should_advance_respects_threshold(self):
        """Verify should_advance respects feasibility threshold."""
        config = CurriculumConfig(
            enabled=True, min_cycles_per_stage=2, advancement_threshold=0.3
        )
        scheduler = CurriculumScheduler(config)

        for _ in range(3):
            scheduler.tick()

        # Below threshold
        assert not scheduler.should_advance(feasibility_rate=0.2)
        # At threshold
        assert scheduler.should_advance(feasibility_rate=0.3)
        # Above threshold
        assert scheduler.should_advance(feasibility_rate=0.5)

    def test_should_advance_disabled_returns_false(self):
        """Verify should_advance returns False when disabled."""
        config = CurriculumConfig(enabled=False)
        scheduler = CurriculumScheduler(config)

        for _ in range(10):
            scheduler.tick()
        assert not scheduler.should_advance(feasibility_rate=0.9)

    def test_advance_p1_to_p2(self):
        """Verify advance transitions P1 → P2."""
        config = CurriculumConfig(enabled=True)
        scheduler = CurriculumScheduler(config)
        assert scheduler.current_stage == CurriculumStage.P1

        new_stage = scheduler.advance()
        assert new_stage == CurriculumStage.P2
        assert scheduler.current_stage == CurriculumStage.P2

    def test_advance_p2_to_p3(self):
        """Verify advance transitions P2 → P3."""
        config = CurriculumConfig(enabled=True, initial_stage="p2")
        scheduler = CurriculumScheduler(config)
        assert scheduler.current_stage == CurriculumStage.P2

        new_stage = scheduler.advance()
        assert new_stage == CurriculumStage.P3
        assert scheduler.current_stage == CurriculumStage.P3

    def test_advance_stays_at_p3(self):
        """Verify advance stays at P3 (final stage)."""
        config = CurriculumConfig(enabled=True, initial_stage="p3")
        scheduler = CurriculumScheduler(config)
        assert scheduler.current_stage == CurriculumStage.P3

        new_stage = scheduler.advance()
        assert new_stage == CurriculumStage.P3
        assert scheduler.current_stage == CurriculumStage.P3

    def test_should_advance_false_at_p3(self):
        """Verify should_advance returns False at P3."""
        config = CurriculumConfig(
            enabled=True, initial_stage="p3", min_cycles_per_stage=1
        )
        scheduler = CurriculumScheduler(config)
        scheduler.tick()
        scheduler.tick()

        # Even with high feasibility, should not advance from P3
        assert not scheduler.should_advance(feasibility_rate=0.9)

    def test_advance_resets_cycle_counter(self):
        """Verify advance resets cycles_in_stage to 0."""
        config = CurriculumConfig(enabled=True)
        scheduler = CurriculumScheduler(config)

        for _ in range(5):
            scheduler.tick()
        assert scheduler.cycles_in_stage == 5

        scheduler.advance()
        assert scheduler.cycles_in_stage == 0

    def test_advance_caches_seeds(self):
        """Verify advance caches best seeds for warm-start."""
        config = CurriculumConfig(enabled=True)
        scheduler = CurriculumScheduler(config)

        seeds = [{"params": {"r_cos": [[1.0]]}}, {"params": {"r_cos": [[2.0]]}}]
        scheduler.advance(best_seeds=seeds)

        assert scheduler.get_warm_start_seeds() == seeds

    def test_get_effective_problem(self):
        """Verify get_effective_problem returns current stage string."""
        config = CurriculumConfig(enabled=True)
        scheduler = CurriculumScheduler(config)

        assert scheduler.get_effective_problem() == "p1"
        scheduler.advance()
        assert scheduler.get_effective_problem() == "p2"
        scheduler.advance()
        assert scheduler.get_effective_problem() == "p3"


class TestCurriculumConfigInExperiment:
    """Tests for CurriculumConfig in ExperimentConfig."""

    def test_config_has_curriculum(self):
        """Verify ExperimentConfig has curriculum field."""
        from ai_scientist.config import load_experiment_config

        cfg = load_experiment_config("configs/experiment.example.yaml")
        assert hasattr(cfg, "curriculum")
        assert cfg.curriculum is not None

    def test_curriculum_defaults(self):
        """Verify CurriculumConfig has expected defaults."""
        config = CurriculumConfig()
        assert config.enabled is False
        assert config.advancement_threshold == 0.3
        assert config.min_cycles_per_stage == 5
        assert config.initial_stage == "p1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
