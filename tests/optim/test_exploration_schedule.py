"""Tests for exploration schedule utilities (Issue #11).

Verifies that UCB exploration ratio decay:
1. Starts at initial value
2. Decays linearly over decay_cycles
3. Stays at final value after full decay
4. Is correctly wired to config
"""

import pytest

from ai_scientist.exploration import (
    compute_exploration_ratio,
    compute_exploration_ratio_from_config,
)


class TestExplorationRatioDecay:
    """Tests for compute_exploration_ratio function."""

    def test_initial_value_at_cycle_zero(self):
        """Verify exploration ratio equals initial at cycle 0."""
        ratio = compute_exploration_ratio(0, initial=2.0, final=0.1, decay_cycles=20)
        assert ratio == 2.0

    def test_final_value_at_decay_cycles(self):
        """Verify exploration ratio equals final at decay_cycles."""
        ratio = compute_exploration_ratio(20, initial=2.0, final=0.1, decay_cycles=20)
        assert ratio == 0.1

    def test_final_value_beyond_decay_cycles(self):
        """Verify exploration ratio stays at final after decay_cycles."""
        ratio = compute_exploration_ratio(30, initial=2.0, final=0.1, decay_cycles=20)
        assert ratio == 0.1
        ratio = compute_exploration_ratio(100, initial=2.0, final=0.1, decay_cycles=20)
        assert ratio == 0.1

    def test_linear_decay_midpoint(self):
        """Verify linear decay at midpoint."""
        # At cycle 10 (half of 20), should be (2.0 + 0.1) / 2 = 1.05
        ratio = compute_exploration_ratio(10, initial=2.0, final=0.1, decay_cycles=20)
        assert abs(ratio - 1.05) < 1e-6

    def test_custom_parameters(self):
        """Verify function works with custom parameters."""
        ratio = compute_exploration_ratio(5, initial=1.0, final=0.0, decay_cycles=10)
        assert abs(ratio - 0.5) < 1e-6

    def test_zero_decay_cycles(self):
        """Verify immediate decay to final with zero decay_cycles."""
        ratio = compute_exploration_ratio(0, initial=2.0, final=0.1, decay_cycles=0)
        assert ratio == 0.1

    def test_negative_decay_cycles(self):
        """Verify immediate decay to final with negative decay_cycles."""
        ratio = compute_exploration_ratio(0, initial=2.0, final=0.1, decay_cycles=-1)
        assert ratio == 0.1


class TestExplorationRatioFromConfig:
    """Tests for compute_exploration_ratio_from_config wrapper."""

    def test_wrapper_matches_direct_call(self):
        """Verify wrapper produces same results as direct function."""
        for cycle in [0, 5, 10, 15, 20, 25]:
            direct = compute_exploration_ratio(
                cycle, initial=2.0, final=0.1, decay_cycles=20
            )
            wrapped = compute_exploration_ratio_from_config(
                cycle,
                ucb_exploration_initial=2.0,
                ucb_exploration_final=0.1,
                ucb_decay_cycles=20,
            )
            assert direct == wrapped


class TestProposalMixConfigUCB:
    """Tests for UCB parameters in ProposalMixConfig."""

    def test_config_has_ucb_parameters(self):
        """Verify ProposalMixConfig has UCB parameters."""
        from ai_scientist.config import ProposalMixConfig

        config = ProposalMixConfig(
            constraint_ratio=0.7,
            exploration_ratio=0.3,
        )
        assert hasattr(config, "ucb_exploration_initial")
        assert hasattr(config, "ucb_exploration_final")
        assert hasattr(config, "ucb_decay_cycles")

    def test_ucb_default_values(self):
        """Verify UCB parameters have expected defaults."""
        from ai_scientist.config import ProposalMixConfig

        config = ProposalMixConfig(
            constraint_ratio=0.7,
            exploration_ratio=0.3,
        )
        assert config.ucb_exploration_initial == 2.0
        assert config.ucb_exploration_final == 0.1
        assert config.ucb_decay_cycles == 20

    def test_ucb_custom_values(self):
        """Verify UCB parameters can be customized."""
        from ai_scientist.config import ProposalMixConfig

        config = ProposalMixConfig(
            constraint_ratio=0.7,
            exploration_ratio=0.3,
            ucb_exploration_initial=3.0,
            ucb_exploration_final=0.5,
            ucb_decay_cycles=10,
        )
        assert config.ucb_exploration_initial == 3.0
        assert config.ucb_exploration_final == 0.5
        assert config.ucb_decay_cycles == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
