"""Exploration schedule utilities for UCB acquisition (Issue #11).

This module provides utilities for computing exploration ratios
used in Upper Confidence Bound (UCB) acquisition functions.

UCB formula:
    score = mean + exploration_ratio * std

Where exploration_ratio decays from initial (high exploration) to
final (low exploration) over a configurable number of cycles.
"""

from __future__ import annotations


def compute_exploration_ratio(
    cycle: int,
    *,
    initial: float = 2.0,
    final: float = 0.1,
    decay_cycles: int = 20,
) -> float:
    """Compute exploration ratio with linear decay.

    The exploration ratio starts high (encouraging exploration of uncertain
    regions) and decays linearly to a low value (favoring exploitation of
    known good regions).

    Args:
        cycle: Current cycle number (0-indexed).
        initial: Initial exploration ratio (default: 2.0).
        final: Final exploration ratio after decay (default: 0.1).
        decay_cycles: Number of cycles to reach final value (default: 20).

    Returns:
        The exploration ratio for the given cycle.

    Examples:
        >>> compute_exploration_ratio(0)   # cycle 0: full exploration
        2.0
        >>> compute_exploration_ratio(10)  # cycle 10: half way
        1.05
        >>> compute_exploration_ratio(20)  # cycle 20: minimum exploration
        0.1
        >>> compute_exploration_ratio(30)  # beyond decay: stays at final
        0.1
    """
    if decay_cycles <= 0:
        return final
    if cycle >= decay_cycles:
        return final

    # Linear decay: ratio = initial - (initial - final) * (cycle / decay_cycles)
    progress = cycle / decay_cycles
    return initial - (initial - final) * progress


def compute_exploration_ratio_from_config(
    cycle: int,
    *,
    ucb_exploration_initial: float,
    ucb_exploration_final: float,
    ucb_decay_cycles: int,
) -> float:
    """Compute exploration ratio using config parameters.

    Convenience wrapper for compute_exploration_ratio that uses
    parameter names matching ProposalMixConfig.

    Args:
        cycle: Current cycle number.
        ucb_exploration_initial: Initial exploration ratio.
        ucb_exploration_final: Final exploration ratio.
        ucb_decay_cycles: Decay cycles.

    Returns:
        The exploration ratio for the given cycle.
    """
    return compute_exploration_ratio(
        cycle,
        initial=ucb_exploration_initial,
        final=ucb_exploration_final,
        decay_cycles=ucb_decay_cycles,
    )
