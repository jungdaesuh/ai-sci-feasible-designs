"""Tests for seed expansion and canonicalization fixes."""

import numpy as np
import pytest

from ai_scientist.cycle_executor import _expand_matrix_to_mode


class TestExpandMatrixToMode:
    """Test suite for the _expand_matrix_to_mode helper function."""

    def test_expand_matrix_preserves_nzero_coefficients(self) -> None:
        """Verify n=0 mode (center column) is preserved after expansion."""
        # 2x3 matrix with ntor=1 (columns: n=-1, n=0, n=+1)
        small = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        # Expand to mpol=3, ntor=2 (4x5 matrix)
        expanded = _expand_matrix_to_mode(
            small, max_poloidal_mode=3, max_toroidal_mode=2
        )

        assert expanded.shape == (4, 5)
        # In original: n=0 is at column 1
        # In expanded: n=0 is at column 2 (center of 5-column matrix)
        assert expanded[0, 2] == pytest.approx(2.0)  # m=0, n=0
        assert expanded[1, 2] == pytest.approx(5.0)  # m=1, n=0

    def test_expand_matrix_zero_pads_new_modes(self) -> None:
        """Verify new mode positions are zero-padded."""
        small = np.array([[1.0, 2.0, 3.0]])  # 1x3, mpol=0, ntor=1

        expanded = _expand_matrix_to_mode(
            small, max_poloidal_mode=2, max_toroidal_mode=3
        )

        assert expanded.shape == (3, 7)
        # Original n=-1,0,+1 should be at columns 2,3,4 in 7-wide matrix
        # New rows (m=1, m=2) should be all zeros
        assert np.all(expanded[1, :] == 0.0)
        assert np.all(expanded[2, :] == 0.0)
        # New edge columns (n=-3,-2,+2,+3) should be zeros
        assert expanded[0, 0] == 0.0  # n=-3
        assert expanded[0, 1] == 0.0  # n=-2
        assert expanded[0, 5] == 0.0  # n=+2
        assert expanded[0, 6] == 0.0  # n=+3

    def test_expand_matrix_returns_same_if_matching_shape(self) -> None:
        """Verify no-op when matrix already matches target shape."""
        matrix = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]])

        result = _expand_matrix_to_mode(
            matrix, max_poloidal_mode=1, max_toroidal_mode=2
        )

        assert result.shape == (2, 5)
        # Should return the same array (not a copy)
        assert np.array_equal(result, matrix)

    def test_expand_matrix_handles_single_element(self) -> None:
        """Verify handling of minimal 1x1 matrix (m=0, n=0 only)."""
        tiny = np.array([[42.0]])

        expanded = _expand_matrix_to_mode(
            tiny, max_poloidal_mode=1, max_toroidal_mode=1
        )

        assert expanded.shape == (2, 3)
        # n=0 in 1-column matrix is at column 0
        # n=0 in 3-column matrix is at column 1
        assert expanded[0, 1] == pytest.approx(42.0)
        assert expanded[1, 1] == 0.0  # New row
        assert expanded[0, 0] == 0.0  # New n=-1
        assert expanded[0, 2] == 0.0  # New n=+1

    def test_expand_matrix_preserves_off_center_modes(self) -> None:
        """Verify non-zero toroidal modes are correctly positioned."""
        # 2x3 matrix with known values at all positions
        matrix = np.array(
            [
                [1.0, 2.0, 3.0],  # m=0: n=-1, n=0, n=+1
                [4.0, 5.0, 6.0],  # m=1: n=-1, n=0, n=+1
            ]
        )

        expanded = _expand_matrix_to_mode(
            matrix, max_poloidal_mode=1, max_toroidal_mode=2
        )

        assert expanded.shape == (2, 5)
        # Original columns should shift: col 0->1, col 1->2, col 2->3
        assert expanded[0, 1] == pytest.approx(1.0)  # m=0, n=-1
        assert expanded[0, 2] == pytest.approx(2.0)  # m=0, n=0
        assert expanded[0, 3] == pytest.approx(3.0)  # m=0, n=+1
        assert expanded[1, 1] == pytest.approx(4.0)  # m=1, n=-1
        assert expanded[1, 2] == pytest.approx(5.0)  # m=1, n=0
        assert expanded[1, 3] == pytest.approx(6.0)  # m=1, n=+1
        # Edge columns should be zero
        assert expanded[0, 0] == 0.0  # n=-2
        assert expanded[0, 4] == 0.0  # n=+2
