"""Tests for robust_zscore scaling."""

import numpy as np
import pytest

from anomsmith.primitives.scaling import robust_zscore


class TestRobustZscore:
    """Tests for robust_zscore."""

    def test_basic_functionality(self) -> None:
        """Test robust_zscore with normal data."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        z_scores = robust_zscore(values)
        assert z_scores.shape == values.shape
        assert np.all(np.isfinite(z_scores))

    def test_constant_series(self) -> None:
        """Test robust_zscore with constant series (should be stable)."""
        values = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        z_scores = robust_zscore(values)
        assert z_scores.shape == values.shape
        # All should be approximately zero (median-centered)
        assert np.allclose(z_scores, 0.0, atol=1e-6)

    def test_empty_array(self) -> None:
        """Test robust_zscore with empty array."""
        values = np.array([])
        z_scores = robust_zscore(values)
        assert len(z_scores) == 0

    def test_single_value(self) -> None:
        """Test robust_zscore with single value."""
        values = np.array([42.0])
        z_scores = robust_zscore(values)
        assert z_scores.shape == values.shape
        assert np.allclose(z_scores, 0.0, atol=1e-6)

    def test_epsilon_guard(self) -> None:
        """Test that epsilon prevents division by zero."""
        # Create series where MAD would be zero without epsilon
        values = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        z_scores = robust_zscore(values, epsilon=1e-8)
        assert np.all(np.isfinite(z_scores))

    def test_output_shape(self) -> None:
        """Test that output shape matches input shape."""
        for shape in [(10,), (100,), (1000,)]:
            values = np.random.randn(*shape)
            z_scores = robust_zscore(values)
            assert z_scores.shape == values.shape

