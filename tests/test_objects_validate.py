"""Tests for Layer 1 validators."""

import numpy as np
import pandas as pd
import pytest

from anomsmith.objects.validate import (
    assert_aligned,
    assert_monotonic_index,
    assert_panel,
    assert_series,
)
from anomsmith.objects.views import LabelView, PanelView, ScoreView, SeriesView


class TestAssertSeries:
    """Tests for assert_series."""

    def test_valid_series(self) -> None:
        """Test assert_series with valid input."""
        index = pd.RangeIndex(0, 10)
        values = np.random.randn(10)
        series = SeriesView(index=index, values=values)
        assert_series(series)  # Should not raise

    def test_invalid_type(self) -> None:
        """Test assert_series with invalid type."""
        with pytest.raises(ValueError, match="Expected SeriesView"):
            assert_series("not a series")  # type: ignore

    def test_non_monotonic_index(self) -> None:
        """Test assert_series with non-monotonic index."""
        index = pd.Index([3, 1, 2, 4, 5])
        values = np.array([1, 2, 3, 4, 5])
        series = SeriesView(index=index, values=values)
        with pytest.raises(ValueError, match="monotonic"):
            assert_series(series)


class TestAssertPanel:
    """Tests for assert_panel."""

    def test_valid_panel(self) -> None:
        """Test assert_panel with valid input."""
        entity_key = pd.Index(["A", "B"])
        time_index = pd.RangeIndex(0, 10)
        values = np.random.randn(2, 10)
        panel = PanelView(entity_key=entity_key, time_index=time_index, values=values)
        assert_panel(panel)  # Should not raise

    def test_invalid_type(self) -> None:
        """Test assert_panel with invalid type."""
        with pytest.raises(ValueError, match="Expected PanelView"):
            assert_panel("not a panel")  # type: ignore


class TestAssertAligned:
    """Tests for assert_aligned."""

    def test_aligned_views(self) -> None:
        """Test assert_aligned with aligned views."""
        index = pd.RangeIndex(0, 10)
        series = SeriesView(index=index, values=np.random.randn(10))
        scores = ScoreView(index=index, scores=np.random.randn(10))
        assert_aligned(series, scores)  # Should not raise

    def test_mismatched_length(self) -> None:
        """Test assert_aligned with mismatched lengths."""
        index1 = pd.RangeIndex(0, 10)
        index2 = pd.RangeIndex(0, 5)
        series = SeriesView(index=index1, values=np.random.randn(10))
        scores = ScoreView(index=index2, scores=np.random.randn(5))
        with pytest.raises(ValueError, match="same length"):
            assert_aligned(series, scores)

    def test_mismatched_index(self) -> None:
        """Test assert_aligned with mismatched indices."""
        index1 = pd.RangeIndex(0, 10)
        index2 = pd.RangeIndex(1, 11)
        series = SeriesView(index=index1, values=np.random.randn(10))
        scores = ScoreView(index=index2, scores=np.random.randn(10))
        with pytest.raises(ValueError, match="equal"):
            assert_aligned(series, scores)


class TestAssertMonotonicIndex:
    """Tests for assert_monotonic_index."""

    def test_monotonic_index(self) -> None:
        """Test assert_monotonic_index with monotonic index."""
        index = pd.RangeIndex(0, 10)
        assert_monotonic_index(index)  # Should not raise

    def test_non_monotonic_index(self) -> None:
        """Test assert_monotonic_index with non-monotonic index."""
        index = pd.Index([3, 1, 2, 4, 5])
        with pytest.raises(ValueError, match="monotonic"):
            assert_monotonic_index(index)

    def test_invalid_type(self) -> None:
        """Test assert_monotonic_index with invalid type."""
        with pytest.raises(ValueError, match="Expected pd.Index"):
            assert_monotonic_index([1, 2, 3])  # type: ignore

