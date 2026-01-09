"""Tests for thresholding primitives."""

import numpy as np
import pandas as pd
import pytest

from anomsmith.objects.views import LabelView, ScoreView
from anomsmith.primitives.thresholding import ThresholdRule, apply_threshold


class TestThresholdRule:
    """Tests for ThresholdRule."""

    def test_absolute_method(self) -> None:
        """Test ThresholdRule with absolute method."""
        rule = ThresholdRule(method="absolute", value=3.0)
        assert rule.method == "absolute"
        assert rule.value == 3.0
        assert rule.quantile is None

    def test_quantile_method(self) -> None:
        """Test ThresholdRule with quantile method."""
        rule = ThresholdRule(method="quantile", value=0.95, quantile=0.95)
        assert rule.method == "quantile"
        assert rule.quantile == 0.95

    def test_quantile_required(self) -> None:
        """Test that quantile is required for quantile method."""
        with pytest.raises(ValueError, match="quantile must be provided"):
            ThresholdRule(method="quantile", value=0.95)

    def test_quantile_range(self) -> None:
        """Test that quantile must be in [0, 1]."""
        with pytest.raises(ValueError, match="quantile must be in"):
            ThresholdRule(method="quantile", value=1.5, quantile=1.5)

    def test_absolute_no_quantile(self) -> None:
        """Test that quantile should not be provided for absolute method."""
        with pytest.raises(ValueError, match="quantile should not be provided"):
            ThresholdRule(method="absolute", value=3.0, quantile=0.95)


class TestApplyThreshold:
    """Tests for apply_threshold."""

    def test_absolute_threshold(self) -> None:
        """Test apply_threshold with absolute threshold."""
        index = pd.RangeIndex(0, 5)
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        score_view = ScoreView(index=index, scores=scores)
        rule = ThresholdRule(method="absolute", value=3.0)

        label_view = apply_threshold(score_view, rule)

        assert isinstance(label_view, LabelView)
        assert np.array_equal(label_view.labels, np.array([0, 0, 1, 1, 1]))

    def test_quantile_threshold(self) -> None:
        """Test apply_threshold with quantile threshold."""
        index = pd.RangeIndex(0, 10)
        scores = np.arange(10, dtype=float)
        score_view = ScoreView(index=index, scores=scores)
        rule = ThresholdRule(method="quantile", value=0.9, quantile=0.9)

        label_view = apply_threshold(score_view, rule)

        assert isinstance(label_view, LabelView)
        # 90th percentile of 0-9 is 8.1, so 9 should be flagged
        assert label_view.labels[9] == 1

    def test_binary_output(self) -> None:
        """Test that output labels are binary."""
        index = pd.RangeIndex(0, 5)
        scores = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
        score_view = ScoreView(index=index, scores=scores)
        rule = ThresholdRule(method="absolute", value=2.0)

        label_view = apply_threshold(score_view, rule)

        unique_vals = np.unique(label_view.labels)
        assert np.all(np.isin(unique_vals, [0, 1]))

    def test_aligned_index(self) -> None:
        """Test that output index matches input index."""
        index = pd.RangeIndex(5, 10)
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        score_view = ScoreView(index=index, scores=scores)
        rule = ThresholdRule(method="absolute", value=3.0)

        label_view = apply_threshold(score_view, rule)

        assert label_view.index.equals(score_view.index)

