"""Tests for Layer 4 workflows."""

import numpy as np
import pandas as pd
import pytest

from anomsmith import RobustZScoreScorer, detect_anomalies, score_anomalies
from anomsmith.primitives.detectors.change_point import ChangePointDetector
from anomsmith.primitives.scorers.robust_zscore import RobustZScoreScorer
from anomsmith.primitives.thresholding import ThresholdRule
from anomsmith.workflows.detect import sweep_thresholds


class TestScoreAnomalies:
    """Tests for score_anomalies workflow."""

    def test_basic_scoring(self) -> None:
        """Test score_anomalies returns pandas Series."""
        y = pd.Series(np.random.randn(100), index=pd.RangeIndex(100))
        scorer = RobustZScoreScorer()
        scorer.fit(y.values)

        scores = score_anomalies(y, scorer)

        assert isinstance(scores, pd.Series)
        assert len(scores) == len(y)
        assert scores.index.equals(y.index)

    def test_aligned_index(self) -> None:
        """Test that scores have aligned index with input."""
        index = pd.date_range("2020-01-01", periods=50, freq="D")
        y = pd.Series(np.random.randn(50), index=index)
        scorer = RobustZScoreScorer()
        scorer.fit(y.values)

        scores = score_anomalies(y, scorer)

        assert scores.index.equals(y.index)


class TestDetectAnomalies:
    """Tests for detect_anomalies workflow."""

    def test_basic_detection(self) -> None:
        """Test detect_anomalies returns DataFrame with score and flag."""
        y = pd.Series(np.random.randn(100), index=pd.RangeIndex(100))
        scorer = RobustZScoreScorer()
        scorer.fit(y.values)
        threshold_rule = ThresholdRule(method="quantile", value=0.95, quantile=0.95)

        result = detect_anomalies(y, scorer, threshold_rule)

        assert isinstance(result, pd.DataFrame)
        assert "score" in result.columns
        assert "flag" in result.columns
        assert len(result) == len(y)
        assert result.index.equals(y.index)

    def test_detector_with_threshold(self) -> None:
        """Test detect_anomalies with BaseDetector."""
        y = pd.Series(np.random.randn(100), index=pd.RangeIndex(100))
        detector = ChangePointDetector(window_size=10, threshold_multiplier=2.0)
        detector.fit(y.values)
        threshold_rule = ThresholdRule(method="absolute", value=2.0)

        result = detect_anomalies(y, detector, threshold_rule)

        assert isinstance(result, pd.DataFrame)
        assert "score" in result.columns
        assert "flag" in result.columns
        assert np.all(np.isin(result["flag"].values, [0, 1]))


class TestSweepThresholds:
    """Tests for sweep_thresholds workflow."""

    def test_sweep_with_series_labels(self) -> None:
        """Test sweep_thresholds with pandas Series labels."""
        y = pd.Series(np.random.randn(100), index=pd.RangeIndex(100))
        labels = pd.Series((np.random.rand(100) > 0.9).astype(int), index=y.index)
        scorer = RobustZScoreScorer()
        scorer.fit(y.values)
        result = sweep_thresholds(y, scorer, [0.5, 1.0, 2.0], labels=labels)
        assert "threshold" in result.columns
        assert "precision" in result.columns
        assert "recall" in result.columns
        assert "f1" in result.columns
        assert len(result) == 3

    def test_sweep_with_ndarray_labels(self) -> None:
        """Test sweep_thresholds with numpy array labels (regression for TD-004)."""
        y = pd.Series(np.random.randn(100), index=pd.RangeIndex(100))
        labels = (np.random.rand(100) > 0.9).astype(int)
        scorer = RobustZScoreScorer()
        scorer.fit(y.values)
        result = sweep_thresholds(y, scorer, [0.5, 1.0, 2.0], labels=labels)
        assert "threshold" in result.columns
        assert "f1" in result.columns
        assert len(result) == 3
        assert not np.isnan(result["f1"]).all()

    def test_sweep_without_labels(self) -> None:
        """Test sweep_thresholds returns NaN metrics when labels not provided."""
        y = pd.Series(np.random.randn(50))
        scorer = RobustZScoreScorer()
        scorer.fit(y.values)
        result = sweep_thresholds(y, scorer, [1.0, 2.0])
        assert result["precision"].isna().all()
        assert result["recall"].isna().all()


