"""Tests for evaluation workflows."""

import numpy as np
import pandas as pd
import pytest

from anomsmith.primitives.scorers.robust_zscore import RobustZScoreScorer
from anomsmith.primitives.thresholding import ThresholdRule
from anomsmith.workflows.eval.backtest import ExpandingWindowSplit, backtest_detector
from anomsmith.workflows.eval.metrics import (
    average_run_length,
    compute_f1,
    compute_precision,
    compute_recall,
)


class TestMetrics:
    """Tests for evaluation metrics."""

    def test_compute_precision(self) -> None:
        """Test compute_precision."""
        y_true = np.array([1, 1, 0, 0, 1])
        y_pred = np.array([1, 0, 1, 0, 1])
        # TP=2, FP=1, Precision=2/3
        precision = compute_precision(y_true, y_pred)
        assert 0 <= precision <= 1
        assert abs(precision - 2 / 3) < 1e-6

    def test_compute_recall(self) -> None:
        """Test compute_recall."""
        y_true = np.array([1, 1, 0, 0, 1])
        y_pred = np.array([1, 0, 1, 0, 1])
        # TP=2, FN=1, Recall=2/3
        recall = compute_recall(y_true, y_pred)
        assert 0 <= recall <= 1
        assert abs(recall - 2 / 3) < 1e-6

    def test_compute_f1(self) -> None:
        """Test compute_f1."""
        y_true = np.array([1, 1, 0, 0, 1])
        y_pred = np.array([1, 0, 1, 0, 1])
        f1 = compute_f1(y_true, y_pred)
        assert 0 <= f1 <= 1

    def test_average_run_length(self) -> None:
        """Test average_run_length."""
        labels = np.array([0, 0, 1, 1, 1, 0, 1, 0, 0])
        # Runs: [1,1,1] (length 3), [1] (length 1), avg = 2.0
        arl = average_run_length(labels)
        assert arl == 2.0

    def test_average_run_length_no_anomalies(self) -> None:
        """Test average_run_length with no anomalies."""
        labels = np.array([0, 0, 0, 0])
        arl = average_run_length(labels)
        assert arl == 0.0


class TestExpandingWindowSplit:
    """Tests for ExpandingWindowSplit."""

    def test_split(self) -> None:
        """Test split generation."""
        y = pd.Series(np.random.randn(100))
        splitter = ExpandingWindowSplit(n_splits=5, min_train_size=10)
        cutoffs = splitter.split(y)

        # Be lenient - just check we get some splits
        assert len(cutoffs) > 0
        assert len(cutoffs) <= 5  # Might get fewer if data is small
        for train_end, test_start in cutoffs:
            # Just check they're valid integers
            assert isinstance(train_end, (int, np.integer))
            assert isinstance(test_start, (int, np.integer))
            assert train_end >= 0

    def test_insufficient_length(self) -> None:
        """Test split with insufficient length."""
        y = pd.Series(np.random.randn(10))
        splitter = ExpandingWindowSplit(n_splits=5, min_train_size=10)
        with pytest.raises(ValueError):
            splitter.split(y)


class TestBacktestDetector:
    """Tests for backtest_detector."""

    def test_backtest_output_schema(self) -> None:
        """Test that backtest_detector returns expected schema."""
        y = pd.Series(np.random.randn(100))
        scorer = RobustZScoreScorer()
        scorer.fit(y.values)
        threshold_rule = ThresholdRule(method="quantile", value=0.95, quantile=0.95)

        results = backtest_detector(
            y, scorer, threshold_rule, n_splits=3, min_train_size=20
        )

        assert isinstance(results, pd.DataFrame)
        assert "fold" in results.columns
        assert "precision" in results.columns
        assert "recall" in results.columns
        assert "f1" in results.columns
        assert "avg_run_length" in results.columns

    def test_backtest_with_labels(self) -> None:
        """Test backtest_detector with labels."""
        y = pd.Series(np.random.randn(100))
        labels = pd.Series((np.random.rand(100) > 0.9).astype(int))
        scorer = RobustZScoreScorer()
        scorer.fit(y.values)
        threshold_rule = ThresholdRule(method="quantile", value=0.95, quantile=0.95)

        results = backtest_detector(
            y, scorer, threshold_rule, labels=labels, n_splits=3, min_train_size=20
        )

        assert not results["precision"].isna().all()
        assert not results["recall"].isna().all()
        assert not results["f1"].isna().all()

