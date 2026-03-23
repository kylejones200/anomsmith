"""Tests for batch inference workflows."""

import numpy as np
import pandas as pd
import pytest

from anomsmith import batch_predict, batch_score
from anomsmith.primitives.detectors.ml import IsolationForestDetector
from anomsmith.primitives.scorers.robust_zscore import RobustZScoreScorer
from anomsmith.workflows.batch_inference import process_s3_batch


def _data_stream(design_matrix: np.ndarray) -> list:
    """Helper to create batches from 2D array."""
    batches = []
    for i in range(0, len(design_matrix), 50):
        batch = design_matrix[i : i + 50]
        if batch.ndim == 1:
            batch = batch.reshape(-1, 1)
        batches.append(pd.DataFrame(batch))
    return batches


class TestBatchScore:
    """Tests for batch_score."""

    def test_batch_score_yields_score_views(self) -> None:
        """Test batch_score yields ScoreView for each batch."""
        scorer = RobustZScoreScorer()
        y = np.random.randn(100)
        scorer.fit(y)

        def stream():
            yield pd.Series(y[:50])
            yield pd.Series(y[50:])

        results = list(batch_score(stream(), scorer))
        assert len(results) == 2
        assert len(results[0].scores) == 50
        assert len(results[1].scores) == 50

    def test_batch_score_raises_if_not_fitted(self) -> None:
        """Test batch_score raises if scorer not fitted."""
        scorer = RobustZScoreScorer()

        def stream():
            yield pd.Series(np.random.randn(10))

        with pytest.raises(ValueError, match="must be fitted"):
            list(batch_score(stream(), scorer))


class TestBatchPredict:
    """Tests for batch_predict."""

    def test_batch_predict_yields_labels_and_scores(self) -> None:
        """Test batch_predict yields (LabelView, ScoreView) tuples."""
        detector = IsolationForestDetector(contamination=0.05, random_state=42)
        X = np.random.randn(100, 2)
        detector.fit(X)

        def stream():
            yield pd.DataFrame(X[:50])
            yield pd.DataFrame(X[50:])

        results = list(batch_predict(stream(), detector))
        assert len(results) == 2
        labels, scores = results[0]
        assert len(labels.labels) == 50
        assert len(scores.scores) == 50

    def test_batch_predict_raises_if_not_fitted(self) -> None:
        """Test batch_predict raises if detector not fitted."""
        detector = IsolationForestDetector(contamination=0.05)

        def stream():
            yield np.random.randn(10, 2)

        with pytest.raises(ValueError, match="must be fitted"):
            list(batch_predict(stream(), detector))


class TestProcessS3Batch:
    """Tests for process_s3_batch."""

    def test_process_s3_batch_raises_on_empty_keys(self) -> None:
        """Test process_s3_batch raises ValueError when s3_keys is empty."""
        scorer = RobustZScoreScorer()
        scorer.fit(np.random.randn(50))
        with pytest.raises(ValueError, match="non-empty"):
            process_s3_batch([], scorer, bucket="dummy-bucket")
