"""Tests for anomsmith.platform evaluation helpers."""

import numpy as np
import pandas as pd

from anomsmith.platform.evaluation import (
    calculate_confusion_matrix_metrics,
    calculate_lead_time,
    compare_detectors,
    evaluate_detector,
)
from anomsmith.primitives.detectors.ml import IsolationForestDetector
from anomsmith.primitives.detectors.pca import PCADetector


class TestPlatformEvaluation:
    def test_evaluate_detector(self) -> None:
        detector = IsolationForestDetector(random_state=42)
        X = np.random.randn(100, 3)
        y_true = np.zeros(100)
        y_true[10:15] = 1

        detector.fit(X)
        metrics = evaluate_detector(detector, X, y_true)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert all(
            0 <= float(v) <= 1
            for v in [
                metrics["accuracy"],
                metrics["precision"],
                metrics["recall"],
                metrics["f1"],
            ]
        )

    def test_evaluate_detector_with_scores(self) -> None:
        detector = IsolationForestDetector(random_state=42)
        X = np.random.randn(100, 2)
        y_true = np.zeros(100)
        y_true[10:15] = 1

        detector.fit(X)
        scores = np.asarray(detector.score(X).scores, dtype=float).ravel()
        metrics = evaluate_detector(detector, X, y_true, scores=scores)

        assert "roc_auc" in metrics
        assert 0 <= float(metrics["roc_auc"]) <= 1

    def test_calculate_lead_time(self) -> None:
        predictions = np.zeros(20)
        predictions[[5, 10, 15]] = 1

        y_true = np.zeros(20)
        y_true[[8, 12, 18]] = 1

        timestamps = np.arange(20)
        metrics = calculate_lead_time(predictions, y_true, timestamps)

        assert "mean_lead_time" in metrics
        assert "median_lead_time" in metrics
        assert "early_detections" in metrics
        assert "late_detections" in metrics

    def test_calculate_lead_time_no_events(self) -> None:
        predictions = np.zeros(100)
        y_true = np.zeros(100)

        metrics = calculate_lead_time(predictions, y_true)

        assert metrics["mean_lead_time"] == 0.0
        assert metrics["early_detections"] == 0

    def test_calculate_confusion_matrix_metrics(self) -> None:
        predictions = np.array([1, 1, 1, 0, 1, 0])
        y_true = np.array([1, 0, 1, 0, 0, 1])

        metrics = calculate_confusion_matrix_metrics(predictions, y_true)

        assert "true_positives" in metrics
        assert "true_negatives" in metrics
        assert "false_positives" in metrics
        assert "false_negatives" in metrics
        assert all(v >= 0 for v in metrics.values())

    def test_compare_detectors(self) -> None:
        X = np.random.randn(100, 5)
        y_true = np.zeros(100)
        y_true[10:20] = 1

        detectors: dict[str, PCADetector | IsolationForestDetector] = {
            "PCA": PCADetector(n_components=0.95, random_state=42),
            "IF": IsolationForestDetector(random_state=43),
        }

        detectors["PCA"].fit(X)
        detectors["IF"].fit(X)

        comparison_df = compare_detectors(detectors, X, y_true)

        assert isinstance(comparison_df, pd.DataFrame)
        assert len(comparison_df) == len(detectors)
        assert "detector" in comparison_df.columns
        assert "accuracy" in comparison_df.columns
        assert "precision" in comparison_df.columns
        assert "recall" in comparison_df.columns
        assert "f1" in comparison_df.columns
