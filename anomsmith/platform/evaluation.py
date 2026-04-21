"""Evaluation helpers aligned with anomsmith detectors (LabelView / ScoreView).

Ported from the former *Anomaly Detection Toolkit* evaluation module. Metrics assume
binary ground truth where ``1`` indicates an anomaly event and detector labels use
anomsmith's convention (``1`` = anomaly, ``0`` = normal).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from anomsmith.primitives.base import BaseDetector


def _as_tabular(y: np.ndarray | pd.DataFrame) -> np.ndarray | pd.DataFrame:
    arr = np.asarray(y)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def _prediction_labels(
    detector: BaseDetector, X: np.ndarray | pd.DataFrame
) -> np.ndarray:
    y = _as_tabular(X)
    lv = detector.predict(y)
    return np.asarray(lv.labels).ravel()


def _anomaly_scores(detector: BaseDetector, X: np.ndarray | pd.DataFrame) -> np.ndarray:
    y = _as_tabular(X)
    sv = detector.score(y)
    return np.asarray(sv.scores).ravel()


def calculate_lead_time(
    predictions: np.ndarray,
    true_labels: np.ndarray,
    timestamps: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Lead time between anomaly detections and failure events.

    Args:
        predictions: Detector labels (``1`` = anomaly, ``0`` = normal).
        true_labels: Ground truth (``1`` = anomaly, ``0`` = normal).
        timestamps: Optional timestamps aligned to predictions.

    Returns:
        Dictionary with mean/median/min/max lead time and early/late detection counts.
    """
    if timestamps is None:
        timestamps = np.arange(len(predictions))

    pred_binary = (predictions == 1).astype(int)
    true_binary = (
        (true_labels == 1).astype(int) if np.any(true_labels == 1) else true_labels
    )

    event_indices = np.where(np.diff(true_binary) == 1)[0] + 1

    if len(event_indices) == 0:
        return {
            "mean_lead_time": 0.0,
            "median_lead_time": 0.0,
            "min_lead_time": 0.0,
            "max_lead_time": 0.0,
            "early_detections": 0,
            "late_detections": 0,
        }

    lead_times: list[float] = []
    early_count = 0
    late_count = 0

    for event_idx in event_indices:
        detections_before = np.where(pred_binary[: event_idx + 1] == 1)[0]

        if len(detections_before) > 0:
            first_detection_idx = detections_before[-1]
            lead_time = float(timestamps[event_idx] - timestamps[first_detection_idx])

            if lead_time > 0:
                lead_times.append(lead_time)
                early_count += 1
            elif lead_time < 0:
                late_count += 1
                lead_times.append(lead_time)

    if len(lead_times) == 0:
        return {
            "mean_lead_time": 0.0,
            "median_lead_time": 0.0,
            "min_lead_time": 0.0,
            "max_lead_time": 0.0,
            "early_detections": 0,
            "late_detections": 0,
        }

    lead_arr = np.array(lead_times)

    return {
        "mean_lead_time": (
            float(np.mean(lead_arr[lead_arr > 0])) if np.any(lead_arr > 0) else 0.0
        ),
        "median_lead_time": (
            float(np.median(lead_arr[lead_arr > 0])) if np.any(lead_arr > 0) else 0.0
        ),
        "min_lead_time": float(np.min(lead_arr[lead_arr > 0]))
        if np.any(lead_arr > 0)
        else 0.0,
        "max_lead_time": float(np.max(lead_arr[lead_arr > 0]))
        if np.any(lead_arr > 0)
        else 0.0,
        "early_detections": int(early_count),
        "late_detections": int(late_count),
    }


def evaluate_detector(
    detector: BaseDetector,
    X: np.ndarray | pd.DataFrame,
    y_true: np.ndarray,
    scores: np.ndarray | None = None,
    timestamps: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Evaluate a fitted anomsmith detector on tabular test data."""
    predictions = _prediction_labels(detector, X)

    pred_binary = (predictions == 1).astype(int)
    true_binary = (
        (y_true == 1).astype(int) if np.any(y_true == 1) else y_true.astype(int)
    )

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(true_binary, pred_binary)),
        "precision": float(precision_score(true_binary, pred_binary, zero_division=0)),
        "recall": float(recall_score(true_binary, pred_binary, zero_division=0)),
        "f1": float(f1_score(true_binary, pred_binary, zero_division=0)),
    }

    if scores is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(true_binary, scores))
        except ValueError:
            metrics["roc_auc"] = 0.0

    if timestamps is not None:
        lead_time_metrics = calculate_lead_time(predictions, y_true, timestamps)
        metrics.update(lead_time_metrics)

    return metrics


def compare_detectors(
    detectors: dict[str, BaseDetector],
    X: np.ndarray | pd.DataFrame,
    y_true: np.ndarray,
    timestamps: np.ndarray | None = None,
) -> pd.DataFrame:
    """Compare multiple fitted detectors side-by-side."""
    results: list[dict[str, Any]] = []

    for name, detector in detectors.items():
        scores = _anomaly_scores(detector, X)
        metrics = evaluate_detector(
            detector, X, y_true, scores=scores, timestamps=timestamps
        )
        row: dict[str, Any] = {**metrics, "detector": name}
        results.append(row)

    return pd.DataFrame(results)


def calculate_confusion_matrix_metrics(
    predictions: np.ndarray, y_true: np.ndarray
) -> dict[str, int]:
    """Confusion matrix counts with ``1`` = predicted / true anomaly."""
    pred_binary = (predictions == 1).astype(int)
    true_binary = (
        (y_true == 1).astype(int) if np.any(y_true == 1) else y_true.astype(int)
    )

    tp = int(np.sum((pred_binary == 1) & (true_binary == 1)))
    tn = int(np.sum((pred_binary == 0) & (true_binary == 0)))
    fp = int(np.sum((pred_binary == 1) & (true_binary == 0)))
    fn = int(np.sum((pred_binary == 0) & (true_binary == 1)))

    return {
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
    }
