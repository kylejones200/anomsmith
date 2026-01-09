"""Public workflow functions for anomaly detection."""

import logging
from typing import TYPE_CHECKING, Any, Union

import numpy as np
import pandas as pd

from anomsmith.primitives.base import BaseDetector, BaseScorer
from anomsmith.primitives.thresholding import ThresholdRule, apply_threshold
from anomsmith.tasks.detect import run_detection, run_scoring

if TYPE_CHECKING:
    try:
        from timesmith.typing import SeriesLike
    except ImportError:
        SeriesLike = None

logger = logging.getLogger(__name__)


def score_anomalies(
    y: Union[pd.Series, np.ndarray, "SeriesLike"],
    scorer: BaseScorer,
) -> pd.Series:
    """Score anomalies in a time series.

    Args:
        y: Time series to score
        scorer: BaseScorer instance

    Returns:
        pandas Series of anomaly scores with same index as y
    """
    logger.info(f"Scoring anomalies with {scorer.__class__.__name__}")
    score_view = run_scoring(y, scorer)
    return pd.Series(score_view.scores, index=score_view.index)


def detect_anomalies(
    y: Union[pd.Series, np.ndarray, "SeriesLike"],
    detector: BaseDetector | BaseScorer,
    threshold_rule: ThresholdRule,
) -> pd.DataFrame:
    """Detect anomalies in a time series.

    Args:
        y: Time series to detect anomalies in
        detector: BaseDetector or BaseScorer instance
        threshold_rule: ThresholdRule to apply

    Returns:
        pandas DataFrame with 'score' and 'flag' columns, indexed by y's index
    """
    logger.info(
        f"Detecting anomalies with {detector.__class__.__name__} "
        f"and threshold {threshold_rule}"
    )
    label_view, score_view = run_detection(y, detector)

    # Apply threshold if scorer was provided
    if isinstance(detector, BaseScorer):
        label_view = apply_threshold(score_view, threshold_rule)

    result = pd.DataFrame(
        {
            "score": score_view.scores,
            "flag": label_view.labels,
        },
        index=score_view.index,
    )
    return result


def sweep_thresholds(
    y: Union[pd.Series, np.ndarray, "SeriesLike"],
    scorer: BaseScorer,
    threshold_values: list[float] | np.ndarray,
    labels: Union[pd.Series, np.ndarray, "SeriesLike", None] = None,
) -> pd.DataFrame:
    """Evaluate multiple threshold values and return metrics.

    Args:
        y: Time series to score
        scorer: BaseScorer instance
        threshold_values: List of threshold values to evaluate
        labels: Optional ground truth labels

    Returns:
        pandas DataFrame with columns: threshold, precision, recall, f1
        (metrics are NaN if labels not provided)
    """
    logger.info(f"Sweeping {len(threshold_values)} threshold values")
    score_view = run_scoring(y, scorer)

    # Convert threshold_values to numpy array for vectorized operations
    thresholds = np.asarray(threshold_values)
    scores = score_view.scores

    # Pre-compute aligned labels once if provided
    if labels is not None:
        aligned_labels = labels.reindex(score_view.index, fill_value=0).values
        aligned_labels = (aligned_labels != 0).astype(int)

        from anomsmith.workflows.eval.metrics import (
            compute_f1,
            compute_precision,
            compute_recall,
        )

        # Vectorized: apply all thresholds at once using broadcasting
        # Shape: (n_thresholds, n_samples)
        # For each threshold, create binary predictions
        predictions = (scores[:, np.newaxis] >= thresholds[np.newaxis, :]).astype(int)

        # Compute metrics for all thresholds at once
        # True positives: predictions AND aligned_labels (both 1)
        # Shape: (n_samples, n_thresholds)
        tp = ((aligned_labels[:, np.newaxis] == 1) & (predictions == 1)).sum(axis=0)
        # Predicted positives: sum of predictions per threshold
        pred_pos = predictions.sum(axis=0)
        # Actual positives: sum of true labels
        actual_pos = aligned_labels.sum()

        # Vectorized precision, recall, f1 for all thresholds
        precision = np.where(pred_pos > 0, tp / pred_pos, 0.0)
        recall = np.where(actual_pos > 0, tp / actual_pos, 0.0)
        f1 = np.where(
            precision + recall > 0, 2 * (precision * recall) / (precision + recall), 0.0
        )
    else:
        precision = np.full(len(thresholds), np.nan)
        recall = np.full(len(thresholds), np.nan)
        f1 = np.full(len(thresholds), np.nan)

    # Build results DataFrame directly (vectorized)
    return pd.DataFrame(
        {
            "threshold": thresholds,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )


def report_detection(
    y: Union[pd.Series, np.ndarray, "SeriesLike"],
    detector: BaseDetector | BaseScorer,
    threshold_rule: ThresholdRule,
) -> dict[str, Any]:
    """Generate detection report with summary stats.

    Args:
        y: Time series that was analyzed
        detector: BaseDetector or BaseScorer instance used
        threshold_rule: ThresholdRule applied

    Returns:
        Dictionary with summary stats and top anomaly timestamps
    """
    logger.info("Generating detection report")
    result_df = detect_anomalies(y, detector, threshold_rule)

    n_anomalies = result_df["flag"].sum()
    n_total = len(result_df)
    anomaly_rate = n_anomalies / n_total if n_total > 0 else 0.0

    # Top anomalies by score
    top_anomalies = (
        result_df[result_df["flag"] == 1]
        .nlargest(10, "score")
        .index.tolist()
    )

    report = {
        "n_anomalies": int(n_anomalies),
        "n_total": int(n_total),
        "anomaly_rate": float(anomaly_rate),
        "mean_score": float(result_df["score"].mean()),
        "max_score": float(result_df["score"].max()),
        "min_score": float(result_df["score"].min()),
        "top_anomaly_timestamps": [str(ts) for ts in top_anomalies],
    }

    return report

