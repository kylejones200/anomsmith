"""Model monitoring and performance tracking utilities.

Designed for integration with monitoring systems like AWS CloudWatch,
Azure Monitor, or GCP Cloud Monitoring.
"""

import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import pandas as pd

from anomsmith.objects.views import LabelView, ScoreView
from anomsmith.workflows.eval.metrics import (
    average_run_length,
    compute_f1,
    compute_precision,
    compute_recall,
)

if TYPE_CHECKING:
    try:
        from timesmith.typing import SeriesLike
    except ImportError:
        SeriesLike = None

logger = logging.getLogger(__name__)


def compute_performance_metrics(
    true_labels: Union[np.ndarray, pd.Series],
    predicted_labels: Union[np.ndarray, pd.Series],
    scores: Optional[Union[np.ndarray, pd.Series]] = None,
) -> dict[str, float]:
    """Compute comprehensive performance metrics for model monitoring.

    Returns metrics suitable for CloudWatch, Prometheus, or similar
    monitoring systems.

    Args:
        true_labels: Ground truth binary labels (0 = normal, 1 = anomaly)
        predicted_labels: Predicted binary labels
        scores: Optional anomaly scores (for threshold-independent metrics)

    Returns:
        Dictionary with metrics:
        - precision: Precision score
        - recall: Recall score
        - f1: F1 score
        - true_positives: Number of true positives
        - false_positives: Number of false positives
        - false_negatives: Number of false negatives
        - true_negatives: Number of true negatives
        - anomaly_rate: Proportion of predicted anomalies
        - avg_run_length: Average length of anomaly runs (if scores provided)

    Examples:
        >>> metrics = compute_performance_metrics(true_labels, pred_labels, scores)
        >>> # Send to CloudWatch
        >>> cloudwatch.put_metric_data(
        ...     Namespace="AnomalyDetection",
        ...     MetricData=[{"MetricName": "F1", "Value": metrics["f1"]}]
        ... )
    """
    # Convert to numpy arrays
    if isinstance(true_labels, pd.Series):
        y_true = true_labels.values
    else:
        y_true = np.asarray(true_labels)

    if isinstance(predicted_labels, pd.Series):
        y_pred = predicted_labels.values
    else:
        y_pred = np.asarray(predicted_labels)

    # Compute confusion matrix components
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))

    # Compute metrics
    precision = compute_precision(y_true, y_pred)
    recall = compute_recall(y_true, y_pred)
    f1 = compute_f1(y_true, y_pred)
    anomaly_rate = y_pred.mean()

    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "anomaly_rate": float(anomaly_rate),
    }

    # Add average run length if scores provided
    if scores is not None:
        if isinstance(scores, pd.Series):
            scores_array = scores.values
        else:
            scores_array = np.asarray(scores)

        # Use predictions to compute run length
        avg_run_len = average_run_length(y_pred)
        metrics["avg_run_length"] = float(avg_run_len)

        # Add score statistics
        metrics["mean_score"] = float(np.mean(scores_array))
        metrics["max_score"] = float(np.max(scores_array))
        metrics["min_score"] = float(np.min(scores_array))
        metrics["score_std"] = float(np.std(scores_array))

    return metrics


def detect_concept_drift(
    recent_scores: Union[np.ndarray, pd.Series],
    historical_scores: Union[np.ndarray, pd.Series],
    threshold: float = 2.0,
) -> dict[str, Any]:
    """Detect concept drift in model scores.

    Compares recent score distribution to historical distribution
    using statistical tests. Useful for triggering model retraining.

    Args:
        recent_scores: Recent anomaly scores (last N samples)
        historical_scores: Historical anomaly scores (training/baseline period)
        threshold: Threshold for drift detection (default 2.0 std devs)

    Returns:
        Dictionary with drift detection results:
        - drift_detected: Boolean indicating if drift detected
        - recent_mean: Mean of recent scores
        - historical_mean: Mean of historical scores
        - drift_magnitude: Difference in means normalized by historical std
        - ks_statistic: Kolmogorov-Smirnov test statistic (if scipy available)
        - p_value: P-value from KS test (if scipy available)

    Examples:
        >>> drift_info = detect_concept_drift(
        ...     recent_scores=model_scores[-1000:],
        ...     historical_scores=baseline_scores
        ... )
        >>> if drift_info["drift_detected"]:
        ...     trigger_model_retraining()
    """
    # Convert to numpy arrays
    if isinstance(recent_scores, pd.Series):
        recent = recent_scores.values
    else:
        recent = np.asarray(recent_scores)

    if isinstance(historical_scores, pd.Series):
        historical = historical_scores.values
    else:
        historical = np.asarray(historical_scores)

    # Compute statistics
    recent_mean = np.mean(recent)
    historical_mean = np.mean(historical)
    historical_std = np.std(historical)

    # Normalized drift magnitude
    if historical_std > 0:
        drift_magnitude = abs(recent_mean - historical_mean) / historical_std
    else:
        drift_magnitude = 0.0

    drift_detected = drift_magnitude > threshold

    result = {
        "drift_detected": bool(drift_detected),
        "recent_mean": float(recent_mean),
        "historical_mean": float(historical_mean),
        "drift_magnitude": float(drift_magnitude),
        "threshold": float(threshold),
    }

    # Try to compute KS test if scipy available
    try:
        from scipy import stats

        ks_statistic, p_value = stats.ks_2samp(historical, recent)
        result["ks_statistic"] = float(ks_statistic)
        result["p_value"] = float(p_value)
    except ImportError:
        logger.debug("scipy not available, skipping KS test")

    return result


def aggregate_metrics_for_cloudwatch(
    metrics_list: list[dict[str, float]],
    namespace: str = "AnomalyDetection",
    model_name: Optional[str] = None,
    timestamp: Optional[datetime] = None,
) -> list[dict[str, Any]]:
    """Format metrics for AWS CloudWatch PutMetricData API.

    Aggregates multiple metric dictionaries into CloudWatch format.

    Args:
        metrics_list: List of metric dictionaries from compute_performance_metrics
        namespace: CloudWatch namespace (default "AnomalyDetection")
        model_name: Optional model name for dimension
        timestamp: Optional timestamp (default: now)

    Returns:
        List of CloudWatch metric data dictionaries

    Examples:
        >>> metrics = [compute_performance_metrics(y1, pred1), ...]
        >>> cw_metrics = aggregate_metrics_for_cloudwatch(metrics, model_name="IsolationForest")
        >>> cloudwatch.put_metric_data(
        ...     Namespace="AnomalyDetection",
        ...     MetricData=cw_metrics
        ... )
    """
    if timestamp is None:
        timestamp = datetime.utcnow()

    cloudwatch_metrics = []

    # Aggregate across all metrics
    aggregated = {}
    for metrics in metrics_list:
        for key, value in metrics.items():
            if key not in aggregated:
                aggregated[key] = []
            aggregated[key].append(value)

    # Create CloudWatch metric for each aggregated metric
    for metric_name, values in aggregated.items():
        metric_data = {
            "MetricName": metric_name,
            "Timestamp": timestamp,
            "Value": float(np.mean(values)),
            "Unit": "None",
        }

        if model_name:
            metric_data["Dimensions"] = [{"Name": "ModelName", "Value": model_name}]

        cloudwatch_metrics.append(metric_data)

        # Add statistics
        if len(values) > 1:
            metric_data_min = {
                "MetricName": f"{metric_name}_Min",
                "Timestamp": timestamp,
                "Value": float(np.min(values)),
                "Unit": "None",
            }
            metric_data_max = {
                "MetricName": f"{metric_name}_Max",
                "Timestamp": timestamp,
                "Value": float(np.max(values)),
                "Unit": "None",
            }
            if model_name:
                metric_data_min["Dimensions"] = [{"Name": "ModelName", "Value": model_name}]
                metric_data_max["Dimensions"] = [{"Name": "ModelName", "Value": model_name}]

            cloudwatch_metrics.extend([metric_data_min, metric_data_max])

    return cloudwatch_metrics


class ModelPerformanceTracker:
    """Track model performance over time for monitoring and alerting.

    Maintains a rolling window of performance metrics and can detect
    degradation or drift.

    Attributes:
        window_size: Number of recent predictions to keep in window
        metrics_history: DataFrame with historical metrics
    """

    def __init__(self, window_size: int = 1000, model_name: Optional[str] = None) -> None:
        """Initialize performance tracker.

        Args:
            window_size: Number of recent samples to track
            model_name: Optional model name for identification
        """
        self.window_size = window_size
        self.model_name = model_name
        self.metrics_history: pd.DataFrame = pd.DataFrame()
        self.score_history: list[float] = []
        self.label_history: list[int] = []
        self.true_label_history: list[int] = []

    def update(
        self,
        scores: Union[np.ndarray, pd.Series],
        predicted_labels: Union[np.ndarray, pd.Series],
        true_labels: Optional[Union[np.ndarray, pd.Series]] = None,
        timestamp: Optional[datetime] = None,
    ) -> dict[str, float]:
        """Update tracker with new predictions.

        Args:
            scores: Anomaly scores
            predicted_labels: Predicted binary labels
            true_labels: Optional ground truth labels
            timestamp: Optional timestamp for this update

        Returns:
            Current performance metrics
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        # Convert to arrays
        if isinstance(scores, pd.Series):
            scores_array = scores.values
        else:
            scores_array = np.asarray(scores)

        if isinstance(predicted_labels, pd.Series):
            labels_array = predicted_labels.values
        else:
            labels_array = np.asarray(predicted_labels)

        # Update history (keep only recent window)
        self.score_history.extend(scores_array.tolist())
        self.label_history.extend(labels_array.tolist())

        if len(self.score_history) > self.window_size:
            self.score_history = self.score_history[-self.window_size :]
            self.label_history = self.label_history[-self.window_size :]

        # Compute metrics if true labels provided
        if true_labels is not None:
            if isinstance(true_labels, pd.Series):
                true_array = true_labels.values
            else:
                true_array = np.asarray(true_labels)

            self.true_label_history.extend(true_array.tolist())
            if len(self.true_label_history) > self.window_size:
                self.true_label_history = self.true_label_history[-self.window_size :]

            metrics = compute_performance_metrics(
                np.array(self.true_label_history),
                np.array(self.label_history),
                np.array(self.score_history),
            )
        else:
            # Just track score statistics
            metrics = {
                "mean_score": float(np.mean(self.score_history)),
                "max_score": float(np.max(self.score_history)),
                "min_score": float(np.min(self.score_history)),
                "score_std": float(np.std(self.score_history)),
                "anomaly_rate": float(np.mean(self.label_history)),
            }

        # Add timestamp
        metrics["timestamp"] = timestamp

        # Update history DataFrame
        new_row = pd.DataFrame([metrics])
        self.metrics_history = pd.concat([self.metrics_history, new_row], ignore_index=True)

        # Keep only recent window in DataFrame
        if len(self.metrics_history) > self.window_size:
            self.metrics_history = self.metrics_history.tail(self.window_size).reset_index(drop=True)

        return metrics

    def get_current_metrics(self) -> dict[str, float]:
        """Get current performance metrics.

        Returns:
            Dictionary with latest metrics
        """
        if self.metrics_history.empty:
            return {}

        return self.metrics_history.iloc[-1].to_dict()

    def detect_degradation(self, baseline_metrics: dict[str, float], threshold: float = 0.1) -> bool:
        """Detect if performance has degraded compared to baseline.

        Args:
            baseline_metrics: Baseline metrics (e.g., from training)
            threshold: Relative degradation threshold (default 0.1 = 10%)

        Returns:
            True if degradation detected
        """
        current = self.get_current_metrics()

        if "f1" in baseline_metrics and "f1" in current:
            baseline_f1 = baseline_metrics["f1"]
            current_f1 = current["f1"]
            degradation = (baseline_f1 - current_f1) / baseline_f1 if baseline_f1 > 0 else 0.0
            return degradation > threshold

        return False

