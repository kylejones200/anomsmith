"""Evaluation metrics for anomaly detection."""

import numpy as np


def compute_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute precision score.

    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels

    Returns:
        Precision score (0-1)
    """
    if y_pred.sum() == 0:
        return 0.0
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    return tp / y_pred.sum()


def compute_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute recall score.

    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels

    Returns:
        Recall score (0-1)
    """
    if y_true.sum() == 0:
        return 0.0
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    return tp / y_true.sum()


def compute_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute F1 score.

    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels

    Returns:
        F1 score (0-1)
    """
    precision = compute_precision(y_true, y_pred)
    recall = compute_recall(y_true, y_pred)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def average_run_length(labels: np.ndarray) -> float:
    """Compute average run length of anomaly segments.

    Args:
        labels: Binary labels (1 = anomaly, 0 = normal)

    Returns:
        Average length of consecutive anomaly segments
    """
    if labels.sum() == 0:
        return 0.0

    # Find runs of 1s
    diff = np.diff(np.concatenate(([0], labels, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    if len(starts) == 0:
        return 0.0

    run_lengths = ends - starts
    return float(run_lengths.mean())


