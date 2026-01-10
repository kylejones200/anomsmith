"""PCA-based predictive maintenance workflows.

Uses Principal Component Analysis and Mahalanobis distance to track
equipment health and classify health states (healthy, warning, critical).
"""

import logging
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import pandas as pd

from anomsmith.objects.health_state import HealthState, HealthStateView
from anomsmith.primitives.detectors.pca import PCADetector

if TYPE_CHECKING:
    try:
        from timesmith.typing import SeriesLike
    except ImportError:
        SeriesLike = None

logger = logging.getLogger(__name__)


def track_mahalanobis_distance(
    X: Union[np.ndarray, pd.DataFrame],
    detector: PCADetector,
    index: Optional[pd.Index] = None,
) -> pd.Series:
    """Track Mahalanobis distance over time as a single metric.

    Computes Mahalanobis distance from the "normal" center in PCA space
    for each time point. This provides a single metric that can be tracked
    as a time series to monitor equipment health drift.

    Args:
        X: Feature matrix (n_samples, n_features) with sensor readings
        detector: Fitted PCADetector (fitted detector with PCA and mean/covariance computed)
        index: Optional index for the resulting Series

    Returns:
        pandas Series with Mahalanobis distance values, indexed by time

    Examples:
        >>> detector = PCADetector(n_components=3, score_method='mahalanobis')
        >>> detector.fit(X_train)  # Fit on healthy operation data
        >>> distances = track_mahalanobis_distance(X_monitor, detector)
        >>> # Track distance over time to detect drift
    """
    if not detector._fitted:
        raise ValueError("PCADetector must be fitted before tracking Mahalanobis distance.")
    
    # Score using Mahalanobis distance
    if isinstance(X, pd.DataFrame):
        index = X.index
        X_data = X.values
    else:
        X_data = np.asarray(X)
        if index is None:
            index = pd.RangeIndex(start=0, stop=len(X_data))

    # Get Mahalanobis distance scores directly
    # Transform data to PCA space and compute Mahalanobis distance manually
    X_scaled = detector.scaler_.transform(X_data)  # type: ignore
    X_pca = detector.pca_.transform(X_scaled)  # type: ignore
    
    # Compute Mahalanobis distance in PC space
    if detector.mean_ is None or detector.cov_ is None:
        raise ValueError(
            "PCADetector must be fitted with mean_ and cov_ computed. "
            "Ensure detector was fitted properly with mahalanobis support."
        )
    
    # Vectorized Mahalanobis distance calculation
    # Mahalanobis distance: sqrt((x - mu)^T * Sigma^-1 * (x - mu))
    diff = X_pca - detector.mean_  # Shape: (n_samples, n_components)
    
    try:
        inv_cov = np.linalg.inv(detector.cov_)
    except np.linalg.LinAlgError:
        # If covariance is singular, use pseudo-inverse
        inv_cov = np.linalg.pinv(detector.cov_)

    # Vectorized: compute quadratic form for all samples
    # For each sample: diff[i] @ inv_cov @ diff[i].T (scalar)
    # Efficient: compute diff @ inv_cov, then element-wise multiply with diff and sum
    quad_form = (diff @ inv_cov) * diff  # Shape: (n_samples, n_components)
    quad_form = np.sum(quad_form, axis=1)  # Shape: (n_samples,)
    distances = np.sqrt(quad_form)

    logger.debug(
        f"Computed Mahalanobis distances: mean={np.mean(distances):.3f}, "
        f"max={np.max(distances):.3f}, min={np.min(distances):.3f}"
    )
    return pd.Series(distances, index=index, name="mahalanobis_distance")


def classify_health_from_distance(
    distances: Union[pd.Series, np.ndarray, "SeriesLike"],
    healthy_threshold: float,
    warning_threshold: float,
    index: Optional[pd.Index] = None,
) -> HealthStateView:
    """Classify health states from Mahalanobis distance thresholds.

    Maps Mahalanobis distance values to health states:
    - distance <= healthy_threshold: Healthy (0)
    - healthy_threshold < distance <= warning_threshold: Warning (1)
    - distance > warning_threshold: Critical/Distress (2)

    This creates probabilistic zones of "normality" based on distance
    from the healthy center, minimizing false positives by having a
    wide decision space for normal operation.

    Args:
        distances: Mahalanobis distance values (n_samples,)
        healthy_threshold: Distance threshold for Healthy state
        warning_threshold: Distance threshold for Warning state (must be > healthy_threshold)
        index: Optional index for the health states

    Returns:
        HealthStateView with classified health states

    Examples:
        >>> distances = track_mahalanobis_distance(X_monitor, detector)
        >>> # Set thresholds based on training data (e.g., percentiles)
        >>> healthy_threshold = np.percentile(distances, 75)
        >>> warning_threshold = np.percentile(distances, 95)
        >>> health_states = classify_health_from_distance(
        ...     distances, healthy_threshold, warning_threshold
        ... )
    """
    if warning_threshold <= healthy_threshold:
        raise ValueError(
            f"warning_threshold ({warning_threshold}) must be greater than "
            f"healthy_threshold ({healthy_threshold})"
        )

    if isinstance(distances, pd.Series):
        index = distances.index
        dist_values = distances.values
    else:
        dist_values = np.asarray(distances)
        if index is None:
            index = pd.RangeIndex(start=0, stop=len(dist_values))

    # Classify health states based on distance thresholds - vectorized
    states = np.zeros(len(dist_values), dtype=int)
    states[(dist_values > healthy_threshold) & (dist_values <= warning_threshold)] = (
        HealthState.WARNING.value
    )
    states[dist_values > warning_threshold] = HealthState.DISTRESS.value

    logger.debug(
        f"Classified health states from distances: "
        f"Healthy={(states == HealthState.HEALTHY.value).sum()}, "
        f"Warning={(states == HealthState.WARNING.value).sum()}, "
        f"Distress={(states == HealthState.DISTRESS.value).sum()}"
    )
    return HealthStateView(index=index, states=states)


def assess_health_with_pca(
    X: Union[np.ndarray, pd.DataFrame],
    detector: PCADetector,
    healthy_threshold: float,
    warning_threshold: float,
    index: Optional[pd.Index] = None,
) -> pd.DataFrame:
    """Assess equipment health using PCA and Mahalanobis distance.

    Complete workflow for PCA-based predictive maintenance:
    1. Compute Mahalanobis distance from healthy center
    2. Classify health states based on distance thresholds
    3. Return results as a DataFrame for easy tracking

    Args:
        X: Feature matrix (n_samples, n_features) with sensor readings
        detector: Fitted PCADetector (must use score_method='mahalanobis')
        healthy_threshold: Distance threshold for Healthy state
        warning_threshold: Distance threshold for Warning state
        index: Optional index for the results

    Returns:
        DataFrame with columns: 'mahalanobis_distance', 'health_state'

    Examples:
        >>> detector = PCADetector(n_components=3, score_method='mahalanobis')
        >>> detector.fit(X_train)  # Fit on healthy operation data
        >>> # Set thresholds based on training data
        >>> healthy_threshold = np.percentile(detector.score(X_train).scores, 75)
        >>> warning_threshold = np.percentile(detector.score(X_train).scores, 95)
        >>> health_df = assess_health_with_pca(
        ...     X_monitor, detector, healthy_threshold, warning_threshold
        ... )
        >>> # Track health over time
        >>> critical_units = health_df[health_df['health_state'] == 2]
    """
    # Track Mahalanobis distance
    distances = track_mahalanobis_distance(X, detector, index=index)

    # Classify health states
    health_states = classify_health_from_distance(
        distances, healthy_threshold, warning_threshold, index=distances.index
    )

    # Combine into DataFrame
    result_df = pd.DataFrame(
        {
            "mahalanobis_distance": distances.values,
            "health_state": health_states.states,
        },
        index=distances.index,
    )

    logger.info(
        f"Assessed health for {len(result_df)} samples: "
        f"Healthy={(result_df['health_state'] == HealthState.HEALTHY.value).sum()}, "
        f"Warning={(result_df['health_state'] == HealthState.WARNING.value).sum()}, "
        f"Critical={(result_df['health_state'] == HealthState.DISTRESS.value).sum()}"
    )
    return result_df


def compute_pca_health_thresholds(
    X_train: Union[np.ndarray, pd.DataFrame],
    detector: PCADetector,
    healthy_percentile: float = 75.0,
    warning_percentile: float = 95.0,
) -> tuple[float, float]:
    """Compute health state thresholds from training data.

    Determines distance thresholds for health state classification based on
    percentiles of Mahalanobis distances in the training (healthy) data.

    Args:
        X_train: Training data (should be healthy operation data)
        detector: Fitted PCADetector (must use score_method='mahalanobis')
        healthy_percentile: Percentile for healthy threshold (default 75.0)
        warning_percentile: Percentile for warning threshold (default 95.0)

    Returns:
        Tuple of (healthy_threshold, warning_threshold)

    Examples:
        >>> detector = PCADetector(n_components=3, score_method='mahalanobis')
        >>> detector.fit(X_train)  # Fit on healthy operation data
        >>> healthy_threshold, warning_threshold = compute_pca_health_thresholds(
        ...     X_train, detector, healthy_percentile=75, warning_percentile=95
        ... )
    """
    if healthy_percentile >= warning_percentile:
        raise ValueError(
            f"healthy_percentile ({healthy_percentile}) must be less than "
            f"warning_percentile ({warning_percentile})"
        )

    # Compute distances on training data
    distances = track_mahalanobis_distance(X_train, detector)

    # Compute percentiles
    healthy_threshold = np.percentile(distances, healthy_percentile)
    warning_threshold = np.percentile(distances, warning_percentile)

    logger.info(
        f"Computed health thresholds: healthy={healthy_threshold:.3f} "
        f"(percentile {healthy_percentile}), warning={warning_threshold:.3f} "
        f"(percentile {warning_percentile})"
    )
    return float(healthy_threshold), float(warning_threshold)

