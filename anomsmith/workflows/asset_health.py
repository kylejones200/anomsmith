"""Asset health workflows for grid assets and industrial equipment.

Combines classification and anomaly detection to assess asset health and
prioritize maintenance actions.
"""

import logging
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import pandas as pd

from anomsmith.objects.health_state import HealthState, HealthStateView
from anomsmith.primitives.classifiers.failure_risk import FailureRiskClassifier
from anomsmith.primitives.detectors.ml import IsolationForestDetector

if TYPE_CHECKING:
    try:
        from timesmith.typing import SeriesLike
    except ImportError:
        SeriesLike = None

logger = logging.getLogger(__name__)


def assess_asset_health(
    sensor_data: pd.DataFrame,
    asset_ids: Optional[pd.Series] = None,
    feature_cols: Optional[list[str]] = None,
    failure_labels: Optional[pd.Series | np.ndarray] = None,
    use_classification: bool = True,
    use_anomaly_detection: bool = True,
    contamination: float = 0.05,
    n_estimators: int = 100,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Assess asset health using classification and anomaly detection.

    Combines failure risk classification with anomaly detection to provide
    comprehensive asset health assessment. Results can be used to prioritize
    maintenance actions.

    Args:
        sensor_data: DataFrame with sensor readings (columns are features, rows are assets)
        asset_ids: Optional Series of asset IDs (defaults to sensor_data index)
        feature_cols: Optional list of feature column names (defaults to all numeric columns)
        failure_labels: Optional binary labels for training classifier (1 = failure, 0 = healthy)
        use_classification: Whether to use failure risk classification (default True)
        use_anomaly_detection: Whether to use anomaly detection (default True)
        contamination: Expected proportion of anomalies (default 0.05)
        n_estimators: Number of trees for Random Forest (default 100)
        random_state: Random state for reproducibility

    Returns:
        DataFrame with columns:
        - asset_id: Asset identifier
        - failure_risk: Probability of failure (if classification used)
        - health_state: Predicted health state (0=Healthy, 1=Warning, 2=Distress)
        - is_anomaly: Binary anomaly flag (if anomaly detection used)
        - anomaly_score: Anomaly score (if anomaly detection used)
        - combined_risk: Combined risk score (higher = more urgent)

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> sensor_data = pd.DataFrame({
        ...     'temperature': [60, 65, 70, 80],
        ...     'vibration': [0.2, 0.25, 0.3, 0.4],
        ...     'pressure': [25, 24, 23, 20]
        ... })
        >>> result = assess_asset_health(sensor_data)
        >>> result.head()
    """
    if feature_cols is None:
        # Use all numeric columns
        feature_cols = sensor_data.select_dtypes(include=[np.number]).columns.tolist()

    if asset_ids is None:
        asset_ids = sensor_data.index

    results = {"asset_id": asset_ids}
    X = sensor_data[feature_cols]

    # Classification-based failure risk prediction
    failure_risks = np.zeros(len(X))
    health_states = np.zeros(len(X), dtype=int)

    if use_classification and failure_labels is not None:
        classifier = FailureRiskClassifier(n_estimators=n_estimators, random_state=random_state)
        classifier.fit(X, failure_labels)
        probas = classifier.predict_proba(X)
        if probas.shape[1] > 1:
            failure_risks = probas[:, 1]  # Probability of failure class
        else:
            failure_risks = probas[:, 0]

        health_state_view = classifier.predict_health_states(
            X, index=pd.RangeIndex(len(X)), risk_threshold=0.5, distress_threshold=0.8
        )
        health_states = health_state_view.states
    elif use_classification:
        logger.warning(
            "Classification requested but no failure_labels provided. "
            "Skipping classification-based assessment."
        )
        use_classification = False

    results["failure_risk"] = failure_risks if use_classification else np.nan
    results["health_state"] = health_states if use_classification else np.zeros(len(X), dtype=int)

    # Anomaly detection
    anomaly_flags = np.zeros(len(X), dtype=int)
    anomaly_scores = np.zeros(len(X))

    if use_anomaly_detection:
        # Use Isolation Forest for multivariate anomaly detection
        detector = IsolationForestDetector(contamination=contamination, random_state=random_state)

        # Fit on sensor data (treating each row as a sample)
        # For Isolation Forest, we need to fit on all data
        detector.fit(X.values)

        # Score each asset
        score_view = detector.score(X.values)
        anomaly_scores = score_view.scores

        # Predict anomalies
        label_view = detector.predict(X.values)
        anomaly_flags = label_view.labels

    results["is_anomaly"] = anomaly_flags
    results["anomaly_score"] = anomaly_scores if use_anomaly_detection else np.nan

    # Combined risk score (normalize and combine both signals)
    combined_risks = np.zeros(len(X))
    if use_classification and use_anomaly_detection:
        # Normalize failure risk to [0, 1]
        if failure_risks.max() > failure_risks.min():
            norm_failure_risk = (failure_risks - failure_risks.min()) / (
                failure_risks.max() - failure_risks.min()
            )
        else:
            norm_failure_risk = failure_risks

        # Normalize anomaly scores to [0, 1]
        if anomaly_scores.max() > anomaly_scores.min():
            norm_anomaly_score = (anomaly_scores - anomaly_scores.min()) / (
                anomaly_scores.max() - anomaly_scores.min()
            )
        else:
            norm_anomaly_score = anomaly_scores

        # Combine (weighted average: 60% classification, 40% anomaly)
        combined_risks = 0.6 * norm_failure_risk + 0.4 * norm_anomaly_score
    elif use_classification:
        # Normalize failure risk
        if failure_risks.max() > failure_risks.min():
            combined_risks = (failure_risks - failure_risks.min()) / (
                failure_risks.max() - failure_risks.min()
            )
        else:
            combined_risks = failure_risks
    elif use_anomaly_detection:
        # Normalize anomaly scores
        if anomaly_scores.max() > anomaly_scores.min():
            combined_risks = (anomaly_scores - anomaly_scores.min()) / (
                anomaly_scores.max() - anomaly_scores.min()
            )
        else:
            combined_risks = anomaly_scores

    results["combined_risk"] = combined_risks

    # Create DataFrame with results
    result_df = pd.DataFrame(results)
    # Sort by combined risk (highest first) for prioritization
    result_df = result_df.sort_values("combined_risk", ascending=False).reset_index(drop=True)

    return result_df


def rank_assets_by_risk(
    asset_health: pd.DataFrame,
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    """Rank assets by combined risk score.

    Args:
        asset_health: DataFrame from assess_asset_health()
        top_n: Optional number of top assets to return (default None = all)

    Returns:
        DataFrame ranked by combined_risk (highest first)
    """
    ranked = asset_health.sort_values("combined_risk", ascending=False)
    if top_n is not None:
        ranked = ranked.head(top_n)
    return ranked

