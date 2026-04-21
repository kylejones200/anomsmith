"""Predictive maintenance features for time series data.

This module provides tools for predictive maintenance including:
- Feature extraction (rolling statistics, change detection, frequency domain)
- Remaining Useful Life (RUL) estimation
- Time-to-failure prediction
- Alert systems with escalation rules
- Integration with anomaly detection
"""

from __future__ import annotations

import logging
import warnings
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from anomsmith.primitives.base import BaseDetector

logger = logging.getLogger(__name__)


def _anomaly_score_and_label_row(
    detector: BaseDetector, anomaly_row: np.ndarray
) -> tuple[float, int]:
    """Run a fitted anomsmith detector on a single feature row (2D shape (1, n_features)).

    Returns:
        (anomaly_score, label) where label is ``1`` for anomaly and ``0`` for normal,
        matching :class:`~anomsmith.objects.views.LabelView` conventions.
    """
    y = np.asarray(anomaly_row, dtype=float)
    if y.ndim == 1:
        y = y.reshape(1, -1)
    score_view = detector.score(y)
    label_view = detector.predict(y)
    score = float(np.ravel(np.asarray(score_view.scores, dtype=float))[-1])
    label = int(np.ravel(np.asarray(label_view.labels))[-1])
    return score, label


class AlertLevel(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILURE = "failure"


@dataclass
class Alert:
    """Represents a predictive maintenance alert."""

    timestamp: datetime
    level: AlertLevel
    message: str
    feature: str
    value: float
    threshold: float
    asset_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class FeatureExtractor:
    """Extract predictive maintenance features from time series data."""

    def __init__(
        self,
        rolling_windows: list[int] | None = None,
        frequency_features: bool = True,
        change_detection: bool = True,
    ):
        """
        Initialize the feature extractor.

        Parameters
        ----------
        rolling_windows : list of int, optional
            Window sizes for rolling statistics. Default: [5, 10, 20, 50, 100]
        frequency_features : bool, default=True
            Whether to extract frequency domain features.
        change_detection : bool, default=True
            Whether to extract change detection features.
        """
        self.rolling_windows = rolling_windows or [5, 10, 20, 50, 100]
        self.frequency_features = frequency_features
        self.change_detection = change_detection
        self.feature_names_: list[str] = []

    def extract(
        self,
        data: np.ndarray | pd.Series | pd.DataFrame,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Extract features from time series data.

        Parameters
        ----------
        data : array-like
            Time series data. Can be 1D array, Series, or DataFrame.
        columns : list of str, optional
            Column names if data is a DataFrame. If None, uses 'value' for 1D data.

        Returns
        -------
        features : DataFrame
            Extracted features with named columns.
        """
        if isinstance(data, pd.DataFrame):
            df = data.copy()
        elif isinstance(data, pd.Series):
            df = pd.DataFrame({data.name if data.name else "value": data})
        else:
            data = np.asarray(data)
            if data.ndim == 1:
                df = pd.DataFrame({"value": data})
            else:
                if columns is None:
                    columns = [f"feature_{i}" for i in range(data.shape[1])]
                df = pd.DataFrame(data, columns=columns)

        feature_list = []

        # Rolling statistics for each column
        for col in df.columns:
            series = df[col]

            # Rolling statistics
            for window in self.rolling_windows:
                if len(series) >= window:
                    rolling_mean = series.rolling(window=window, min_periods=1).mean()
                    rolling_std = series.rolling(window=window, min_periods=1).std()
                    rolling_min = series.rolling(window=window, min_periods=1).min()
                    rolling_max = series.rolling(window=window, min_periods=1).max()
                    rolling_median = series.rolling(
                        window=window, min_periods=1
                    ).median()

                    feature_list.append(
                        pd.DataFrame(
                            {
                                f"{col}_rolling_mean_{window}": rolling_mean,
                                f"{col}_rolling_std_{window}": rolling_std,
                                f"{col}_rolling_min_{window}": rolling_min,
                                f"{col}_rolling_max_{window}": rolling_max,
                                f"{col}_rolling_median_{window}": rolling_median,
                            }
                        )
                    )

                    # Rolling percentiles
                    rolling_q25 = series.rolling(window=window, min_periods=1).quantile(
                        0.25
                    )
                    rolling_q75 = series.rolling(window=window, min_periods=1).quantile(
                        0.75
                    )
                    rolling_iqr = rolling_q75 - rolling_q25

                    feature_list.append(
                        pd.DataFrame(
                            {
                                f"{col}_rolling_q25_{window}": rolling_q25,
                                f"{col}_rolling_q75_{window}": rolling_q75,
                                f"{col}_rolling_iqr_{window}": rolling_iqr,
                            }
                        )
                    )

            # Change detection features
            if self.change_detection:
                diff = series.diff()
                diff2 = diff.diff()  # Second derivative

                feature_list.append(
                    pd.DataFrame(
                        {
                            f"{col}_diff": diff,
                            f"{col}_diff2": diff2,
                            f"{col}_diff_abs": diff.abs(),
                            f"{col}_diff2_abs": diff2.abs(),
                            f"{col}_pct_change": series.pct_change(),
                        }
                    )
                )

                # Rate of change
                for window in [5, 10, 20]:
                    if len(series) >= window:
                        roc = series.pct_change(periods=window)
                        feature_list.append(pd.DataFrame({f"{col}_roc_{window}": roc}))

            # Frequency domain features
            if self.frequency_features and len(series) > 10:
                # FFT features
                fft_vals = np.fft.rfft(series.dropna().values)
                fft_power = np.abs(fft_vals) ** 2
                freqs = np.fft.rfftfreq(len(series.dropna()))

                if len(fft_power) > 0:
                    dominant_freq_idx = (
                        np.argmax(fft_power[1:]) + 1
                    )  # Skip DC component
                    dominant_freq = (
                        freqs[dominant_freq_idx]
                        if dominant_freq_idx < len(freqs)
                        else 0.0
                    )
                    spectral_centroid = np.sum(freqs * fft_power) / (
                        np.sum(fft_power) + 1e-10
                    )
                    spectral_rolloff = self._spectral_rolloff(freqs, fft_power)

                    # Create feature vectors (repeated for each time step)
                    n = len(series)
                    feature_list.append(
                        pd.DataFrame(
                            {
                                f"{col}_dominant_freq": np.full(n, dominant_freq),
                                f"{col}_spectral_centroid": np.full(
                                    n, spectral_centroid
                                ),
                                f"{col}_spectral_rolloff": np.full(n, spectral_rolloff),
                            },
                            index=series.index,
                        )
                    )

        # Combine all features
        if feature_list:
            features = pd.concat(feature_list, axis=1)
            features = features.bfill().fillna(
                0
            )  # Backward fill then fill remaining with 0
            self.feature_names_ = features.columns.tolist()
            return features
        else:
            return pd.DataFrame(index=df.index)

    @staticmethod
    def _spectral_rolloff(
        freqs: np.ndarray, power: np.ndarray, rolloff_percent: float = 0.85
    ) -> float:
        """Calculate spectral rolloff frequency."""
        cumsum_power = np.cumsum(power)
        total_power = cumsum_power[-1]
        if total_power == 0:
            return 0.0
        rolloff_threshold = total_power * rolloff_percent
        rolloff_idx = np.where(cumsum_power >= rolloff_threshold)[0]
        if len(rolloff_idx) > 0:
            return float(freqs[rolloff_idx[0]])
        return float(freqs[-1]) if len(freqs) > 0 else 0.0


class RULEstimator:
    """Estimate Remaining Useful Life (RUL) for assets."""

    def __init__(
        self,
        method: str = "regression",
        n_estimators: int = 100,
        max_depth: int | None = None,
        random_state: int | None = None,
    ):
        """
        Initialize RUL estimator.

        Parameters
        ----------
        method : str, default='regression'
            Method for RUL estimation. Options: 'regression', 'degradation'
        n_estimators : int, default=100
            Number of trees for Random Forest (if using regression).
        max_depth : int, optional
            Maximum depth of trees.
        random_state : int, optional
            Random state for reproducibility.
        """
        self.method = method
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model_: RandomForestRegressor | None = None
        self.scaler_: StandardScaler | None = None
        self.is_fitted_ = False

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        degradation_threshold: float | None = None,
    ):
        """
        Fit the RUL estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix (e.g., from FeatureExtractor).
        y : array-like of shape (n_samples,)
            RUL values (time until failure) or degradation values.
        degradation_threshold : float, optional
            Threshold for degradation-based method. If provided, converts
            degradation values to RUL.
        """
        X = self._validate_input(X)
        y = np.asarray(y).ravel()

        if degradation_threshold is not None:
            # Convert degradation to RUL
            y = np.maximum(0, degradation_threshold - y)

        if self.method == "regression":
            self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X)

            self.model_ = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1,
            )
            self.model_.fit(X_scaled, y)
        elif self.method == "degradation":
            # Simple linear degradation model
            self.model_ = None  # Degradation model stored differently
            warnings.warn("Degradation method not fully implemented, using regression")
            self.method = "regression"
            self.fit(X, y)

        self.is_fitted_ = True

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Predict RUL for new data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        rul : ndarray of shape (n_samples,)
            Predicted RUL values.
        """
        if not self.is_fitted_:
            raise ValueError("Estimator must be fitted before prediction")

        X = self._validate_input(X)

        if (
            self.method == "regression"
            and self.model_ is not None
            and self.scaler_ is not None
        ):
            X_scaled = self.scaler_.transform(X)
            rul = self.model_.predict(X_scaled)
            return np.maximum(0, rul)  # RUL cannot be negative
        else:
            raise ValueError("Model not properly fitted")

    def _validate_input(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Validate and convert input to numpy array."""
        if isinstance(X, pd.DataFrame):
            return X.values
        return np.asarray(X)


class FailureClassifier:
    """Classify normal vs. failure states."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        random_state: int | None = None,
    ):
        """
        Initialize failure classifier.

        Parameters
        ----------
        n_estimators : int, default=100
            Number of trees for Random Forest.
        max_depth : int, optional
            Maximum depth of trees.
        random_state : int, optional
            Random state for reproducibility.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model_: RandomForestClassifier | None = None
        self.scaler_: StandardScaler | None = None
        self.is_fitted_ = False

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
    ):
        """
        Fit the failure classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : array-like of shape (n_samples,)
            Binary labels: 0 for normal, 1 for failure.
        """
        X = self._validate_input(X)
        y = np.asarray(y).ravel()

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        self.model_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model_.fit(X_scaled, y)
        self.is_fitted_ = True

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Predict failure states.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        predictions : ndarray of shape (n_samples,)
            Binary predictions: 0 for normal, 1 for failure.
        """
        if not self.is_fitted_:
            raise ValueError("Classifier must be fitted before prediction")

        X = self._validate_input(X)

        if self.model_ is not None and self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X)
            return self.model_.predict(X_scaled)
        else:
            raise ValueError("Model not properly fitted")

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Predict failure probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        probabilities : ndarray of shape (n_samples, 2)
            Probability of [normal, failure] for each sample.
        """
        if not self.is_fitted_:
            raise ValueError("Classifier must be fitted before prediction")

        X = self._validate_input(X)

        if self.model_ is not None and self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X)
            return self.model_.predict_proba(X_scaled)
        else:
            raise ValueError("Model not properly fitted")

    def _validate_input(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Validate and convert input to numpy array."""
        if isinstance(X, pd.DataFrame):
            return X.values
        return np.asarray(X)


class AlertSystem:
    """Alert system for predictive maintenance with escalation rules."""

    def __init__(
        self,
        thresholds: dict[str, dict[str, float]] | None = None,
        escalation_rules: dict[str, dict[str, Any]] | None = None,
    ):
        """
        Initialize alert system.

        Parameters
        ----------
        thresholds : dict, optional
            Dictionary mapping feature names to threshold configurations.
            Format: {feature: {level: threshold_value}}
            Example: {'temperature': {'warning': 80.0, 'critical': 90.0, 'failure': 100.0}}
        escalation_rules : dict, optional
            Escalation rules for alerts. Format: {level: {condition: action}}
        """
        self.thresholds = thresholds or {}
        self.escalation_rules = escalation_rules or {}
        self.alert_history: deque = deque(maxlen=1000)  # Keep last 1000 alerts
        self.alert_counts: dict[str, int] = {}  # Count alerts per asset/feature

    def check_thresholds(
        self,
        features: np.ndarray | pd.DataFrame | pd.Series,
        feature_names: list[str] | None = None,
        timestamp: datetime | None = None,
        asset_id: str | None = None,
    ) -> list[Alert]:
        """
        Check features against thresholds and generate alerts.

        Parameters
        ----------
        features : array-like
            Feature values to check. Can be single value, array, or DataFrame.
        feature_names : list of str, optional
            Names of features. Required if features is array.
        timestamp : datetime, optional
            Timestamp for alerts. Defaults to current time.
        asset_id : str, optional
            Asset identifier.

        Returns
        -------
        alerts : list of Alert
            List of generated alerts.
        """
        if timestamp is None:
            timestamp = datetime.now()

        alerts = []

        # Convert to DataFrame for easier handling
        if isinstance(features, pd.DataFrame):
            df = features
        elif isinstance(features, pd.Series):
            df = pd.DataFrame(features).T
        else:
            features = np.asarray(features)
            if features.ndim == 0:
                features = features.reshape(1, -1)
            elif features.ndim == 1:
                features = features.reshape(1, -1)

            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(features.shape[1])]
            df = pd.DataFrame(features, columns=feature_names)

        # Check each feature against thresholds
        for col in df.columns:
            if col not in self.thresholds:
                continue

            values = df[col].values
            thresholds = self.thresholds[col]

            for value in values:
                if pd.isna(value):
                    continue

                # Determine alert level
                level = self._determine_level(value, thresholds)

                if level is not None:
                    threshold_value = thresholds.get(level.value, float("inf"))
                    message = (
                        f"{col} = {value:.2f} exceeds {level.value} threshold "
                        f"({threshold_value:.2f})"
                    )

                    alert = Alert(
                        timestamp=timestamp,
                        level=level,
                        message=message,
                        feature=col,
                        value=float(value),
                        threshold=threshold_value,
                        asset_id=asset_id,
                    )

                    alerts.append(alert)
                    self.alert_history.append(alert)

                    # Update alert counts
                    key = f"{asset_id}_{col}" if asset_id else col
                    self.alert_counts[key] = self.alert_counts.get(key, 0) + 1

        # Apply escalation rules
        escalated_alerts = self._apply_escalation(alerts)
        return escalated_alerts

    def _determine_level(
        self, value: float, thresholds: dict[str, float]
    ) -> AlertLevel | None:
        """Determine alert level based on value and thresholds."""
        # Check in order: failure, critical, warning, info
        for level in [
            AlertLevel.FAILURE,
            AlertLevel.CRITICAL,
            AlertLevel.WARNING,
            AlertLevel.INFO,
        ]:
            if level.value in thresholds:
                threshold = thresholds[level.value]
                if value >= threshold:
                    return level
        return None

    def _apply_escalation(self, alerts: list[Alert]) -> list[Alert]:
        """Apply escalation rules to alerts."""
        escalated = []
        for alert in alerts:
            # Check escalation rules
            if alert.level.value in self.escalation_rules:
                rules = self.escalation_rules[alert.level.value]
                # Example: escalate if multiple alerts for same feature
                key = (
                    f"{alert.asset_id}_{alert.feature}"
                    if alert.asset_id
                    else alert.feature
                )
                count = self.alert_counts.get(key, 0)

                if "min_count" in rules and count >= rules["min_count"]:
                    # Escalate to next level
                    if alert.level == AlertLevel.WARNING:
                        alert.level = AlertLevel.CRITICAL
                    elif alert.level == AlertLevel.CRITICAL:
                        alert.level = AlertLevel.FAILURE

            escalated.append(alert)
        return escalated

    def get_recent_alerts(
        self,
        n: int = 10,
        level: AlertLevel | None = None,
        asset_id: str | None = None,
    ) -> list[Alert]:
        """
        Get recent alerts.

        Parameters
        ----------
        n : int, default=10
            Number of recent alerts to return.
        level : AlertLevel, optional
            Filter by alert level.
        asset_id : str, optional
            Filter by asset ID.

        Returns
        -------
        alerts : list of Alert
            Recent alerts matching criteria.
        """
        alerts = list(self.alert_history)
        if level is not None:
            alerts = [a for a in alerts if a.level == level]
        if asset_id is not None:
            alerts = [a for a in alerts if a.asset_id == asset_id]
        return alerts[-n:]


class PredictiveMaintenanceSystem:
    """Complete predictive maintenance system integrating all components."""

    def __init__(
        self,
        feature_extractor: FeatureExtractor | None = None,
        rul_estimator: RULEstimator | None = None,
        failure_classifier: FailureClassifier | None = None,
        alert_system: AlertSystem | None = None,
        anomaly_detector: BaseDetector | None = None,
    ):
        """
        Initialize predictive maintenance system.

        Parameters
        ----------
        feature_extractor : FeatureExtractor, optional
            Feature extractor. If None, creates default.
        rul_estimator : RULEstimator, optional
            RUL estimator. If None, creates default.
        failure_classifier : FailureClassifier, optional
            Failure classifier. If None, creates default.
        alert_system : AlertSystem, optional
            Alert system. If None, creates default.
        anomaly_detector : anomsmith.primitives.base.BaseDetector, optional
            Fitted detector from ``anomsmith.primitives.detectors`` (or custom subclass)
            applied to the **last row** of extracted features (same feature space used
            when the detector was fit).
        """
        self.feature_extractor = feature_extractor or FeatureExtractor()
        self.rul_estimator = rul_estimator or RULEstimator()
        self.failure_classifier = failure_classifier or FailureClassifier()
        self.alert_system = alert_system or AlertSystem()
        self.anomaly_detector = anomaly_detector

    def process(
        self,
        data: np.ndarray | pd.Series | pd.DataFrame,
        timestamp: datetime | None = None,
        asset_id: str | None = None,
        return_features: bool = False,
    ) -> dict[str, Any]:
        """
        Process new data and generate predictions/alerts.

        Parameters
        ----------
        data : array-like
            Time series data to process.
        timestamp : datetime, optional
            Timestamp for the data.
        asset_id : str, optional
            Asset identifier.
        return_features : bool, default=False
            Whether to return extracted features.

        Returns
        -------
        results : dict
            Dictionary containing:
            - 'features': extracted features (if return_features=True)
            - 'rul': predicted RUL
            - 'failure_probability': probability of failure
            - 'failure_prediction': binary failure prediction
            - 'anomaly_score': anomaly score from :meth:`anomsmith.primitives.base.BaseDetector.score`
            - 'anomaly_prediction': ``0`` (normal) or ``1`` (anomaly) from ``LabelView`` labels
            - 'alerts': list of alerts
        """
        # Extract features
        features = self.feature_extractor.extract(data)

        results = {}

        if return_features:
            results["features"] = features

        # RUL estimation
        if self.rul_estimator.is_fitted_:
            try:
                rul = self.rul_estimator.predict(features.iloc[[-1]])  # Last row
                results["rul"] = float(rul[0])
            except Exception as e:
                warnings.warn(f"RUL prediction failed: {e}")
                results["rul"] = None
        else:
            results["rul"] = None

        # Failure classification
        if self.failure_classifier.is_fitted_:
            try:
                failure_proba = self.failure_classifier.predict_proba(
                    features.iloc[[-1]]
                )
                failure_pred = self.failure_classifier.predict(features.iloc[[-1]])
                results["failure_probability"] = float(
                    failure_proba[0][1]
                )  # Probability of failure
                results["failure_prediction"] = int(failure_pred[0])
            except Exception as e:
                warnings.warn(f"Failure classification failed: {e}")
                results["failure_probability"] = None
                results["failure_prediction"] = None
        else:
            results["failure_probability"] = None
            results["failure_prediction"] = None

        # Anomaly detection (anomsmith BaseDetector: score / predict on tabular row)
        if self.anomaly_detector is not None:
            try:
                if len(features) > 0:
                    anomaly_data = features.iloc[[-1]].values
                else:
                    data_array = np.asarray(data)
                    if data_array.ndim == 1:
                        anomaly_data = data_array.reshape(1, -1)
                    else:
                        anomaly_data = data_array[-1:].reshape(1, -1)

                ascore, alabel = _anomaly_score_and_label_row(
                    self.anomaly_detector, anomaly_data
                )
                results["anomaly_score"] = ascore
                results["anomaly_prediction"] = int(alabel)
            except Exception as e:
                warnings.warn(f"Anomaly detection failed: {e}")
                results["anomaly_score"] = None
                results["anomaly_prediction"] = None
        else:
            results["anomaly_score"] = None
            results["anomaly_prediction"] = None

        # Generate alerts
        alerts = self.alert_system.check_thresholds(
            features,
            feature_names=features.columns.tolist(),
            timestamp=timestamp,
            asset_id=asset_id,
        )
        results["alerts"] = alerts

        return results


# Utility functions for predictive maintenance workflows


def calculate_rul(
    df: pd.DataFrame,
    asset_id_col: str = "asset_id",
    cycle_col: str = "cycle",
    failure_cycle_col: str | None = None,
) -> pd.Series:
    """
    Calculate Remaining Useful Life (RUL) for each record.

    RUL is calculated as: max_cycle - current_cycle for each asset.

    Parameters
    ----------
    df : DataFrame
        DataFrame with asset_id and cycle columns.
    asset_id_col : str, default='asset_id'
        Column name for asset/equipment identifier.
    cycle_col : str, default='cycle'
        Column name for cycle/time step.
    failure_cycle_col : str, optional
        Column name for failure cycle. If provided, uses this instead of max cycle.

    Returns
    -------
    rul : Series
        Remaining Useful Life for each record.
    """
    df = df.copy()

    if failure_cycle_col and failure_cycle_col in df.columns:
        # Use explicit failure cycle
        df["max_cycle"] = df.groupby(asset_id_col)[failure_cycle_col].transform("max")
    else:
        # Calculate max cycle per asset
        df["max_cycle"] = df.groupby(asset_id_col)[cycle_col].transform("max")

    rul = df["max_cycle"] - df[cycle_col]
    return rul.clip(lower=0)  # RUL cannot be negative


def create_rul_labels(
    df: pd.DataFrame,
    rul_col: str = "RUL",
    warning_threshold: int = 30,
    critical_threshold: int = 15,
) -> pd.DataFrame:
    """
    Create health status labels based on RUL values.

    Parameters
    ----------
    df : DataFrame
        DataFrame with RUL column.
    rul_col : str, default='RUL'
        Column name for RUL values.
    warning_threshold : int, default=30
        RUL threshold for warning state.
    critical_threshold : int, default=15
        RUL threshold for critical state.

    Returns
    -------
    df : DataFrame
        DataFrame with added columns:
        - health_status: 'healthy', 'warning', 'critical', 'failed'
        - binary_label: 0 (healthy) or 1 (failure/warning/critical)
        - multi_class_label: 0 (healthy), 1 (warning), 2 (critical), 3 (failed)
    """
    df = df.copy()

    # Health status
    conditions = [
        df[rul_col] > warning_threshold,
        (df[rul_col] > critical_threshold) & (df[rul_col] <= warning_threshold),
        (df[rul_col] > 0) & (df[rul_col] <= critical_threshold),
        df[rul_col] == 0,
    ]
    choices = ["healthy", "warning", "critical", "failed"]
    df["health_status"] = np.select(conditions, choices, default="unknown")

    # Binary label (0 = healthy, 1 = failure/warning/critical)
    df["binary_label"] = (df["health_status"] != "healthy").astype(int)

    # Multi-class label (0=healthy, 1=warning, 2=critical, 3=failed)
    label_map = {"healthy": 0, "warning": 1, "critical": 2, "failed": 3}
    df["multi_class_label"] = df["health_status"].map(label_map).fillna(-1).astype(int)

    return df


def add_rolling_statistics(
    df: pd.DataFrame,
    feature_cols: list[str],
    asset_id_col: str = "asset_id",
    cycle_col: str = "cycle",
    window: int = 5,
    stats: list[str] | None = None,
) -> pd.DataFrame:
    """
    Add rolling window statistics for feature columns, grouped by asset.

    Parameters
    ----------
    df : DataFrame
        DataFrame with asset and feature columns.
    feature_cols : list of str
        Feature column names to compute rolling statistics for.
    asset_id_col : str, default='asset_id'
        Column name for asset identifier.
    cycle_col : str, default='cycle'
        Column name for cycle/time step (used for sorting).
    window : int, default=5
        Rolling window size.
    stats : list of str, optional
        Statistics to compute. Default: ['mean', 'std', 'min', 'max'].

    Returns
    -------
    df : DataFrame
        DataFrame with added rolling statistic columns.
    """
    if stats is None:
        stats = ["mean", "std", "min", "max"]

    df = df.copy()
    df = df.sort_values([asset_id_col, cycle_col]).reset_index(drop=True)

    new_cols = {}

    for col in feature_cols:
        if col not in df.columns:
            continue

        for stat in stats:
            col_name = f"{col}_rolling_{stat}_{window}"

            if stat == "mean":
                new_cols[col_name] = df.groupby(asset_id_col)[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
            elif stat == "std":
                new_cols[col_name] = df.groupby(asset_id_col)[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).std()
                )
            elif stat == "min":
                new_cols[col_name] = df.groupby(asset_id_col)[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).min()
                )
            elif stat == "max":
                new_cols[col_name] = df.groupby(asset_id_col)[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).max()
                )
            elif stat == "median":
                new_cols[col_name] = df.groupby(asset_id_col)[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).median()
                )

    # Add new columns to dataframe
    for col_name, values in new_cols.items():
        df[col_name] = values

    return df


def add_degradation_rates(
    df: pd.DataFrame,
    feature_cols: list[str],
    asset_id_col: str = "asset_id",
    cycle_col: str = "cycle",
    periods: list[int] | None = None,
) -> pd.DataFrame:
    """
    Add degradation rate features (rate of change) for feature columns.

    Parameters
    ----------
    df : DataFrame
        DataFrame with asset and feature columns.
    feature_cols : list of str
        Feature column names to compute degradation rates for.
    asset_id_col : str, default='asset_id'
        Column name for asset identifier.
    cycle_col : str, default='cycle'
        Column name for cycle/time step.
    periods : list of int, optional
        Periods for rate of change calculation. Default: [1, 3, 5].

    Returns
    -------
    df : DataFrame
        DataFrame with added degradation rate columns.
    """
    if periods is None:
        periods = [1, 3, 5]

    df = df.copy()
    df = df.sort_values([asset_id_col, cycle_col]).reset_index(drop=True)

    new_cols = {}

    for col in feature_cols:
        if col not in df.columns:
            continue

        for period in periods:
            col_name = f"{col}_degradation_rate_{period}"
            # Calculate rate of change grouped by asset
            new_cols[col_name] = df.groupby(asset_id_col)[col].transform(
                lambda x: x.pct_change(periods=period)
            )

    # Add new columns to dataframe
    for col_name, values in new_cols.items():
        df[col_name] = values

    return df


def prepare_pm_features(
    df: pd.DataFrame,
    asset_id_col: str = "asset_id",
    cycle_col: str = "cycle",
    feature_cols: list[str] | None = None,
    calculate_rul_flag: bool = True,
    add_labels: bool = True,
    add_rolling_stats: bool = True,
    include_degradation_rates: bool = False,
    rolling_window: int = 5,
    warning_threshold: int = 30,
    critical_threshold: int = 15,
    failure_cycle_col: str | None = None,
) -> pd.DataFrame:
    """
    Prepare predictive maintenance features from raw sensor data.

    This is a convenience function that combines:
    - RUL calculation
    - Health status labeling
    - Rolling statistics
    - Degradation rates

    Parameters
    ----------
    df : DataFrame
        Input DataFrame with asset_id, cycle, and sensor/feature columns.
    asset_id_col : str, default='asset_id'
        Column name for asset identifier.
    cycle_col : str, default='cycle'
        Column name for cycle/time step.
    feature_cols : list of str, optional
        Feature column names. If None, auto-detects (excludes asset_id, cycle, RUL, etc.).
    calculate_rul_flag : bool, default=True
        Whether to calculate RUL.
    add_labels : bool, default=True
        Whether to add health status labels.
    add_rolling_stats : bool, default=True
        Whether to add rolling statistics.
    include_degradation_rates : bool, default=False
        Whether to add degradation rate features.
    rolling_window : int, default=5
        Window size for rolling statistics.
    warning_threshold : int, default=30
        RUL threshold for warning state.
    critical_threshold : int, default=15
        RUL threshold for critical state.
    failure_cycle_col : str, optional
        Column name for failure cycle (if available).

    Returns
    -------
    df : DataFrame
        DataFrame with all engineered features.
    """
    df = df.copy()

    # Auto-detect feature columns if not provided
    if feature_cols is None:
        exclude_cols = [
            asset_id_col,
            cycle_col,
            "RUL",
            "health_status",
            "binary_label",
            "multi_class_label",
            "max_cycle",
        ]
        feature_cols = [col for col in df.columns if col not in exclude_cols]

    # Calculate RUL
    if calculate_rul_flag:
        df["RUL"] = calculate_rul(
            df,
            asset_id_col=asset_id_col,
            cycle_col=cycle_col,
            failure_cycle_col=failure_cycle_col,
        )

    # Add labels
    if add_labels and "RUL" in df.columns:
        df = create_rul_labels(
            df,
            rul_col="RUL",
            warning_threshold=warning_threshold,
            critical_threshold=critical_threshold,
        )

    # Add rolling statistics
    if add_rolling_stats and feature_cols:
        df = add_rolling_statistics(
            df,
            feature_cols=feature_cols,
            asset_id_col=asset_id_col,
            cycle_col=cycle_col,
            window=rolling_window,
        )

    # Add degradation rates
    if include_degradation_rates and feature_cols:
        df = add_degradation_rates(
            df,
            feature_cols=feature_cols,
            asset_id_col=asset_id_col,
            cycle_col=cycle_col,
        )

    return df


# Real-time data ingestion and monitoring


class RealTimeIngestion:
    """Real-time data ingestion system for predictive maintenance."""

    def __init__(
        self,
        pm_system: PredictiveMaintenanceSystem,
        window_size: int = 100,
        update_frequency: int | None = None,
    ):
        """
        Initialize real-time ingestion system.

        Parameters
        ----------
        pm_system : PredictiveMaintenanceSystem
            Predictive maintenance system to use for processing.
        window_size : int, default=100
            Size of sliding window for processing.
        update_frequency : int, optional
            Frequency in seconds for periodic updates. If None, processes on demand.
        """
        self.pm_system = pm_system
        self.window_size = window_size
        self.update_frequency = update_frequency

        # Data buffers per asset
        self.data_buffers: dict[str, deque] = {}
        self.timestamp_buffers: dict[str, deque] = {}
        self.results_history: dict[str, list[dict[str, Any]]] = {}

    def ingest(
        self,
        data: float | np.ndarray | pd.Series,
        asset_id: str,
        timestamp: datetime | None = None,
        sensor_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Ingest new data point and process if window is full.

        Parameters
        ----------
        data : float, array-like, or Series
            New sensor reading(s).
        asset_id : str
            Asset identifier.
        timestamp : datetime, optional
            Timestamp for the data. Defaults to current time.
        sensor_name : str, optional
            Name of sensor/feature. Required if data is scalar.

        Returns
        -------
        results : dict
            Processing results if window is processed, else None.
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Initialize buffers for asset if needed
        if asset_id not in self.data_buffers:
            self.data_buffers[asset_id] = deque(maxlen=self.window_size)
            self.timestamp_buffers[asset_id] = deque(maxlen=self.window_size)
            self.results_history[asset_id] = []

        # Convert data to appropriate format
        if isinstance(data, (int, float)):
            if sensor_name is None:
                raise ValueError("sensor_name required when data is scalar")
            data_dict = {sensor_name: data}
        elif isinstance(data, pd.Series):
            data_dict = data.to_dict()
        elif isinstance(data, np.ndarray):
            if data.ndim == 0:
                if sensor_name is None:
                    raise ValueError("sensor_name required when data is scalar")
                data_dict = {sensor_name: float(data)}
            else:
                # Assume it's a feature vector - convert to dict with generic names
                data_dict = {f"feature_{i}": float(data[i]) for i in range(len(data))}
        else:
            data_dict = (
                dict(data) if hasattr(data, "items") else {sensor_name or "value": data}
            )

        # Add to buffer
        self.data_buffers[asset_id].append(data_dict)
        self.timestamp_buffers[asset_id].append(timestamp)

        # Process if window is full
        if len(self.data_buffers[asset_id]) >= self.window_size:
            return self.process_window(asset_id)
        else:
            return {
                "status": "buffering",
                "buffer_size": len(self.data_buffers[asset_id]),
            }

    def process_window(self, asset_id: str) -> dict[str, Any]:
        """
        Process current window for an asset.

        Parameters
        ----------
        asset_id : str
            Asset identifier.

        Returns
        -------
        results : dict
            Processing results.
        """
        if asset_id not in self.data_buffers:
            raise ValueError(f"No data for asset {asset_id}")

        # Convert buffer to DataFrame
        buffer_data = list(self.data_buffers[asset_id])
        timestamps = list(self.timestamp_buffers[asset_id])

        # Create DataFrame from buffer
        df = pd.DataFrame(buffer_data, index=timestamps)

        # Process with PM system
        latest_timestamp = timestamps[-1]
        results = self.pm_system.process(
            df.iloc[-1],  # Use latest row
            timestamp=latest_timestamp,
            asset_id=asset_id,
            return_features=False,
        )

        # Store in history
        self.results_history[asset_id].append(results)

        return results

    def get_latest_results(self, asset_id: str, n: int = 1) -> list[dict[str, Any]]:
        """
        Get latest processing results for an asset.

        Parameters
        ----------
        asset_id : str
            Asset identifier.
        n : int, default=1
            Number of latest results to return.

        Returns
        -------
        results : list of dict
            Latest results.
        """
        if asset_id not in self.results_history:
            return []
        return self.results_history[asset_id][-n:]

    def get_all_assets(self) -> list[str]:
        """Get list of all asset IDs being monitored."""
        return list(self.data_buffers.keys())


class DashboardVisualizer:
    """Dashboard visualization utilities for predictive maintenance monitoring."""

    def __init__(self, figsize: tuple[int, int] = (15, 10)):
        """
        Initialize dashboard visualizer.

        Parameters
        ----------
        figsize : tuple of int, default=(15, 10)
            Figure size for plots.
        """
        self.figsize = figsize

    def create_dashboard(
        self,
        results_history: dict[str, list[dict[str, Any]]],
        sensor_data: dict[str, pd.DataFrame] | None = None,
        save_path: str | None = None,
    ):
        """
        Create comprehensive dashboard visualization.

        Parameters
        ----------
        results_history : dict
            Dictionary mapping asset_id to list of processing results.
        sensor_data : dict, optional
            Dictionary mapping asset_id to DataFrame with sensor readings.
        save_path : str, optional
            Path to save the dashboard figure.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Dashboard figure.
        """
        try:
            import matplotlib.dates as mdates
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for dashboard visualization")

        # Apply signalplot style if available
        try:
            import signalplot

            signalplot.apply()
            colors = {
                "primary": "black",
                "secondary": "#666666",
                "accent": signalplot.ACCENT
                if hasattr(signalplot, "ACCENT")
                else "#d62728",
                "normal": "#333333",
                "anomaly": signalplot.ACCENT
                if hasattr(signalplot, "ACCENT")
                else "#d62728",
            }
        except ImportError:
            # Fallback to minimal matplotlib defaults
            plt.rcParams.update(
                {
                    "figure.facecolor": "white",
                    "axes.facecolor": "white",
                    "axes.spines.top": False,
                    "axes.spines.right": False,
                    "axes.linewidth": 0.8,
                    "axes.grid": True,
                    "axes.grid.axis": "y",
                    "grid.alpha": 0.3,
                }
            )
            colors = {
                "primary": "black",
                "secondary": "#666666",
                "accent": "#d62728",
                "normal": "#333333",
                "anomaly": "#d62728",
            }

        n_assets = len(results_history)
        if n_assets == 0:
            raise ValueError("No results to visualize")

        # Create subplots: 2 rows, 2 columns per asset (or adjust layout)
        n_cols = min(2, n_assets)
        n_rows = (n_assets + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows * 2, n_cols * 2, figsize=self.figsize)
        if n_assets == 1:
            axes = axes.reshape(2, 2)
        elif n_rows == 1:
            axes = axes.reshape(2, n_cols * 2)
        else:
            axes = axes.reshape(n_rows * 2, n_cols * 2)

        asset_ids = list(results_history.keys())

        for idx, asset_id in enumerate(asset_ids):
            row = (idx // n_cols) * 2
            col = (idx % n_cols) * 2

            results = results_history[asset_id]

            if len(results) == 0:
                continue

            # Extract time series data
            timestamps = [
                (
                    r.get("timestamp", datetime.now())
                    if isinstance(r.get("timestamp"), datetime)
                    else datetime.now()
                )
                for r in results
            ]
            rul_values = [r.get("rul") for r in results if r.get("rul") is not None]
            failure_probs = [
                r.get("failure_probability")
                for r in results
                if r.get("failure_probability") is not None
            ]
            anomaly_scores = [
                r.get("anomaly_score")
                for r in results
                if r.get("anomaly_score") is not None
            ]

            # Plot 1: RUL over time
            ax1 = axes[row, col]
            if rul_values:
                ax1.plot(
                    timestamps[: len(rul_values)],
                    rul_values,
                    color=colors["primary"],
                    linewidth=2,
                    label="RUL",
                )
                ax1.axhline(
                    y=30,
                    color=colors["secondary"],
                    linestyle="--",
                    linewidth=1,
                    label="Warning Threshold",
                    alpha=0.7,
                )
                ax1.axhline(
                    y=15,
                    color=colors["accent"],
                    linestyle="--",
                    linewidth=1,
                    label="Critical Threshold",
                    alpha=0.7,
                )
                ax1.set_xlabel("Time")
                ax1.set_ylabel("RUL")
                ax1.set_title(f"{asset_id} - Remaining Useful Life")
                ax1.legend(frameon=False, loc="best")
                ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

            # Plot 2: Failure Probability
            ax2 = axes[row, col + 1]
            if failure_probs:
                ax2.plot(
                    timestamps[: len(failure_probs)],
                    failure_probs,
                    color=colors["accent"],
                    linewidth=2,
                    label="Failure Probability",
                )
                ax2.axhline(
                    y=0.5,
                    color=colors["secondary"],
                    linestyle="--",
                    linewidth=1,
                    label="50% Threshold",
                    alpha=0.7,
                )
                ax2.set_xlabel("Time")
                ax2.set_ylabel("Probability")
                ax2.set_title(f"{asset_id} - Failure Probability")
                ax2.legend(frameon=False, loc="best")
                ax2.set_ylim(0, 1)
                ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

            # Plot 3: Anomaly Scores
            ax3 = axes[row + 1, col]
            if anomaly_scores:
                ax3.plot(
                    timestamps[: len(anomaly_scores)],
                    anomaly_scores,
                    color=colors["primary"],
                    linewidth=2,
                    label="Anomaly Score",
                )
                # Mark anomalies
                anomaly_preds = [
                    r.get("anomaly_prediction")
                    for r in results
                    if r.get("anomaly_prediction") is not None
                ]
                if anomaly_preds:
                    anomaly_times = [
                        t
                        for t, p in zip(timestamps[: len(anomaly_preds)], anomaly_preds)
                        if p == 1
                    ]
                    if anomaly_times:
                        ax3.scatter(
                            anomaly_times,
                            [
                                anomaly_scores[timestamps.index(t)]
                                for t in anomaly_times
                            ],
                            color=colors["anomaly"],
                            s=100,
                            marker="x",
                            label="Anomaly",
                            zorder=5,
                            linewidths=1.5,
                        )
                ax3.set_xlabel("Time")
                ax3.set_ylabel("Score")
                ax3.set_title(f"{asset_id} - Anomaly Detection")
                ax3.legend(frameon=False, loc="best")
                ax3.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

            # Plot 4: Alerts
            ax4 = axes[row + 1, col + 1]
            all_alerts = [alert for r in results for alert in r.get("alerts", [])]
            if all_alerts:
                alert_levels = {"info": 1, "warning": 2, "critical": 3, "failure": 4}
                alert_times = [alert.timestamp for alert in all_alerts]
                alert_values = [
                    alert_levels.get(alert.level.value, 0) for alert in all_alerts
                ]
                alert_color_map = {
                    "info": colors["secondary"],
                    "warning": colors["secondary"],
                    "critical": colors["accent"],
                    "failure": colors["accent"],
                }
                alert_colors = [
                    alert_color_map.get(alert.level.value, colors["secondary"])
                    for alert in all_alerts
                ]

                ax4.scatter(
                    alert_times,
                    alert_values,
                    c=alert_colors,
                    s=100,
                    alpha=0.7,
                    edgecolors="none",
                )
                ax4.set_xlabel("Time")
                ax4.set_ylabel("Alert Level")
                ax4.set_title(f"{asset_id} - Alerts")
                ax4.set_yticks([1, 2, 3, 4])
                ax4.set_yticklabels(["Info", "Warning", "Critical", "Failure"])
                ax4.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
            else:
                ax4.text(
                    0.5,
                    0.5,
                    "No Alerts",
                    ha="center",
                    va="center",
                    transform=ax4.transAxes,
                )
                ax4.set_title(f"{asset_id} - Alerts")

        plt.tight_layout()

        if save_path:
            fig.savefig(
                save_path,
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )
            logger.info("Dashboard saved to: %s", save_path)

        return fig

    def create_summary_dashboard(
        self,
        results_history: dict[str, list[dict[str, Any]]],
        save_path: str | None = None,
    ):
        """
        Create summary dashboard with key metrics.

        Parameters
        ----------
        results_history : dict
            Dictionary mapping asset_id to list of processing results.
        save_path : str, optional
            Path to save the dashboard figure.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Summary dashboard figure.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for dashboard visualization")

        # Apply signalplot style if available
        try:
            import signalplot

            signalplot.apply()
            colors = {
                "primary": "black",
                "secondary": "#666666",
                "accent": signalplot.ACCENT
                if hasattr(signalplot, "ACCENT")
                else "#d62728",
                "normal": "#333333",
                "anomaly": signalplot.ACCENT
                if hasattr(signalplot, "ACCENT")
                else "#d62728",
            }
        except ImportError:
            # Fallback to minimal matplotlib defaults
            plt.rcParams.update(
                {
                    "figure.facecolor": "white",
                    "axes.facecolor": "white",
                    "axes.spines.top": False,
                    "axes.spines.right": False,
                    "axes.linewidth": 0.8,
                    "axes.grid": True,
                    "axes.grid.axis": "y",
                    "grid.alpha": 0.3,
                }
            )
            colors = {
                "primary": "black",
                "secondary": "#666666",
                "accent": "#d62728",
                "normal": "#333333",
                "anomaly": "#d62728",
            }

        n_assets = len(results_history)
        if n_assets == 0:
            raise ValueError("No results to visualize")

        fig, axes = plt.subplots(2, 2, figsize=self.figsize)

        asset_ids = list(results_history.keys())

        # Plot 1: Current RUL by Asset
        ax1 = axes[0, 0]
        current_rul = []
        asset_labels = []
        for asset_id in asset_ids:
            results = results_history[asset_id]
            if results:
                latest_rul = results[-1].get("rul")
                if latest_rul is not None:
                    current_rul.append(latest_rul)
                    asset_labels.append(asset_id)

        if current_rul:
            bar_colors = [
                colors["primary"]
                if r > 30
                else colors["secondary"]
                if r > 15
                else colors["accent"]
                for r in current_rul
            ]
            ax1.barh(
                asset_labels,
                current_rul,
                color=bar_colors,
                alpha=0.8,
                edgecolor=colors["primary"],
                linewidth=0.5,
            )
            ax1.axvline(
                x=30,
                color=colors["secondary"],
                linestyle="--",
                linewidth=1,
                label="Warning",
                alpha=0.7,
            )
            ax1.axvline(
                x=15,
                color=colors["accent"],
                linestyle="--",
                linewidth=1,
                label="Critical",
                alpha=0.7,
            )
            ax1.set_xlabel("RUL")
            ax1.set_title("Current RUL by Asset")
            ax1.legend(frameon=False, loc="best")

        # Plot 2: Failure Probability Distribution
        ax2 = axes[0, 1]
        all_failure_probs = []
        for asset_id in asset_ids:
            results = results_history[asset_id]
            failure_probs = [
                r.get("failure_probability")
                for r in results
                if r.get("failure_probability") is not None
            ]
            all_failure_probs.extend(failure_probs)

        if all_failure_probs:
            ax2.hist(
                all_failure_probs,
                bins=20,
                color=colors["accent"],
                alpha=0.8,
                edgecolor=colors["primary"],
                linewidth=0.5,
            )
            ax2.set_xlabel("Failure Probability")
            ax2.set_ylabel("Frequency")
            ax2.set_title("Failure Probability Distribution")

        # Plot 3: Alert Count by Level
        ax3 = axes[1, 0]
        alert_counts = {"info": 0, "warning": 0, "critical": 0, "failure": 0}
        for asset_id in asset_ids:
            results = results_history[asset_id]
            for r in results:
                for alert in r.get("alerts", []):
                    level = alert.level.value
                    alert_counts[level] = alert_counts.get(level, 0) + 1

        if any(alert_counts.values()):
            levels = list(alert_counts.keys())
            counts = [alert_counts[level] for level in levels]
            bar_colors = [
                colors["secondary"]
                if level in ["info", "warning"]
                else colors["accent"]
                for level in levels
            ]
            ax3.bar(
                levels,
                counts,
                color=bar_colors,
                alpha=0.8,
                edgecolor=colors["primary"],
                linewidth=0.5,
            )
            ax3.set_ylabel("Count")
            ax3.set_title("Alert Count by Level")

        # Plot 4: Asset Health Status
        ax4 = axes[1, 1]
        health_status = []
        for asset_id in asset_ids:
            results = results_history[asset_id]
            if results:
                latest = results[-1]
                rul = latest.get("rul")
                if rul is not None:
                    if rul > 30:
                        health_status.append("Healthy")
                    elif rul > 15:
                        health_status.append("Warning")
                    elif rul > 0:
                        health_status.append("Critical")
                    else:
                        health_status.append("Failed")

        if health_status:
            status_counts: dict[str, int] = {}
            for status in health_status:
                status_counts[status] = status_counts.get(status, 0) + 1

            statuses = list(status_counts.keys())
            counts = [status_counts[s] for s in statuses]
            pie_colors_map = {
                "Healthy": colors["primary"],
                "Warning": colors["secondary"],
                "Critical": colors["accent"],
                "Failed": colors["accent"],
            }
            pie_colors = [pie_colors_map.get(s, colors["secondary"]) for s in statuses]
            ax4.pie(
                counts,
                labels=statuses,
                colors=pie_colors,
                autopct="%1.1f%%",
                startangle=90,
            )
            ax4.set_title("Asset Health Status Distribution")

        plt.tight_layout()

        if save_path:
            fig.savefig(
                save_path,
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )
            logger.info("Summary dashboard saved to: %s", save_path)

        return fig
