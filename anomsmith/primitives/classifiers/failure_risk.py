"""Failure risk classifiers for asset health prediction.

Classifies assets as healthy or at risk of failure based on sensor readings
and asset attributes.
"""

import logging
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from anomsmith.objects.health_state import HealthState, HealthStateView

if TYPE_CHECKING:
    try:
        from timesmith.typing import SeriesLike
    except ImportError:
        SeriesLike = None

logger = logging.getLogger(__name__)


class FailureRiskClassifier:
    """Random Forest-based classifier for predicting asset failure risk.

    Uses sensor readings and asset attributes to classify assets as healthy
    or at risk of failure. Designed for grid assets and industrial equipment.

    Args:
        n_estimators: Number of trees in Random Forest (default 100)
        random_state: Random state for reproducibility
    """

    def __init__(
        self,
        n_estimators: int = 100,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize FailureRiskClassifier.

        Args:
            n_estimators: Number of trees in Random Forest
            random_state: Random state for reproducibility
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.classifier_ = RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state
        )
        self.scaler_ = StandardScaler()
        self._fitted = False

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
    ) -> "FailureRiskClassifier":
        """Fit the failure risk classifier.

        Args:
            X: Feature array or DataFrame with sensor readings and asset attributes
            y: Binary labels (1 = failure/at risk, 0 = healthy)

        Returns:
            Self for method chaining
        """
        if isinstance(X, pd.DataFrame):
            X_data = X.values
        else:
            X_data = np.asarray(X)

        if isinstance(y, pd.Series):
            y_data = y.values
        else:
            y_data = np.asarray(y)

        # Scale features
        X_scaled = self.scaler_.fit_transform(X_data)
        self.classifier_.fit(X_scaled, y_data)
        self._fitted = True

        logger.debug(
            f"Fitted FailureRiskClassifier: n_estimators={self.n_estimators}, "
            f"n_samples={len(X_data)}, n_features={X_data.shape[1]}"
        )
        return self

    def predict_health_states(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        index: pd.Index | None = None,
        risk_threshold: float = 0.5,
        distress_threshold: float = 0.8,
    ) -> HealthStateView:
        """Predict health states from features.

        Args:
            X: Feature array or DataFrame
            index: Optional index for the health states
            risk_threshold: Probability threshold for Warning state (default 0.5)
            distress_threshold: Probability threshold for Distress state (default 0.8)

        Returns:
            HealthStateView with predicted health states
        """
        if not self._fitted:
            raise ValueError("Classifier must be fitted before prediction.")

        if isinstance(X, pd.DataFrame):
            X_data = X.values
            if index is None:
                index = X.index
        else:
            X_data = np.asarray(X)
            if index is None:
                index = pd.RangeIndex(start=0, stop=len(X_data))

        # Scale features
        X_scaled = self.scaler_.transform(X_data)

        # Predict probabilities
        probas = self.classifier_.predict_proba(X_scaled)
        # Get probability of failure/risk class
        if probas.shape[1] > 1:
            risk_proba = probas[:, 1]  # Probability of failure class
        else:
            risk_proba = probas[:, 0]

        # Convert to health states
        states = np.zeros(len(risk_proba), dtype=int)
        states[risk_proba > distress_threshold] = HealthState.DISTRESS
        states[(risk_proba > risk_threshold) & (risk_proba <= distress_threshold)] = HealthState.WARNING

        return HealthStateView(index=index, states=states)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict failure risk probabilities.

        Args:
            X: Feature array or DataFrame

        Returns:
            Array of shape (n_samples, n_classes) with class probabilities
        """
        if not self._fitted:
            raise ValueError("Classifier must be fitted before prediction.")

        if isinstance(X, pd.DataFrame):
            X_data = X.values
        else:
            X_data = np.asarray(X)

        X_scaled = self.scaler_.transform(X_data)
        return self.classifier_.predict_proba(X_scaled)

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict binary failure labels.

        Args:
            X: Feature array or DataFrame

        Returns:
            Array of shape (n_samples,) with binary predictions (1 = failure, 0 = healthy)
        """
        if not self._fitted:
            raise ValueError("Classifier must be fitted before prediction.")

        if isinstance(X, pd.DataFrame):
            X_data = X.values
        else:
            X_data = np.asarray(X)

        X_scaled = self.scaler_.transform(X_data)
        return self.classifier_.predict(X_scaled)

