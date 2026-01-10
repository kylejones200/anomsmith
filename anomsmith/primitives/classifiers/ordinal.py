"""Ordinal classification models for predictive maintenance.

Ordinal models respect the natural ordering of health states
(Healthy < Warning < Distress) rather than treating them as independent classes.
"""

import logging
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import pandas as pd

try:
    import mord

    MORD_AVAILABLE = True
except ImportError:
    MORD_AVAILABLE = False
    mord = None  # type: ignore

from anomsmith.objects.health_state import HealthState, HealthStateView
from anomsmith.primitives.base import BaseEstimator

if TYPE_CHECKING:
    try:
        from timesmith.typing import SeriesLike
    except ImportError:
        SeriesLike = None

logger = logging.getLogger(__name__)


class OrdinalLogisticClassifier(BaseEstimator):
    """Ordinal logistic regression using mord library.

    Captures long-term degradation patterns with smooth ordinal boundaries.
    Respects the natural ordering of health states.

    Args:
        alpha: Regularization parameter (default 0.0)
        random_state: Random state for reproducibility
    """

    def __init__(
        self,
        alpha: float = 0.0,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize ordinal logistic classifier."""
        if not MORD_AVAILABLE:
            raise ImportError(
                "mord is required for OrdinalLogisticClassifier. "
                "Install with: pip install mord"
            )

        super().__init__(alpha=alpha, random_state=random_state)
        self.alpha = alpha
        self.random_state = random_state
        self.model_: Optional[mord.LogisticIT] = None  # type: ignore
        self._fitted = False

    def fit(
        self,
        y: Union[np.ndarray, pd.Series, "SeriesLike"],
        X: Union[np.ndarray, pd.DataFrame, None] = None,
    ) -> "OrdinalLogisticClassifier":
        """Fit the ordinal logistic regression model.

        Args:
            y: Target health states as integers (0=Healthy, 1=Warning, 2=Distress)
            X: Feature matrix (n_samples, n_features) - required for ordinal logistic regression

        Returns:
            Self for method chaining
        """
        if X is None:
            raise ValueError("X (feature matrix) is required for OrdinalLogisticClassifier")
        
        if isinstance(X, pd.DataFrame):
            X_data = X.values
        else:
            X_data = np.asarray(X)

        if isinstance(y, pd.Series):
            y_data = y.values
        else:
            y_data = np.asarray(y)

        # Ensure y is integer type
        y_data = y_data.astype(int)

        # Validate health state values
        unique_vals = np.unique(y_data)
        if not np.all(np.isin(unique_vals, [s.value for s in HealthState])):
            raise ValueError(
                f"Health states must be in [0, 1, 2], got unique values: {unique_vals}"
            )

        # Fit model
        self.model_ = mord.LogisticIT(alpha=self.alpha)  # type: ignore
        self.model_.fit(X_data, y_data)  # type: ignore

        self._fitted = True
        logger.debug(
            f"Fitted OrdinalLogisticClassifier: n_samples={len(X_data)}, "
            f"n_features={X_data.shape[1]}"
        )
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict health states.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Array of predicted health states (0=Healthy, 1=Warning, 2=Distress)
        """
        self._check_fitted()

        if isinstance(X, pd.DataFrame):
            X_data = X.values
        else:
            X_data = np.asarray(X)

        predictions = self.model_.predict(X_data)  # type: ignore
        # Ensure predictions are in valid range [0, 2]
        predictions = np.clip(predictions.astype(int), 0, 2)
        return predictions

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict health state probabilities.

        Note: mord doesn't provide predict_proba, so this returns
        approximate probabilities based on predicted class.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Array of shape (n_samples, 3) with probabilities for each health state
        """
        self._check_fitted()

        if isinstance(X, pd.DataFrame):
            X_data = X.values
        else:
            X_data = np.asarray(X)

        # mord doesn't provide predict_proba, so we use a workaround
        # Predict probabilities for ordinal thresholds
        predictions = self.predict(X_data)
        n_samples = len(X_data)

        # Vectorized probability assignment
        # Probability templates for each class: [p(Healthy), p(Warning), p(Distress)]
        proba_templates = np.array([
            [0.7, 0.2, 0.1],  # Class 0 (Healthy)
            [0.2, 0.6, 0.2],  # Class 1 (Warning)
            [0.1, 0.2, 0.7],  # Class 2 (Distress)
        ])

        # Use advanced indexing to assign probabilities (vectorized)
        proba = proba_templates[predictions]

        return proba

    def predict_health_states(
        self, X: Union[np.ndarray, pd.DataFrame], index: Optional[pd.Index] = None
    ) -> HealthStateView:
        """Predict health states as HealthStateView.

        Args:
            X: Feature matrix (n_samples, n_features)
            index: Optional index for the health states

        Returns:
            HealthStateView with predicted health states
        """
        if index is None:
            if isinstance(X, pd.DataFrame):
                index = X.index
            else:
                index = pd.RangeIndex(start=0, stop=len(X))

        predictions = self.predict(X)
        return HealthStateView(index=index, states=predictions)

