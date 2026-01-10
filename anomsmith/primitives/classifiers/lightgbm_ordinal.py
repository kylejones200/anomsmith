"""LightGBM-based ordinal regressor for health state prediction.

Uses LightGBM regression to predict ordinal health states, then rounds
and clips to valid health state values. Handles nonlinearities and sharp thresholds.
"""

import logging
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None  # type: ignore

from anomsmith.objects.health_state import HealthState, HealthStateView
from anomsmith.primitives.base import BaseEstimator

if TYPE_CHECKING:
    try:
        from timesmith.typing import SeriesLike
    except ImportError:
        SeriesLike = None

logger = logging.getLogger(__name__)


class LightGBMOrdinalClassifier(BaseEstimator):
    """LightGBM ordinal regressor for health state prediction.

    Uses regression objective with rounding/clipping to predict ordinal health states.
    Handles nonlinearities and sharp thresholds quickly.

    Args:
        n_estimators: Number of boosting rounds (default 100)
        learning_rate: Learning rate (default 0.1)
        max_depth: Maximum tree depth (default -1, unlimited)
        random_state: Random state for reproducibility
        verbosity: Verbosity level (-1 = silent)
        **kwargs: Additional LightGBM parameters
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = -1,
        random_state: Optional[int] = None,
        verbosity: int = -1,
        **kwargs,
    ) -> None:
        """Initialize LightGBM ordinal classifier."""
        if not LIGHTGBM_AVAILABLE:
            raise ImportError(
                "lightgbm is required for LightGBMOrdinalClassifier. "
                "Install with: pip install lightgbm"
            )

        super().__init__(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            verbosity=verbosity,
            **kwargs,
        )
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.verbosity = verbosity
        self.kwargs = kwargs
        self.model_: Optional[lgb.Booster] = None  # type: ignore
        self._fitted = False

    def fit(
        self,
        y: Union[np.ndarray, pd.Series, "SeriesLike"],
        X: Union[np.ndarray, pd.DataFrame, None] = None,
    ) -> "LightGBMOrdinalClassifier":
        """Fit the LightGBM ordinal regressor.

        Args:
            y: Target health states as integers (0=Healthy, 1=Warning, 2=Distress)
            X: Feature matrix (n_samples, n_features) - required for LightGBM

        Returns:
            Self for method chaining
        """
        if X is None:
            raise ValueError("X (feature matrix) is required for LightGBMOrdinalClassifier")
        
        if isinstance(X, pd.DataFrame):
            X_data = X.values
            feature_names = list(X.columns)
        else:
            X_data = np.asarray(X)
            feature_names = [f"feature_{i}" for i in range(X_data.shape[1])]

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

        # Prepare LightGBM dataset
        train_data = lgb.Dataset(X_data, label=y_data, feature_name=feature_names)  # type: ignore

        # LightGBM parameters
        params = {
            "objective": "regression_l1",  # L1 regression for ordinal prediction
            "verbosity": self.verbosity,
            "random_state": self.random_state,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            **self.kwargs,
        }

        # Train model
        self.model_ = lgb.train(  # type: ignore
            params, train_data, num_boost_round=self.n_estimators
        )

        self._fitted = True
        logger.debug(
            f"Fitted LightGBMOrdinalClassifier: n_estimators={self.n_estimators}, "
            f"n_samples={len(X_data)}, n_features={X_data.shape[1]}"
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

        # Predict continuous values, then round and clip to valid range
        predictions = self.model_.predict(X_data)  # type: ignore
        predictions = np.round(predictions).clip(0, 2).astype(int)
        return predictions

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict health state probabilities (approximated from regression).

        Converts continuous regression predictions to ordinal probabilities
        using soft assignment based on distance to each class.

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

        # Predict continuous values
        continuous_preds = self.model_.predict(X_data)  # type: ignore
        continuous_preds = continuous_preds.reshape(-1, 1)  # Shape: (n_samples, 1)

        # Vectorized: compute distance from each class for all samples
        class_centers = np.arange(3, dtype=float).reshape(1, -1)  # Shape: (1, 3)
        distances = np.abs(continuous_preds - class_centers)  # Shape: (n_samples, 3)

        # Softmax-like probabilities (closer = higher probability)
        exp_distances = np.exp(-distances)  # Shape: (n_samples, 3)
        proba = exp_distances / exp_distances.sum(axis=1, keepdims=True)  # Normalize

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

