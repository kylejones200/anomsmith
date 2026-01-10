"""Cox Proportional Hazards survival model.

Base implementation for Cox regression-based survival analysis.
"""

import logging
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import pandas as pd

from anomsmith.primitives.base import BaseEstimator

if TYPE_CHECKING:
    try:
        from timesmith.typing import SeriesLike
    except ImportError:
        SeriesLike = None

logger = logging.getLogger(__name__)


class CoxSurvivalModel(BaseEstimator):
    """Base Cox Proportional Hazards survival model.

    Abstract base class for Cox regression models. Subclasses implement
    specific variants (linear CoxPH, neural DeepSurv, etc.).

    Args:
        random_state: Random state for reproducibility
    """

    def __init__(self, random_state: Optional[int] = None) -> None:
        """Initialize Cox survival model."""
        self.random_state = random_state
        super().__init__(random_state=random_state)
        self._fitted = False

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame, None],
        durations: Union[np.ndarray, pd.Series],
        events: Union[np.ndarray, pd.Series, None] = None,
        y: Union[np.ndarray, pd.Series, "SeriesLike", None] = None,
    ) -> "CoxSurvivalModel":
        """Fit the survival model.

        Args:
            X: Feature matrix (n_samples, n_features)
            durations: Time-to-event values (n_samples,)
            events: Event indicators (1 = event occurred, 0 = censored)
            y: Optional target (not used, kept for interface compatibility)

        Returns:
            Self for method chaining
        """
        raise NotImplementedError("Subclasses must implement fit method")

    def predict_survival_function(
        self, X: Union[np.ndarray, pd.DataFrame], time_points: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """Predict survival function S(t|X).

        Args:
            X: Feature matrix (n_samples, n_features)
            time_points: Optional time points for prediction

        Returns:
            DataFrame with survival probabilities (rows = time points, cols = samples)
        """
        raise NotImplementedError("Subclasses must implement predict_survival_function method")

    def predict_risk_score(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict relative risk scores (hazard ratios).

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Array of risk scores (higher = higher failure risk)
        """
        raise NotImplementedError("Subclasses must implement predict_risk_score method")

    def predict_time_to_failure(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Predict median time-to-failure (where survival probability = threshold).

        Args:
            X: Feature matrix (n_samples, n_features)
            threshold: Survival probability threshold (default 0.5)

        Returns:
            Array of predicted median time-to-failure
        """
        surv_df = self.predict_survival_function(X)
        # Vectorized: find time point where survival probability crosses threshold
        time_index = surv_df.index.values
        surv_values = surv_df.values  # Shape: (n_time_points, n_samples)
        
        # For each sample (column), find first time where survival < threshold
        below_threshold = surv_values < threshold  # Shape: (n_time_points, n_samples)
        
        # Find first index where below_threshold is True for each sample
        # argmax finds first True (since False=0, True=1)
        # If no True values, argmax returns 0, which we need to handle
        ttf_indices = np.argmax(below_threshold, axis=0)  # Shape: (n_samples,)
        
        # Handle case where no value is below threshold: use last index
        all_above_threshold = ~below_threshold.any(axis=0)
        ttf_indices[all_above_threshold] = len(time_index) - 1
        
        # Handle case where first value is below threshold correctly
        # If first is below, it's valid - use it. Otherwise argmax handles it.
        first_below = below_threshold[0, :]
        # Only adjust if first is not below but argmax returned 0 (no True found)
        # Actually, argmax behavior: returns 0 if no True, so we need to check
        # If first is False but argmax is 0, then no values below threshold
        # This is already handled by all_above_threshold check
        
        ttf = time_index[ttf_indices]
        return ttf

