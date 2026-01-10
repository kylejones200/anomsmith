"""Time series drift detection using forecasting models.

Detects drift by comparing actual values to forecasts from statistical models.
"""

import logging
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    ARIMA = None  # type: ignore
    sm = None  # type: ignore

from anomsmith.objects.views import LabelView, ScoreView
from anomsmith.primitives.base import BaseDetector

if TYPE_CHECKING:
    try:
        from timesmith.typing import SeriesLike
    except ImportError:
        SeriesLike = None

logger = logging.getLogger(__name__)


class ARIMADriftDetector(BaseDetector):
    """ARIMA-based drift detector for time series.

    Uses ARIMA forecasting to detect drift. If actual values diverge significantly
    from forecasts, the series is flagged as drifting.

    Args:
        order: ARIMA order (p, d, q). Default (1, 1, 1)
        threshold_std: Number of standard deviations for drift threshold (default 2.0)
        random_state: Random state for reproducibility (not used, kept for compatibility)
    """

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 1, 1),
        threshold_std: float = 2.0,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize ARIMADriftDetector.

        Args:
            order: ARIMA order (p, d, q)
            threshold_std: Number of standard deviations for drift threshold
            random_state: Random state (not used, kept for compatibility)
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError(
                "statsmodels is required for ARIMADriftDetector. "
                "Install with: pip install statsmodels"
            )

        self.order = order
        self.threshold_std = threshold_std
        self.random_state = random_state
        self.model_: ARIMA | None = None  # type: ignore
        self.fitted_model_: Any | None = None  # type: ignore
        self.residual_std_: float = 0.0
        super().__init__(order=order, threshold_std=threshold_std, random_state=random_state)
        self._fitted = False

    def fit(
        self,
        y: Union[np.ndarray, pd.Series, "SeriesLike"],
        X: Union[np.ndarray, pd.DataFrame, "PanelLike", None] = None,
    ) -> "ARIMADriftDetector":
        """Fit the ARIMA model on training data.

        Args:
            y: Training time series (1D)
            X: Optional features (not used for ARIMA)

        Returns:
            Self for method chaining
        """
        if isinstance(y, pd.Series):
            values = y.values
        else:
            values = np.asarray(y)

        if values.ndim > 1:
            if values.shape[1] > 1:
                raise ValueError("ARIMADriftDetector only supports univariate time series.")
            values = values.flatten()

        # Fit ARIMA model
        try:
            self.model_ = ARIMA(values, order=self.order)  # type: ignore
            self.fitted_model_ = self.model_.fit()  # type: ignore
            # Compute residual standard deviation for threshold
            residuals = self.fitted_model_.resid  # type: ignore
            self.residual_std_ = np.std(residuals)
            self._fitted = True
            logger.debug(f"Fitted ARIMADriftDetector: residual_std={self.residual_std_:.4f}")
        except Exception as e:
            logger.error(f"Error fitting ARIMA model: {e}")
            raise

        return self

    def score(self, y: Union[np.ndarray, pd.Series, "SeriesLike"]) -> ScoreView:
        """Score drift using ARIMA residuals.

        Args:
            y: Time series to score

        Returns:
            ScoreView with drift scores (residual magnitudes)
        """
        self._check_fitted()

        if isinstance(y, pd.Series):
            index = y.index
            values = y.values
        else:
            index = pd.RangeIndex(start=0, stop=len(y))
            values = np.asarray(y)

        if values.ndim > 1:
            if values.shape[1] > 1:
                raise ValueError("ARIMADriftDetector only supports univariate time series.")
            values = values.flatten()

        # Generate forecasts using fitted model
        if self.fitted_model_ is None:
            raise ValueError("Model must be fitted before scoring. Call fit() first.")

        try:
            # Use fitted model's predict method
            forecast = self.fitted_model_.predict(start=1, end=len(values), dynamic=False)  # type: ignore
            # Compute residuals (actual - forecast)
            residuals = values[1:] - forecast
            # Score is absolute residual normalized by residual std
            scores = np.abs(residuals) / (self.residual_std_ + 1e-10)
            # Pad first value (no prediction for first point)
            scores = np.concatenate([[0.0], scores])
        except Exception as e:
            logger.error(f"Error generating ARIMA forecast: {e}")
            # Fallback: return zeros
            scores = np.zeros(len(values))
            logger.warning("Failed to compute ARIMA scores, returning zeros")

        return ScoreView(index=index, scores=scores)

    def predict(self, y: Union[np.ndarray, pd.Series, "SeriesLike"]) -> LabelView:
        """Predict drift labels.

        Args:
            y: Time series to detect drift in

        Returns:
            LabelView with binary labels (1 = drift, 0 = normal)
        """
        score_view = self.score(y)
        # Flag as drift if score exceeds threshold
        labels = (score_view.scores > self.threshold_std).astype(int)
        return LabelView(index=score_view.index, labels=labels)

