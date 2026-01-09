"""Change point based anomaly detector."""

import logging
from typing import TYPE_CHECKING, Union

import numpy as np
import pandas as pd

from anomsmith.objects.views import LabelView, ScoreView
from anomsmith.primitives.base import BaseDetector

if TYPE_CHECKING:
    try:
        from timesmith.typing import SeriesLike
    except ImportError:
        SeriesLike = None

logger = logging.getLogger(__name__)


class ChangePointDetector(BaseDetector):
    """Change point detector using rolling window statistics.

    Detects anomalies by comparing local statistics to global statistics.
    Uses mean and std deviation in rolling windows.
    """

    def __init__(self, window_size: int = 10, threshold_multiplier: float = 3.0) -> None:
        """Initialize ChangePointDetector.

        Args:
            window_size: Size of rolling window for local statistics
            threshold_multiplier: Multiplier for standard deviation threshold
        """
        self.window_size = window_size
        self.threshold_multiplier = threshold_multiplier
        self._global_mean: float = 0.0
        self._global_std: float = 1.0
        super().__init__(window_size=window_size, threshold_multiplier=threshold_multiplier)
        self._fitted = False

    def fit(
        self,
        y: Union[np.ndarray, pd.Series, "SeriesLike"],
        X: np.ndarray | pd.DataFrame | None = None,
    ) -> "ChangePointDetector":
        """Fit the detector on training data.

        Args:
            y: Training time series
            X: Optional features (not used)

        Returns:
            Self for method chaining
        """
        if isinstance(y, pd.Series):
            values = y.values
        else:
            values = y

        self._global_mean = float(np.mean(values))
        self._global_std = float(np.std(values))
        if self._global_std == 0:
            self._global_std = 1.0

        self._fitted = True
        logger.debug(
            f"Fitted ChangePointDetector: mean={self._global_mean:.3f}, "
            f"std={self._global_std:.3f}"
        )
        return self

    def score(self, y: np.ndarray | pd.Series) -> ScoreView:
        """Score anomalies using change point detection.

        Args:
            y: Time series to score

        Returns:
            ScoreView with anomaly scores
        """
        self._check_fitted()

        if isinstance(y, pd.Series):
            index = y.index
            values = y.values
        else:
            index = pd.RangeIndex(start=0, stop=len(y))
            values = y

        n = len(values)
        scores = np.zeros(n)

        # Vectorized computation using pandas rolling for efficiency
        # Convert to Series for rolling operations
        series = pd.Series(values)
        
        # Compute rolling statistics (vectorized)
        rolling_mean = series.rolling(window=self.window_size, center=True, min_periods=1).mean()
        rolling_std = series.rolling(window=self.window_size, center=True, min_periods=1).std()
        # Handle zero std (avoid division by zero)
        rolling_std = rolling_std.fillna(1.0).replace(0.0, 1.0).values

        # Vectorized deviation calculations
        global_dev = np.abs(values - self._global_mean) / self._global_std
        local_dev = np.abs(values - rolling_mean.values) / rolling_std

        # Combine global and local deviations (element-wise max)
        scores = np.maximum(global_dev, local_dev)

        return ScoreView(index=index, scores=scores)

    def predict(self, y: np.ndarray | pd.Series) -> LabelView:
        """Predict anomaly labels.

        Args:
            y: Time series to detect anomalies in

        Returns:
            LabelView with binary labels
        """
        score_view = self.score(y)
        threshold = self.threshold_multiplier
        labels = (score_view.scores >= threshold).astype(int)

        return LabelView(index=score_view.index, labels=labels)

