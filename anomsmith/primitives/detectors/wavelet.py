"""Wavelet-based anomaly detection detector.

Uses wavelet decomposition to detect anomalies in time series by identifying
large coefficients in detail levels, which indicate sudden changes or anomalies.
"""

import logging
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import pandas as pd

try:
    import pywt
except ImportError:
    pywt = None  # type: ignore

from anomsmith.objects.views import LabelView, ScoreView
from anomsmith.primitives.base import BaseDetector

if TYPE_CHECKING:
    try:
        from timesmith.typing import SeriesLike
    except ImportError:
        SeriesLike = None

logger = logging.getLogger(__name__)


class WaveletDetector(BaseDetector):
    """Wavelet-based anomaly detector for time series.

    Detects anomalies by identifying large coefficients in wavelet detail levels,
    which indicate sudden changes or anomalies.

    Args:
        wavelet: Wavelet type (e.g., 'db4', 'haar', 'bior2.2'). Default 'db4'.
        threshold_factor: Threshold factor for anomaly detection (in terms of MAD). Default 3.0.
        level: Decomposition level. Default 5.
        random_state: Random state for reproducibility (not used, kept for compatibility)
    """

    def __init__(
        self,
        wavelet: str = "db4",
        threshold_factor: float = 3.0,
        level: int = 5,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize WaveletDetector.

        Args:
            wavelet: Wavelet type (e.g., 'db4', 'haar', 'bior2.2')
            threshold_factor: Threshold factor for anomaly detection (in terms of MAD)
            level: Decomposition level
            random_state: Random state (not used, kept for compatibility)
        """
        if pywt is None:
            raise ImportError(
                "PyWavelets is required for WaveletDetector. "
                "Install with: pip install PyWavelets"
            )

        self.wavelet = wavelet
        self.threshold_factor = threshold_factor
        self.level = level
        self.random_state = random_state
        super().__init__(
            wavelet=wavelet, threshold_factor=threshold_factor, level=level, random_state=random_state
        )
        self._fitted = False

    def fit(
        self,
        y: Union[np.ndarray, pd.Series, "SeriesLike"],
        X: Union[np.ndarray, pd.DataFrame, "PanelLike", None] = None,
    ) -> "WaveletDetector":
        """Fit the wavelet detector.

        Args:
            y: Time series data (1D)
            X: Optional features (not used)

        Returns:
            Self for method chaining
        """
        if isinstance(y, pd.Series):
            values = y.values
        else:
            values = np.asarray(y)

        if values.ndim > 1:
            if values.shape[1] > 1:
                raise ValueError("WaveletDetector only supports univariate time series.")
            values = values.flatten()

        # Store data for scoring
        self.data_ = values
        self._fitted = True
        logger.debug(f"Fitted WaveletDetector with wavelet={self.wavelet}, level={self.level}")
        return self

    def score(self, y: Union[np.ndarray, pd.Series, "SeriesLike"]) -> ScoreView:
        """Score anomalies using wavelet decomposition.

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
            values = np.asarray(y)

        if values.ndim > 1:
            if values.shape[1] > 1:
                raise ValueError("WaveletDetector only supports univariate time series.")
            values = values.flatten()

        # Perform wavelet decomposition
        coeffs = pywt.wavedec(values, self.wavelet, level=self.level)

        # Focus on detail coefficients (high-frequency anomalies)
        detail_coeffs = coeffs[1:]

        # Calculate threshold for each detail level
        anomaly_scores = np.zeros(len(values))

        for detail in detail_coeffs:
            if len(detail) == 0:
                continue

            # Use robust statistics (median, MAD) - vectorized
            detail_abs = np.abs(detail)
            median_detail = np.median(detail_abs)
            mad = np.median(np.abs(detail_abs - median_detail))
            threshold = median_detail + self.threshold_factor * (mad / 0.6745)

            # Find anomalies in this detail level - vectorized
            anomaly_mask = detail_abs > threshold

            if not np.any(anomaly_mask):
                continue

            # Map back to original time indices - vectorized
            scale_factor = len(values) // len(detail)
            anomaly_indices = np.where(anomaly_mask)[0]

            # Vectorized mapping using broadcasting
            start_indices = anomaly_indices * scale_factor
            end_indices = np.minimum((anomaly_indices + 1) * scale_factor, len(values))

            # Add scores efficiently using vectorized operations
            for start_idx, end_idx, score in zip(
                start_indices, end_indices, detail_abs[anomaly_mask]
            ):
                anomaly_scores[start_idx:end_idx] += score

        return ScoreView(index=index, scores=anomaly_scores)

    def predict(self, y: Union[np.ndarray, pd.Series, "SeriesLike"]) -> LabelView:
        """Predict anomaly labels.

        Args:
            y: Time series to detect anomalies in

        Returns:
            LabelView with binary labels
        """
        score_view = self.score(y)
        # Threshold based on percentile
        scores = score_view.scores
        threshold = np.percentile(scores[scores > 0], 95) if np.any(scores > 0) else 0
        labels = (scores > threshold).astype(int)

        return LabelView(index=score_view.index, labels=labels)


