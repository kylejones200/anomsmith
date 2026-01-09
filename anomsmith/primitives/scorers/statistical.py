"""Statistical anomaly detection scorers."""

import logging
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import pandas as pd

from anomsmith.objects.views import ScoreView
from anomsmith.primitives.base import BaseScorer

if TYPE_CHECKING:
    try:
        from timesmith.typing import SeriesLike
    except ImportError:
        SeriesLike = None

logger = logging.getLogger(__name__)


class ZScoreScorer(BaseScorer):
    """Z-score based anomaly scorer.

    Computes absolute Z-scores relative to mean and standard deviation.
    Higher scores indicate more anomalous points.

    Args:
        n_std: Number of standard deviations (used for thresholding, not scoring)
        random_state: Random state for reproducibility (not used, kept for compatibility)
    """

    def __init__(self, n_std: float = 3.0, random_state: Optional[int] = None) -> None:
        self.n_std = n_std
        self.random_state = random_state
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        super().__init__(n_std=n_std, random_state=random_state)
        self._fitted = False

    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    ) -> "ZScoreScorer":
        """Fit the scorer by computing mean and standard deviation.

        Args:
            y: Training data
            X: Optional features (not used)

        Returns:
            Self for method chaining
        """
        if isinstance(y, pd.Series):
            values = y.values
        else:
            values = y

        # Handle 1D and 2D cases
        if values.ndim == 1:
            values = values.reshape(-1, 1)

        self.mean_ = np.mean(values, axis=0)
        self.std_ = np.std(values, axis=0)
        # Avoid division by zero
        self.std_ = np.where(self.std_ == 0, 1.0, self.std_)

        self._fitted = True
        logger.debug(f"Fitted ZScoreScorer: mean={self.mean_}, std={self.std_}")
        return self

    def score(self, y: Union[np.ndarray, pd.Series, "SeriesLike"]) -> ScoreView:
        """Score anomalies using Z-scores.

        Args:
            y: Time series to score

        Returns:
            ScoreView with absolute Z-scores
        """
        self._check_fitted()

        if isinstance(y, pd.Series):
            index = y.index
            values = y.values
        else:
            index = pd.RangeIndex(start=0, stop=len(y))
            values = y

        # Handle 1D and 2D cases
        if values.ndim == 1:
            values = values.reshape(-1, 1)

        z_scores = np.abs((values - self.mean_) / self.std_)
        # Return maximum Z-score across features
        if z_scores.ndim > 1:
            scores = np.max(z_scores, axis=1)
        else:
            scores = z_scores.flatten()

        return ScoreView(index=index, scores=scores)


class IQRScorer(BaseScorer):
    """Interquartile Range (IQR) based outlier scorer.

    Computes outlier scores based on IQR bounds.
    Higher scores indicate more anomalous points.

    Args:
        factor: IQR multiplier for outlier bounds (default: 1.5)
        random_state: Random state for reproducibility (not used, kept for compatibility)
    """

    def __init__(self, factor: float = 1.5, random_state: Optional[int] = None) -> None:
        self.factor = factor
        self.random_state = random_state
        self.q1_: np.ndarray | None = None
        self.q3_: np.ndarray | None = None
        self.iqr_: np.ndarray | None = None
        super().__init__(factor=factor, random_state=random_state)
        self._fitted = False

    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    ) -> "IQRScorer":
        """Fit the scorer by computing quartiles.

        Args:
            y: Training data
            X: Optional features (not used)

        Returns:
            Self for method chaining
        """
        if isinstance(y, pd.Series):
            values = y.values
        else:
            values = y

        # Handle 1D and 2D cases
        if values.ndim == 1:
            values = values.reshape(-1, 1)

        self.q1_ = np.percentile(values, 25, axis=0)
        self.q3_ = np.percentile(values, 75, axis=0)
        self.iqr_ = self.q3_ - self.q1_
        # Avoid division by zero
        self.iqr_ = np.where(self.iqr_ == 0, 1.0, self.iqr_)

        self._fitted = True
        logger.debug(f"Fitted IQRScorer: q1={self.q1_}, q3={self.q3_}")
        return self

    def score(self, y: Union[np.ndarray, pd.Series]) -> ScoreView:
        """Score anomalies using IQR bounds.

        Args:
            y: Time series to score

        Returns:
            ScoreView with IQR-based scores
        """
        self._check_fitted()

        if isinstance(y, pd.Series):
            index = y.index
            values = y.values
        else:
            index = pd.RangeIndex(start=0, stop=len(y))
            values = y

        # Handle 1D and 2D cases
        if values.ndim == 1:
            values = values.reshape(-1, 1)

        # Check if outside bounds for any feature
        lower_bound = self.q1_ - self.factor * self.iqr_
        upper_bound = self.q3_ + self.factor * self.iqr_

        outlier_mask = (values < lower_bound) | (values > upper_bound)
        if outlier_mask.ndim > 1:
            # Vectorized scoring: distance from bounds for all samples
            # Compute distances for all samples at once
            dist_lower = np.maximum(0, lower_bound - values)  # (n_samples, n_features)
            dist_upper = np.maximum(0, values - upper_bound)  # (n_samples, n_features)
            
            # Max distance across features for each sample
            max_dist_lower = np.max(dist_lower, axis=1)  # (n_samples,)
            max_dist_upper = np.max(dist_upper, axis=1)  # (n_samples,)
            
            # Check if any feature is outside bounds for each sample
            is_outlier = np.any(outlier_mask, axis=1)  # (n_samples,)
            
            # For outliers: use max distance from bounds
            # For normal: use negative min distance to nearest bound
            outlier_scores = np.maximum(max_dist_lower, max_dist_upper)
            normal_scores = -np.minimum(max_dist_lower, max_dist_upper)
            
            # Normalize by max IQR
            scale = self.iqr_.max() + 1e-10
            scores = np.where(is_outlier, outlier_scores / scale, normal_scores / scale)
        else:
            # 1D case: simple binary scoring
            scores = np.where(outlier_mask, 1.0, 0.0).astype(float)

        return ScoreView(index=index, scores=scores)

