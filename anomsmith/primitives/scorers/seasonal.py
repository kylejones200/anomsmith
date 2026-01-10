"""Seasonal baseline anomaly scorer.

Detects anomalies by comparing values to seasonal baselines (e.g., weekly, monthly patterns).
"""

import logging
from typing import TYPE_CHECKING, Literal, Optional, Union

import numpy as np
import pandas as pd

from anomsmith.objects.views import ScoreView
from anomsmith.primitives.base import BaseScorer

if TYPE_CHECKING:
    try:
        from timesmith.typing import PanelLike, SeriesLike
    except ImportError:
        SeriesLike = None
        PanelLike = None

logger = logging.getLogger(__name__)


class SeasonalBaselineScorer(BaseScorer):
    """Seasonal baseline anomaly scorer.

    Calculates seasonal baselines (e.g., weekly, monthly) and scores points
    that deviate significantly from expected seasonal patterns.

    Args:
        seasonality: Seasonality to use. Options: 'week', 'month', 'day', 'hour'.
        random_state: Random state for reproducibility (not used, kept for compatibility)
    """

    def __init__(
        self,
        seasonality: Literal["week", "month", "day", "hour"] = "week",
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize SeasonalBaselineScorer.

        Args:
            seasonality: Seasonality to use ('week', 'month', 'day', 'hour')
            random_state: Random state (not used, kept for compatibility)
        """
        self.seasonality = seasonality
        self.random_state = random_state
        self.seasonal_stats_: pd.DataFrame | None = None
        super().__init__(seasonality=seasonality, random_state=random_state)
        self._fitted = False

    def _get_seasonal_key(self, dates: pd.DatetimeIndex) -> pd.Series:
        """Extract seasonal key from dates based on seasonality type."""
        seasonality_map = {
            "week": lambda d: d.isocalendar().week,
            "month": lambda d: d.month,
            "day": lambda d: d.dayofyear,
            "hour": lambda d: d.hour,
        }
        if self.seasonality not in seasonality_map:
            raise ValueError(f"Unknown seasonality: {self.seasonality}")
        return pd.Series(seasonality_map[self.seasonality](dates), index=dates)

    def fit(
        self,
        y: Union[np.ndarray, pd.Series, "SeriesLike"],
        X: Union[np.ndarray, pd.DataFrame, "PanelLike", None] = None,
    ) -> "SeasonalBaselineScorer":
        """Fit the scorer by computing seasonal baselines.

        Args:
            y: Time series with datetime index
            X: Optional features (not used)

        Returns:
            Self for method chaining
        """
        if isinstance(y, pd.Series):
            index = y.index
            values = y.values
        else:
            # For numpy arrays, create a default datetime index
            index = pd.date_range("2020-01-01", periods=len(y), freq="D")
            values = y

        if not isinstance(index, pd.DatetimeIndex):
            raise ValueError(
                "SeasonalBaselineScorer requires a DatetimeIndex. "
                f"Got {type(index)} instead."
            )

        # Compute seasonal statistics
        df = pd.DataFrame({"value": values, "date": index})
        df["seasonal_key"] = self._get_seasonal_key(index)

        seasonal_stats = df.groupby("seasonal_key").agg({"value": ["mean", "std"]}).reset_index()
        seasonal_stats.columns = ["seasonal_key", "mean", "std"]
        seasonal_stats["std"] = seasonal_stats["std"].fillna(0)
        seasonal_stats["std"] = seasonal_stats["std"].replace(0, 1.0)  # Avoid division by zero
        self.seasonal_stats_ = seasonal_stats

        self._fitted = True
        logger.debug(
            f"Fitted SeasonalBaselineScorer with {len(seasonal_stats)} seasonal periods"
        )
        return self

    def score(self, y: Union[np.ndarray, pd.Series, "SeriesLike"]) -> ScoreView:
        """Score anomalies using seasonal baseline.

        Args:
            y: Time series to score

        Returns:
            ScoreView with seasonal z-scores
        """
        self._check_fitted()

        if isinstance(y, pd.Series):
            index = y.index
            values = y.values
        else:
            index = pd.RangeIndex(start=0, stop=len(y))
            values = y

        if not isinstance(index, pd.DatetimeIndex):
            raise ValueError(
                "SeasonalBaselineScorer requires a DatetimeIndex. "
                f"Got {type(index)} instead."
            )

        # Get seasonal keys for scoring data
        seasonal_keys = self._get_seasonal_key(index)

        # Merge with seasonal stats - vectorized pandas operation
        df = pd.DataFrame({"value": values, "seasonal_key": seasonal_keys}, index=index)
        df = df.merge(self.seasonal_stats_, on="seasonal_key", how="left")

        # Compute Z-scores relative to seasonal baseline - vectorized
        z_scores = np.abs((df["value"] - df["mean"]) / df["std"])
        scores = z_scores.fillna(0).values

        return ScoreView(index=index, scores=scores)

