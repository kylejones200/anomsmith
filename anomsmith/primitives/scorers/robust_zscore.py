"""Robust Z-Score based anomaly scorer."""

import logging
from typing import TYPE_CHECKING, Union

import numpy as np
import pandas as pd

from anomsmith.objects.views import ScoreView
from anomsmith.primitives.base import BaseScorer
from anomsmith.primitives.scaling import robust_zscore

if TYPE_CHECKING:
    try:
        from timesmith.typing import SeriesLike
    except ImportError:
        SeriesLike = None

logger = logging.getLogger(__name__)


class RobustZScoreScorer(BaseScorer):
    """Robust Z-Score anomaly scorer.

    Uses median and MAD for robust scaling, then computes absolute z-scores.
    Higher scores indicate more anomalous points.
    """

    def __init__(self, epsilon: float = 1e-8) -> None:
        """Initialize RobustZScoreScorer.

        Args:
            epsilon: Small value to prevent division by zero in MAD
        """
        self.epsilon = epsilon
        super().__init__(epsilon=epsilon)
        self._fitted = False

    def fit(
        self,
        y: Union[np.ndarray, pd.Series, "SeriesLike"],
        X: np.ndarray | pd.DataFrame | None = None,
    ) -> "RobustZScoreScorer":
        """Fit the scorer (no-op for this scorer).

        Args:
            y: Target values (not used, kept for interface compatibility)
            X: Optional features (not used)

        Returns:
            Self for method chaining
        """
        logger.debug("Fitting RobustZScoreScorer (no-op)")
        self._fitted = True
        return self

    def score(self, y: Union[np.ndarray, pd.Series, "SeriesLike"]) -> ScoreView:
        """Score anomalies using robust z-scores.

        Args:
            y: Time series to score

        Returns:
            ScoreView with absolute robust z-scores
        """
        if isinstance(y, pd.Series):
            index = y.index
            values = y.values
        else:
            index = pd.RangeIndex(start=0, stop=len(y))
            values = y

        z_scores = robust_zscore(values, epsilon=self.epsilon)
        abs_z_scores = np.abs(z_scores)

        return ScoreView(index=index, scores=abs_z_scores)

