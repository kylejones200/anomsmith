"""Ensemble methods for anomaly detection."""

import logging
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
import pandas as pd

from anomsmith.objects.views import LabelView, ScoreView
from anomsmith.primitives.base import BaseDetector, BaseScorer

if TYPE_CHECKING:
    try:
        from timesmith.typing import PanelLike, SeriesLike
    except ImportError:
        SeriesLike = None
        PanelLike = None

logger = logging.getLogger(__name__)


class VotingEnsembleDetector(BaseDetector):
    """Voting ensemble that combines predictions from multiple detectors.

    An anomaly is flagged if at least `voting_threshold` detectors agree.

    Args:
        detectors: List of anomaly detectors or scorers to ensemble
        voting_threshold: Minimum number of detectors that must flag a sample as anomalous
        random_state: Random state for reproducibility (not used, kept for compatibility)
    """

    def __init__(
        self,
        detectors: List[BaseDetector | BaseScorer],
        voting_threshold: int = 2,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize VotingEnsembleDetector.

        Args:
            detectors: List of BaseDetector or BaseScorer instances
            voting_threshold: Minimum number of detectors that must agree
            random_state: Random state (not used, kept for compatibility)
        """
        if not detectors:
            raise ValueError("At least one detector is required")
        if voting_threshold < 1 or voting_threshold > len(detectors):
            raise ValueError(
                f"voting_threshold must be between 1 and {len(detectors)}. Got {voting_threshold}"
            )

        self.detectors = detectors
        self.voting_threshold = voting_threshold
        self.random_state = random_state
        super().__init__(
            detectors=detectors, voting_threshold=voting_threshold, random_state=random_state
        )
        self._fitted = False

    def fit(
        self,
        y: Union[np.ndarray, pd.Series, "SeriesLike"],
        X: Union[np.ndarray, pd.DataFrame, "PanelLike", None] = None,
    ) -> "VotingEnsembleDetector":
        """Fit all detectors in the ensemble.

        Args:
            y: Training time series
            X: Optional features (not used)

        Returns:
            Self for method chaining
        """
        for detector in self.detectors:
            detector.fit(y, X)

        self._fitted = True
        logger.debug(
            f"Fitted VotingEnsembleDetector with {len(self.detectors)} detectors, "
            f"voting_threshold={self.voting_threshold}"
        )
        return self

    def score(self, y: Union[np.ndarray, pd.Series, "SeriesLike"]) -> ScoreView:
        """Compute ensemble scores as mean of individual detector scores.

        Args:
            y: Time series to score

        Returns:
            ScoreView with average anomaly scores
        """
        self._check_fitted()

        if isinstance(y, pd.Series):
            index = y.index
        else:
            index = pd.RangeIndex(start=0, stop=len(y))

        # Get scores from all detectors - vectorized
        all_scores = []
        for detector in self.detectors:
            if isinstance(detector, BaseScorer):
                score_view = detector.score(y)
            else:
                score_view = detector.score(y)
            all_scores.append(score_view.scores)

        # Average scores - vectorized
        scores_array = np.array(all_scores)
        ensemble_scores = np.mean(scores_array, axis=0)

        return ScoreView(index=index, scores=ensemble_scores)

    def predict(self, y: Union[np.ndarray, pd.Series, "SeriesLike"]) -> LabelView:
        """Predict anomalies using voting.

        Args:
            y: Time series to detect anomalies in

        Returns:
            LabelView with binary labels
        """
        self._check_fitted()

        if isinstance(y, pd.Series):
            index = y.index
        else:
            index = pd.RangeIndex(start=0, stop=len(y))

        # Get predictions from all detectors - vectorized
        all_predictions = []
        for detector in self.detectors:
            if isinstance(detector, BaseScorer):
                # For scorers, we need to score and then apply a threshold
                # Use a simple percentile threshold
                score_view = detector.score(y)
                scores = score_view.scores
                threshold = np.percentile(scores, 95) if len(scores) > 0 else 0
                predictions = (scores > threshold).astype(int)
            else:
                label_view = detector.predict(y)
                predictions = label_view.labels
            all_predictions.append(predictions)

        # Count votes (how many detectors flagged as anomaly) - vectorized
        predictions_array = np.array(all_predictions)
        votes = np.sum(predictions_array == 1, axis=0)

        # Flag as anomaly if voting_threshold or more detectors agree
        labels = (votes >= self.voting_threshold).astype(int)

        return LabelView(index=index, labels=labels)

    def get_vote_counts(self, y: Union[np.ndarray, pd.Series, "SeriesLike"]) -> np.ndarray:
        """Get vote counts for each sample.

        Args:
            y: Time series to analyze

        Returns:
            Array of vote counts (number of detectors that flagged each sample as anomalous)
        """
        self._check_fitted()

        # Get predictions from all detectors - vectorized
        all_predictions = []
        for detector in self.detectors:
            if isinstance(detector, BaseScorer):
                score_view = detector.score(y)
                scores = score_view.scores
                threshold = np.percentile(scores, 95) if len(scores) > 0 else 0
                predictions = (scores > threshold).astype(int)
            else:
                label_view = detector.predict(y)
                predictions = label_view.labels
            all_predictions.append(predictions)

        predictions_array = np.array(all_predictions)
        vote_counts = np.sum(predictions_array == 1, axis=0)
        return vote_counts

