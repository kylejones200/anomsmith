"""Ensemble methods for anomaly detection."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

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
        detectors: list[BaseDetector | BaseScorer],
        voting_threshold: int = 2,
        random_state: int | None = None,
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
            detectors=detectors,
            voting_threshold=voting_threshold,
            random_state=random_state,
        )
        self._fitted = False

    def fit(
        self,
        y: np.ndarray | pd.Series | SeriesLike,
        X: np.ndarray | pd.DataFrame | PanelLike | None = None,
    ) -> VotingEnsembleDetector:
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

    def score(self, y: np.ndarray | pd.Series | SeriesLike) -> ScoreView:
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

    def predict(self, y: np.ndarray | pd.Series | SeriesLike) -> LabelView:
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

    def get_vote_counts(self, y: np.ndarray | pd.Series | SeriesLike) -> np.ndarray:
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


# Back-compat alias from the former Anomaly Detection Toolkit
VotingEnsemble = VotingEnsembleDetector


class ScoreCombiningEnsembleDetector(BaseDetector):
    """Combine scores from multiple detectors/scorers (mean, max, min, or median).

    Replaces the former toolkit ``EnsembleDetector`` score-combination path: labels
    are produced by thresholding the **combined** score at a fixed percentile.
    For hard voting over member *predictions*, use :class:`VotingEnsembleDetector`
    instead.
    """

    def __init__(
        self,
        detectors: list[BaseDetector | BaseScorer],
        combination_method: str = "mean",
        score_percentile: float = 95.0,
        random_state: int | None = None,
    ) -> None:
        if not detectors:
            raise ValueError("At least one detector or scorer is required")
        if combination_method not in ("mean", "max", "min", "median"):
            raise ValueError(
                f"combination_method must be mean|max|min|median, got {combination_method!r}"
            )
        self.detectors = detectors
        self.combination_method = combination_method
        self.score_percentile = score_percentile
        self.random_state = random_state
        self._combiner = cast(
            Callable[..., Any],
            {
                "mean": np.mean,
                "max": np.max,
                "min": np.min,
                "median": np.median,
            }[combination_method],
        )
        super().__init__(
            detectors=detectors,
            combination_method=combination_method,
            score_percentile=score_percentile,
            random_state=random_state,
        )

    def fit(
        self,
        y: np.ndarray | pd.Series | SeriesLike,
        X: np.ndarray | pd.DataFrame | PanelLike | None = None,
    ) -> ScoreCombiningEnsembleDetector:
        for detector in self.detectors:
            detector.fit(y, X)
        self._fitted = True
        return self

    def score(self, y: np.ndarray | pd.Series | SeriesLike) -> ScoreView:
        self._check_fitted()
        if isinstance(y, pd.Series):
            index = y.index
        else:
            index = pd.RangeIndex(start=0, stop=len(y))
        rows = []
        for detector in self.detectors:
            sv = detector.score(y)
            rows.append(np.asarray(sv.scores, dtype=float))
        stacked = np.stack(rows, axis=0)
        combined = self._combiner(stacked, axis=0)
        return ScoreView(index=index, scores=combined)

    def predict(self, y: np.ndarray | pd.Series | SeriesLike) -> LabelView:
        sv = self.score(y)
        s = np.asarray(sv.scores, dtype=float)
        thr = float(np.percentile(s, self.score_percentile)) if len(s) else 0.0
        labels = (s > thr).astype(int)
        return LabelView(index=sv.index, labels=labels)


EnsembleDetector = ScoreCombiningEnsembleDetector
