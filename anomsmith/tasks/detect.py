"""Detection task definitions and runners."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from anomsmith.objects.validate import assert_series, assert_aligned
from anomsmith.objects.views import LabelView, ScoreView
from anomsmith.objects.window import WindowSpec
from anomsmith.primitives.base import BaseDetector, BaseScorer
from anomsmith.tasks.helpers import make_series_view

try:
    from timesmith.typing import SeriesLike
except ImportError:
    SeriesLike = None  # type: ignore

# Try to use timesmith's DetectTask if available
try:
    from timesmith.tasks import DetectTask as TimesmithDetectTask
    # Use timesmith's DetectTask if it exists and is compatible
    DetectTask = TimesmithDetectTask  # type: ignore
except (ImportError, AttributeError):
    # Fallback to our own DetectTask
    from typing import Union
    _SeriesLikeType = Union[pd.Series, np.ndarray] if SeriesLike is None else Union[pd.Series, np.ndarray, SeriesLike]
    
    @dataclass
    class DetectTask:
        """Task specification for anomaly detection.

        Attributes:
            y: Target time series
            X: Optional features (not used in first pass)
            labels: Optional ground truth labels
            window_spec: Optional window specification
            cutoff: Optional cutoff point for train/test split
        """

        y: _SeriesLikeType  # type: ignore
        X: pd.DataFrame | np.ndarray | None = None
        labels: _SeriesLikeType | None = None  # type: ignore
        window_spec: WindowSpec | None = None
        cutoff: int | None = None


def run_scoring(
    y: pd.Series | np.ndarray | "SeriesLike",
    scorer: BaseScorer,
) -> ScoreView:
    """Run scoring task with a scorer.

    Args:
        y: Time series to score
        scorer: BaseScorer instance

    Returns:
        ScoreView with anomaly scores

    Raises:
        ValueError: If inputs are invalid
    """
    series_view = make_series_view(y)
    assert_series(series_view)

    # Pass original y to preserve index if it's a Series
    score_view = scorer.score(y)

    assert_aligned(series_view, score_view)

    return score_view


def run_detection(
    y: pd.Series | np.ndarray | "SeriesLike",
    detector: BaseDetector | BaseScorer,
) -> tuple[LabelView, ScoreView]:
    """Run detection task with a detector or scorer.

    If a BaseScorer is provided, only scoring is performed (no labels).

    Args:
        y: Time series to detect anomalies in
        detector: BaseDetector or BaseScorer instance

    Returns:
        Tuple of (LabelView, ScoreView). LabelView may be empty if scorer provided.

    Raises:
        ValueError: If inputs are invalid
    """
    series_view = make_series_view(y)
    assert_series(series_view)

    if isinstance(detector, BaseDetector):
        # Pass original y to preserve index if it's a Series
        label_view = detector.predict(y)
        score_view = detector.score(y)
        assert_aligned(series_view, label_view)
        assert_aligned(series_view, score_view)
        return label_view, score_view
    elif isinstance(detector, BaseScorer):
        # Pass original y to preserve index if it's a Series
        score_view = detector.score(y)
        assert_aligned(series_view, score_view)
        # Return empty labels if only scorer provided
        # series_view is now a pd.Series, so use .index directly
        empty_labels = np.zeros(len(series_view.index), dtype=int)
        label_view = LabelView(index=series_view.index, labels=empty_labels)
        return label_view, score_view
    else:
        raise ValueError(
            f"Expected BaseDetector or BaseScorer, got {type(detector)}"
        )

