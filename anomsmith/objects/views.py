"""Immutable view dataclasses for time series data.

This module uses timesmith's SeriesLike and PanelLike types for time series data.
ScoreView and LabelView are kept for anomaly-specific outputs.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

# Use timesmith types directly (now a required dependency)
from timesmith.typing import PanelLike, SeriesLike

# Type aliases for backward compatibility and clarity
SeriesView = SeriesLike  # Alias for timesmith SeriesLike
PanelView = PanelLike  # Alias for timesmith PanelLike


@dataclass(frozen=True)
class ScoreView:
    """Immutable view of anomaly scores aligned to an index.

    Attributes:
        index: Time index (must match input series index)
        scores: Anomaly scores as 1D array (higher = more anomalous)
    """

    index: pd.Index
    scores: np.ndarray

    def __post_init__(self) -> None:
        """Validate inputs after initialization."""
        if len(self.index) != len(self.scores):
            raise ValueError(
                f"Index length ({len(self.index)}) must match scores length "
                f"({len(self.scores)})"
            )
        if self.scores.ndim != 1:
            raise ValueError(f"Scores must be 1D, got shape {self.scores.shape}")


@dataclass(frozen=True)
class LabelView:
    """Immutable view of binary anomaly labels aligned to an index.

    Attributes:
        index: Time index (must match input series index)
        labels: Binary flags as 1D array (1 = anomaly, 0 = normal)
    """

    index: pd.Index
    labels: np.ndarray

    def __post_init__(self) -> None:
        """Validate inputs after initialization."""
        if len(self.index) != len(self.labels):
            raise ValueError(
                f"Index length ({len(self.index)}) must match labels length "
                f"({len(self.labels)})"
            )
        if self.labels.ndim != 1:
            raise ValueError(f"Labels must be 1D, got shape {self.labels.shape}")
        unique_vals = np.unique(self.labels)
        if not np.all(np.isin(unique_vals, [0, 1])):
            raise ValueError(
                f"Labels must be binary (0 or 1), got unique values {unique_vals}"
            )

