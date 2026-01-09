"""Validators for Layer 1 objects.

All validators raise ValueError with clear messages.
"""

import numpy as np
import pandas as pd

try:
    from timesmith.typing import (
        PanelLike,
        SeriesLike,
        assert_panel as timesmith_assert_panel,
        assert_series as timesmith_assert_series,
    )
except ImportError:
    raise ImportError(
        "timesmith is required. Install with: pip install timesmith or pip install anomsmith[timesmith]"
    )

from anomsmith.objects.views import LabelView, ScoreView

# Type aliases
SeriesView = SeriesLike
PanelView = PanelLike


def assert_series(series: SeriesLike) -> None:
    """Validate a SeriesLike (using timesmith validation).

    Args:
        series: SeriesLike to validate

    Raises:
        ValueError: If series is invalid
    """
    timesmith_assert_series(series)


def assert_panel(panel: PanelLike) -> None:
    """Validate a PanelLike (using timesmith validation).

    Args:
        panel: PanelLike to validate

    Raises:
        ValueError: If panel is invalid
    """
    timesmith_assert_panel(panel)


def assert_aligned(
    view1: SeriesLike | ScoreView | LabelView,
    view2: SeriesLike | ScoreView | LabelView,
) -> None:
    """Assert two views have aligned indices.

    Args:
        view1: First view
        view2: Second view

    Raises:
        ValueError: If indices are not aligned
    """
    # Check types - SeriesLike can be pd.Series or pd.DataFrame
    if not isinstance(view1, (pd.Series, pd.DataFrame, ScoreView, LabelView)):
        raise ValueError(f"Expected SeriesLike, ScoreView, or LabelView, got {type(view1)}")
    if not isinstance(view2, (pd.Series, pd.DataFrame, ScoreView, LabelView)):
        raise ValueError(f"Expected SeriesLike, ScoreView, or LabelView, got {type(view2)}")

    index1 = view1.index
    index2 = view2.index

    if len(index1) != len(index2):
        raise ValueError(
            f"Indices must have same length: {len(index1)} vs {len(index2)}"
        )

    if not index1.equals(index2):
        raise ValueError(
            f"Indices must be equal. First index: {index1}, Second index: {index2}"
        )


def assert_monotonic_index(index: pd.Index) -> None:
    """Assert index is monotonic (non-decreasing).

    Args:
        index: Index to validate

    Raises:
        ValueError: If index is not monotonic
    """
    if not isinstance(index, pd.Index):
        raise ValueError(f"Expected pd.Index, got {type(index)}")

    if not index.is_monotonic_increasing:
        raise ValueError(f"Index must be monotonic (non-decreasing), got {index}")

