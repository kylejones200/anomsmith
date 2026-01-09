"""Helpers for converting inputs into timesmith-compatible types."""

from typing import Union

import numpy as np
import pandas as pd

try:
    from timesmith.typing import PanelLike, SeriesLike, assert_series, assert_panel
except ImportError:
    raise ImportError(
        "timesmith is required. Install with: pip install timesmith or pip install anomsmith[timesmith]"
    )

from anomsmith.objects.validate import assert_monotonic_index

# Type aliases
SeriesView = SeriesLike
PanelView = PanelLike


def make_series_view(y: Union[pd.Series, np.ndarray, SeriesLike]) -> SeriesLike:
    """Convert pandas Series or numpy array into timesmith SeriesLike.

    Args:
        y: Time series data (pd.Series, np.ndarray, or SeriesLike)

    Returns:
        SeriesLike (pd.Series) compatible with timesmith

    Raises:
        ValueError: If input is invalid
    """
    # If already a SeriesLike, validate and return
    if isinstance(y, (pd.Series, pd.DataFrame)):
        assert_series(y)
        # Convert DataFrame to Series if needed
        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                raise ValueError(f"DataFrame must have exactly one column, got {y.shape[1]}")
            return y.iloc[:, 0]
        return y
    
    # Convert numpy array to Series
    if isinstance(y, np.ndarray):
        if y.ndim != 1:
            raise ValueError(f"Input must be 1D, got shape {y.shape}")
        index = pd.RangeIndex(start=0, stop=len(y))
        series = pd.Series(y, index=index)
        assert_series(series)
        return series
    
    raise ValueError(f"Expected pd.Series, np.ndarray, or SeriesLike, got {type(y)}")


def make_panel_view(
    y: Union[pd.DataFrame, np.ndarray, PanelLike],
    entity_key: pd.Index | None = None,
    time_index: pd.Index | None = None,
) -> PanelLike:
    """Convert pandas DataFrame or numpy array into timesmith PanelLike.

    Args:
        y: Panel data (entities x time)
        entity_key: Optional entity identifiers
        time_index: Optional time index

    Returns:
        PanelLike (pd.DataFrame) compatible with timesmith

    Raises:
        ValueError: If input is invalid
    """
    # If already a PanelLike, validate and return
    if isinstance(y, pd.DataFrame):
        # Set index/columns if provided
        if entity_key is not None:
            y = y.set_index(entity_key) if not y.index.equals(entity_key) else y
        if time_index is not None:
            y = y.reindex(columns=time_index) if not y.columns.equals(time_index) else y
        assert_panel(y)
        return y
    
    # Convert numpy array to DataFrame
    if isinstance(y, np.ndarray):
        if y.ndim != 2:
            raise ValueError(f"Input must be 2D, got shape {y.shape}")
        if entity_key is None:
            entity_key = pd.RangeIndex(start=0, stop=y.shape[0])
        if time_index is None:
            time_index = pd.RangeIndex(start=0, stop=y.shape[1])
        df = pd.DataFrame(y, index=entity_key, columns=time_index)
        assert_panel(df)
        return df
    
    raise ValueError(f"Expected pd.DataFrame, np.ndarray, or PanelLike, got {type(y)}")

