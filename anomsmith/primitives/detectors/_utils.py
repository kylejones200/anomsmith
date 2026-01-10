"""Utility functions for detectors to reduce code duplication."""

from typing import Optional, Union

import numpy as np
import pandas as pd


def prepare_input_data(
    y: Union[np.ndarray, pd.Series],
    X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
) -> np.ndarray:
    """Prepare input data for detectors/scorers.

    Handles conversion of Series/DataFrame to numpy arrays and ensures 2D shape.

    Args:
        y: Target data (Series or array)
        X: Optional feature data (DataFrame or array)

    Returns:
        2D numpy array ready for sklearn/scaling operations
    """
    # Use X if provided, otherwise use y
    if X is not None:
        if isinstance(X, pd.DataFrame):
            X_data = X.values
        else:
            X_data = X
        if X_data.ndim == 1:
            X_data = X_data.reshape(-1, 1)
        return X_data
    else:
        if isinstance(y, pd.Series):
            X_data = y.values.reshape(-1, 1)
        else:
            X_data = y.reshape(-1, 1) if y.ndim == 1 else y
        return X_data


def extract_index_and_values(y: Union[np.ndarray, pd.Series]) -> tuple[pd.Index, np.ndarray]:
    """Extract index and values from input.

    Args:
        y: Input data (Series or array)

    Returns:
        Tuple of (index, values)
    """
    if isinstance(y, pd.Series):
        return y.index, y.values
    else:
        return pd.RangeIndex(start=0, stop=len(y)), y


