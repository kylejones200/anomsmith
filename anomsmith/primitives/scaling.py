"""Robust score scaling primitives."""

import numpy as np


def robust_zscore(
    values: np.ndarray,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Compute robust z-scores using median and MAD.

    Uses median as center and Median Absolute Deviation (MAD) as scale.
    Includes epsilon guard to prevent division by zero.

    Args:
        values: Input values to scale
        epsilon: Small value to prevent division by zero

    Returns:
        Robust z-scores (same shape as input)
    """
    if values.size == 0:
        return np.array([])

    median = np.median(values)
    mad = np.median(np.abs(values - median))
    scale = mad + epsilon

    z_scores = (values - median) / scale
    return z_scores

