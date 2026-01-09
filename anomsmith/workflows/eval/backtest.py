"""Backtesting utilities for anomaly detection."""

import logging
from typing import TYPE_CHECKING, Any, Union

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    try:
        from timesmith.typing import SeriesLike
    except ImportError:
        SeriesLike = None

from anomsmith.primitives.base import BaseDetector, BaseScorer
from anomsmith.primitives.thresholding import ThresholdRule
from anomsmith.workflows.detect import detect_anomalies
from anomsmith.workflows.eval.metrics import (
    average_run_length,
    compute_f1,
    compute_precision,
    compute_recall,
)

logger = logging.getLogger(__name__)

# Use timesmith's ExpandingWindowSplit and SlidingWindowSplit if available
try:
    from timesmith.eval import (
        ExpandingWindowSplit as TimesmithExpandingWindowSplit,
        SlidingWindowSplit as TimesmithSlidingWindowSplit,
    )
    ExpandingWindowSplit = TimesmithExpandingWindowSplit  # type: ignore
    SlidingWindowSplit = TimesmithSlidingWindowSplit  # type: ignore
except (ImportError, AttributeError):
    # Fallback to our own implementation
    class ExpandingWindowSplit:
        """Expanding window splitter for time series backtesting.

        Attributes:
            n_splits: Number of splits to generate
            min_train_size: Minimum training set size
        """

        def __init__(self, n_splits: int = 5, min_train_size: int = 10) -> None:
        """Initialize splitter.

        Args:
            n_splits: Number of splits to generate
            min_train_size: Minimum training set size
        """
            self.n_splits = n_splits
            self.min_train_size = min_train_size

        def split(self, y: Union[pd.Series, np.ndarray, "SeriesLike"]) -> list[tuple[int, int]]:
            """Generate train/test cutoff points.

            Args:
                y: Time series to split

            Returns:
                List of (train_end, test_start) tuples
            """
            n = len(y)
            if n < self.min_train_size + self.n_splits:
                raise ValueError(
                    f"Series length ({n}) must be at least "
                    f"{self.min_train_size + self.n_splits}"
                )

            # Vectorized cutoff generation
            step = (n - self.min_train_size) // self.n_splits
            indices = np.arange(self.n_splits)
            train_ends = self.min_train_size + indices * step
            # Convert to list of tuples as expected by consumers
            cutoffs = [(int(te), int(te)) for te in train_ends]

            return cutoffs

    # SlidingWindowSplit not available from timesmith, define our own if needed
    class SlidingWindowSplit:
        """Sliding window splitter for time series backtesting.
        
        Similar to ExpandingWindowSplit but uses fixed-size windows.
        """
        def __init__(self, n_splits: int = 5, train_size: int = 20, test_size: int = 10) -> None:
            """Initialize splitter.
            
            Args:
                n_splits: Number of splits to generate
                train_size: Size of training window
                test_size: Size of test window
            """
            self.n_splits = n_splits
            self.train_size = train_size
            self.test_size = test_size
        
        def split(self, y: Union[pd.Series, np.ndarray, "SeriesLike"]) -> list[tuple[int, int]]:
            """Generate train/test cutoff points.
            
            Args:
                y: Time series to split
                
            Returns:
                List of (train_end, test_start) tuples
            """
            n = len(y)
            if n < self.train_size + self.test_size:
                raise ValueError(
                    f"Series length ({n}) must be at least "
                    f"{self.train_size + self.test_size}"
                )
            
            # Generate sliding windows
            max_start = n - self.train_size - self.test_size
            step = max(1, max_start // self.n_splits) if max_start > 0 else 1
            
            cutoffs = []
            for i in range(self.n_splits):
                train_start = min(i * step, max_start)
                train_end = train_start + self.train_size
                test_start = train_end
                cutoffs.append((train_end, test_start))
            
            return cutoffs


def backtest_detector(
    y: Union[pd.Series, np.ndarray, "SeriesLike"],
    detector: BaseDetector | BaseScorer,
    threshold_rule: ThresholdRule,
    labels: Union[pd.Series, np.ndarray, "SeriesLike", None] = None,
    n_splits: int = 5,
    min_train_size: int = 10,
) -> pd.DataFrame:
    """Run backtest of detector across expanding windows.

    Args:
        y: Time series to backtest on
        detector: BaseDetector or BaseScorer instance
        threshold_rule: ThresholdRule to apply
        labels: Optional ground truth labels
        n_splits: Number of splits
        min_train_size: Minimum training set size

    Returns:
        pandas DataFrame with columns: fold, precision, recall, f1, avg_run_length
    """
    logger.info(f"Running backtest with {n_splits} splits")
    splitter = ExpandingWindowSplit(n_splits=n_splits, min_train_size=min_train_size)
    cutoffs = splitter.split(y)

    results = []
    for fold, (train_end, test_start) in enumerate(cutoffs):
        # Split data
        y_train = y.iloc[:train_end]
        y_test = y.iloc[test_start:]

        # Fit on training data
        detector.fit(y_train.values)

        # Detect on test data
        result_df = detect_anomalies(y_test, detector, threshold_rule)

        if labels is not None:
            # Align labels
            labels_test = labels.reindex(y_test.index, fill_value=0).values
            labels_test = (labels_test != 0).astype(int)

            precision = compute_precision(labels_test, result_df["flag"].values)
            recall = compute_recall(labels_test, result_df["flag"].values)
            f1 = compute_f1(labels_test, result_df["flag"].values)
            avg_run_length = average_run_length(result_df["flag"].values)
        else:
            precision = np.nan
            recall = np.nan
            f1 = np.nan
            avg_run_length = average_run_length(result_df["flag"].values)

        results.append(
            {
                "fold": fold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "avg_run_length": avg_run_length,
            }
        )

    return pd.DataFrame(results)

