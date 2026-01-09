"""Integration test: anomsmith with timesmith typing.

This test validates that anomsmith works correctly with timesmith typing
and that objects can move seamlessly between the two libraries.
"""

import numpy as np
import pandas as pd
import pytest

# Import from timesmith.typing (single source of truth)
from timesmith.typing import SeriesLike

# Import anomsmith
from anomsmith import detect_anomalies, ThresholdRule
from anomsmith.primitives.scorers.robust_zscore import RobustZScoreScorer


def test_timesmith_typing_integration() -> None:
    """Test that anomsmith accepts SeriesLike from timesmith."""
    # Create a pandas Series (SeriesLike)
    np.random.seed(42)
    n = 100
    values = np.random.randn(n) * 2 + 10
    index = pd.date_range("2020-01-01", periods=n, freq="D")
    y: SeriesLike = pd.Series(values, index=index)
    
    # Inject an anomaly
    y.iloc[50] += 10.0
    
    # Use anomsmith to detect anomalies
    scorer = RobustZScoreScorer(epsilon=1e-8)
    scorer.fit(y.values)
    
    threshold_rule = ThresholdRule(method="quantile", value=0.95, quantile=0.95)
    result = detect_anomalies(y, scorer, threshold_rule)
    
    # Verify we got results
    assert len(result) == n
    assert 'score' in result.columns
    assert 'flag' in result.columns
    assert result.index.equals(y.index)
    
    # Verify we detected at least one anomaly (the injected one)
    assert result['flag'].sum() > 0


def test_timesmith_validation_preserved() -> None:
    """Test that timesmith validation is preserved through anomsmith."""
    # Create a pandas Series
    np.random.seed(42)
    n = 50
    values = np.random.randn(n)
    index = pd.date_range("2020-01-01", periods=n, freq="D")
    y: SeriesLike = pd.Series(values, index=index)
    
    # This should work - anomsmith should accept SeriesLike
    scorer = RobustZScoreScorer(epsilon=1e-8)
    scorer.fit(y.values)
    
    threshold_rule = ThresholdRule(method="quantile", value=0.95, quantile=0.95)
    result = detect_anomalies(y, scorer, threshold_rule)
    
    # Verify the result maintains the index
    assert result.index.equals(y.index)

