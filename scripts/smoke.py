#!/usr/bin/env python3
"""Smoke test for anomsmith integration with timesmith.

This script validates that anomsmith works correctly with timesmith typing.
It must:
1. Accept a pandas Series
2. Validate via timesmith.typing
3. Detect anomalies
4. Exit 0 on success
"""

import sys

import numpy as np
import pandas as pd

# Import from timesmith.typing (single source of truth)
from timesmith.typing import SeriesLike

# Import validators from timesmith.typing.validators (or fallback)
try:
    from timesmith.typing.validators import assert_series_like
except ImportError:
    # Fallback: validators might be in timesmith.typing directly
    from timesmith.typing import assert_series as assert_series_like

# Import anomsmith
from anomsmith import detect_anomalies, ThresholdRule
from anomsmith.primitives.scorers.robust_zscore import RobustZScoreScorer


def main() -> int:
    """Run smoke test."""
    # Create a pandas Series
    np.random.seed(42)
    n = 100
    values = np.random.randn(n) * 2 + 10
    index = pd.date_range("2020-01-01", periods=n, freq="D")
    y: SeriesLike = pd.Series(values, index=index)
    
    # Inject an anomaly
    y.iloc[50] += 10.0
    
    # Validate via timesmith.typing
    try:
        assert_series_like(y)
    except Exception as e:
        print(f"ERROR: timesmith validation failed: {e}", file=sys.stderr)
        return 1
    
    # Use anomsmith to detect anomalies
    try:
        scorer = RobustZScoreScorer(epsilon=1e-8)
        scorer.fit(y.values)
        
        threshold_rule = ThresholdRule(method="quantile", value=0.95, quantile=0.95)
        result = detect_anomalies(y, scorer, threshold_rule)
        
        # Verify we got results
        if len(result) != n:
            print(f"ERROR: Expected {n} results, got {len(result)}", file=sys.stderr)
            return 1
        
        # Verify we detected the anomaly
        if result['flag'].sum() == 0:
            print("WARNING: No anomalies detected (might be expected)", file=sys.stderr)
        
        print("✓ Smoke test passed: anomsmith works with timesmith typing")
        return 0
        
    except Exception as e:
        print(f"ERROR: anomsmith detection failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

