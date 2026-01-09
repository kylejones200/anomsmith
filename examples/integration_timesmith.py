"""Integration example: anomsmith with timesmith typing.

This example demonstrates:
1. Creating a pandas Series (SeriesLike)
2. Validating via timesmith.typing
3. Using anomsmith to detect anomalies
4. Showing seamless integration between the two libraries
"""

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


def main() -> None:
    """Run integration example."""
    print("=" * 60)
    print("Anomsmith + Timesmith Integration Example")
    print("=" * 60)
    
    # Create a pandas Series (SeriesLike from timesmith)
    np.random.seed(42)
    n = 200
    trend = np.linspace(0, 2, n)
    noise = np.random.randn(n) * 0.5
    values = trend + noise
    
    # Inject anomalies
    values[50] += 5.0  # Spike
    values[100:] += 3.0  # Level shift
    
    index = pd.date_range("2020-01-01", periods=n, freq="D")
    y: SeriesLike = pd.Series(values, index=index)
    
    print(f"\n1. Created SeriesLike with {len(y)} points")
    print(f"   Date range: {y.index[0]} to {y.index[-1]}")
    
    # Validate via timesmith.typing
    print("\n2. Validating via timesmith.typing.validators.assert_series_like...")
    try:
        assert_series_like(y)
        print("   ✓ Validation passed")
    except Exception as e:
        print(f"   ✗ Validation failed: {e}")
        return
    
    # Use anomsmith to detect anomalies
    print("\n3. Detecting anomalies with anomsmith...")
    scorer = RobustZScoreScorer(epsilon=1e-8)
    scorer.fit(y.values)
    
    threshold_rule = ThresholdRule(method="quantile", value=0.95, quantile=0.95)
    result = detect_anomalies(y, scorer, threshold_rule)
    
    print(f"   ✓ Detection complete")
    print(f"   Anomalies detected: {result['flag'].sum()}")
    print(f"   Anomaly rate: {result['flag'].mean():.2%}")
    
    # Show top anomalies
    top_anomalies = result[result['flag'] == 1].nlargest(5, 'score')
    print("\n4. Top 5 anomalies by score:")
    for idx, row in top_anomalies.iterrows():
        print(f"   {idx}: score={row['score']:.4f}")
    
    print("\n" + "=" * 60)
    print("Integration successful! Objects move seamlessly between")
    print("timesmith (typing/validation) and anomsmith (detection).")
    print("=" * 60)


if __name__ == "__main__":
    main()

