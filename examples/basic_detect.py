"""Basic anomaly detection example.

Creates a synthetic time series with one injected spike and one level shift,
then runs anomaly detection using RobustZScoreScorer with a quantile threshold.
"""

import numpy as np
import pandas as pd

from anomsmith import detect_anomalies, ThresholdRule
from anomsmith.primitives.scorers.robust_zscore import RobustZScoreScorer


def create_synthetic_series(n: int = 200) -> pd.Series:
    """Create synthetic time series with anomalies.

    Args:
        n: Length of series

    Returns:
        pandas Series with injected anomalies
    """
    # Base series: random walk with trend
    np.random.seed(42)
    trend = np.linspace(0, 2, n)
    noise = np.random.randn(n) * 0.5
    y = trend + noise

    # Inject spike anomaly at index 50
    y[50] += 5.0

    # Inject level shift starting at index 100
    y[100:] += 3.0

    index = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.Series(y, index=index)


def main() -> None:
    """Run basic detection example."""
    # Create synthetic data
    y = create_synthetic_series(n=200)

    # Initialize scorer
    scorer = RobustZScoreScorer(epsilon=1e-8)
    scorer.fit(y.values)

    # Define threshold rule
    threshold_rule = ThresholdRule(method="quantile", value=0.95, quantile=0.95)

    # Detect anomalies
    result = detect_anomalies(y, scorer, threshold_rule)

    # Print output head
    print("Detection Results (first 10 rows):")
    print(result.head(10))
    print("\nSummary:")
    print(f"Total points: {len(result)}")
    print(f"Anomalies detected: {result['flag'].sum()}")
    print(f"Anomaly rate: {result['flag'].mean():.2%}")

    # Save to CSV
    output_dir = "examples/out"
    import os

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "basic_detect.csv")
    result.to_csv(output_path)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

