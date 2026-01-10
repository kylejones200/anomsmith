"""PCA-based predictive maintenance using Mahalanobis distance.

Demonstrates Principal Component Analysis for condition-based maintenance
using Mahalanobis distance to track equipment health drift.

Based on "Predictive Maintenance using Principal Component Analysis in Python".

This example shows:
1. Loading and preprocessing turbofan sensor data
2. Fitting PCA on healthy operation data
3. Computing Mahalanobis distance from healthy center
4. Classifying health states (healthy, warning, critical) based on distance thresholds
5. Tracking Mahalanobis distance over time as a single metric
6. Providing early warning (10-14 days) before equipment failure
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import anomsmith components
from anomsmith import (
    assess_health_with_pca,
    classify_health_from_distance,
    compute_pca_health_thresholds,
    track_mahalanobis_distance,
)
from anomsmith.objects.health_state import HealthState
from anomsmith.primitives.detectors.pca import PCADetector


def load_turbofan_data_with_category(
    file_path: str, n_samples: Optional[int] = None
) -> tuple[pd.DataFrame, pd.Series]:
    """Load turbofan data with health state categorization.

    Args:
        file_path: Path to data file (or None to generate synthetic)
        n_samples: Optional limit on number of samples

    Returns:
        Tuple of (X, y) where:
        - X: Feature matrix (sensor readings and settings)
        - y: Health states (0=Healthy, 1=Warning, 2=Critical/Distress)
    """
    if file_path is None or not Path(file_path).exists():
        logger.warning(
            f"File {file_path} not found. Generating synthetic turbofan data for demonstration."
        )
        return generate_synthetic_turbofan_with_category(n_samples=n_samples)

    logger.info(f"Loading turbofan data from {file_path}")

    # Load data
    df = pd.read_csv(file_path)

    # Check if Category column exists (from previous processing)
    if "Category" not in df.columns:
        # Compute RUL and categorize
        if "max" in df.columns or "RUL" in df.columns:
            # RUL already computed
            rul_col = "RUL" if "RUL" in df.columns else "max"
            max_rul = df.groupby(["source", "id"])[rul_col].max().reset_index()
            max_rul.columns = ["source", "id", "max_rul"]
        else:
            # Compute RUL
            rul = df.groupby(["source", "id"])["cycle"].max().reset_index()
            rul.columns = ["source", "id", "max_cycle"]
            df = df.merge(rul, on=["source", "id"])
            df["RUL"] = df["max_cycle"] - df["cycle"]
            max_rul = df.groupby(["source", "id"])["RUL"].max().reset_index()
            max_rul.columns = ["source", "id", "max_rul"]

        # Categorize: Healthy (>30), Warning (10-30), Critical (<=10)
        def categorize_rul(rul_val: float) -> str:
            if rul_val > 30:
                return "Healthy"
            elif rul_val > 10:
                return "Warning"
            else:
                return "Critical"

        max_rul["Category"] = max_rul["max_rul"].apply(categorize_rul)
        df = df.merge(max_rul[["source", "id", "Category"]], on=["source", "id"])

    # Map Category to numeric health states
    category_map = {"Healthy": 0, "Warning": 1, "Critical": 2}
    df["HealthState"] = df["Category"].map(category_map)

    # Select sensor features (assuming standard turbofan column names)
    # If columns don't exist, try to infer from available columns
    sensor_cols = [col for col in df.columns if col.startswith("sensor_") or col.startswith("s")]
    setting_cols = [col for col in df.columns if col.startswith("setting") or col.startswith("op_setting")]

    # Combine settings and sensors
    feature_cols = setting_cols + sensor_cols
    if not feature_cols:
        # Fallback: use all numeric columns except ID columns
        feature_cols = [
            col
            for col in df.columns
            if col not in ["unit", "id", "source", "cycle", "Category", "HealthState", "RUL", "max", "max_cycle"]
            and df[col].dtype in [np.float64, np.int64]
        ]

    X = df[feature_cols].copy()
    y = df["HealthState"].copy()

    if n_samples is not None:
        X = X.head(n_samples)
        y = y.head(n_samples)

    logger.info(f"Loaded {len(X)} samples with {len(feature_cols)} features")
    logger.info(f"Health state distribution:")
    logger.info(y.value_counts().sort_index().to_string())

    return X, y


def generate_synthetic_turbofan_with_category(
    n_engines: int = 100, n_samples: int = 5000
) -> tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic turbofan data with health state categorization.

    Simulates sensor readings and failure patterns with health state categorization.
    """
    np.random.seed(42)
    logger.info("Generating synthetic turbofan data with health states...")

    # Generate engine data
    engine_ids = np.arange(1, n_engines + 1)
    max_cycles = np.random.randint(150, 300, size=n_engines)

    data = []
    for engine_id in engine_ids:
        max_cycle = max_cycles[engine_id - 1]
        # Generate cycles for this engine
        cycles = np.arange(1, max_cycle + 1)
        for cycle in cycles:
            rul = max_cycle - cycle
            degradation = 1.0 - (rul / max_cycle)

            # Generate settings (3) and sensor readings (21)
            settings = np.random.rand(3) * 10 + 50
            sensors = np.random.randn(21) * 5 + 50
            sensors[:5] += degradation * 10  # Temperature sensors increase
            sensors[5:10] += degradation * -5  # Pressure sensors decrease

            # Categorize health state
            if rul > 30:
                health_state = 0  # Healthy
            elif rul > 10:
                health_state = 1  # Warning
            else:
                health_state = 2  # Critical

            row = {
                "id": engine_id,
                "cycle": cycle,
                "RUL": rul,
                "HealthState": health_state,
                **{f"setting_{i+1}": settings[i] for i in range(3)},
                **{f"sensor_{i+1}": sensors[i] for i in range(21)},
            }
            data.append(row)

    df = pd.DataFrame(data)

    # Select feature columns
    feature_cols = [f"setting_{i+1}" for i in range(3)] + [f"sensor_{i+1}" for i in range(21)]
    X = df[feature_cols].copy()
    y = df["HealthState"].copy()

    if n_samples is not None:
        X = X.head(n_samples)
        y = y.head(n_samples)

    return X, y


def main() -> None:
    """Run PCA-based predictive maintenance example."""
    logger.info("=" * 70)
    logger.info("PCA-Based Predictive Maintenance using Mahalanobis Distance")
    logger.info("=" * 70)
    logger.info(
        "This example demonstrates:\n"
        "- Dimensionality reduction using PCA\n"
        "- Mahalanobis distance as a single health metric\n"
        "- Health state classification (healthy, warning, critical)\n"
        "- Time series tracking for early warning (10-14 days before failure)"
    )

    # Step 1: Load and preprocess data
    logger.info("\n" + "=" * 70)
    logger.info("Step 1: Loading and Preprocessing Data")
    logger.info("=" * 70)

    data_path = "data/predictive maintenance trainALL.csv"  # Update this path as needed
    try:
        X, y = load_turbofan_data_with_category(data_path, n_samples=10000)
    except FileNotFoundError:
        logger.warning("Real dataset not found, using synthetic data")
        X, y = generate_synthetic_turbofan_with_category(n_engines=100, n_samples=5000)

    # Split data (use healthy data for training PCA model)
    # For proper evaluation, split by health state: train on healthy, test on all
    healthy_mask = y == HealthState.HEALTHY.value
    X_train_healthy = X[healthy_mask]
    X_test = X[~healthy_mask]
    y_test = y[~healthy_mask]

    logger.info(f"Training set (healthy only): {len(X_train_healthy)} samples")
    logger.info(f"Test set (all states): {len(X_test)} samples")

    # Step 2: Fit PCA on healthy operation data
    logger.info("\n" + "=" * 70)
    logger.info("Step 2: Fitting PCA on Healthy Operation Data")
    logger.info("=" * 70)

    # Fit PCA detector on healthy data (models "normal" operation)
    detector = PCADetector(
        n_components=3,  # Use 3 principal components as in article
        score_method="mahalanobis",  # Use Mahalanobis distance for health tracking
        contamination=0.05,  # 5% contamination for threshold (not used for health states)
        random_state=42,
    )

    # Fit on healthy operation data (this establishes the "normal" center)
    # PCADetector.fit accepts X as y (features as target) or X separately
    # For PCA, we pass features as the target (y)
    detector.fit(X_train_healthy)

    logger.info(f"✓ PCA fitted on {len(X_train_healthy)} healthy samples")
    logger.info(f"  Number of components: {detector.pca_.n_components_}")  # type: ignore
    logger.info(
        f"  Explained variance: {detector.pca_.explained_variance_ratio_.sum():.3f}"  # type: ignore
    )

    # Step 3: Compute health state thresholds
    logger.info("\n" + "=" * 70)
    logger.info("Step 3: Computing Health State Thresholds")
    logger.info("=" * 70)

    # Compute thresholds based on training data distances
    healthy_threshold, warning_threshold = compute_pca_health_thresholds(
        X_train_healthy, detector, healthy_percentile=75.0, warning_percentile=95.0
    )

    logger.info(f"Health state thresholds:")
    logger.info(f"  Healthy: distance <= {healthy_threshold:.3f}")
    logger.info(f"  Warning: {healthy_threshold:.3f} < distance <= {warning_threshold:.3f}")
    logger.info(f"  Critical: distance > {warning_threshold:.3f}")

    # Step 4: Track Mahalanobis distance on test data
    logger.info("\n" + "=" * 70)
    logger.info("Step 4: Tracking Mahalanobis Distance Over Time")
    logger.info("=" * 70)

    # Track Mahalanobis distance as a time series metric
    distances = track_mahalanobis_distance(X_test, detector, index=X_test.index)

    logger.info(f"Mahalanobis distance statistics:")
    logger.info(f"  Mean: {distances.mean():.3f}")
    logger.info(f"  Median: {distances.median():.3f}")
    logger.info(f"  Min: {distances.min():.3f}")
    logger.info(f"  Max: {distances.max():.3f}")
    logger.info(f"  Std: {distances.std():.3f}")

    # Step 5: Classify health states from distances
    logger.info("\n" + "=" * 70)
    logger.info("Step 5: Classifying Health States from Distances")
    logger.info("=" * 70)

    health_states = classify_health_from_distance(
        distances, healthy_threshold, warning_threshold, index=distances.index
    )

    # Compare with true health states (if available)
    state_counts = pd.Series(health_states.states).value_counts().sort_index()
    state_names = {0: "Healthy", 1: "Warning", 2: "Critical"}
    logger.info(f"Predicted health state distribution:")
    for state, count in state_counts.items():
        logger.info(f"  {state_names[state]}: {count} ({count/len(health_states.states)*100:.1f}%)")

    if y_test is not None:
        true_state_counts = pd.Series(y_test.values).value_counts().sort_index()
        logger.info(f"\nTrue health state distribution:")
        for state, count in true_state_counts.items():
            logger.info(f"  {state_names[state]}: {count} ({count/len(y_test)*100:.1f}%)")

        # Compute accuracy if true states available
        from sklearn.metrics import accuracy_score

        accuracy = accuracy_score(y_test.values, health_states.states)
        logger.info(f"\nClassification Accuracy: {accuracy:.3f}")

    # Step 6: Complete health assessment
    logger.info("\n" + "=" * 70)
    logger.info("Step 6: Complete Health Assessment")
    logger.info("=" * 70)

    health_df = assess_health_with_pca(
        X_test, detector, healthy_threshold, warning_threshold, index=X_test.index
    )

    logger.info(f"\nHealth assessment results (first 10 samples):")
    logger.info(health_df.head(10).to_string())

    # Step 7: Time series tracking example
    logger.info("\n" + "=" * 70)
    logger.info("Step 7: Time Series Tracking for Early Warning")
    logger.info("=" * 70)

    # Track distance over time and identify threshold crossings
    logger.info(f"\nTracking Mahalanobis distance over time:")
    logger.info(f"  Total samples: {len(distances)}")
    logger.info(f"  Samples above healthy threshold: {(distances > healthy_threshold).sum()}")
    logger.info(f"  Samples above warning threshold: {(distances > warning_threshold).sum()}")

    # Find first crossing of warning threshold (early warning signal)
    warning_crossings = distances[distances > warning_threshold]
    if len(warning_crossings) > 0:
        first_warning_idx = warning_crossings.index[0]
        first_warning_pos = distances.index.get_loc(first_warning_idx) if hasattr(distances.index, 'get_loc') else list(distances.index).index(first_warning_idx)
        logger.info(
            f"  First warning signal at index {first_warning_idx} "
            f"(distance={distances.loc[first_warning_idx]:.3f})"
        )
        
        # Calculate samples before first warning (early warning lead time)
        samples_before_warning = first_warning_pos
        logger.info(
            f"  Early warning lead time: {samples_before_warning} samples "
            f"(~{samples_before_warning/24:.1f} days if hourly data, ~{samples_before_warning:.0f} cycles for turbofan)"
        )
    else:
        logger.info(f"  No warnings detected (max distance={distances.max():.3f})")

    # Show distance progression over time (first 20 samples)
    logger.info(f"\nMahalanobis distance progression (first 20 samples):")
    for i in range(min(20, len(distances))):
        idx = distances.index[i]
        dist_val = distances.iloc[i]
        if i < len(health_states.states):
            state_val = health_states.states[i]
            state_name = state_names[state_val]
            logger.info(f"  {idx}: distance={dist_val:.3f}, state={state_name}")
        else:
            logger.info(f"  {idx}: distance={dist_val:.3f}")

    # Step 8: Summary and insights
    logger.info("\n" + "=" * 70)
    logger.info("Step 8: Summary and Insights")
    logger.info("=" * 70)

    logger.info("\nKey Insights:")
    logger.info("  ✓ PCA reduces dimensionality from high-frequency IoT data")
    logger.info("  ✓ Mahalanobis distance provides a single metric to track")
    logger.info("  ✓ Probabilistic zones minimize false positives (wide 'normal' space)")
    logger.info("  ✓ Distance thresholds create health state boundaries")
    logger.info("  ✓ Time series tracking enables early warning (10-14 days before failure)")

    logger.info("\nAdvantages of this approach:")
    logger.info("  - Easily interpreted by control room operators")
    logger.info("  - Single metric (distance) instead of multiple sensors")
    logger.info("  - Reduces false positives with wide decision space for normal")
    logger.info("  - Provides early warning for maintenance planning")
    logger.info("  - Can be extended with forecasting (e.g., AWS Forecast)")

    logger.info("\nNext Steps:")
    logger.info("  - Set up real-time monitoring of Mahalanobis distance")
    logger.info("  - Configure alerting when distance exceeds thresholds")
    logger.info("  - Integrate with forecasting to predict threshold crossings")
    logger.info("  - Connect to decision policies for maintenance scheduling")
    logger.info("  - Validate with historical failure data")

    logger.info("\n" + "=" * 70)
    logger.info("Example Complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

