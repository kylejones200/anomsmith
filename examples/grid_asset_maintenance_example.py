"""Grid asset predictive maintenance example.

Demonstrates classification and anomaly detection for transformer health assessment.
Based on "Predictive Maintenance for Grid Assets" article.
"""

import numpy as np
import pandas as pd

from anomsmith import assess_asset_health, rank_assets_by_risk
from anomsmith.primitives.classifiers.failure_risk import FailureRiskClassifier
from anomsmith.primitives.detectors.ml import IsolationForestDetector


def generate_synthetic_scada_data(n_samples: int = 2000, n_assets: int = 100) -> tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic SCADA data for transformer monitoring.

    Args:
        n_samples: Total number of samples
        n_assets: Number of assets

    Returns:
        Tuple of (sensor_data, failure_labels) where:
        - sensor_data: DataFrame with sensor readings
        - failure_labels: Series with binary failure labels
    """
    np.random.seed(42)

    # Generate asset IDs
    asset_ids = np.random.choice(range(1, n_assets + 1), size=n_samples)

    # Generate sensor readings
    temperature = np.random.normal(60, 5, n_samples)
    vibration = np.random.normal(0.2, 0.05, n_samples)
    oil_pressure = np.random.normal(25, 3, n_samples)
    load = np.random.normal(800, 100, n_samples)

    # Simulate failures: elevated temp/vibration correlated with failures
    failure_prob = 1 / (1 + np.exp(-(0.05 * (temperature - 65) + 8 * (vibration - 0.25))))
    failures = np.random.binomial(1, failure_prob)

    sensor_data = pd.DataFrame(
        {
            "asset_id": asset_ids,
            "Temperature_C": temperature,
            "Vibration_g": vibration,
            "OilPressure_psi": oil_pressure,
            "Load_kVA": load,
        }
    )

    failure_labels = pd.Series(failures, name="Failure")

    return sensor_data, failure_labels


def main() -> None:
    """Run grid asset maintenance example."""
    print("=" * 60)
    print("Grid Asset Predictive Maintenance Example")
    print("=" * 60)

    # Generate synthetic SCADA data
    print("\n1. Generating synthetic SCADA data...")
    sensor_data, failure_labels = generate_synthetic_scada_data(n_samples=2000, n_assets=100)
    print(f"   Created {len(sensor_data)} samples for {sensor_data['asset_id'].nunique()} assets")
    print(f"   Failure rate: {failure_labels.mean():.2%}")

    # Feature columns (excluding asset_id)
    feature_cols = ["Temperature_C", "Vibration_g", "OilPressure_psi", "Load_kVA"]
    X = sensor_data[feature_cols]

    # Classification-based failure risk prediction
    print("\n2. Training failure risk classifier...")
    classifier = FailureRiskClassifier(n_estimators=100, random_state=42)
    classifier.fit(X, failure_labels)
    probas = classifier.predict_proba(X)
    failure_risks = probas[:, 1]  # Probability of failure class

    print(f"   Mean failure risk: {failure_risks.mean():.4f}")
    print(f"   Max failure risk: {failure_risks.max():.4f}")
    print(f"   Min failure risk: {failure_risks.min():.4f}")

    # Anomaly detection
    print("\n3. Running anomaly detection...")
    detector = IsolationForestDetector(contamination=0.05, random_state=42)
    detector.fit(X.values)
    anomaly_scores = detector.score(X.values).scores
    anomaly_labels = detector.predict(X.values).labels

    print(f"   Anomalies detected: {anomaly_labels.sum()} ({anomaly_labels.mean():.2%})")
    print(f"   Mean anomaly score: {anomaly_scores.mean():.4f}")

    # Assess asset health (combining both approaches)
    print("\n4. Assessing asset health (combined classification + anomaly detection)...")
    asset_health = assess_asset_health(
        sensor_data,
        asset_ids=sensor_data["asset_id"],
        feature_cols=feature_cols,
        failure_labels=failure_labels,
        use_classification=True,
        use_anomaly_detection=True,
        contamination=0.05,
        n_estimators=100,
        random_state=42,
    )

    print(f"   Assets assessed: {len(asset_health)}")
    print(f"\n   Health state distribution:")
    state_counts = asset_health["health_state"].value_counts().sort_index()
    state_names = {0: "Healthy", 1: "Warning", 2: "Distress"}
    for state, count in state_counts.items():
        print(f"     {state_names[state]}: {count} ({count/len(asset_health)*100:.1f}%)")

    print(f"\n   Anomaly detection results:")
    print(f"     Anomalies flagged: {asset_health['is_anomaly'].sum()} ({asset_health['is_anomaly'].mean():.2%})")

    # Rank assets by risk
    print("\n5. Ranking assets by combined risk...")
    top_assets = rank_assets_by_risk(asset_health, top_n=10)
    print(f"\n   Top 10 assets requiring attention:")
    print(top_assets[["asset_id", "failure_risk", "health_state", "is_anomaly", "combined_risk"]].to_string(index=False))

    # Show correlation between classification and anomaly detection
    print("\n6. Comparing classification vs anomaly detection...")
    classification_at_risk = asset_health["health_state"] > 0
    anomaly_at_risk = asset_health["is_anomaly"] == 1

    both_at_risk = (classification_at_risk & anomaly_at_risk).sum()
    classification_only = (classification_at_risk & ~anomaly_at_risk).sum()
    anomaly_only = (~classification_at_risk & anomaly_at_risk).sum()
    neither_at_risk = (~classification_at_risk & ~anomaly_at_risk).sum()

    print(f"   Both methods flag as risk: {both_at_risk} ({both_at_risk/len(asset_health)*100:.1f}%)")
    print(f"   Classification only: {classification_only} ({classification_only/len(asset_health)*100:.1f}%)")
    print(f"   Anomaly detection only: {anomaly_only} ({anomaly_only/len(asset_health)*100:.1f}%)")
    print(f"   Neither flags as risk: {neither_at_risk} ({neither_at_risk/len(asset_health)*100:.1f}%)")

    # Aggregate by asset (if multiple samples per asset)
    print("\n7. Aggregating results by asset...")
    asset_summary = asset_health.groupby("asset_id").agg(
        {
            "failure_risk": "mean",
            "health_state": "max",  # Use worst state
            "is_anomaly": "max",  # Flag if any anomaly detected
            "anomaly_score": "max",  # Use highest anomaly score
            "combined_risk": "max",  # Use highest combined risk
        }
    ).reset_index()

    asset_summary = asset_summary.sort_values("combined_risk", ascending=False)
    print(f"\n   Top 10 assets (aggregated):")
    print(
        asset_summary[["asset_id", "failure_risk", "health_state", "is_anomaly", "combined_risk"]]
        .head(10)
        .to_string(index=False)
    )

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)
    print("\nKey takeaways:")
    print("  - Classification predicts failure risk based on historical patterns")
    print("  - Anomaly detection flags unusual behavior without labels")
    print("  - Combined approach provides comprehensive asset health assessment")
    print("  - Risk ranking enables targeted maintenance prioritization")
    print("\nNext steps:")
    print("  - Integrate with asset registry data (age, manufacturer, etc.)")
    print("  - Add time series features for trend-based assessment")
    print("  - Connect to maintenance action workflows")
    print("  - Monitor model performance and retrain periodically")


if __name__ == "__main__":
    main()

