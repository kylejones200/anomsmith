"""AWS-based predictive maintenance pipeline example.

Demonstrates how to use anomsmith in an AWS-based predictive maintenance
system following the architecture described in:
"Implementing a Predictive Maintenance System for Oil and Gas using AWS"

This example shows:
1. Model training and serialization for SageMaker
2. Batch inference for Kinesis/S3 data processing
3. Model monitoring and CloudWatch integration
4. Concept drift detection for retraining triggers
"""

import logging
from datetime import datetime

import numpy as np
import pandas as pd

from anomsmith import (
    detect_anomalies,
    ThresholdRule,
    batch_score,
    compute_performance_metrics,
    detect_concept_drift,
    aggregate_metrics_for_cloudwatch,
    ModelPerformanceTracker,
)
from anomsmith.primitives.detectors.ml import IsolationForestDetector
from anomsmith.primitives.model_persistence import (
    export_model_for_sagemaker,
    save_model,
)
from anomsmith.primitives.scorers.robust_zscore import RobustZScoreScorer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def simulate_plant_data(n_samples: int = 10000, n_features: int = 10) -> pd.DataFrame:
    """Simulate plant sensor data (temperature, pressure, vibration, etc.)."""
    np.random.seed(42)

    # Simulate normal operating conditions
    data = np.random.randn(n_samples, n_features)
    data[:, 0] = data[:, 0] * 5 + 60  # Temperature (C)
    data[:, 1] = data[:, 1] * 3 + 25  # Pressure (psi)
    data[:, 2] = np.abs(data[:, 2]) * 0.05 + 0.2  # Vibration (g)

    # Inject some anomalies (equipment failures)
    n_anomalies = int(n_samples * 0.05)
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
    data[anomaly_indices, 0] += 15  # Elevated temperature
    data[anomaly_indices, 1] -= 10  # Pressure drop
    data[anomaly_indices, 2] += 0.3  # Increased vibration

    columns = [f"sensor_{i}" for i in range(n_features)]
    index = pd.date_range("2024-01-01", periods=n_samples, freq="H")

    return pd.DataFrame(data, columns=columns, index=index)


def step1_model_training():
    """Step 1: Train model and prepare for SageMaker deployment."""
    logger.info("=" * 60)
    logger.info("Step 1: Model Training and SageMaker Export")
    logger.info("=" * 60)

    # Generate training data
    logger.info("Generating training data...")
    train_data = simulate_plant_data(n_samples=5000, n_features=10)
    train_labels = (train_data["sensor_0"] > 70).astype(int)  # Simple label based on temperature

    # Train model (Isolation Forest for multivariate anomaly detection)
    logger.info("Training Isolation Forest detector...")
    detector = IsolationForestDetector(contamination=0.05, random_state=42)
    detector.fit(train_data.values)

    # Save model locally
    logger.info("Saving model for deployment...")
    save_model(
        detector,
        "models/isolation_forest_v1",
        metadata={
            "version": "1.0",
            "training_date": datetime.now().isoformat(),
            "n_samples": len(train_data),
            "n_features": train_data.shape[1],
        },
    )

    # Export for SageMaker
    logger.info("Exporting model for SageMaker deployment...")
    export_info = export_model_for_sagemaker(
        detector,
        s3_path="s3://my-predictive-maintenance-bucket/models/isolation-forest/v1.0",
        metadata={"version": "1.0", "deployment_date": datetime.now().isoformat()},
        local_path="models/sagemaker_export",
    )

    logger.info(f"Model exported to: {export_info['local_path']}")
    logger.info(f"S3 upload command: {export_info['upload_command']}")
    logger.info(f"Inference code saved to: {export_info['local_path']}/inference.py")

    return detector, export_info


def step2_batch_inference(detector: IsolationForestDetector):
    """Step 2: Batch inference for processing Kinesis/S3 data."""
    logger.info("\n" + "=" * 60)
    logger.info("Step 2: Batch Inference (Simulating Kinesis/S3 Processing)")
    logger.info("=" * 60)

    # Simulate data stream (e.g., from Kinesis or S3 batch job)
    def data_stream():
        """Simulate streaming data from Kinesis or S3."""
        for i in range(0, 5000, 1000):
            batch_data = simulate_plant_data(n_samples=1000, n_features=10)
            # Simulate arrival with timestamp
            batch_data.index = pd.date_range(f"2024-01-01", periods=1000, freq="H") + pd.Timedelta(
                hours=i
            )
            yield batch_data

    # Process batches
    logger.info("Processing data in batches...")
    all_results = []
    for batch_idx, batch_scores in enumerate(batch_score(data_stream(), detector)):
        # Simulate sending scores to downstream system (e.g., DynamoDB, S3)
        result_df = pd.DataFrame(
            {"score": batch_scores.scores, "timestamp": batch_scores.index}
        )
        all_results.append(result_df)
        logger.info(f"Processed batch {batch_idx + 1}: {len(result_df)} samples")

    combined_results = pd.concat(all_results, ignore_index=False)
    logger.info(f"Total samples processed: {len(combined_results)}")
    logger.info(f"Average score: {combined_results['score'].mean():.4f}")
    logger.info(f"Max score: {combined_results['score'].max():.4f}")

    return combined_results


def step3_model_monitoring(detector: IsolationForestDetector):
    """Step 3: Model monitoring and CloudWatch integration."""
    logger.info("\n" + "=" * 60)
    logger.info("Step 3: Model Monitoring and CloudWatch Integration")
    logger.info("=" * 60)

    # Simulate production data with some drift
    logger.info("Simulating production data...")
    baseline_data = simulate_plant_data(n_samples=2000, n_features=10)
    recent_data = simulate_plant_data(n_samples=1000, n_features=10)
    # Add drift: equipment is degrading
    recent_data["sensor_0"] += 5  # Temperature gradually increasing

    # Get predictions
    baseline_labels = detector.predict(baseline_data.values).labels
    baseline_scores = detector.score(baseline_data.values).scores
    recent_labels = detector.predict(recent_data.values).labels
    recent_scores = detector.score(recent_data.values).scores

    # Compute performance metrics (for monitoring)
    logger.info("Computing performance metrics...")
    # Simulate ground truth labels
    baseline_true = (baseline_data["sensor_0"] > 70).astype(int)
    recent_true = (recent_data["sensor_0"] > 75).astype(int)  # Higher threshold due to drift

    baseline_metrics = compute_performance_metrics(
        baseline_true.values, baseline_labels, baseline_scores
    )
    recent_metrics = compute_performance_metrics(recent_true.values, recent_labels, recent_scores)

    logger.info(f"Baseline F1: {baseline_metrics['f1']:.4f}")
    logger.info(f"Recent F1: {recent_metrics['f1']:.4f}")
    logger.info(f"Baseline Precision: {baseline_metrics['precision']:.4f}")
    logger.info(f"Recent Precision: {recent_metrics['precision']:.4f}")

    # Detect concept drift
    logger.info("\nDetecting concept drift...")
    drift_info = detect_concept_drift(recent_scores, baseline_scores, threshold=2.0)
    logger.info(f"Drift detected: {drift_info['drift_detected']}")
    logger.info(f"Drift magnitude: {drift_info['drift_magnitude']:.4f}")
    if drift_info.get("p_value"):
        logger.info(f"KS test p-value: {drift_info['p_value']:.6f}")

    # Format metrics for CloudWatch
    logger.info("\nFormatting metrics for CloudWatch...")
    cloudwatch_metrics = aggregate_metrics_for_cloudwatch(
        [baseline_metrics, recent_metrics], namespace="PredictiveMaintenance", model_name="IsolationForest"
    )

    logger.info(f"Prepared {len(cloudwatch_metrics)} CloudWatch metrics")
    logger.info("Example metric:")
    logger.info(f"  {cloudwatch_metrics[0]}")

    # Simulate sending to CloudWatch (commented out - requires boto3)
    # import boto3
    # cloudwatch = boto3.client('cloudwatch')
    # cloudwatch.put_metric_data(
    #     Namespace="PredictiveMaintenance",
    #     MetricData=cloudwatch_metrics
    # )

    return baseline_metrics, recent_metrics, drift_info


def step4_performance_tracking(detector: IsolationForestDetector):
    """Step 4: Continuous performance tracking."""
    logger.info("\n" + "=" * 60)
    logger.info("Step 4: Continuous Performance Tracking")
    logger.info("=" * 60)

    # Initialize tracker
    tracker = ModelPerformanceTracker(window_size=1000, model_name="IsolationForest")

    # Simulate continuous updates (e.g., from real-time inference)
    logger.info("Simulating continuous performance updates...")
    for i in range(5):
        # Generate new batch of data
        data = simulate_plant_data(n_samples=200, n_features=10)
        true_labels = (data["sensor_0"] > 70).astype(int)

        # Get predictions
        pred_labels = detector.predict(data.values).labels
        scores = detector.score(data.values).scores

        # Update tracker
        metrics = tracker.update(scores, pred_labels, true_labels.values)

        logger.info(f"Update {i + 1}: F1={metrics.get('f1', 0):.4f}, "
                   f"Anomaly Rate={metrics.get('anomaly_rate', 0):.4f}")

    # Get current metrics
    current = tracker.get_current_metrics()
    logger.info(f"\nCurrent metrics: {current}")

    # Check for degradation
    baseline_metrics = {"f1": 0.85, "precision": 0.82, "recall": 0.88}
    degradation_detected = tracker.detect_degradation(baseline_metrics, threshold=0.1)
    logger.info(f"Degradation detected: {degradation_detected}")

    if degradation_detected:
        logger.info("⚠️  ALERT: Model performance degraded - retraining recommended!")

    return tracker


def main() -> None:
    """Run complete AWS predictive maintenance pipeline example."""
    logger.info("AWS-Based Predictive Maintenance Pipeline Example")
    logger.info("=" * 60)
    logger.info(
        "This example demonstrates how to use anomsmith in an AWS-based\n"
        "predictive maintenance system following the 7-step architecture:\n"
        "1. Model Training & SageMaker Export\n"
        "2. Batch Inference (Kinesis/S3)\n"
        "3. Model Monitoring (CloudWatch)\n"
        "4. Performance Tracking & Drift Detection\n"
    )

    # Step 1: Model training and export
    detector, export_info = step1_model_training()

    # Step 2: Batch inference
    batch_results = step2_batch_inference(detector)

    # Step 3: Model monitoring
    baseline_metrics, recent_metrics, drift_info = step3_model_monitoring(detector)

    # Step 4: Performance tracking
    tracker = step4_performance_tracking(detector)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Pipeline Summary")
    logger.info("=" * 60)
    logger.info("✓ Model trained and exported for SageMaker deployment")
    logger.info("✓ Batch inference processed 5000 samples")
    logger.info(f"✓ Concept drift detected: {drift_info['drift_detected']}")
    logger.info(f"✓ Performance tracking active (window: {tracker.window_size} samples)")

    logger.info("\nNext Steps for AWS Deployment:")
    logger.info("1. Upload model to S3 using: " + export_info["upload_command"])
    logger.info("2. Create SageMaker endpoint with inference.py code")
    logger.info("3. Set up Lambda functions for Kinesis stream processing")
    logger.info("4. Configure CloudWatch alarms for drift detection")
    logger.info("5. Set up EventBridge rules to trigger retraining pipelines")
    logger.info("6. Deploy QuickSight dashboards for visualization")

    logger.info("\n" + "=" * 60)
    logger.info("Example complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

