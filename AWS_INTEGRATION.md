# AWS Integration for Predictive Maintenance

This document describes the AWS integration features added to anomsmith based on the architecture described in "Implementing a Predictive Maintenance System for Oil and Gas using AWS".

## Overview

Anomsmith now includes utilities and workflows designed to integrate seamlessly with AWS services for production-scale predictive maintenance systems. These features follow anomsmith's 4-layer architecture while providing cloud-ready functionality.

## Features Added

### 1. Model Persistence (`anomsmith.primitives.model_persistence`)

**Purpose**: Serialize and export models for cloud deployment (AWS SageMaker, Azure ML, GCP Vertex AI).

**Key Functions**:
- `save_model()`: Save anomsmith models to disk with metadata
- `load_model()`: Load saved models
- `export_model_for_sagemaker()`: Export models in SageMaker-compatible format
  - Generates inference code template (`inference.py`)
  - Creates `requirements.txt` with dependencies
  - Provides S3 upload commands
  - Includes metadata for deployment tracking

**Example**:
```python
from anomsmith.primitives.model_persistence import export_model_for_sagemaker

export_info = export_model_for_sagemaker(
    model=detector,
    s3_path="s3://my-bucket/models/anomaly-detector/v1.0",
    metadata={"version": "1.0", "training_date": "2024-01-01"}
)

# Upload to S3
# aws s3 sync models/sagemaker_export s3://my-bucket/models/anomaly-detector/v1.0
```

### 2. Batch Inference (`anomsmith.workflows.batch_inference`)

**Purpose**: Process large datasets efficiently for Kinesis streams, S3 batch jobs, or Lambda functions.

**Key Functions**:
- `batch_score()`: Score anomalies in batches from an iterator
- `batch_predict()`: Predict anomalies in batches
- `process_s3_batch()`: Process multiple S3 files with anomaly detection

**Use Cases**:
- AWS Kinesis Data Streams processing
- S3 batch transformation jobs
- Lambda function batch processing
- SageMaker Batch Transform jobs

**Example**:
```python
from anomsmith.workflows.batch_inference import batch_score

def data_stream():
    # Simulate Kinesis or S3 data stream
    for batch in s3_batches:
        yield pd.read_csv(batch)

for batch_scores in batch_score(data_stream(), scorer):
    # Process scores (e.g., write to DynamoDB, S3, etc.)
    write_results(batch_scores)
```

### 3. Model Monitoring (`anomsmith.workflows.model_monitoring`)

**Purpose**: Track model performance and detect degradation for CloudWatch integration.

**Key Functions**:
- `compute_performance_metrics()`: Compute comprehensive metrics (precision, recall, F1, etc.)
- `detect_concept_drift()`: Detect distribution drift using statistical tests
- `aggregate_metrics_for_cloudwatch()`: Format metrics for AWS CloudWatch PutMetricData API
- `ModelPerformanceTracker`: Class for continuous performance tracking

**Use Cases**:
- CloudWatch metrics publishing
- Model performance monitoring
- Automated retraining triggers
- Alert generation for performance degradation

**Example**:
```python
from anomsmith.workflows.model_monitoring import (
    compute_performance_metrics,
    aggregate_metrics_for_cloudwatch,
    ModelPerformanceTracker
)

# Compute metrics
metrics = compute_performance_metrics(true_labels, pred_labels, scores)

# Format for CloudWatch
cw_metrics = aggregate_metrics_for_cloudwatch(
    [metrics],
    namespace="PredictiveMaintenance",
    model_name="IsolationForest"
)

# Send to CloudWatch (requires boto3)
import boto3
cloudwatch = boto3.client('cloudwatch')
cloudwatch.put_metric_data(
    Namespace="PredictiveMaintenance",
    MetricData=cw_metrics
)

# Continuous tracking
tracker = ModelPerformanceTracker(window_size=1000)
tracker.update(scores, pred_labels, true_labels)
if tracker.detect_degradation(baseline_metrics):
    trigger_retraining()
```

## Architecture Alignment

All AWS integration features follow anomsmith's strict 4-layer architecture:

### Layer 1 (Objects)
- Uses existing `ScoreView`, `LabelView` objects
- No new objects needed

### Layer 2 (Primitives)
- `model_persistence.py`: Model serialization utilities
- No AWS-specific dependencies (uses standard pickle/json)

### Layer 4 (Workflows)
- `batch_inference.py`: Batch processing workflows
- `model_monitoring.py`: Monitoring and metrics workflows
- Optional AWS dependencies (boto3) handled gracefully with try/except

## AWS Services Integration

### SageMaker Integration
1. **Model Export**: Use `export_model_for_sagemaker()` to create deployment package
2. **Inference Endpoints**: Generated `inference.py` works with SageMaker endpoints
3. **Batch Transform**: Use `batch_score()` or `batch_predict()` in batch jobs
4. **Training Jobs**: Train models using SageMaker Studio/Pipelines

### Kinesis Integration
1. **Stream Processing**: Use `batch_score()` to process Kinesis records in Lambda
2. **Firehose**: Process Firehose output with batch inference functions
3. **Analytics**: Stream results to Kinesis Analytics for real-time dashboards

### S3 Integration
1. **Batch Jobs**: Use `process_s3_batch()` for S3 batch transformation
2. **Model Storage**: Save models to S3 using `save_model()` + S3 sync
3. **Data Pipeline**: Process S3 data in Lambda or ECS tasks

### CloudWatch Integration
1. **Metrics**: Use `aggregate_metrics_for_cloudwatch()` to publish metrics
2. **Alarms**: Set up CloudWatch alarms based on performance metrics
3. **Dashboards**: Visualize metrics in CloudWatch dashboards

### Step Functions Integration
1. **Workflow Orchestration**: Use anomsmith workflows in Step Functions state machines
2. **Retraining Pipelines**: Trigger retraining based on drift detection
3. **Batch Processing**: Orchestrate S3 batch jobs with Step Functions

### EventBridge Integration
1. **Drift Detection**: Use `detect_concept_drift()` to trigger EventBridge events
2. **Scheduled Retraining**: Schedule periodic model retraining
3. **Alerts**: Send notifications when degradation detected

## Example Pipeline

See `examples/aws_predictive_maintenance_example.py` for a complete example showing:

1. **Step 1**: Model training and SageMaker export
2. **Step 2**: Batch inference simulation (Kinesis/S3)
3. **Step 3**: Model monitoring and CloudWatch metrics
4. **Step 4**: Performance tracking and drift detection

## Optional Dependencies

AWS integration features use optional dependencies:

```toml
[project.optional-dependencies]
aws = [
    "boto3>=1.26.0",
]
stats = [
    "statsmodels>=0.13.0",
    "scipy>=1.9.0",
]
```

Install with:
```bash
pip install anomsmith[aws]  # For S3 batch processing
pip install anomsmith[stats]  # For drift detection with KS test
pip install anomsmith[all]  # All optional dependencies
```

## Best Practices

1. **Model Versioning**: Always include version metadata when saving models
2. **Batch Sizing**: Use appropriate batch sizes (1000-10000 samples) for Lambda functions
3. **Error Handling**: Batch functions yield empty results on error - handle gracefully
4. **Monitoring**: Track metrics continuously, not just at deployment
5. **Drift Detection**: Set up automated alerts for concept drift
6. **Retraining**: Use CloudWatch alarms or EventBridge rules to trigger retraining

## Next Steps

1. **Deploy to SageMaker**: Use exported model package to create endpoint
2. **Set up Kinesis**: Configure stream processing with Lambda functions
3. **Configure CloudWatch**: Create dashboards and alarms for monitoring
4. **Automate Retraining**: Set up SageMaker Pipelines for periodic retraining
5. **Visualize Results**: Deploy QuickSight dashboards for operations team

## Integration with Existing Workflows

All AWS integration features work seamlessly with existing anomsmith workflows:

```python
# Standard workflow
from anomsmith import detect_anomalies, ThresholdRule
result = detect_anomalies(y, detector, threshold_rule)

# AWS-enhanced workflow
from anomsmith import batch_score, compute_performance_metrics
for scores in batch_score(kinesis_stream(), detector):
    metrics = compute_performance_metrics(y_true, y_pred, scores.scores)
    # Send to CloudWatch, trigger alerts, etc.
```

The AWS integration extends anomsmith's capabilities without changing core functionality, maintaining backward compatibility while enabling cloud-scale deployments.

