# Predictive Maintenance Implementation

This document summarizes the implementation of predictive maintenance techniques from the article "Predictive Maintenance with Time Series in Python using PCA, Statistics and LSTMs" into anomsmith.

## Overview

Anomsmith now includes comprehensive predictive maintenance capabilities following the 4-layer architecture:

1. **Layer 1 (Objects)**: Health state objects (`HealthState`, `HealthStateView`, `ActionView`, `PolicyResult`)
2. **Layer 2 (Primitives)**: Detection and classification algorithms
3. **Layer 3 (Tasks)**: Task orchestration (implicit via workflows)
4. **Layer 4 (Workflows)**: Public API for predictive maintenance

## Implemented Techniques

### 1. PCA for Healthy Bounds
**Location**: `anomsmith.primitives.detectors.pca.PCADetector` (already existed, enhanced)

- Uses PCA to define healthy operating space
- Detects drift from healthy cluster
- Supports reconstruction error and Mahalanobis distance scoring

### 2. Time Series Forecasting (ARIMA Drift Detection)
**Location**: `anomsmith.primitives.detectors.drift.ARIMADriftDetector` (NEW)

- Uses ARIMA models to forecast sensor values
- Detects drift by comparing actual values to forecasts
- Flags anomalies when residuals exceed threshold
- **Requires**: `statsmodels` (optional dependency)

### 3. Isolation Forest Anomaly Detection
**Location**: `anomsmith.primitives.detectors.ml.IsolationForestDetector` (already existed)

- Automatically flags abnormal behavior in multivariate sensor readings
- Works well when no labels are available

### 4. Sequence-Based Distress Classification (Random Forest)
**Location**: `anomsmith.primitives.classifiers.sequence.SequenceDistressClassifier` (NEW)

- Uses sliding window approach to classify sequences
- Random Forest classifier for distress detection
- Helper function `create_sequences()` for data preparation

### 5. LSTM-Based Distress Classification
**Location**: `anomsmith.primitives.classifiers.lstm.LSTMDistressClassifier` (NEW)

- Long Short-Term Memory networks for sequence modeling
- Learns patterns in sensor readings that indicate distress
- **Requires**: `tensorflow` (optional dependency)

### 6. Attention-Based LSTM Classification
**Location**: `anomsmith.primitives.classifiers.lstm.AttentionLSTMDistressClassifier` (NEW)

- LSTM with attention mechanism
- Focuses on important parts of the sequence
- Provides attention weights for interpretability
- **Requires**: `tensorflow` (optional dependency)

## Health State Discretization

**Location**: `anomsmith.primitives.health_state.discretize.discretize_rul_to_health_states()`

- Converts RUL (Remaining Useful Life) values to health states:
  - Healthy (RUL > 30 cycles)
  - Warning (10 < RUL <= 30 cycles)
  - Distress (RUL <= 10 cycles)

## Decision Policies

**Location**: `anomsmith.primitives.policy.simple.SimpleHealthPolicy`

- Maps health states to actions:
  - Distress (2) → Intervene
  - Healthy (0) → Warning (1) transition → Review
  - Otherwise → Wait
- Configurable costs and risk reductions
- Evaluates policy performance metrics

## Workflows

**Location**: `anomsmith.workflows.pm`

- `discretize_rul()`: Convert RUL to health states
- `apply_policy()`: Apply decision policy to health states
- `evaluate_policy()`: Evaluate policy performance

## Examples

1. **Basic Predictive Maintenance Example**
   - Location: `examples/predictive_maintenance_example.py`
   - Demonstrates RUL discretization, policy application, and evaluation

2. **Comprehensive Notebook** (TODO)
   - Location: `examples/notebooks/predictive_maintenance_comprehensive.ipynb`
   - Will demonstrate all techniques: PCA, ARIMA, Isolation Forest, Sequence models, LSTM, Attention

## Usage Example

```python
import pandas as pd
import numpy as np
from anomsmith import discretize_rul, apply_policy, evaluate_policy
from anomsmith.primitives.classifiers.sequence import (
    SequenceDistressClassifier,
    create_sequences,
)
from anomsmith.primitives.detectors.pca import PCADetector
from anomsmith.primitives.detectors.drift import ARIMADriftDetector

# Load data (NASA turbofan dataset format)
df = pd.read_csv("train_FD001.txt", sep=r"\s+", header=None)
# ... preprocess data ...

# 1. Discretize RUL to health states
df["health_state"] = discretize_rul(df["RUL"], healthy_threshold=30, warning_threshold=10)

# 2. PCA for healthy bounds
pca_detector = PCADetector(n_components=3, score_method="mahalanobis")
healthy_data = df[df["time"] <= 30][sensor_cols]
pca_detector.fit(healthy_data.values)

# 3. ARIMA drift detection
arima_detector = ARIMADriftDetector(order=(1, 1, 1), threshold_std=2.0)
for unit_id in df["unit"].unique():
    unit_data = df[df["unit"] == unit_id].sort_values("time")
    arima_detector.fit(unit_data["sensor_9"].values)
    drift_scores = arima_detector.score(unit_data["sensor_9"])

# 4. Sequence-based classification
sequences, labels = create_sequences(
    df, sensor_cols=["sensor_9"], target_col="distress", window_size=30
)
classifier = SequenceDistressClassifier(window_size=30, n_estimators=100)
classifier.fit(sequences, labels)
health_states = classifier.predict_health_states(sequences)

# 5. Apply decision policy
policy_result = apply_policy(
    health_states.to_series(),
    previous_states=None,
    intervene_cost=100.0,
    review_cost=30.0,
    base_risks=(0.01, 0.1, 0.3),
)

# 6. Evaluate policy
metrics = evaluate_policy(
    health_states.to_series(),
    intervene_cost=100.0,
    review_cost=30.0,
    base_risks=(0.01, 0.1, 0.3),
)
print(f"Total cost: ${metrics['total_cost']:.2f}")
print(f"Total risk: {metrics['total_risk']:.4f}")
```

## Optional Dependencies

- **statsmodels**: Required for `ARIMADriftDetector`
  - Install: `pip install statsmodels`
  
- **tensorflow**: Required for `LSTMDistressClassifier` and `AttentionLSTMDistressClassifier`
  - Install: `pip install tensorflow`

## Architecture Compliance

All implementations follow anomsmith's strict 4-layer architecture:
- **Layer 1**: Immutable objects only (numpy, pandas)
- **Layer 2**: Algorithm interfaces (numpy, pandas, sklearn, optional deep learning)
- **Layer 3**: Task orchestration (implicit via workflows)
- **Layer 4**: Public workflows (can use matplotlib for plots)

## Next Steps

1. Create comprehensive example notebook demonstrating all techniques
2. Add unit tests for new primitives
3. Add integration tests for workflows
4. Document best practices for combining methods

