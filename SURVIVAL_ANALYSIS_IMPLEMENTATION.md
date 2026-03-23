# Survival Analysis Implementation

This document summarizes the implementation of survival analysis techniques from "Predictive Maintenance Modeling Time-to-Failure using Survival Analysis in Python (Turbofan Engines)" into anomsmith.

## Overview

Anomsmith now includes comprehensive survival analysis capabilities for predictive maintenance, following the 4-layer architecture:

1. **Layer 1 (Objects)**: Health state objects (existing)
2. **Layer 2 (Primitives)**: Survival model implementations
3. **Layer 3 (Tasks)**: Task orchestration (implicit via workflows)
4. **Layer 4 (Workflows)**: Public API for survival analysis

## Implemented Models

### 1. Lifelines CoxPH (`LifelinesCoxModel`)
**Location**: `anomsmith.primitives.survival.lifelines_cox.LifelinesCoxModel`

- Linear Cox Proportional Hazards regression
- Good for interpretability and sparse data
- Assumes proportional hazards and log-linear relationship
- **Requires**: `lifelines` (optional dependency)

### 2. LogisticHazard (`LogisticHazardModel`)
**Location**: `anomsmith.primitives.survival.neural.LogisticHazardModel`

- Discrete-time neural survival model
- Models time-to-failure in discrete intervals
- Well-suited for sensor data with nonlinear degradation
- Makes fewer assumptions than CoxPH
- **Requires**: `pycox`, `torch` (optional dependencies)

### 3. DeepSurv (`DeepSurvModel`)
**Location**: `anomsmith.primitives.survival.neural.DeepSurvModel`

- Continuous-time neural Cox model
- Learns flexible nonlinear mapping from features to risk
- Captures feature interactions but assumes proportional hazards
- **Requires**: `pycox`, `torch` (optional dependencies)

## Base Class

### `CoxSurvivalModel` (Abstract Base)
**Location**: `anomsmith.primitives.survival.cox.CoxSurvivalModel`

Provides common interface:
- `fit(X, durations, events)`: Train model
- `predict_survival_function(X, time_points)`: Get survival curves
- `predict_risk_score(X)`: Get relative risk scores
- `predict_time_to_failure(X, threshold)`: Get median TTF (vectorized)

## Evaluation Metrics

**Location**: `anomsmith.workflows.eval.survival_metrics`

- `compute_concordance_index()`: C-index for ranking ability (0.5 = random, 1.0 = perfect)
- `evaluate_survival_model()`: Comprehensive metrics including C-index, MAE, etc.

## Workflows

**Location**: `anomsmith.workflows.survival`

- `fit_survival_model_for_maintenance()`: Convenience function to fit models with defaults
- `predict_rul_from_survival()`: Convert survival predictions to RUL
- `predict_health_states_from_survival()`: Convert survival → RUL → health states
- `compare_survival_models()`: Evaluate multiple models side-by-side

## Integration with Existing Infrastructure

Survival models integrate seamlessly with existing anomsmith components:

1. **Health States**: `predict_health_states_from_survival()` converts survival predictions to `HealthStateView`
2. **Decision Policies**: Health states from survival models can be fed directly to `apply_policy()`
3. **Evaluation**: C-index and other metrics work with existing evaluation workflows

## Example

**Location**: `examples/survival_analysis_turbofan_example.py`

Complete example demonstrating:
1. Loading/preprocessing NASA C-MAPSS FD003 data (or synthetic equivalent)
2. Training multiple survival models (CoxPH, LogisticHazard, DeepSurv)
3. Evaluating models using C-index
4. Predicting RUL and health states
5. Applying decision policies
6. Comparing survival curves

## Usage Example

```python
from anomsmith import (
    fit_survival_model_for_maintenance,
    predict_health_states_from_survival,
    compare_survival_models,
    apply_policy,
)
from sklearn.model_selection import train_test_split

# Load data
X, durations, events = load_turbofan_data("train_FD003.txt")

# Train-test split
X_train, X_test, durations_train, durations_test, events_train, events_test = (
    train_test_split(X, durations, events, test_size=0.2, random_state=42)
)

# Train models
cox_model = fit_survival_model_for_maintenance(
    X_train, durations_train, events_train, model_type="cox"
)
lhaz_model = fit_survival_model_for_maintenance(
    X_train, durations_train, events_train,
    model_type="logistic_hazard", n_bins=50, epochs=50
)
deepsurv_model = fit_survival_model_for_maintenance(
    X_train, durations_train, events_train,
    model_type="deepsurv", epochs=50
)

# Compare models
models = {"CoxPH": cox_model, "LogisticHazard": lhaz_model, "DeepSurv": deepsurv_model}
comparison = compare_survival_models(models, X_test, durations_test, events_test)
print(comparison)  # C-index, MAE, etc.

# Predict health states
health_states = predict_health_states_from_survival(
    best_model, X_test, healthy_threshold=30, warning_threshold=10
)

# Apply decision policy
policy_result = apply_policy(health_states)
print(f"Total cost: ${policy_result.costs.sum():.2f}")
print(f"Total risk: {policy_result.risks.sum():.4f}")
```

## Optional Dependencies

- **lifelines**: Required for `LifelinesCoxModel`
  - Install: `pip install lifelines`
  
- **pycox + torch**: Required for `LogisticHazardModel` and `DeepSurvModel`
  - Install: `pip install pycox torch`
  
- **scipy**: Optional for enhanced drift detection (KS test)
  - Install: `pip install scipy`

All models are optional - anomsmith works without them, but survival analysis features require the appropriate dependencies.

## Architecture Compliance

All implementations follow anomsmith's strict 4-layer architecture:
- **Layer 1**: Immutable objects only (numpy, pandas)
- **Layer 2**: Algorithm interfaces (numpy, pandas, sklearn, optional deep learning)
- **Layer 3**: Task orchestration (implicit via workflows)
- **Layer 4**: Public workflows (can use matplotlib for plots if needed)

## Performance Notes

- `predict_time_to_failure()` is vectorized for efficiency
- Survival curve prediction uses efficient pandas operations
- Models leverage existing vectorization in pycox/lifelines

## Next Steps

1. Add more survival model variants (AFT models, etc.)
2. Add survival-specific feature engineering (time-dependent covariates)
3. Add survival curve visualization utilities
4. Integrate with AWS SageMaker for production deployment
5. Add survival model persistence for cloud deployment

