"""Survival analysis for turbofan engine predictive maintenance.

Demonstrates survival analysis using NASA C-MAPSS turbofan dataset (FD003).
Based on "Predictive Maintenance Modeling Time-to-Failure using Survival Analysis in Python".

This example shows:
1. Loading and preprocessing turbofan sensor data
2. Training multiple survival models (CoxPH, LogisticHazard, DeepSurv)
3. Evaluating models using C-index
4. Predicting health states from survival models
5. Integration with decision policies
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from anomsmith import (
    apply_policy,
    compare_survival_models,
    discretize_rul,
    evaluate_policy,
    fit_survival_model_for_maintenance,
    predict_health_states_from_survival,
    predict_rul_from_survival,
)
from anomsmith.objects.health_state import HealthState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_turbofan_data(
    file_path: str, n_samples: Optional[int] = None
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load and preprocess NASA C-MAPSS turbofan data (FD003 format).

    Args:
        file_path: Path to train_FD003.txt file (or None to generate synthetic)
        n_samples: Optional limit on number of samples to load

    Returns:
        Tuple of (X, durations, events) where:
        - X: Feature matrix with sensor readings
        - durations: Time-to-failure values (RUL)
        - events: Event indicators (all 1 for FD003 - all engines run until failure)
    """
    if file_path is None or not Path(file_path).exists():
        logger.warning(
            f"File {file_path} not found. Generating synthetic turbofan data for demonstration."
        )
        return generate_synthetic_turbofan_data(n_samples=n_samples)

    logger.info(f"Loading turbofan data from {file_path}")

    # Column names for FD003 format
    cols = (
        ["unit", "cycle"]
        + [f"op_setting_{i}" for i in range(1, 4)]
        + [f"sensor_{i}" for i in range(1, 22)]
    )

    # Load data
    df = pd.read_csv(file_path, sep=r"\s+", header=None, names=cols)

    # Compute RUL (time-to-failure) for each engine
    rul = df.groupby("unit")["cycle"].max().reset_index()
    rul.columns = ["unit", "max_cycle"]
    df = df.merge(rul, on="unit")
    df["event"] = 1  # All engines run until failure in FD003
    df["time"] = df["max_cycle"] - df["cycle"]  # RUL = time to failure

    # Select sensor features (exclude operational settings for simplicity)
    feature_cols = [f"sensor_{i}" for i in range(1, 22)]
    X = df[feature_cols].copy()
    durations = df["time"].values
    events = df["event"].values

    if n_samples is not None:
        X = X.head(n_samples)
        durations = durations[:n_samples]
        events = events[:n_samples]

    logger.info(f"Loaded {len(X)} samples with {X.shape[1]} features")
    logger.info(f"Duration range: {durations.min()} to {durations.max()} cycles")
    logger.info(f"Event rate: {events.mean():.2%}")

    return X, pd.Series(durations, name="duration"), pd.Series(events, name="event")


def generate_synthetic_turbofan_data(
    n_samples: int = 5000, n_engines: int = 100
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Generate synthetic turbofan data for demonstration.

    Simulates sensor readings and failure patterns similar to NASA C-MAPSS dataset.
    """
    np.random.seed(42)
    logger.info("Generating synthetic turbofan data...")

    # Generate engine IDs
    engine_ids = np.random.choice(range(1, n_engines + 1), size=n_samples)
    max_cycles = np.random.randint(150, 300, size=n_engines)

    # Create sensor data
    data = []
    durations = []
    events = []

    for i, engine_id in enumerate(engine_ids):
        cycle = i % max_cycles[engine_id - 1] + 1
        max_cycle = max_cycles[engine_id - 1]
        rul = max_cycle - cycle

        # Simulate sensor readings with degradation over time
        # Sensors degrade as engine approaches failure
        degradation = 1.0 - (rul / max_cycle)  # 0 to 1 as engine degrades

        # Base sensor readings with noise
        sensors = np.random.randn(21) * 5 + 50
        # Add degradation signal
        sensors[:5] += degradation * 10  # Temperature sensors increase
        sensors[5:10] += degradation * -5  # Pressure sensors decrease
        sensors[10:] += degradation * np.random.randn(11) * 2  # Other sensors vary

        data.append(sensors)
        durations.append(rul)
        events.append(1)  # All engines run until failure

    X = pd.DataFrame(data, columns=[f"sensor_{i}" for i in range(1, 22)])
    durations = pd.Series(durations, name="duration")
    events = pd.Series(events, name="event")

    return X, durations, events


def preprocess_features(X: pd.DataFrame) -> tuple[pd.DataFrame, object]:
    """Preprocess features for survival modeling.

    Standardizes features and optionally applies variance threshold.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import VarianceThreshold

    # Standardize features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X), columns=X.columns, index=X.index
    ).astype("float32")

    # Apply variance threshold (remove low-variance features)
    selector = VarianceThreshold(threshold=1e-4)
    X_selected = selector.fit_transform(X_scaled)
    selected_features = [f for f, keep in zip(X.columns, selector.get_support()) if keep]
    X_final = pd.DataFrame(X_selected, columns=selected_features, index=X.index).astype("float32")

    logger.info(f"Preprocessed features: {X.shape[1]} -> {X_final.shape[1]} features")
    return X_final, scaler


def main() -> None:
    """Run survival analysis example for turbofan predictive maintenance."""
    logger.info("=" * 70)
    logger.info("Survival Analysis for Turbofan Engine Predictive Maintenance")
    logger.info("=" * 70)
    logger.info(
        "This example demonstrates survival analysis using NASA C-MAPSS\n"
        "turbofan dataset (FD003) or synthetic equivalent.\n"
        "Models: CoxPH (lifelines), LogisticHazard (pycox), DeepSurv (pycox)"
    )

    # Step 1: Load and preprocess data
    logger.info("\n" + "=" * 70)
    logger.info("Step 1: Loading and Preprocessing Data")
    logger.info("=" * 70)

    # Try to load real data, fall back to synthetic
    data_path = "train_FD003.txt"  # Update this path as needed
    try:
        X, durations, events = load_turbofan_data(data_path, n_samples=5000)
    except FileNotFoundError:
        logger.warning("Real dataset not found, using synthetic data")
        X, durations, events = generate_synthetic_turbofan_data(n_samples=5000, n_engines=100)

    # Preprocess features
    X_processed, scaler = preprocess_features(X)

    # Train-test split
    from sklearn.model_selection import train_test_split

    X_train, X_test, durations_train, durations_test, events_train, events_test = (
        train_test_split(
            X_processed,
            durations,
            events,
            test_size=0.2,
            random_state=42,
        )
    )

    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")

    # Step 2: Train survival models
    logger.info("\n" + "=" * 70)
    logger.info("Step 2: Training Survival Models")
    logger.info("=" * 70)

    models = {}

    # Train CoxPH model (lifelines) if available
    try:
        logger.info("\nTraining CoxPH model (lifelines)...")
        cox_model = fit_survival_model_for_maintenance(
            X_train,
            durations_train.values,
            events_train.values,
            model_type="cox",
            penalizer=0.1,
        )
        models["CoxPH (lifelines)"] = cox_model
        logger.info("✓ CoxPH model trained")
    except ImportError:
        logger.warning("lifelines not available, skipping CoxPH model")
    except Exception as e:
        logger.error(f"Error training CoxPH: {e}")

    # Train LogisticHazard model (pycox) if available
    try:
        logger.info("\nTraining LogisticHazard model (pycox)...")
        lhaz_model = fit_survival_model_for_maintenance(
            X_train,
            durations_train.values,
            events_train.values,
            model_type="logistic_hazard",
            n_bins=50,
            num_nodes=[32, 32],
            batch_size=128,
            epochs=50,
        )
        models["LogisticHazard (pycox)"] = lhaz_model
        logger.info("✓ LogisticHazard model trained")
    except ImportError:
        logger.warning("pycox not available, skipping LogisticHazard model")
    except Exception as e:
        logger.error(f"Error training LogisticHazard: {e}")

    # Train DeepSurv model (pycox) if available
    try:
        logger.info("\nTraining DeepSurv model (pycox)...")
        deepsurv_model = fit_survival_model_for_maintenance(
            X_train,
            durations_train.values,
            events_train.values,
            model_type="deepsurv",
            num_nodes=[32, 32],
            batch_size=128,
            epochs=50,
        )
        models["DeepSurv (pycox)"] = deepsurv_model
        logger.info("✓ DeepSurv model trained")
    except ImportError:
        logger.warning("pycox not available, skipping DeepSurv model")
    except Exception as e:
        logger.error(f"Error training DeepSurv: {e}")

    if not models:
        logger.error(
            "No models could be trained. Install required dependencies:\n"
            "  pip install lifelines  # For CoxPH\n"
            "  pip install pycox torch  # For LogisticHazard and DeepSurv"
        )
        return

    # Step 3: Evaluate models
    logger.info("\n" + "=" * 70)
    logger.info("Step 3: Evaluating Survival Models")
    logger.info("=" * 70)

    comparison = compare_survival_models(
        models, X_test, durations_test.values, events_test.values
    )

    logger.info("\nModel Comparison (C-index):")
    logger.info(comparison[["model", "c_index"]].to_string(index=False))

    if "mean_absolute_error" in comparison.columns:
        logger.info("\nModel Comparison (MAE):")
        logger.info(comparison[["model", "mean_absolute_error"]].to_string(index=False))

    # Select best model
    best_model_name = comparison.loc[comparison["c_index"].idxmax(), "model"]
    best_model = models[best_model_name]
    best_c_index = comparison.loc[comparison["c_index"].idxmax(), "c_index"]

    logger.info(f"\nBest model: {best_model_name} (C-index: {best_c_index:.3f})")

    # Step 4: Predict RUL and health states
    logger.info("\n" + "=" * 70)
    logger.info("Step 4: Predicting RUL and Health States")
    logger.info("=" * 70)

    # Predict RUL for test set
    logger.info("Predicting RUL from survival model...")
    rul_predictions = predict_rul_from_survival(best_model, X_test, threshold=0.5)

    logger.info(f"\nRUL predictions (first 10):")
    logger.info(rul_predictions.head(10).to_string())

    logger.info(f"\nRUL statistics:")
    logger.info(f"  Mean: {rul_predictions.mean():.2f} cycles")
    logger.info(f"  Median: {rul_predictions.median():.2f} cycles")
    logger.info(f"  Min: {rul_predictions.min():.2f} cycles")
    logger.info(f"  Max: {rul_predictions.max():.2f} cycles")

    # Predict health states
    logger.info("\nPredicting health states from survival model...")
    health_states = predict_health_states_from_survival(
        best_model, X_test, healthy_threshold=30.0, warning_threshold=10.0, threshold=0.5
    )

    # Count health states
    state_counts = pd.Series(health_states.states).value_counts().sort_index()
    state_names = {0: "Healthy", 1: "Warning", 2: "Distress"}
    logger.info(f"\nHealth state distribution:")
    for state, count in state_counts.items():
        logger.info(f"  {state_names[state]}: {count} ({count/len(health_states)*100:.1f}%)")

    # Step 5: Apply decision policy
    logger.info("\n" + "=" * 70)
    logger.info("Step 5: Applying Decision Policy")
    logger.info("=" * 70)

    policy_result = apply_policy(health_states)

    action_counts = pd.Series(policy_result.actions.actions).value_counts().sort_index()
    action_names = {0: "Wait", 1: "Review", 2: "Intervene"}
    logger.info(f"\nActions recommended:")
    for action, count in action_counts.items():
        logger.info(f"  {action_names[action]}: {count} ({count/len(policy_result)*100:.1f}%)")

    if policy_result.costs is not None:
        logger.info(f"\nTotal cost: ${policy_result.costs.sum():.2f}")
    if policy_result.risks is not None:
        logger.info(f"Total risk: {policy_result.risks.sum():.4f}")

    # Evaluate policy
    evaluation = evaluate_policy(policy_result)
    logger.info(f"\nPolicy evaluation:")
    for key, value in evaluation.items():
        logger.info(f"  {key}: {value}")

    # Step 6: Survival curves for sample engines
    logger.info("\n" + "=" * 70)
    logger.info("Step 6: Survival Curves for Sample Engines")
    logger.info("=" * 70)

    sample_indices = [0, 10, 50]
    for idx in sample_indices:
        if idx >= len(X_test):
            continue

        sample_X = X_test.iloc[[idx]]
        true_duration = durations_test.iloc[idx]
        true_rul = true_duration

        # Predict survival function
        surv_df = best_model.predict_survival_function(sample_X)
        predicted_median_rul = rul_predictions.iloc[idx]

        logger.info(f"\nSample {idx}:")
        logger.info(f"  True RUL: {true_rul:.1f} cycles")
        logger.info(f"  Predicted median RUL: {predicted_median_rul:.1f} cycles")
        logger.info(f"  Error: {abs(predicted_median_rul - true_rul):.1f} cycles")

        # Get survival probabilities at key time points
        time_points = [10, 30, 50, 100]
        for t in time_points:
            if t <= surv_df.index.max():
                surv_prob = surv_df.loc[t, 0] if t in surv_df.index else surv_df.iloc[
                    surv_df.index.get_indexer([t], method="nearest")[0], 0
                ]
                logger.info(f"  Survival probability at {t} cycles: {surv_prob:.3f}")

    logger.info("\n" + "=" * 70)
    logger.info("Example Complete!")
    logger.info("=" * 70)
    logger.info("\nKey Takeaways:")
    logger.info("  - Survival analysis provides time-to-failure predictions")
    logger.info("  - Neural models (LogisticHazard, DeepSurv) often outperform linear CoxPH")
    logger.info("  - Survival curves show probability of operational status over time")
    logger.info("  - RUL predictions integrate with health state and decision policies")
    logger.info("\nNext Steps:")
    logger.info("  - Tune model hyperparameters (n_bins, network architecture)")
    logger.info("  - Add feature engineering (rolling statistics, differences)")
    logger.info("  - Integrate with AWS SageMaker for production deployment")
    logger.info("  - Set up monitoring and retraining pipelines")


if __name__ == "__main__":
    main()

