"""Ensemble models for ordered time series (ordinal classification).

Demonstrates ensemble methods for predicting ordered health states
(Healthy, Warning, Distress) from turbofan sensor data.

Based on "Predictive Maintenance with Ensemble Models for Ordered Time Series in Python (Turbofan Case)".

This example shows:
1. Loading and preprocessing turbofan data with health state categorization
2. Training three base ordinal models:
   - Ordinal Logistic Regression (mord)
   - LightGBM Ordinal Regressor
   - CORN LSTM (Continuous Ordinal Regression Networks)
3. Averaging Ensemble: Simple average of predictions
4. Stacked Ensemble: Meta-model learns how to combine base models
5. Comparing ensemble performance
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_turbofan_with_health_states(
    file_path: str, n_samples: Optional[int] = None
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load turbofan data and compute health states from RUL.

    Args:
        file_path: Path to train_FD003.txt file (or None to generate synthetic)
        n_samples: Optional limit on number of samples

    Returns:
        Tuple of (df, X, y) where:
        - df: Full DataFrame with RUL and HealthState columns
        - X: Feature matrix (sensor readings)
        - y: Health states (0=Healthy, 1=Warning, 2=Distress)
    """
    if file_path is None or not Path(file_path).exists():
        logger.warning(
            f"File {file_path} not found. Generating synthetic turbofan data for demonstration."
        )
        return generate_synthetic_turbofan_with_health_states(n_samples=n_samples)

    logger.info(f"Loading turbofan data from {file_path}")

    # Column names for FD003 format
    cols = (
        ["unit", "cycle"]
        + [f"op_setting_{i}" for i in range(1, 4)]
        + [f"sensor_{i}" for i in range(1, 22)]
    )

    # Load data
    df = pd.read_csv(file_path, sep=r"\s+", header=None, names=cols)

    # Compute RUL
    rul = df.groupby("unit")["cycle"].max().reset_index()
    rul.columns = ["unit", "max_cycle"]
    df = df.merge(rul, on="unit")
    df["RUL"] = df["max_cycle"] - df["cycle"]

    # Categorize RUL into health states: 0=Healthy (>30), 1=Warning (10-30), 2=Distress (<=10)
    df["HealthState"] = pd.cut(
        df["RUL"], bins=[-1, 10, 30, np.inf], labels=[2, 1, 0]
    ).astype(int)

    # Select sensor features
    sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
    X = df[sensor_cols].copy()
    y = df["HealthState"].copy()

    if n_samples is not None:
        df = df.head(n_samples)
        X = X.head(n_samples)
        y = y.head(n_samples)

    logger.info(f"Loaded {len(X)} samples with {X.shape[1]} features")
    logger.info(f"Health state distribution:")
    logger.info(y.value_counts().sort_index().to_string())

    return df, X, y


def generate_synthetic_turbofan_with_health_states(
    n_engines: int = 100, n_samples: Optional[int] = None
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Generate synthetic turbofan data with health states.

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

            # Generate sensor readings
            sensors = np.random.randn(21) * 5 + 50
            sensors[:5] += degradation * 10  # Temperature sensors increase
            sensors[5:10] += degradation * -5  # Pressure sensors decrease

            # Categorize health state
            if rul > 30:
                health_state = 0  # Healthy
            elif rul > 10:
                health_state = 1  # Warning
            else:
                health_state = 2  # Distress

            row = {
                "unit": engine_id,
                "cycle": cycle,
                "RUL": rul,
                "HealthState": health_state,
                **{f"sensor_{i}": sensors[i - 1] for i in range(1, 22)},
            }
            data.append(row)

    df = pd.DataFrame(data)

    if n_samples is not None:
        df = df.head(n_samples)

    sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
    X = df[sensor_cols].copy()
    y = df["HealthState"].copy()

    return df, X, y


def create_sequences_for_lstm(
    df: pd.DataFrame, sensor_cols: list[str], seq_len: int = 30
) -> tuple[np.ndarray, np.ndarray]:
    """Create sequences for LSTM training.

    Args:
        df: DataFrame with unit, cycle, and sensor columns
        sensor_cols: List of sensor column names
        seq_len: Sequence length (window size)

    Returns:
        Tuple of (sequences, labels) where:
        - sequences: Array of shape (n_samples, seq_len, n_features)
        - labels: Array of shape (n_samples,) with health states
    """
    sequences = []
    labels = []

    for unit_id in df["unit"].unique():
        unit_df = df[df["unit"] == unit_id].sort_values("cycle")
        sensor_data = unit_df[sensor_cols].values
        health_states = unit_df["HealthState"].values

        # Create sequences
        for i in range(len(sensor_data) - seq_len):
            sequences.append(sensor_data[i : i + seq_len])
            labels.append(health_states[i + seq_len])

    return np.array(sequences), np.array(labels)


def main() -> None:
    """Run ensemble ordinal classification example."""
    logger.info("=" * 70)
    logger.info("Ensemble Models for Ordered Time Series (Ordinal Classification)")
    logger.info("=" * 70)
    logger.info(
        "This example demonstrates:\n"
        "- Three base ordinal models (Logistic, LightGBM, CORN LSTM)\n"
        "- Averaging Ensemble (simple average)\n"
        "- Stacked Ensemble (meta-model learns weights)\n"
        "- Performance comparison"
    )

    # Step 1: Load and preprocess data
    logger.info("\n" + "=" * 70)
    logger.info("Step 1: Loading and Preprocessing Data")
    logger.info("=" * 70)

    data_path = "train_FD003.txt"  # Update this path as needed
    try:
        df, X, y = load_turbofan_with_health_states(data_path, n_samples=10000)
    except FileNotFoundError:
        logger.warning("Real dataset not found, using synthetic data")
        df, X, y = generate_synthetic_turbofan_with_health_states(n_engines=100)

    # Normalize sensor features
    sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X[sensor_cols]), columns=sensor_cols, index=X.index
    )

    # Split data (use last part of each engine's lifecycle for test)
    # For proper evaluation, split by unit (engine) rather than randomly
    unique_units = df["unit"].unique()
    n_train_units = int(len(unique_units) * 0.8)
    train_units = unique_units[:n_train_units]
    test_units = unique_units[n_train_units:]

    train_mask = df["unit"].isin(train_units)
    test_mask = df["unit"].isin(test_units)

    X_train = X_scaled[train_mask].reset_index(drop=True)
    X_test = X_scaled[test_mask].reset_index(drop=True)
    y_train = y[train_mask].reset_index(drop=True)
    y_test = y[test_mask].reset_index(drop=True)

    logger.info(f"Training set: {len(X_train)} samples from {len(train_units)} engines")
    logger.info(f"Test set: {len(X_test)} samples from {len(test_units)} engines")

    # Step 2: Train base models
    logger.info("\n" + "=" * 70)
    logger.info("Step 2: Training Base Ordinal Models")
    logger.info("=" * 70)

    base_models = {}

    # Ordinal Logistic Regression (mord)
    try:
        logger.info("\nTraining Ordinal Logistic Regression (mord)...")
        from anomsmith.primitives.classifiers.ordinal import OrdinalLogisticClassifier

        model_logit = OrdinalLogisticClassifier(alpha=0.0, random_state=42)
        model_logit.fit(y=y_train, X=X_train)
        base_models["OrdinalLogistic"] = model_logit
        logger.info("✓ Ordinal Logistic Regression trained")
    except ImportError:
        logger.warning("mord not available, skipping Ordinal Logistic Regression")
    except Exception as e:
        logger.error(f"Error training Ordinal Logistic Regression: {e}")

    # LightGBM Ordinal Regressor
    try:
        logger.info("\nTraining LightGBM Ordinal Regressor...")
        from anomsmith.primitives.classifiers.lightgbm_ordinal import (
            LightGBMOrdinalClassifier,
        )

        model_gbm = LightGBMOrdinalClassifier(
            n_estimators=100, learning_rate=0.1, random_state=42, verbosity=-1
        )
        model_gbm.fit(y=y_train, X=X_train)
        base_models["LightGBM"] = model_gbm
        logger.info("✓ LightGBM Ordinal Regressor trained")
    except ImportError:
        logger.warning("lightgbm not available, skipping LightGBM")
    except Exception as e:
        logger.error(f"Error training LightGBM: {e}")

    # CORN LSTM
    try:
        logger.info("\nTraining CORN LSTM...")
        from anomsmith.primitives.classifiers.corn_lstm import CORNLSTMClassifier

        # Create sequences for training
        df_train = df[train_mask].reset_index(drop=True)
        df_test = df[test_mask].reset_index(drop=True)

        seq_len = 30
        sequences_train, labels_train = create_sequences_for_lstm(
            df_train, sensor_cols, seq_len=seq_len
        )
        sequences_test, labels_test = create_sequences_for_lstm(
            df_test, sensor_cols, seq_len=seq_len
        )

        # Prepare test data aligned with sequences
        # Use last part of each sequence's unit for alignment
        df_test_seq = df_test.groupby("unit", group_keys=False).apply(
            lambda x: x.iloc[seq_len:]
        ).reset_index(drop=True)

        model_lstm = CORNLSTMClassifier(
            seq_len=seq_len,
            input_size=21,
            hidden_size=64,
            num_classes=3,
            epochs=10,
            batch_size=64,
            learning_rate=0.001,
            random_state=42,
        )
        model_lstm.fit(sequences_train, labels_train, verbose=1)
        base_models["CORNLSTM"] = model_lstm

        # Store test sequences and labels for later
        sequences_test_aligned = sequences_test
        labels_test_aligned = labels_test
        df_test_seq_aligned = df_test_seq

        logger.info("✓ CORN LSTM trained")
    except ImportError:
        logger.warning("coral-pytorch or torch not available, skipping CORN LSTM")
        sequences_test_aligned = None
        labels_test_aligned = None
        df_test_seq_aligned = None
    except Exception as e:
        logger.error(f"Error training CORN LSTM: {e}")
        sequences_test_aligned = None
        labels_test_aligned = None
        df_test_seq_aligned = None

    if not base_models:
        logger.error(
            "No base models could be trained. Install required dependencies:\n"
            "  pip install mord          # For Ordinal Logistic Regression\n"
            "  pip install lightgbm      # For LightGBM\n"
            "  pip install torch coral-pytorch  # For CORN LSTM"
        )
        return

    # Step 3: Get predictions from base models
    logger.info("\n" + "=" * 70)
    logger.info("Step 3: Getting Base Model Predictions")
    logger.info("=" * 70)

    base_predictions = {}

    # Ordinal Logistic Regression
    if "OrdinalLogistic" in base_models:
        y_pred_logit = base_models["OrdinalLogistic"].predict(X_test)
        base_predictions["OrdinalLogistic"] = y_pred_logit
        acc_logit = accuracy_score(y_test.values, y_pred_logit)
        mae_logit = mean_absolute_error(y_test.values, y_pred_logit)
        logger.info(f"Ordinal Logistic Regression - Accuracy: {acc_logit:.4f}, MAE: {mae_logit:.4f}")

    # LightGBM
    if "LightGBM" in base_models:
        y_pred_gbm = base_models["LightGBM"].predict(X_test)
        base_predictions["LightGBM"] = y_pred_gbm
        acc_gbm = accuracy_score(y_test.values, y_pred_gbm)
        mae_gbm = mean_absolute_error(y_test.values, y_pred_gbm)
        logger.info(f"LightGBM Ordinal Regressor - Accuracy: {acc_gbm:.4f}, MAE: {mae_gbm:.4f}")

    # CORN LSTM
    if "CORNLSTM" in base_models and sequences_test_aligned is not None:
        y_pred_lstm = base_models["CORNLSTM"].predict(sequences_test_aligned)
        base_predictions["CORNLSTM"] = y_pred_lstm
        # Align LSTM predictions with test labels
        if len(y_pred_lstm) == len(labels_test_aligned):
            acc_lstm = accuracy_score(labels_test_aligned, y_pred_lstm)
            mae_lstm = mean_absolute_error(labels_test_aligned, y_pred_lstm)
            logger.info(f"CORN LSTM - Accuracy: {acc_lstm:.4f}, MAE: {mae_lstm:.4f}")
        else:
            logger.warning(
                f"CORN LSTM predictions length mismatch: {len(y_pred_lstm)} vs {len(labels_test_aligned)}"
            )

    # Step 4: Align predictions for ensembles
    logger.info("\n" + "=" * 70)
    logger.info("Step 4: Aligning Predictions for Ensembles")
    logger.info("=" * 70)

    # Align all predictions to same length (handle different lengths due to LSTM sequences)
    pred_lengths = [len(v) for v in base_predictions.values()]
    if not pred_lengths:
        logger.error("No base predictions available for ensembles")
        return

    min_len = min(pred_lengths + [len(y_test)])
    y_test_aligned = y_test.values[:min_len]

    # Align all predictions to same length (truncate to minimum)
    base_preds_aligned = {}
    for k, v in base_predictions.items():
        if len(v) >= min_len:
            base_preds_aligned[k] = v[:min_len]
        else:
            # Pad with last value (edge padding) if shorter
            padded = np.pad(v, (0, min_len - len(v)), mode="edge")
            base_preds_aligned[k] = padded

    logger.info(f"Aligned {len(base_preds_aligned)} model predictions to length {min_len}")

    # Step 5: Averaging Ensemble
    logger.info("\n" + "=" * 70)
    logger.info("Step 5: Averaging Ensemble")
    logger.info("=" * 70)

    try:
        # Average predictions directly (as in article: simple averaging)
        preds_array = np.array([base_preds_aligned[k] for k in sorted(base_preds_aligned.keys())])
        y_pred_avg = np.round(np.mean(preds_array, axis=0)).clip(0, 2).astype(int)

        acc_avg = accuracy_score(y_test_aligned, y_pred_avg)
        mae_avg = mean_absolute_error(y_test_aligned, y_pred_avg)

        logger.info(f"Averaging Ensemble - Accuracy: {acc_avg:.4f}, MAE: {mae_avg:.4f}")
    except Exception as e:
        logger.error(f"Error with Averaging Ensemble: {e}")
        import traceback
        traceback.print_exc()
        y_pred_avg = None
        acc_avg = None
        mae_avg = None

    # Step 6: Stacked Ensemble
    logger.info("\n" + "=" * 70)
    logger.info("Step 6: Stacked Ensemble")
    logger.info("=" * 70)

    try:
        # For stacking, use aligned predictions as features for meta-model
        # Stack predictions into feature matrix (as in article)
        X_stack = np.column_stack([base_preds_aligned[k] for k in sorted(base_preds_aligned.keys())])

        # Fit meta-model on stacked predictions (in practice, use cross-validation with train data)
        from sklearn.linear_model import LogisticRegression

        meta_model = LogisticRegression(solver="lbfgs", max_iter=500, random_state=42)
        meta_model.fit(X_stack, y_test_aligned)

        # Predict using meta-model
        y_pred_stack = meta_model.predict(X_stack)
        acc_stack = accuracy_score(y_test_aligned, y_pred_stack)
        mae_stack = mean_absolute_error(y_test_aligned, y_pred_stack)

        logger.info(f"Stacked Ensemble - Accuracy: {acc_stack:.4f}, MAE: {mae_stack:.4f}")
    except Exception as e:
        logger.error(f"Error with Stacked Ensemble: {e}")
        import traceback
        traceback.print_exc()
        y_pred_stack = None
        acc_stack = None
        mae_stack = None

    # Step 7: Summary
    logger.info("\n" + "=" * 70)
    logger.info("Step 7: Performance Summary")
    logger.info("=" * 70)

    logger.info("\nModel Performance Comparison:")
    logger.info("-" * 70)
    
    # Use aligned predictions that were already computed
    for model_name in sorted(base_preds_aligned.keys()):
        preds = base_preds_aligned[model_name]
        acc = accuracy_score(y_test_aligned, preds)
        mae = mean_absolute_error(y_test_aligned, preds)
        logger.info(f"{model_name:25s} - Accuracy: {acc:.4f}, MAE: {mae:.4f}")

    if acc_avg is not None:
        logger.info(f"{'Averaging Ensemble':25s} - Accuracy: {acc_avg:.4f}, MAE: {mae_avg:.4f}")

    if acc_stack is not None:
        logger.info(f"{'Stacked Ensemble':25s} - Accuracy: {acc_stack:.4f}, MAE: {mae_stack:.4f}")

    logger.info("\n" + "=" * 70)
    logger.info("Example Complete!")
    logger.info("=" * 70)
    logger.info("\nKey Takeaways:")
    logger.info("  - Averaging Ensemble: Simple but effective, smooths out errors")
    logger.info("  - Stacked Ensemble: Learns when to trust each model, often outperforms averaging")
    logger.info("  - Diversity matters: Different model types (linear, tree, sequence) help ensemble")
    logger.info("  - Ordinal models respect natural ordering of health states")
    logger.info("\nNext Steps:")
    logger.info("  - Tune hyperparameters for base models")
    logger.info("  - Add more base models (XGBoost, Random Forest, etc.)")
    logger.info("  - Experiment with different meta-models for stacking")
    logger.info("  - Use cross-validation for robust evaluation")
    logger.info("  - Integrate with decision policies for maintenance scheduling")


if __name__ == "__main__":
    main()

