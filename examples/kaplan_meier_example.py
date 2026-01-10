"""Kaplan-Meier and parametric survival models for predictive maintenance.

Demonstrates Kaplan-Meier non-parametric survival estimation and
parametric survival models (Weibull, Exponential, LogNormal, etc.)
using NASA C-MAPSS turbofan dataset.

Based on "Survival Analysis for predictive maintenance with data from NASA Turbofan example in Python".

This example shows:
1. Loading and preprocessing turbofan data
2. Computing Remaining Useful Life (RUL) and categorization (Healthy, Warning, Critical)
3. Fitting Kaplan-Meier survival estimator
4. Calculating survival probabilities at specific time points (e.g., 99% survival time)
5. Fitting multiple parametric survival models (Weibull, Exponential, etc.)
6. Comparing different survival models
7. Predicting survival probabilities at specific time points (e.g., 275 periods)
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from anomsmith.primitives.survival.kaplan_meier import KaplanMeierModel
from anomsmith.primitives.survival.parametric import ParametricSurvivalModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_turbofan_data_with_rul(
    file_path: str, n_samples: Optional[int] = None
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load turbofan data and compute RUL and categorization.

    Args:
        file_path: Path to data file (or None to generate synthetic)
        n_samples: Optional limit on number of samples

    Returns:
        Tuple of (df, max_rul, categories) where:
        - df: DataFrame with all data including Category (Healthy, Warning, Critical)
        - max_rul: Series of maximum RUL for each engine
        - categories: Series of categories for each engine
    """
    if file_path is None or not Path(file_path).exists():
        logger.warning(
            f"File {file_path} not found. Generating synthetic turbofan data for demonstration."
        )
        return generate_synthetic_turbofan_with_categories(n_samples=n_samples)

    logger.info(f"Loading turbofan data from {file_path}")

    # Load data
    df = pd.read_csv(file_path)

    # Check if Category column exists (from previous processing)
    if "Category" not in df.columns:
        # Compute RUL and categorize
        if "max" in df.columns:
            # RUL already computed
            max_rul = df.groupby(["source", "id"])["max"].max().reset_index()
            max_rul.columns = ["source", "id", "max_rul"]
        else:
            # Compute RUL
            rul = df.groupby(["source", "id"])["cycle"].max().reset_index()
            rul.columns = ["source", "id", "max_cycle"]
            df = df.merge(rul, on=["source", "id"])
            df["max"] = df["max_cycle"] - df["cycle"]  # RUL
            max_rul = df.groupby(["source", "id"])["max"].max().reset_index()
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

    # Get max RUL for each engine (for survival analysis)
    max_rul_series = df.groupby(["source", "id"])["max"].max() if "max" in df.columns else max_rul["max_rul"]
    categories_series = max_rul["Category"] if "Category" in max_rul.columns else df["Category"].unique()

    if n_samples is not None:
        max_rul_series = max_rul_series.head(n_samples)

    logger.info(f"Loaded {len(max_rul_series)} engines")
    logger.info(f"Categories: {pd.Series(categories_series).value_counts().to_dict()}")

    return df, max_rul_series, categories_series


def generate_synthetic_turbofan_with_categories(
    n_engines: int = 100, n_samples: int = 5000
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Generate synthetic turbofan data with RUL categorization.

    Simulates sensor readings and failure patterns similar to NASA C-MAPSS dataset,
    including categorization into Healthy, Warning, and Critical states.
    """
    np.random.seed(42)
    logger.info("Generating synthetic turbofan data with categories...")

    # Generate engine data
    engine_ids = np.arange(1, n_engines + 1)
    max_cycles = np.random.randint(150, 300, size=n_engines)

    # Create data with categories
    data = []
    max_rul_dict = {}
    categories_dict = {}

    for engine_id, max_cycle in zip(engine_ids, max_cycles):
        # Generate sensor data for this engine
        n_cycles = np.random.randint(50, max_cycle + 1)
        for cycle in range(1, n_cycles + 1):
            rul = max_cycle - cycle
            degradation = 1.0 - (rul / max_cycle)

            # Generate sensor readings
            sensors = np.random.randn(21) * 5 + 50
            sensors[:5] += degradation * 10
            sensors[5:10] += degradation * -5

            data.append({
                "source": "FD003",
                "id": engine_id,
                "cycle": cycle,
                "max": rul,
                **{f"sensor_{i}": sensors[i-1] for i in range(1, 22)}
            })

        # Categorize: Healthy (>30), Warning (10-30), Critical (<=10)
        if rul > 30:
            category = "Healthy"
        elif rul > 10:
            category = "Warning"
        else:
            category = "Critical"

        max_rul_dict[engine_id] = rul
        categories_dict[engine_id] = category

    df = pd.DataFrame(data)
    df["Category"] = df["id"].map(categories_dict)

    max_rul = pd.Series(max_rul_dict, name="max_rul")
    categories = pd.Series(categories_dict, name="Category")

    return df, max_rul, categories


def main() -> None:
    """Run Kaplan-Meier and parametric survival models example."""
    logger.info("=" * 70)
    logger.info("Kaplan-Meier and Parametric Survival Models for Predictive Maintenance")
    logger.info("=" * 70)
    logger.info(
        "This example demonstrates:\n"
        "- Kaplan-Meier non-parametric survival estimation\n"
        "- Parametric survival models (Weibull, Exponential, LogNormal, etc.)\n"
        "- Survival probability predictions at specific time points\n"
        "- Qth survival time (e.g., 99% survival time)"
    )

    # Step 1: Load and preprocess data
    logger.info("\n" + "=" * 70)
    logger.info("Step 1: Loading and Preprocessing Data")
    logger.info("=" * 70)

    # Try to load real data, fall back to synthetic
    data_path = "data/predictive maintenance trainALL.csv"  # Update this path as needed
    try:
        df, max_rul, categories = load_turbofan_data_with_rul(data_path, n_samples=100)
    except FileNotFoundError:
        logger.warning("Real dataset not found, using synthetic data")
        df, max_rul, categories = generate_synthetic_turbofan_with_categories(n_engines=100)

    # Check categories
    logger.info(f"\nCategories in data:")
    logger.info(pd.Series(categories).value_counts().to_string())

    # Prepare data for survival analysis
    # Each engine's max RUL is the "duration" (time to failure)
    # All engines are "observed" (they all fail)
    durations = max_rul.values
    events = np.ones(len(max_rul))  # All engines fail (observed=1)

    logger.info(f"\nMax RUL statistics:")
    logger.info(f"  Median: {np.median(durations):.1f} periods")
    logger.info(f"  Mean: {np.mean(durations):.1f} periods")
    logger.info(f"  Min: {np.min(durations):.1f} periods")
    logger.info(f"  Max: {np.max(durations):.1f} periods")

    # Step 2: Fit Kaplan-Meier model
    logger.info("\n" + "=" * 70)
    logger.info("Step 2: Fitting Kaplan-Meier Non-Parametric Model")
    logger.info("=" * 70)

    kmf = KaplanMeierModel(alpha=0.05)
    kmf.fit(X=None, durations=durations, events=events)

    logger.info("✓ Kaplan-Meier model fitted")

    # Step 3: Calculate survival probabilities and qth survival time
    logger.info("\n" + "=" * 70)
    logger.info("Step 3: Survival Probabilities and Qth Survival Time")
    logger.info("=" * 70)

    # Find 99% survival time (q=0.99)
    q99_time = kmf.qth_survival_time(0.99)
    logger.info(f"\n99% survival time: {q99_time:.1f} periods")
    logger.info(f"This means 99% of machines are still working after {q99_time:.1f} periods.")

    # Predict survival at specific time points
    time_points = [100, 200, 275, 300]
    logger.info(f"\nSurvival probabilities at specific time points:")
    for t in time_points:
        prob = kmf.predict(t)
        logger.info(f"  At {t} periods: {prob:.3f} ({prob*100:.1f}% survival probability)")

    # Step 4: Fit parametric survival models
    logger.info("\n" + "=" * 70)
    logger.info("Step 4: Fitting Parametric Survival Models")
    logger.info("=" * 70)

    parametric_models = {}

    # Weibull (bathtub curve - high failure at start and end)
    try:
        logger.info("\nFitting Weibull model...")
        weibull = ParametricSurvivalModel(model_type="weibull", alpha=0.05)
        weibull.fit(X=None, durations=durations, events=events)
        parametric_models["Weibull"] = weibull
        logger.info("✓ Weibull model fitted")
    except Exception as e:
        logger.error(f"Error fitting Weibull: {e}")

    # Exponential (constant failure rate)
    try:
        logger.info("\nFitting Exponential model...")
        exponential = ParametricSurvivalModel(model_type="exponential", alpha=0.05)
        exponential.fit(X=None, durations=durations, events=events)
        parametric_models["Exponential"] = exponential
        logger.info("✓ Exponential model fitted")
    except Exception as e:
        logger.error(f"Error fitting Exponential: {e}")

    # LogNormal (early failures)
    try:
        logger.info("\nFitting LogNormal model...")
        lognormal = ParametricSurvivalModel(model_type="lognormal", alpha=0.05)
        lognormal.fit(X=None, durations=durations, events=events)
        parametric_models["LogNormal"] = lognormal
        logger.info("✓ LogNormal model fitted")
    except Exception as e:
        logger.error(f"Error fitting LogNormal: {e}")

    # LogLogistic (S-shaped hazard)
    try:
        logger.info("\nFitting LogLogistic model...")
        loglogistic = ParametricSurvivalModel(model_type="loglogistic", alpha=0.05)
        loglogistic.fit(X=None, durations=durations, events=events)
        parametric_models["LogLogistic"] = loglogistic
        logger.info("✓ LogLogistic model fitted")
    except Exception as e:
        logger.error(f"Error fitting LogLogistic: {e}")

    # Piecewise Exponential
    try:
        logger.info("\nFitting Piecewise Exponential model...")
        piecewise = ParametricSurvivalModel(
            model_type="piecewise_exponential", breakpoints=[40, 60], alpha=0.05
        )
        piecewise.fit(X=None, durations=durations, events=events)
        parametric_models["PiecewiseExponential"] = piecewise
        logger.info("✓ Piecewise Exponential model fitted")
    except Exception as e:
        logger.error(f"Error fitting Piecewise Exponential: {e}")

    # Generalized Gamma
    try:
        logger.info("\nFitting Generalized Gamma model...")
        gamma = ParametricSurvivalModel(model_type="generalized_gamma", alpha=0.05)
        gamma.fit(X=None, durations=durations, events=events)
        parametric_models["GeneralizedGamma"] = gamma
        logger.info("✓ Generalized Gamma model fitted")
    except Exception as e:
        logger.error(f"Error fitting Generalized Gamma: {e}")

    # Spline
    try:
        logger.info("\nFitting Spline model...")
        spline = ParametricSurvivalModel(
            model_type="spline", breakpoints=[6, 20, 40, 75], alpha=0.05
        )
        spline.fit(X=None, durations=durations, events=events)
        parametric_models["Spline"] = spline
        logger.info("✓ Spline model fitted")
    except Exception as e:
        logger.error(f"Error fitting Spline: {e}")

    if not parametric_models:
        logger.error(
            "No parametric models could be fitted. Install required dependencies:\n"
            "  pip install lifelines"
        )
        return

    # Step 5: Compare models - predict survival at 275 periods
    logger.info("\n" + "=" * 70)
    logger.info("Step 5: Comparing Models - Survival at 275 Periods")
    logger.info("=" * 70)

    target_time = 275
    logger.info(f"\nSurvival probability at {target_time} periods:")

    # Kaplan-Meier
    km_prob = kmf.predict(target_time)
    logger.info(f"  Kaplan-Meier: {km_prob:.3f} ({km_prob*100:.1f}%)")

    # Parametric models
    for model_name, model in parametric_models.items():
        try:
            prob = model.predict(target_time)
            logger.info(f"  {model_name}: {prob:.3f} ({prob*100:.1f}%)")
        except Exception as e:
            logger.warning(f"  {model_name}: Error predicting - {e}")

    # Step 6: Compare median survival times
    logger.info("\n" + "=" * 70)
    logger.info("Step 6: Comparing Median Survival Times")
    logger.info("=" * 70)

    logger.info(f"\nMedian survival times:")

    # Kaplan-Meier
    km_median = kmf.model_.median_survival_time_  # type: ignore
    logger.info(f"  Kaplan-Meier: {km_median:.1f} periods")

    # Parametric models
    for model_name, model in parametric_models.items():
        try:
            median = model.model_.median_survival_time_  # type: ignore
            logger.info(f"  {model_name}: {median:.1f} periods")
        except Exception as e:
            logger.warning(f"  {model_name}: Error getting median - {e}")

    logger.info("\n" + "=" * 70)
    logger.info("Example Complete!")
    logger.info("=" * 70)
    logger.info("\nKey Takeaways:")
    logger.info("  - Kaplan-Meier provides non-parametric baseline survival estimates")
    logger.info("  - Parametric models assume specific distributions (e.g., Weibull = bathtub curve)")
    logger.info("  - Different models capture different failure patterns")
    logger.info("  - Qth survival time (e.g., 99%) helps with maintenance planning")
    logger.info("\nNext Steps:")
    logger.info("  - Compare models using AIC/BIC for model selection")
    logger.info("  - Use Cox PH regression to incorporate features (sensor readings)")
    logger.info("  - Integrate with decision policies for maintenance scheduling")
    logger.info("  - Add visualization of survival curves for model comparison")


if __name__ == "__main__":
    main()

