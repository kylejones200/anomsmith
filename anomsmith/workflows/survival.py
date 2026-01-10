"""Survival analysis workflows for predictive maintenance.

Integrates survival models with anomsmith's health state and decision
policy framework for comprehensive predictive maintenance.
"""

import logging
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import pandas as pd

from anomsmith.objects.health_state import HealthState, HealthStateView
from anomsmith.primitives.survival.cox import CoxSurvivalModel
from anomsmith.workflows.eval.survival_metrics import (
    compute_concordance_index,
    evaluate_survival_model,
)
from anomsmith.workflows.pm import discretize_rul

if TYPE_CHECKING:
    try:
        from timesmith.typing import SeriesLike
    except ImportError:
        SeriesLike = None

logger = logging.getLogger(__name__)


def predict_rul_from_survival(
    model: CoxSurvivalModel,
    X: Union[np.ndarray, pd.DataFrame],
    threshold: float = 0.5,
) -> pd.Series:
    """Predict Remaining Useful Life (RUL) from survival model.

    Uses median survival time (where survival probability = threshold)
    as predicted RUL.

    Args:
        model: Fitted survival model
        X: Feature matrix (n_samples, n_features)
        threshold: Survival probability threshold for median (default 0.5)

    Returns:
        Series of predicted RUL values

    Examples:
        >>> rul_predictions = predict_rul_from_survival(survival_model, X_test)
        >>> health_states = predict_health_states_from_survival(
        ...     survival_model, X_test, healthy_threshold=30, warning_threshold=10
        ... )
    """
    logger.info(f"Predicting RUL from survival model (threshold={threshold})")
    rul_array = model.predict_time_to_failure(X, threshold=threshold)
    return pd.Series(rul_array, name="rul")


def predict_health_states_from_survival(
    model: CoxSurvivalModel,
    X: Union[np.ndarray, pd.DataFrame],
    healthy_threshold: float = 30.0,
    warning_threshold: float = 10.0,
    threshold: float = 0.5,
) -> HealthStateView:
    """Predict health states from survival model.

    Converts survival model predictions to health states by:
    1. Predicting RUL from survival model
    2. Discretizing RUL into health states

    Args:
        model: Fitted survival model
        X: Feature matrix (n_samples, n_features)
        healthy_threshold: RUL threshold for Healthy state (default 30)
        warning_threshold: RUL threshold for Warning state (default 10)
        threshold: Survival probability threshold for median RUL (default 0.5)

    Returns:
        HealthStateView with predicted health states

    Examples:
        >>> health_states = predict_health_states_from_survival(
        ...     model, X_test, healthy_threshold=30, warning_threshold=10
        ... )
    """
    # Determine index for RUL series
    if isinstance(X, pd.DataFrame):
        index = X.index
    else:
        index = pd.RangeIndex(start=0, stop=len(X))

    # Predict RUL
    rul_series = predict_rul_from_survival(model, X, threshold=threshold, index=index)

    # Discretize to health states using the primitive directly
    from anomsmith.primitives.health_state.discretize import discretize_rul_to_health_states

    health_states = discretize_rul_to_health_states(
        rul_series, healthy_threshold=healthy_threshold, warning_threshold=warning_threshold
    )

    logger.info(f"Predicted health states from survival model: {len(health_states.states)} samples")
    return health_states


def fit_survival_model_for_maintenance(
    X: Union[np.ndarray, pd.DataFrame],
    durations: Union[np.ndarray, pd.Series],
    events: Union[np.ndarray, pd.Series, None] = None,
    model_type: str = "logistic_hazard",
    **model_kwargs,
) -> CoxSurvivalModel:
    """Fit a survival model for predictive maintenance.

    Convenience function that fits a survival model with sensible defaults
    for predictive maintenance use cases.

    Args:
        X: Feature matrix (n_samples, n_features) - sensor readings
        durations: Time-to-failure values (n_samples,)
        events: Event indicators (1 = failure, 0 = censored), optional
        model_type: Model type - 'cox' (lifelines), 'logistic_hazard', or 'deepsurv'
        **model_kwargs: Additional model parameters

    Returns:
        Fitted survival model

    Examples:
        >>> model = fit_survival_model_for_maintenance(
        ...     X_train, durations_train, events_train,
        ...     model_type="logistic_hazard", n_bins=50
        ... )
    """
    logger.info(f"Fitting {model_type} survival model for predictive maintenance")

    if model_type == "cox" or model_type == "lifelines":
        try:
            from anomsmith.primitives.survival.lifelines_cox import LifelinesCoxModel

            model = LifelinesCoxModel(**model_kwargs)
        except ImportError:
            raise ImportError(
                "lifelines is required for Cox model. Install with: pip install lifelines"
            )
    elif model_type == "logistic_hazard":
        try:
            from anomsmith.primitives.survival.neural import LogisticHazardModel

            model = LogisticHazardModel(**model_kwargs)
        except ImportError:
            raise ImportError(
                "pycox is required for LogisticHazard. Install with: pip install pycox"
            )
    elif model_type == "deepsurv":
        try:
            from anomsmith.primitives.survival.neural import DeepSurvModel

            model = DeepSurvModel(**model_kwargs)
        except ImportError:
            raise ImportError("pycox is required for DeepSurv. Install with: pip install pycox")
    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            "Choose from: 'cox', 'logistic_hazard', 'deepsurv'"
        )

    model.fit(X, durations, events)

    logger.info(f"Fitted {model_type} model: {len(X)} samples, {X.shape[1]} features")
    return model


def compare_survival_models(
    models: dict[str, CoxSurvivalModel],
    X_test: Union[np.ndarray, pd.DataFrame],
    durations_test: Union[np.ndarray, pd.Series],
    events_test: Union[np.ndarray, pd.Series, None] = None,
) -> pd.DataFrame:
    """Compare multiple survival models.

    Evaluates multiple survival models and returns comparison metrics.

    Args:
        models: Dictionary mapping model names to fitted CoxSurvivalModel instances
        X_test: Test feature matrix
        durations_test: Test time-to-event values
        events_test: Test event indicators, optional

    Returns:
        DataFrame with comparison metrics (C-index, MAE, etc.) for each model

    Examples:
        >>> models = {
        ...     "CoxPH": cox_model,
        ...     "LogisticHazard": lhaz_model,
        ...     "DeepSurv": deepsurv_model
        ... }
        >>> comparison = compare_survival_models(models, X_test, durations_test, events_test)
        >>> print(comparison)
    """
    logger.info(f"Comparing {len(models)} survival models")

    results = []

    for model_name, model in models.items():
        try:
            # Predict survival function
            surv_df = model.predict_survival_function(X_test)
            risk_scores = model.predict_risk_score(X_test)

            # Evaluate
            metrics = evaluate_survival_model(
                surv_df, durations_test, events_test, risk_scores=risk_scores
            )
            metrics["model"] = model_name
            results.append(metrics)

            logger.info(f"{model_name}: C-index = {metrics['c_index']:.3f}")

        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
            results.append({"model": model_name, "c_index": np.nan, "error": str(e)})

    return pd.DataFrame(results)

