"""Evaluation metrics for survival analysis models.

Provides metrics for assessing survival model performance, including
concordance index (C-index) for ranking ability.
"""

import logging
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import pandas as pd

try:
    from lifelines.utils import concordance_index as lifelines_c_index

    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False
    lifelines_c_index = None  # type: ignore

try:
    from pycox.evaluation import EvalSurv

    PYCOX_AVAILABLE = True
except ImportError:
    PYCOX_AVAILABLE = False
    EvalSurv = None  # type: ignore

if TYPE_CHECKING:
    try:
        from timesmith.typing import SeriesLike
    except ImportError:
        SeriesLike = None

logger = logging.getLogger(__name__)


def compute_concordance_index(
    durations: Union[np.ndarray, pd.Series],
    risk_scores: Union[np.ndarray, pd.Series],
    events: Union[np.ndarray, pd.Series, None] = None,
) -> float:
    """Compute concordance index (C-index) for survival model evaluation.

    C-index measures how well a model ranks survival times. A score of 0.5
    implies random ordering; 1.0 implies perfect prediction.

    Uses lifelines if available, otherwise computes manually.

    Args:
        durations: Actual time-to-event values (n_samples,)
        risk_scores: Predicted risk scores (n_samples,) - higher = higher risk
        events: Event indicators (1 = event occurred, 0 = censored), optional

    Returns:
        C-index between 0.0 and 1.0

    Examples:
        >>> c_index = compute_concordance_index(true_durations, risk_scores, events)
        >>> print(f"C-index: {c_index:.3f}")
    """
    durations_array = np.asarray(durations)
    risk_array = np.asarray(risk_scores)

    if events is None:
        events_array = np.ones(len(durations_array))  # All events observed
    else:
        events_array = np.asarray(events)

    # Use lifelines if available (more robust)
    if LIFELINES_AVAILABLE:
        c_index = lifelines_c_index(durations_array, -risk_array, events_array)  # type: ignore
        return float(c_index)

    # Manual computation (simplified version)
    # C-index = proportion of concordant pairs
    n_samples = len(durations_array)
    if n_samples < 2:
        return 0.5

    concordant = 0
    comparable = 0

    for i in range(n_samples):
        if events_array[i] == 0:  # Skip censored observations
            continue

        for j in range(i + 1, n_samples):
            # Two observations are comparable if:
            # - Both have events, or
            # - One has event and the other's duration > event duration
            if events_array[j] == 1:
                # Both have events - always comparable
                comparable += 1
                if (durations_array[i] < durations_array[j] and risk_array[i] > risk_array[j]) or (
                    durations_array[i] > durations_array[j] and risk_array[i] < risk_array[j]
                ):
                    concordant += 1
            elif durations_array[j] > durations_array[i]:
                # j is censored but after i's event - comparable
                comparable += 1
                if risk_array[i] > risk_array[j]:
                    concordant += 1

    if comparable == 0:
        return 0.5

    c_index = concordant / comparable
    return float(c_index)


def evaluate_survival_model(
    surv_df: pd.DataFrame,
    durations: Union[np.ndarray, pd.Series],
    events: Union[np.ndarray, pd.Series, None] = None,
    risk_scores: Optional[Union[np.ndarray, pd.Series]] = None,
) -> dict[str, float]:
    """Evaluate survival model performance.

    Computes comprehensive metrics for survival model evaluation.

    Args:
        surv_df: Survival function DataFrame (rows = time points, cols = samples)
        durations: Actual time-to-event values (n_samples,)
        events: Event indicators (1 = event occurred, 0 = censored), optional
        risk_scores: Optional risk scores for C-index (if None, computed from survival)

    Returns:
        Dictionary with evaluation metrics:
        - c_index: Concordance index
        - mean_absolute_error: Mean absolute error in predicted vs actual durations
        - median_survival_error: Error in median survival predictions

    Examples:
        >>> surv_df = model.predict_survival_function(X_test)
        >>> metrics = evaluate_survival_model(surv_df, durations_test, events_test)
        >>> print(f"C-index: {metrics['c_index']:.3f}")
    """
    durations_array = np.asarray(durations)
    events_array = np.asarray(events) if events is not None else np.ones(len(durations))

    # Compute risk scores from survival if not provided
    if risk_scores is None:
        # Use negative median survival time as risk score
        median_survival = surv_df.apply(
            lambda col: col.index[col <= 0.5][0] if (col <= 0.5).any() else col.index[-1]
        )
        risk_scores_array = -median_survival.values
    else:
        risk_scores_array = np.asarray(risk_scores)

    # Compute C-index
    c_index = compute_concordance_index(durations_array, risk_scores_array, events_array)

    # Compute MAE (using median survival as prediction)
    median_survival = surv_df.apply(
        lambda col: col.index[col <= 0.5][0] if (col <= 0.5).any() else col.index[-1]
    )
    mae = np.mean(np.abs(median_survival.values - durations_array))

    # Use pycox evaluation if available (more comprehensive)
    if PYCOX_AVAILABLE:
        try:
            eval_surv = EvalSurv(  # type: ignore
                surv_df,
                durations_array,
                events_array,
                censor_surv="km",  # Kaplan-Meier for censoring
            )
            c_index_td = eval_surv.concordance_td()  # Time-dependent C-index
            return {
                "c_index": float(c_index),
                "c_index_td": float(c_index_td),  # Time-dependent version
                "mean_absolute_error": float(mae),
                "median_survival_error": float(mae),
            }
        except Exception as e:
            logger.warning(f"Error using pycox evaluation: {e}, using manual metrics")

    return {
        "c_index": float(c_index),
        "mean_absolute_error": float(mae),
        "median_survival_error": float(mae),
    }

