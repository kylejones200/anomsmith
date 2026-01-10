"""Kaplan-Meier survival estimator.

Non-parametric survival estimator that estimates survival probability
over time without assuming any particular distribution.
"""

import logging
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import pandas as pd

try:
    from lifelines import KaplanMeierFitter
    from lifelines.utils import qth_survival_time

    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False
    KaplanMeierFitter = None  # type: ignore
    qth_survival_time = None  # type: ignore

from anomsmith.primitives.survival.cox import CoxSurvivalModel

if TYPE_CHECKING:
    try:
        from timesmith.typing import SeriesLike
    except ImportError:
        SeriesLike = None

logger = logging.getLogger(__name__)


class KaplanMeierModel(CoxSurvivalModel):
    """Kaplan-Meier non-parametric survival estimator.

    Estimates survival probability over time without assuming any
    particular distribution. Good baseline model for survival analysis.

    Args:
        alpha: Alpha for confidence intervals (default 0.05)
        random_state: Random state (not used, kept for compatibility)
    """

    def __init__(
        self,
        alpha: float = 0.05,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize Kaplan-Meier model."""
        if not LIFELINES_AVAILABLE:
            raise ImportError(
                "lifelines is required for KaplanMeierModel. "
                "Install with: pip install lifelines"
            )

        super().__init__(random_state=random_state)
        self.alpha = alpha
        self.model_: Optional[KaplanMeierFitter] = None  # type: ignore
        self.durations_: Optional[np.ndarray] = None
        self.events_: Optional[np.ndarray] = None

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame, None],
        durations: Union[np.ndarray, pd.Series],
        events: Union[np.ndarray, pd.Series, None] = None,
        y: Union[np.ndarray, pd.Series, "SeriesLike", None] = None,
    ) -> "KaplanMeierModel":
        """Fit the Kaplan-Meier model.

        Note: X (features) is not used in Kaplan-Meier as it's non-parametric.
        Only durations and events are used. X can be None or any value.

        Args:
            X: Feature matrix (ignored for Kaplan-Meier, kept for interface compatibility). Can be None.
            durations: Time-to-event values (n_samples,)
            events: Event indicators (1 = event occurred, 0 = censored)
            y: Optional target (not used, kept for interface compatibility)

        Returns:
            Self for method chaining
        """
        durations_array = np.asarray(durations)
        events_array = np.asarray(events) if events is not None else np.ones(len(durations))

        # Store for later use
        self.durations_ = durations_array
        self.events_ = events_array

        # Fit model
        self.model_ = KaplanMeierFitter(alpha=self.alpha)  # type: ignore
        self.model_.fit(durations_array, events_array, label="Kaplan-Meier Estimate")  # type: ignore

        self._fitted = True
        logger.debug(
            f"Fitted KaplanMeierModel: n_samples={len(durations_array)}, "
            f"median_survival={self.model_.median_survival_time_:.2f}"  # type: ignore
        )
        return self

    def predict_survival_function(
        self, X: Union[np.ndarray, pd.DataFrame, None] = None, time_points: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """Predict survival function S(t).

        Note: X is ignored as Kaplan-Meier doesn't use features.

        Args:
            X: Feature matrix (ignored, kept for interface compatibility)
            time_points: Optional time points for prediction

        Returns:
            DataFrame with survival probabilities (rows = time points, cols = samples)
            Since Kaplan-Meier doesn't use features, all samples have the same survival curve
        """
        self._check_fitted()

        if time_points is None:
            # Use all time points from the fitted model
            survival_function = self.model_.survival_function_  # type: ignore
            time_index = survival_function.index
            surv_values = survival_function.iloc[:, 0].values  # Single column for all samples
        else:
            # Interpolate to requested time points
            survival_function = self.model_.survival_function_  # type: ignore
            time_index = np.asarray(time_points)
            # Interpolate survival probabilities
            surv_values = self.model_.predict(time_index)  # type: ignore

        # Create DataFrame with same survival for all samples (Kaplan-Meier doesn't use features)
        # For compatibility with interface, return DataFrame with one column per "sample"
        # In practice, all samples would have the same survival curve
        if X is not None:
            n_samples = len(X) if isinstance(X, (pd.DataFrame, np.ndarray)) else 1
        else:
            n_samples = 1

        # Repeat survival curve for each sample
        surv_matrix = np.tile(surv_values.reshape(-1, 1), (1, n_samples))
        return pd.DataFrame(surv_matrix, index=time_index, columns=range(n_samples))

    def predict_risk_score(self, X: Union[np.ndarray, pd.DataFrame, None] = None) -> np.ndarray:
        """Predict risk scores.

        Note: Kaplan-Meier doesn't provide per-sample risk scores since
        it doesn't use features. Returns a constant risk score.

        Args:
            X: Feature matrix (ignored, kept for interface compatibility)

        Returns:
            Array of constant risk scores (since Kaplan-Meier doesn't use features)
        """
        self._check_fitted()

        # Kaplan-Meier doesn't provide per-sample risk scores
        # Return negative median survival time as a risk score
        median_survival = self.model_.median_survival_time_  # type: ignore

        if X is not None:
            n_samples = len(X) if isinstance(X, (pd.DataFrame, np.ndarray)) else 1
        else:
            n_samples = 1

        # Return constant risk score for all samples
        return np.full(n_samples, -median_survival)

    def predict(self, time_points: Union[float, int, np.ndarray, pd.Series]) -> Union[float, np.ndarray]:
        """Predict survival probability at specific time points.

        This is the main prediction method for Kaplan-Meier.

        Args:
            time_points: Time point(s) to predict survival probability at

        Returns:
            Survival probability (float) or array of probabilities

        Examples:
            >>> model.predict(275)
            0.2551797222601467  # 25.5% chance of survival at 275 periods
        """
        self._check_fitted()

        if isinstance(time_points, (float, int)):
            return float(self.model_.predict(time_points))  # type: ignore
        else:
            time_array = np.asarray(time_points)
            predictions = self.model_.predict(time_array)  # type: ignore
            return predictions.values if isinstance(predictions, pd.Series) else predictions

    def qth_survival_time(self, q: float) -> float:
        """Find time point where survival probability equals q (e.g., q=0.99 for 99% survival).

        Args:
            q: Survival probability threshold (e.g., 0.99 for 99% survival)

        Returns:
            Time point where survival probability equals q

        Examples:
            >>> model.qth_survival_time(0.99)
            135.0  # 99% of machines are still working after 135 periods
        """
        self._check_fitted()

        time_point = qth_survival_time(q, self.model_)  # type: ignore
        return float(time_point) if time_point is not None else np.inf

    def plot_survival_function(self, ax=None, ci_show: bool = False, **kwargs):
        """Plot survival function using lifelines plotting.

        Args:
            ax: Matplotlib axes (optional)
            ci_show: Whether to show confidence intervals (default False)
            **kwargs: Additional arguments passed to lifelines plot

        Returns:
            Matplotlib axes
        """
        self._check_fitted()

        return self.model_.plot(ax=ax, ci_show=ci_show, **kwargs)  # type: ignore

