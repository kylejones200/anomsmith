"""Parametric survival models using lifelines.

Parametric models assume a specific distribution for survival times,
such as Weibull (bathtub curve), Exponential, LogNormal, etc.
"""

import logging
from typing import TYPE_CHECKING, Literal, Optional, Union

import numpy as np
import pandas as pd

try:
    from lifelines import (
        ExponentialFitter,
        GeneralizedGammaFitter,
        LogLogisticFitter,
        LogNormalFitter,
        PiecewiseExponentialFitter,
        SplineFitter,
        WeibullFitter,
    )

    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False
    WeibullFitter = None  # type: ignore
    ExponentialFitter = None  # type: ignore
    LogNormalFitter = None  # type: ignore
    LogLogisticFitter = None  # type: ignore
    PiecewiseExponentialFitter = None  # type: ignore
    GeneralizedGammaFitter = None  # type: ignore
    SplineFitter = None  # type: ignore

from anomsmith.primitives.survival.cox import CoxSurvivalModel

if TYPE_CHECKING:
    try:
        from timesmith.typing import SeriesLike
    except ImportError:
        SeriesLike = None

logger = logging.getLogger(__name__)


class ParametricSurvivalModel(CoxSurvivalModel):
    """Base class for parametric survival models.

    Parametric models assume a specific distribution for survival times.
    Different distributions capture different failure patterns:
    - Weibull: Bathtub curve (high failure at start and end)
    - Exponential: Constant failure rate
    - LogNormal: Early failures
    - LogLogistic: S-shaped hazard
    - etc.

    Args:
        model_type: Type of parametric model ('weibull', 'exponential', 'lognormal', etc.)
        **kwargs: Additional parameters for specific model types
        random_state: Random state (not used, kept for compatibility)
    """

    _MODEL_MAP = {
        "weibull": WeibullFitter if LIFELINES_AVAILABLE else None,
        "exponential": ExponentialFitter if LIFELINES_AVAILABLE else None,
        "lognormal": LogNormalFitter if LIFELINES_AVAILABLE else None,
        "loglogistic": LogLogisticFitter if LIFELINES_AVAILABLE else None,
        "piecewise_exponential": PiecewiseExponentialFitter if LIFELINES_AVAILABLE else None,
        "generalized_gamma": GeneralizedGammaFitter if LIFELINES_AVAILABLE else None,
        "spline": SplineFitter if LIFELINES_AVAILABLE else None,
    }

    def __init__(
        self,
        model_type: Literal[
            "weibull",
            "exponential",
            "lognormal",
            "loglogistic",
            "piecewise_exponential",
            "generalized_gamma",
            "spline",
        ] = "weibull",
        breakpoints: Optional[list[float]] = None,  # For piecewise_exponential and spline
        alpha: float = 0.05,  # For confidence intervals
        random_state: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Initialize parametric survival model."""
        if not LIFELINES_AVAILABLE:
            raise ImportError(
                "lifelines is required for ParametricSurvivalModel. "
                "Install with: pip install lifelines"
            )

        if model_type not in self._MODEL_MAP:
            raise ValueError(
                f"Unknown model_type: {model_type}. "
                f"Choose from: {list(self._MODEL_MAP.keys())}"
            )

        if self._MODEL_MAP[model_type] is None:
            raise ImportError(f"{model_type} model not available. Install lifelines.")

        super().__init__(random_state=random_state)
        self.model_type = model_type
        self.breakpoints = breakpoints
        self.alpha = alpha
        self.kwargs = kwargs
        self.model_ = None  # Will be set in fit()
        self.durations_: Optional[np.ndarray] = None
        self.events_: Optional[np.ndarray] = None

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame, None],
        durations: Union[np.ndarray, pd.Series],
        events: Union[np.ndarray, pd.Series, None] = None,
        y: Union[np.ndarray, pd.Series, "SeriesLike", None] = None,
    ) -> "ParametricSurvivalModel":
        """Fit the parametric survival model.

        Note: X (features) is not used in parametric models without covariates.
        Only durations and events are used. X can be None or any value.

        Args:
            X: Feature matrix (ignored for non-covariate models, kept for interface compatibility). Can be None.
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

        # Create model instance based on type
        model_class = self._MODEL_MAP[self.model_type]

        if self.model_type == "piecewise_exponential":
            if self.breakpoints is None:
                # Default breakpoints
                self.breakpoints = [40, 60]
            self.model_ = model_class(self.breakpoints, alpha=self.alpha, **self.kwargs)  # type: ignore
        elif self.model_type == "spline":
            if self.breakpoints is None:
                # Default breakpoints
                self.breakpoints = [6, 20, 40, 75]
            self.model_ = model_class(self.breakpoints, alpha=self.alpha, **self.kwargs)  # type: ignore
        else:
            self.model_ = model_class(alpha=self.alpha, **self.kwargs)  # type: ignore

        # Fit model
        self.model_.fit(durations_array, events_array, label=f"{self.model_type.capitalize()}")  # type: ignore

        self._fitted = True
        logger.debug(
            f"Fitted {self.model_type} model: n_samples={len(durations_array)}, "
            f"median_survival={self.model_.median_survival_time_:.2f}"  # type: ignore
        )
        return self

    def predict_survival_function(
        self, X: Union[np.ndarray, pd.DataFrame, None] = None, time_points: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """Predict survival function S(t).

        Args:
            X: Feature matrix (ignored, kept for interface compatibility)
            time_points: Optional time points for prediction

        Returns:
            DataFrame with survival probabilities (rows = time points, cols = samples)
        """
        self._check_fitted()

        if time_points is None:
            # Use all time points from the fitted model
            survival_function = self.model_.survival_function_  # type: ignore
            time_index = survival_function.index
            surv_values = survival_function.iloc[:, 0].values
        else:
            # Interpolate to requested time points
            time_index = np.asarray(time_points)
            surv_values = self.model_.predict(time_index)  # type: ignore
            if isinstance(surv_values, pd.Series):
                surv_values = surv_values.values

        # Create DataFrame with same survival for all samples (non-covariate models)
        if X is not None:
            n_samples = len(X) if isinstance(X, (pd.DataFrame, np.ndarray)) else 1
        else:
            n_samples = 1

        # Repeat survival curve for each sample
        surv_matrix = np.tile(surv_values.reshape(-1, 1), (1, n_samples))
        return pd.DataFrame(surv_matrix, index=time_index, columns=range(n_samples))

    def predict_risk_score(self, X: Union[np.ndarray, pd.DataFrame, None] = None) -> np.ndarray:
        """Predict risk scores.

        Note: Non-covariate parametric models don't provide per-sample risk scores.
        Returns constant risk score based on median survival.

        Args:
            X: Feature matrix (ignored, kept for interface compatibility)

        Returns:
            Array of constant risk scores
        """
        self._check_fitted()

        # Use negative median survival time as risk score
        median_survival = self.model_.median_survival_time_  # type: ignore

        if X is not None:
            n_samples = len(X) if isinstance(X, (pd.DataFrame, np.ndarray)) else 1
        else:
            n_samples = 1

        return np.full(n_samples, -median_survival)

    def predict(self, time_points: Union[float, int, np.ndarray, pd.Series]) -> Union[float, np.ndarray]:
        """Predict survival probability at specific time points.

        Args:
            time_points: Time point(s) to predict survival probability at

        Returns:
            Survival probability (float) or array of probabilities

        Examples:
            >>> model = ParametricSurvivalModel(model_type="weibull")
            >>> model.fit(durations=durations, events=events)
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

    def plot_survival_function(self, ax=None, **kwargs):
        """Plot survival function using lifelines plotting.

        Args:
            ax: Matplotlib axes (optional)
            **kwargs: Additional arguments passed to lifelines plot

        Returns:
            Matplotlib axes
        """
        self._check_fitted()

        return self.model_.plot_survival_function(ax=ax, **kwargs)  # type: ignore

