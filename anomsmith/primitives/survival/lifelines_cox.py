"""Cox Proportional Hazards model using lifelines library.

Linear Cox regression for survival analysis. Suitable when interpretability
is critical or when data is sparse.
"""

import logging
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import pandas as pd

try:
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index as lifelines_c_index

    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False
    CoxPHFitter = None  # type: ignore
    lifelines_c_index = None  # type: ignore

from anomsmith.primitives.survival.cox import CoxSurvivalModel

if TYPE_CHECKING:
    try:
        from timesmith.typing import SeriesLike
    except ImportError:
        SeriesLike = None

logger = logging.getLogger(__name__)


class LifelinesCoxModel(CoxSurvivalModel):
    """Cox Proportional Hazards model using lifelines.

    Linear Cox regression assumes proportional hazards and a log-linear
    relationship between features and hazard. Good for interpretability.

    Args:
        penalizer: Regularization coefficient (default 0.0)
        l1_ratio: L1/L2 regularization ratio (default 0.0)
        random_state: Random state (not used, kept for compatibility)
    """

    def __init__(
        self,
        penalizer: float = 0.0,
        l1_ratio: float = 0.0,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize Lifelines Cox model."""
        if not LIFELINES_AVAILABLE:
            raise ImportError(
                "lifelines is required for LifelinesCoxModel. "
                "Install with: pip install lifelines"
            )

        super().__init__(random_state=random_state)
        self.penalizer = penalizer
        self.l1_ratio = l1_ratio
        self.model_: Optional[CoxPHFitter] = None  # type: ignore
        self.feature_names_: Optional[list[str]] = None

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame, None],
        durations: Union[np.ndarray, pd.Series],
        events: Union[np.ndarray, pd.Series, None] = None,
        y: Union[np.ndarray, pd.Series, "SeriesLike", None] = None,
    ) -> "LifelinesCoxModel":
        """Fit the Cox model.

        Args:
            X: Feature matrix (n_samples, n_features) - Required for Cox model
            durations: Time-to-event values (n_samples,)
            events: Event indicators (1 = event occurred, 0 = censored)
            y: Optional target (not used, kept for interface compatibility)

        Returns:
            Self for method chaining
        """
        if X is None:
            raise ValueError("X (feature matrix) is required for LifelinesCoxModel")
        
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
            self.feature_names_ = list(X.columns)
        else:
            X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
            self.feature_names_ = list(X_df.columns)

        # Add duration and event columns
        X_df["duration"] = durations
        X_df["event"] = events if events is not None else np.ones(len(durations))

        # Fit model
        self.model_ = CoxPHFitter(penalizer=self.penalizer, l1_ratio=self.l1_ratio)  # type: ignore
        self.model_.fit(X_df, duration_col="duration", event_col="event")  # type: ignore

        self._fitted = True
        logger.debug(f"Fitted LifelinesCoxModel: {len(self.feature_names_)} features")
        return self

    def predict_survival_function(
        self, X: Union[np.ndarray, pd.DataFrame], time_points: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """Predict survival function S(t|X).

        Args:
            X: Feature matrix (n_samples, n_features)
            time_points: Optional time points for prediction

        Returns:
            DataFrame with survival probabilities (rows = time points, cols = samples)
        """
        self._check_fitted()

        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            X_df = pd.DataFrame(X, columns=self.feature_names_)

        # Get baseline survival function
        baseline_surv = self.model_.baseline_survival_  # type: ignore

        # Predict partial hazards (risk scores)
        partial_hazards = self.model_.predict_partial_hazard(X_df)  # type: ignore

        # Compute survival: S(t|X) = S_0(t) ^ exp(partial_hazard)
        if time_points is None:
            time_points = baseline_surv.index.values

        surv_matrix = np.zeros((len(time_points), len(X_df)))
        for i, t in enumerate(time_points):
            # Find closest baseline survival value
            baseline_idx = baseline_surv.index.get_indexer([t], method="nearest")[0]
            baseline_s = baseline_surv.iloc[baseline_idx]
            # Compute survival for each sample
            surv_matrix[i, :] = np.power(baseline_s, np.exp(partial_hazards.values))

        return pd.DataFrame(surv_matrix, index=time_points, columns=range(len(X_df)))

    def predict_risk_score(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict relative risk scores (partial hazards).

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Array of risk scores (higher = higher failure risk)
        """
        self._check_fitted()

        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            X_df = pd.DataFrame(X, columns=self.feature_names_)

        partial_hazards = self.model_.predict_partial_hazard(X_df)  # type: ignore
        return partial_hazards.values.flatten()

