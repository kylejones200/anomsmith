"""Neural survival models using pycox.

Deep learning approaches for survival analysis that capture nonlinear
feature interactions and time-dependent effects.
"""

import logging
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import pandas as pd

try:
    import torch.nn as nn
    import torchtuples as tt
    from pycox.models import CoxPH, LogisticHazard
    from pycox.preprocessing.label_transforms import LabTransDiscreteTime

    PYCOX_AVAILABLE = True
except ImportError:
    PYCOX_AVAILABLE = False
    nn = None  # type: ignore
    tt = None  # type: ignore
    CoxPH = None  # type: ignore
    LogisticHazard = None  # type: ignore
    LabTransDiscreteTime = None  # type: ignore

from anomsmith.primitives.survival.cox import CoxSurvivalModel

if TYPE_CHECKING:
    try:
        from timesmith.typing import SeriesLike
    except ImportError:
        SeriesLike = None

logger = logging.getLogger(__name__)


class LogisticHazardModel(CoxSurvivalModel):
    """Discrete-time neural survival model using LogisticHazard.

    Models time-to-failure in discrete intervals. Well-suited for sensor data
    where degradation follows nonlinear stages. Makes fewer assumptions than CoxPH.

    Args:
        n_bins: Number of time bins for discretization (default 50)
        num_nodes: Hidden layer sizes (default [32, 32])
        batch_size: Batch size for training (default 128)
        epochs: Number of training epochs (default 50)
        random_state: Random state for reproducibility
    """

    def __init__(
        self,
        n_bins: int = 50,
        num_nodes: list[int] = None,
        batch_size: int = 128,
        epochs: int = 50,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize LogisticHazard model."""
        if not PYCOX_AVAILABLE:
            raise ImportError(
                "pycox is required for LogisticHazardModel. "
                "Install with: pip install pycox"
            )

        super().__init__(random_state=random_state)
        self.n_bins = n_bins
        self.num_nodes = num_nodes or [32, 32]
        self.batch_size = batch_size
        self.epochs = epochs
        self.model_: Optional[LogisticHazard] = None  # type: ignore
        self.labtrans_: Optional[LabTransDiscreteTime] = None  # type: ignore
        self.n_features_: Optional[int] = None

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame, None],
        durations: Union[np.ndarray, pd.Series],
        events: Union[np.ndarray, pd.Series, None] = None,
        y: Union[np.ndarray, pd.Series, "SeriesLike", None] = None,
    ) -> "LogisticHazardModel":
        """Fit the LogisticHazard model.

        Args:
            X: Feature matrix (n_samples, n_features) as float32 - Required for LogisticHazard
            durations: Time-to-event values (n_samples,)
            events: Event indicators (1 = event occurred, 0 = censored)
            y: Optional target (not used, kept for interface compatibility)

        Returns:
            Self for method chaining
        """
        if X is None:
            raise ValueError("X (feature matrix) is required for LogisticHazardModel")
        
        if isinstance(X, pd.DataFrame):
            X_data = X.values.astype("float32")
        else:
            X_data = np.asarray(X, dtype="float32")

        durations_array = np.asarray(durations)
        events_array = np.asarray(events) if events is not None else np.ones(len(durations))

        self.n_features_ = X_data.shape[1]

        # Discretize time
        self.labtrans_ = LabTransDiscreteTime(self.n_bins)  # type: ignore
        y_train_disc = self.labtrans_.fit_transform(durations_array, events_array)  # type: ignore

        # Build neural network
        net = tt.practical.MLPVanilla(  # type: ignore
            in_features=self.n_features_,
            num_nodes=self.num_nodes,
            out_features=self.labtrans_.out_features,  # type: ignore
            activation=nn.ReLU,  # type: ignore
        )

        # Create model
        self.model_ = LogisticHazard(  # type: ignore
            net, tt.optim.Adam, duration_index=self.labtrans_.cuts  # type: ignore
        )

        # Train
        self.model_.fit(  # type: ignore
            X_data, y_train_disc, batch_size=self.batch_size, epochs=self.epochs, verbose=False
        )

        self._fitted = True
        logger.debug(
            f"Fitted LogisticHazardModel: n_bins={self.n_bins}, "
            f"n_features={self.n_features_}, n_samples={len(X_data)}"
        )
        return self

    def predict_survival_function(
        self, X: Union[np.ndarray, pd.DataFrame], time_points: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """Predict survival function S(t|X).

        Args:
            X: Feature matrix (n_samples, n_features) as float32
            time_points: Optional time points (uses model bins if None)

        Returns:
            DataFrame with survival probabilities (rows = time points, cols = samples)
        """
        self._check_fitted()

        if isinstance(X, pd.DataFrame):
            X_data = X.values.astype("float32")
        else:
            X_data = np.asarray(X, dtype="float32")

        surv_df = self.model_.predict_surv_df(X_data)  # type: ignore

        if time_points is not None:
            # Interpolate to requested time points
            # Use reindex with nearest method, forward fill for extrapolation
            surv_interp = surv_df.reindex(time_points, method="nearest")
            surv_interp = surv_interp.ffill().bfill().fillna(1.0)
            return surv_interp

        return surv_df

    def predict_risk_score(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict relative risk scores.

        Args:
            X: Feature matrix (n_samples, n_features) as float32

        Returns:
            Array of risk scores (higher = higher failure risk)
        """
        self._check_fitted()

        if isinstance(X, pd.DataFrame):
            X_data = X.values.astype("float32")
        else:
            X_data = np.asarray(X, dtype="float32")

        # Use negative median survival time as risk score
        surv_df = self.predict_survival_function(X_data)
        median_survival = surv_df.apply(lambda col: col.index[col <= 0.5][0] if (col <= 0.5).any() else col.index[-1])
        # Higher median survival = lower risk
        risk_scores = -median_survival.values
        return risk_scores


class DeepSurvModel(CoxSurvivalModel):
    """Continuous-time neural Cox model using DeepSurv.

    Learns flexible nonlinear mapping from features to risk scores.
    Improves over CoxPH by capturing feature interactions but still assumes
    proportional hazards.

    Args:
        num_nodes: Hidden layer sizes (default [32, 32])
        batch_size: Batch size for training (default 128)
        epochs: Number of training epochs (default 50)
        random_state: Random state for reproducibility
    """

    def __init__(
        self,
        num_nodes: list[int] = None,
        batch_size: int = 128,
        epochs: int = 50,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize DeepSurv model."""
        if not PYCOX_AVAILABLE:
            raise ImportError(
                "pycox is required for DeepSurvModel. "
                "Install with: pip install pycox"
            )

        super().__init__(random_state=random_state)
        self.num_nodes = num_nodes or [32, 32]
        self.batch_size = batch_size
        self.epochs = epochs
        self.model_: Optional[CoxPH] = None  # type: ignore
        self.n_features_: Optional[int] = None

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame, None],
        durations: Union[np.ndarray, pd.Series],
        events: Union[np.ndarray, pd.Series, None] = None,
        y: Union[np.ndarray, pd.Series, "SeriesLike", None] = None,
    ) -> "DeepSurvModel":
        """Fit the DeepSurv model.

        Args:
            X: Feature matrix (n_samples, n_features) as float32 - Required for DeepSurv
            durations: Time-to-event values (n_samples,)
            events: Event indicators (1 = event occurred, 0 = censored)
            y: Optional target (not used, kept for interface compatibility)

        Returns:
            Self for method chaining
        """
        if X is None:
            raise ValueError("X (feature matrix) is required for DeepSurvModel")
        
        if isinstance(X, pd.DataFrame):
            X_data = X.values.astype("float32")
        else:
            X_data = np.asarray(X, dtype="float32")

        durations_array = np.asarray(durations)
        events_array = np.asarray(events) if events is not None else np.ones(len(durations))

        self.n_features_ = X_data.shape[1]

        # Build neural network (single output for risk score)
        net = tt.practical.MLPVanilla(  # type: ignore
            in_features=self.n_features_,
            num_nodes=self.num_nodes,
            out_features=1,
            activation=nn.ReLU,  # type: ignore
        )

        # Create model
        self.model_ = CoxPH(net, tt.optim.Adam)  # type: ignore

        # Train
        self.model_.fit(  # type: ignore
            X_data,
            (durations_array, events_array),
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=False,
        )

        # Compute baseline hazards for survival prediction
        self.model_.compute_baseline_hazards()  # type: ignore

        self._fitted = True
        logger.debug(
            f"Fitted DeepSurvModel: n_features={self.n_features_}, n_samples={len(X_data)}"
        )
        return self

    def predict_survival_function(
        self, X: Union[np.ndarray, pd.DataFrame], time_points: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """Predict survival function S(t|X).

        Args:
            X: Feature matrix (n_samples, n_features) as float32
            time_points: Optional time points (uses baseline hazards if None)

        Returns:
            DataFrame with survival probabilities (rows = time points, cols = samples)
        """
        self._check_fitted()

        if isinstance(X, pd.DataFrame):
            X_data = X.values.astype("float32")
        else:
            X_data = np.asarray(X, dtype="float32")

        surv_df = self.model_.predict_surv_df(X_data)  # type: ignore

        if time_points is not None:
            # Interpolate to requested time points
            # Use reindex with nearest method, forward fill for extrapolation
            surv_interp = surv_df.reindex(time_points, method="nearest")
            surv_interp = surv_interp.ffill().bfill().fillna(1.0)
            return surv_interp

        return surv_df

    def predict_risk_score(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict relative risk scores (partial hazards).

        Args:
            X: Feature matrix (n_samples, n_features) as float32

        Returns:
            Array of risk scores (higher = higher failure risk)
        """
        self._check_fitted()

        if isinstance(X, pd.DataFrame):
            X_data = X.values.astype("float32")
        else:
            X_data = np.asarray(X, dtype="float32")

        # DeepSurv outputs log-hazard, exponentiate to get hazard ratio
        risk_scores = self.model_.predict(X_data)  # type: ignore
        return risk_scores.flatten()

