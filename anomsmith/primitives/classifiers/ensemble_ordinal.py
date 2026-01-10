"""Ensemble methods for ordinal classification.

Provides averaging and stacking ensembles for combining ordinal predictions
from multiple models.
"""

import logging
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LogisticRegression

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    LogisticRegression = None  # type: ignore

from anomsmith.objects.health_state import HealthState, HealthStateView
from anomsmith.primitives.base import BaseEstimator

if TYPE_CHECKING:
    try:
        from timesmith.typing import SeriesLike
    except ImportError:
        SeriesLike = None

logger = logging.getLogger(__name__)


class AveragingOrdinalEnsemble(BaseEstimator):
    """Averaging ensemble for ordinal predictions.

    Averages predictions from multiple ordinal models and rounds to nearest class.
    Simple but effective for combining diverse models.

    Args:
        models: List of fitted ordinal classifiers
        random_state: Random state (not used, kept for compatibility)
    """

    def __init__(
        self,
        models: List[BaseEstimator],
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize averaging ensemble."""
        if not models:
            raise ValueError("At least one model must be provided")

        super().__init__(models=models, random_state=random_state)
        self.models = models
        self.random_state = random_state
        self._fitted = True  # Models are assumed to be pre-fitted

    def fit(
        self,
        y: Union[np.ndarray, pd.Series, "SeriesLike"],
        X: Union[np.ndarray, pd.DataFrame],
    ) -> "AveragingOrdinalEnsemble":
        """Fit method for compatibility (models should be pre-fitted).

        Args:
            y: Target values (not used, models are pre-fitted)
            X: Feature matrix (not used, models are pre-fitted)

        Returns:
            Self for method chaining
        """
        # Check that all models are fitted
        for i, model in enumerate(self.models):
            if not hasattr(model, "_fitted") or not model._fitted:
                raise ValueError(f"Model {i} is not fitted. Fit all models before ensembling.")

        self._fitted = True
        logger.debug(f"AveragingOrdinalEnsemble: {len(self.models)} models")
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict health states by averaging model predictions.

        Args:
            X: Feature matrix (n_samples, n_features) or sequences (n_samples, seq_len, n_features)

        Returns:
            Array of predicted health states (0=Healthy, 1=Warning, 2=Distress)
        """
        self._check_fitted()

        # Get predictions from all models
        all_predictions = []
        for i, model in enumerate(self.models):
            if not hasattr(model, 'predict'):
                raise ValueError(f"Model {i} ({type(model).__name__}) does not have a predict method")
            
            try:
                pred = model.predict(X)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"Unable to predict with model {i} ({type(model).__name__}) on input "
                    f"{type(X).__name__}: {e}"
                ) from e
            
            if pred is None or len(pred) == 0:
                raise ValueError(f"Model {i} ({type(model).__name__}) returned empty predictions")
            
            all_predictions.append(pred)

        # Align all predictions to same length (handle different lengths)
        if not all_predictions:
            raise ValueError("No predictions obtained from models")
        
        # Find minimum length across all predictions
        min_len = min(len(p) for p in all_predictions)
        if min_len == 0:
            raise ValueError("At least one model returned empty predictions")
        
        # Align predictions: truncate if longer, pad if shorter
        all_predictions_aligned = [
            p[:min_len] if len(p) >= min_len else np.pad(p, (0, min_len - len(p)), mode="edge")
            for p in all_predictions
        ]

        # Average predictions (as numeric values) and round to nearest class
        predictions_array = np.array(all_predictions_aligned)
        averaged = np.mean(predictions_array, axis=0)
        predictions = np.round(averaged).clip(0, 2).astype(int)

        return predictions

    def predict_health_states(
        self, X: Union[np.ndarray, pd.DataFrame], index: Optional[pd.Index] = None
    ) -> HealthStateView:
        """Predict health states as HealthStateView.

        Args:
            X: Feature matrix or sequences
            index: Optional index for the health states

        Returns:
            HealthStateView with predicted health states
        """
        if index is None:
            if isinstance(X, pd.DataFrame):
                index = X.index
            else:
                index = pd.RangeIndex(start=0, stop=len(X))

        predictions = self.predict(X)
        return HealthStateView(index=index, states=predictions)


class StackedOrdinalEnsemble(BaseEstimator):
    """Stacked ensemble for ordinal predictions.

    Uses predictions from base models as features for a meta-model.
    More sophisticated than averaging - learns when to trust each model.

    Args:
        base_models: List of fitted ordinal classifiers (base models)
        meta_model: Meta-model for stacking (default: LogisticRegression)
        random_state: Random state for reproducibility
    """

    def __init__(
        self,
        base_models: List[BaseEstimator],
        meta_model: Optional[BaseEstimator] = None,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize stacked ensemble."""
        if not base_models:
            raise ValueError("At least one base model must be provided")

        if meta_model is None:
            if not SKLEARN_AVAILABLE:
                raise ImportError(
                    "scikit-learn is required for default meta-model. "
                    "Install with: pip install scikit-learn"
                )
            meta_model = LogisticRegression(solver="lbfgs", max_iter=500, random_state=random_state)  # type: ignore

        super().__init__(
            base_models=base_models, meta_model=meta_model, random_state=random_state
        )
        self.base_models = base_models
        self.meta_model = meta_model
        self.random_state = random_state
        self._fitted = False

    def fit(
        self,
        y: Union[np.ndarray, pd.Series, "SeriesLike"],
        X: Union[np.ndarray, pd.DataFrame],
    ) -> "StackedOrdinalEnsemble":
        """Fit the stacked ensemble.

        Args:
            y: Target health states (n_samples,)
            X: Feature matrix (n_samples, n_features) or sequences (n_samples, seq_len, n_features)

        Returns:
            Self for method chaining
        """
        # Check that all base models are fitted
        for i, model in enumerate(self.base_models):
            if not hasattr(model, "_fitted") or not model._fitted:
                raise ValueError(
                    f"Base model {i} is not fitted. Fit all base models before stacking."
                )

        if isinstance(y, pd.Series):
            y_data = y.values
        else:
            y_data = np.asarray(y)

        # Get predictions from all base models (these become features for meta-model)
        base_predictions = []
        for model in self.base_models:
            pred = model.predict(X)
            base_predictions.append(pred)

        # Stack predictions into feature matrix
        X_stack = np.column_stack(base_predictions)

        # Fit meta-model on stacked predictions
        self.meta_model.fit(X_stack, y_data)  # type: ignore

        self._fitted = True
        logger.debug(
            f"Fitted StackedOrdinalEnsemble: {len(self.base_models)} base models, "
            f"n_samples={len(X_stack)}"
        )
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict health states using stacked ensemble.

        Args:
            X: Feature matrix (n_samples, n_features) or sequences (n_samples, seq_len, n_features)

        Returns:
            Array of predicted health states (0=Healthy, 1=Warning, 2=Distress)
        """
        self._check_fitted()

        # Get predictions from all base models (these become features for meta-model)
        base_predictions = []
        for i, model in enumerate(self.base_models):
            if not hasattr(model, 'predict'):
                raise ValueError(
                    f"Base model {i} ({type(model).__name__}) does not have a predict method"
                )
            
            try:
                pred = model.predict(X)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"Unable to predict with base model {i} ({type(model).__name__}) on input "
                    f"{type(X).__name__}: {e}"
                ) from e
            
            if pred is None or len(pred) == 0:
                raise ValueError(
                    f"Base model {i} ({type(model).__name__}) returned empty predictions"
                )
            
            base_predictions.append(pred)

        # Align all predictions to same length (handle different lengths)
        if not base_predictions:
            raise ValueError("No predictions obtained from base models")
        
        min_len = min(len(p) for p in base_predictions)
        if min_len == 0:
            raise ValueError("At least one base model returned empty predictions")
        
        base_predictions_aligned = [
            p[:min_len] if len(p) >= min_len else np.pad(p, (0, min_len - len(p)), mode="edge")
            for p in base_predictions
        ]

        # Stack predictions into feature matrix
        X_stack = np.column_stack(base_predictions_aligned)

        # Predict using meta-model
        predictions = self.meta_model.predict(X_stack)  # type: ignore
        # Ensure predictions are in valid range [0, 2]
        predictions = np.clip(predictions.astype(int), 0, 2)

        return predictions

    def predict_health_states(
        self, X: Union[np.ndarray, pd.DataFrame], index: Optional[pd.Index] = None
    ) -> HealthStateView:
        """Predict health states as HealthStateView.

        Args:
            X: Feature matrix or sequences
            index: Optional index for the health states

        Returns:
            HealthStateView with predicted health states
        """
        if index is None:
            if isinstance(X, pd.DataFrame):
                index = X.index
            else:
                index = pd.RangeIndex(start=0, stop=len(X))

        predictions = self.predict(X)
        return HealthStateView(index=index, states=predictions)

