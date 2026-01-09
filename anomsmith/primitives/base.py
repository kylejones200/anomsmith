"""Base classes for primitives with parameter management and tags."""

import copy
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Union

import numpy as np
import pandas as pd

from anomsmith.objects.views import LabelView, ScoreView

if TYPE_CHECKING:
    try:
        from timesmith.typing import SeriesLike
    except ImportError:
        SeriesLike = None


class BaseObject(ABC):
    """Base class for all primitives with parameter management.

    Provides get_params, set_params, clone, and repr methods.
    """

    def __init__(self, **params: Any) -> None:
        """Initialize with parameters.

        Args:
            **params: Parameters to set
        """
        self._params: dict[str, Any] = {}
        self.set_params(**params)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this object.

        Args:
            deep: If True, return deep copy of parameters

        Returns:
            Dictionary of parameter names to values
        """
        if deep:
            return copy.deepcopy(self._params)
        return self._params.copy()

    def set_params(self, **params: Any) -> "BaseObject":
        """Set parameters for this object.

        Args:
            **params: Parameters to set

        Returns:
            Self for method chaining
        """
        for key, value in params.items():
            setattr(self, key, value)
            self._params[key] = value
        return self

    def clone(self) -> "BaseObject":
        """Create a deep copy of this object.

        Returns:
            Deep copy of this object
        """
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        """String representation with parameters."""
        params_str = ", ".join(f"{k}={v!r}" for k, v in self._params.items())
        return f"{self.__class__.__name__}({params_str})"

    @property
    def tags(self) -> dict[str, Any]:
        """Get tags for this object.

        Returns:
            Dictionary of tag names to values
        """
        return getattr(self, "_tags", {})


class BaseEstimator(BaseObject):
    """Base class for estimators with fit and fitted state.

    Attributes:
        _fitted: Whether the estimator has been fitted
    """

    def __init__(self, **params: Any) -> None:
        """Initialize estimator."""
        super().__init__(**params)
        self._fitted = False

    @abstractmethod
    def fit(
        self,
        y: Union[np.ndarray, pd.Series, "SeriesLike"],
        X: np.ndarray | pd.DataFrame | None = None,
    ) -> "BaseEstimator":
        """Fit the estimator.

        Args:
            y: Target values
            X: Optional features

        Returns:
            Self for method chaining
        """
        pass

    @property
    def is_fitted(self) -> bool:
        """Check if estimator is fitted.

        Returns:
            True if fitted, False otherwise
        """
        return self._fitted

    def _check_fitted(self) -> None:
        """Check if estimator is fitted, raise if not."""
        if not self._fitted:
            raise ValueError(f"{self.__class__.__name__} has not been fitted yet")


class BaseScorer(BaseEstimator):
    """Base class for anomaly scorers.

    Scorers assign anomaly scores to time series points.
    Higher scores indicate more anomalous points.
    """

    def __init__(self, **params: Any) -> None:
        """Initialize scorer."""
        super().__init__(**params)
        self._tags = {
            "scitype_input": "series",
            "supports_panel": False,
            "requires_sorted_index": True,
            "handles_missing": False,
            "supports_partial_fit": False,
        }

    @abstractmethod
    def score(self, y: Union[np.ndarray, pd.Series, "SeriesLike"]) -> ScoreView:
        """Score anomalies in a time series.

        Args:
            y: Time series to score

        Returns:
            ScoreView with anomaly scores
        """
        pass


class BaseDetector(BaseEstimator):
    """Base class for anomaly detectors.

    Detectors produce both scores and binary labels.
    """

    def __init__(self, **params: Any) -> None:
        """Initialize detector."""
        super().__init__(**params)
        self._tags = {
            "scitype_input": "series",
            "supports_panel": False,
            "requires_sorted_index": True,
            "handles_missing": False,
            "supports_partial_fit": False,
        }

    @abstractmethod
    def predict(self, y: Union[np.ndarray, pd.Series, "SeriesLike"]) -> LabelView:
        """Predict anomaly labels.

        Args:
            y: Time series to detect anomalies in

        Returns:
            LabelView with binary anomaly labels
        """
        pass

    @abstractmethod
    def score(self, y: Union[np.ndarray, pd.Series, "SeriesLike"]) -> ScoreView:
        """Score anomalies in a time series.

        Args:
            y: Time series to score

        Returns:
            ScoreView with anomaly scores
        """
        pass

