"""Machine learning based anomaly detectors.

These detectors use sklearn models and must be in Layer 2 (primitives).
sklearn is allowed in Layer 2 as it's a core ML library.
"""

import logging
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from anomsmith.objects.views import LabelView, ScoreView
from anomsmith.primitives.base import BaseDetector
from anomsmith.primitives.detectors._utils import extract_index_and_values, prepare_input_data

logger = logging.getLogger(__name__)


class IsolationForestDetector(BaseDetector):
    """Isolation Forest anomaly detector.

    Isolation Forest is an ensemble method that isolates anomalies
    by randomly selecting features and splitting values.

    Args:
        contamination: Expected proportion of outliers in the data
        n_estimators: Number of base estimators
        random_state: Random state for reproducibility
        n_jobs: Number of jobs to run in parallel
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 200,
        random_state: Optional[int] = None,
        n_jobs: int = -1,
    ) -> None:
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model_: IsolationForest | None = None
        self.scaler_ = StandardScaler()
        super().__init__(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self._fitted = False

    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    ) -> "IsolationForestDetector":
        """Fit the Isolation Forest detector.

        Args:
            y: Training data (target)
            X: Optional features (if None, uses y)

        Returns:
            Self for method chaining
        """
        X_data = prepare_input_data(y, X)
        X_scaled = self.scaler_.fit_transform(X_data)

        self.model_ = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.model_.fit(X_scaled)

        self._fitted = True
        logger.debug("Fitted IsolationForestDetector")
        return self

    def predict(self, y: Union[np.ndarray, pd.Series]) -> LabelView:
        """Predict anomaly labels.

        Args:
            y: Time series to detect anomalies in

        Returns:
            LabelView with binary labels (1 = anomaly, 0 = normal)
        """
        self._check_fitted()

        index, values = extract_index_and_values(y)
        X_data = prepare_input_data(values)
        X_scaled = self.scaler_.transform(X_data)
        predictions = self.model_.predict(X_scaled)  # Returns -1 (anomaly) or 1 (normal)
        # Convert to 0/1 (anomsmith convention)
        labels = (predictions == -1).astype(int)

        return LabelView(index=index, labels=labels)

    def score(self, y: Union[np.ndarray, pd.Series]) -> ScoreView:
        """Score anomalies.

        Args:
            y: Time series to score

        Returns:
            ScoreView with anomaly scores
        """
        self._check_fitted()

        index, values = extract_index_and_values(y)
        X_data = prepare_input_data(values)
        X_scaled = self.scaler_.transform(X_data)
        scores = self.model_.decision_function(X_scaled)
        # Invert so higher scores = more anomalous
        scores = -scores

        return ScoreView(index=index, scores=scores)


class LOFDetector(BaseDetector):
    """Local Outlier Factor (LOF) anomaly detector.

    LOF measures the local deviation of density of a given sample
    with respect to its neighbors.

    Args:
        contamination: Expected proportion of outliers in the data
        n_neighbors: Number of neighbors to use
        random_state: Random state for reproducibility
        n_jobs: Number of jobs to run in parallel
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_neighbors: int = 20,
        random_state: Optional[int] = None,
        n_jobs: int = -1,
    ) -> None:
        self.contamination = contamination
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model_: LocalOutlierFactor | None = None
        self.scaler_ = StandardScaler()
        super().__init__(
            contamination=contamination,
            n_neighbors=n_neighbors,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self._fitted = False

    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    ) -> "LOFDetector":
        """Fit the LOF detector.

        Args:
            y: Training data (target)
            X: Optional features (if None, uses y)

        Returns:
            Self for method chaining
        """
        X_data = prepare_input_data(y, X)
        X_scaled = self.scaler_.fit_transform(X_data)

        # Set novelty=True to enable predict() method for new data
        self.model_ = LocalOutlierFactor(
            contamination=self.contamination,
            n_neighbors=self.n_neighbors,
            n_jobs=self.n_jobs,
            novelty=True,
        )
        self.model_.fit(X_scaled)

        self._fitted = True
        logger.debug("Fitted LOFDetector")
        return self

    def predict(self, y: Union[np.ndarray, pd.Series]) -> LabelView:
        """Predict anomaly labels.

        Args:
            y: Time series to detect anomalies in

        Returns:
            LabelView with binary labels (1 = anomaly, 0 = normal)
        """
        self._check_fitted()

        index, values = extract_index_and_values(y)
        X_data = prepare_input_data(values)
        X_scaled = self.scaler_.transform(X_data)
        predictions = self.model_.predict(X_scaled)  # Returns -1 (anomaly) or 1 (normal)
        # Convert to 0/1 (anomsmith convention)
        labels = (predictions == -1).astype(int)

        return LabelView(index=index, labels=labels)

    def score(self, y: Union[np.ndarray, pd.Series]) -> ScoreView:
        """Score anomalies.

        Args:
            y: Time series to score

        Returns:
            ScoreView with anomaly scores
        """
        self._check_fitted()

        index, values = extract_index_and_values(y)
        X_data = prepare_input_data(values)
        X_scaled = self.scaler_.transform(X_data)

        # LOF model must be refitted for scoring new data
        model = LocalOutlierFactor(
            contamination=self.contamination, n_neighbors=self.n_neighbors, n_jobs=self.n_jobs
        )
        model.fit(X_scaled)
        scores = model.negative_outlier_factor_
        # Invert so higher scores = more anomalous
        scores = -scores

        return ScoreView(index=index, scores=scores)


class RobustCovarianceDetector(BaseDetector):
    """Robust Covariance (Elliptic Envelope) anomaly detector.

    Assumes that the data is Gaussian distributed and fits an
    elliptic envelope to the data.

    Args:
        contamination: Expected proportion of outliers in the data
        support_fraction: Proportion of points to be used as support
        random_state: Random state for reproducibility
    """

    def __init__(
        self,
        contamination: float = 0.05,
        support_fraction: float = 0.8,
        random_state: Optional[int] = None,
    ) -> None:
        self.contamination = contamination
        self.support_fraction = support_fraction
        self.random_state = random_state
        self.model_: EllipticEnvelope | None = None
        self.scaler_ = StandardScaler()
        super().__init__(
            contamination=contamination,
            support_fraction=support_fraction,
            random_state=random_state,
        )
        self._fitted = False

    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    ) -> "RobustCovarianceDetector":
        """Fit the Robust Covariance detector.

        Args:
            y: Training data (target)
            X: Optional features (if None, uses y)

        Returns:
            Self for method chaining
        """
        X_data = prepare_input_data(y, X)
        X_scaled = self.scaler_.fit_transform(X_data)

        self.model_ = EllipticEnvelope(
            contamination=self.contamination,
            support_fraction=self.support_fraction,
            random_state=self.random_state,
        )
        self.model_.fit(X_scaled)

        self._fitted = True
        logger.debug("Fitted RobustCovarianceDetector")
        return self

    def predict(self, y: Union[np.ndarray, pd.Series]) -> LabelView:
        """Predict anomaly labels.

        Args:
            y: Time series to detect anomalies in

        Returns:
            LabelView with binary labels (1 = anomaly, 0 = normal)
        """
        self._check_fitted()

        index, values = extract_index_and_values(y)
        X_data = prepare_input_data(values)
        X_scaled = self.scaler_.transform(X_data)
        predictions = self.model_.predict(X_scaled)  # Returns -1 (anomaly) or 1 (normal)
        # Convert to 0/1 (anomsmith convention)
        labels = (predictions == -1).astype(int)

        return LabelView(index=index, labels=labels)

    def score(self, y: Union[np.ndarray, pd.Series]) -> ScoreView:
        """Score anomalies.

        Args:
            y: Time series to score

        Returns:
            ScoreView with anomaly scores
        """
        self._check_fitted()

        index, values = extract_index_and_values(y)
        X_data = prepare_input_data(values)
        X_scaled = self.scaler_.transform(X_data)
        scores = self.model_.decision_function(X_scaled)
        # Invert so higher scores = more anomalous
        scores = -scores

        return ScoreView(index=index, scores=scores)

