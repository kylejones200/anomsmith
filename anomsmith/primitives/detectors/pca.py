"""PCA-based anomaly detection detector.

Uses Principal Component Analysis to model healthy operation boundaries.
Anomalies are detected using either:
- Mahalanobis distance in the principal component space
- Reconstruction error (difference between original and reconstructed data)
"""

import logging
from typing import TYPE_CHECKING, Literal, Optional, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from anomsmith.objects.views import LabelView, ScoreView
from anomsmith.primitives.base import BaseDetector

if TYPE_CHECKING:
    try:
        from timesmith.typing import SeriesLike
    except ImportError:
        SeriesLike = None

logger = logging.getLogger(__name__)


class PCADetector(BaseDetector):
    """PCA-based anomaly detector.

    Uses Principal Component Analysis to model healthy operation boundaries.
    Anomalies are detected using either:
    - Mahalanobis distance in the principal component space
    - Reconstruction error (difference between original and reconstructed data)

    Args:
        n_components: Number of components to keep. If 0 < n_components < 1,
            select the number of components such that the amount of variance
            that needs to be explained is greater than the percentage specified.
        score_method: Method for computing anomaly scores:
            - 'reconstruction': Use reconstruction error
            - 'mahalanobis': Use Mahalanobis distance in PC space
            - 'both': Use both and return average
        contamination: Expected proportion of outliers in the data (used for threshold)
        random_state: Random state for reproducibility
    """

    def __init__(
        self,
        n_components: Union[float, int] = 0.95,
        score_method: Literal["reconstruction", "mahalanobis", "both"] = "reconstruction",
        contamination: float = 0.05,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_components = n_components
        self.score_method = score_method
        self.contamination = contamination
        self.random_state = random_state
        self.pca_: PCA | None = None
        self.scaler_ = StandardScaler()
        self.threshold_: float | None = None
        self.mean_: np.ndarray | None = None
        self.cov_: np.ndarray | None = None
        super().__init__(
            n_components=n_components,
            score_method=score_method,
            contamination=contamination,
            random_state=random_state,
        )
        self._fitted = False

    def fit(
        self,
        y: Union[np.ndarray, pd.Series, "SeriesLike"],
        X: np.ndarray | pd.DataFrame | None = None,
    ) -> "PCADetector":
        """Fit the PCA detector on healthy operation data.

        Args:
            y: Training data (target)
            X: Optional features (if None, uses y)

        Returns:
            Self for method chaining
        """
        # Use X if provided, otherwise use y
        if X is not None:
            if isinstance(X, pd.DataFrame):
                X_data = X.values
            else:
                X_data = X
            if X_data.ndim == 1:
                X_data = X_data.reshape(-1, 1)
        else:
            if isinstance(y, pd.Series):
                X_data = y.values.reshape(-1, 1)
            else:
                X_data = y.reshape(-1, 1) if y.ndim == 1 else y

        X_scaled = self.scaler_.fit_transform(X_data)

        # Fit PCA
        self.pca_ = PCA(n_components=self.n_components, random_state=self.random_state)
        X_pca = self.pca_.fit_transform(X_scaled)

        # Compute statistics in PC space for Mahalanobis distance
        self.mean_ = np.mean(X_pca, axis=0)
        self.cov_ = np.cov(X_pca.T)

        # Compute threshold based on training data
        scores = self._compute_scores(X_scaled, X_pca)
        self.threshold_ = np.percentile(scores, 100 * (1 - self.contamination))

        self._fitted = True
        logger.debug(f"Fitted PCADetector: threshold={self.threshold_}")
        return self

    def predict(self, y: np.ndarray | pd.Series) -> LabelView:
        """Predict anomaly labels.

        Args:
            y: Time series to detect anomalies in

        Returns:
            LabelView with binary labels (1 = anomaly, 0 = normal)
        """
        score_view = self.score(y)
        if self.threshold_ is None:
            raise ValueError("Detector must be fitted before prediction.")
        labels = (score_view.scores > self.threshold_).astype(int)

        return LabelView(index=score_view.index, labels=labels)

    def score(self, y: np.ndarray | pd.Series) -> ScoreView:
        """Score anomalies.

        Args:
            y: Time series to score

        Returns:
            ScoreView with anomaly scores
        """
        self._check_fitted()

        if isinstance(y, pd.Series):
            index = y.index
            values = y.values.reshape(-1, 1)
        else:
            index = pd.RangeIndex(start=0, stop=len(y))
            values = y.reshape(-1, 1) if y.ndim == 1 else y

        X_scaled = self.scaler_.transform(values)
        X_pca = self.pca_.transform(X_scaled)  # type: ignore

        scores = self._compute_scores(X_scaled, X_pca)

        return ScoreView(index=index, scores=scores)

    def _compute_scores(self, X_scaled: np.ndarray, X_pca: np.ndarray) -> np.ndarray:
        """Compute anomaly scores using specified method.

        Args:
            X_scaled: Scaled input data
            X_pca: Data transformed to principal component space

        Returns:
            Anomaly scores
        """
        if self.score_method == "reconstruction":
            # Reconstruction error
            X_reconstructed = self.pca_.inverse_transform(X_pca)  # type: ignore
            reconstruction_error = np.sum((X_scaled - X_reconstructed) ** 2, axis=1)
            return reconstruction_error

        elif self.score_method == "mahalanobis":
            # Mahalanobis distance in PC space
            if self.mean_ is None or self.cov_ is None:
                raise ValueError("PCA must be fitted before computing Mahalanobis distance.")

            # Vectorized Mahalanobis distance: sqrt((x - mu)^T * Sigma^-1 * (x - mu))
            diff = X_pca - self.mean_  # Shape: (n_samples, n_components)
            
            try:
                inv_cov = np.linalg.inv(self.cov_)
            except np.linalg.LinAlgError:
                # If covariance is singular, use pseudo-inverse
                inv_cov = np.linalg.pinv(self.cov_)

            # Vectorized: compute quadratic form for all samples at once
            # (diff @ inv_cov) * diff computes element-wise product, then sum over features
            quad_form = (diff @ inv_cov) * diff  # Shape: (n_samples, n_components)
            mahalanobis_dist = np.sqrt(np.sum(quad_form, axis=1))  # Shape: (n_samples,)
            
            return mahalanobis_dist

        elif self.score_method == "both":
            # Average of both methods
            recon_scores = self._compute_scores_with_method(X_scaled, X_pca, "reconstruction")
            maha_scores = self._compute_scores_with_method(X_scaled, X_pca, "mahalanobis")
            # Normalize and average
            recon_norm = (recon_scores - recon_scores.min()) / (
                recon_scores.max() - recon_scores.min() + 1e-10
            )
            maha_norm = (maha_scores - maha_scores.min()) / (
                maha_scores.max() - maha_scores.min() + 1e-10
            )
            return (recon_norm + maha_norm) / 2

        else:
            raise ValueError(f"Unknown score_method: {self.score_method}")

    def _compute_scores_with_method(
        self, X_scaled: np.ndarray, X_pca: np.ndarray, method: str
    ) -> np.ndarray:
        """Compute scores with a specific method."""
        if method == "reconstruction":
            X_reconstructed = self.pca_.inverse_transform(X_pca)  # type: ignore
            return np.sum((X_scaled - X_reconstructed) ** 2, axis=1)
        elif method == "mahalanobis":
            if self.mean_ is None or self.cov_ is None:
                raise ValueError("PCA must be fitted before computing Mahalanobis distance.")
            
            # Vectorized Mahalanobis distance calculation
            diff = X_pca - self.mean_  # Shape: (n_samples, n_components)
            
            try:
                inv_cov = np.linalg.inv(self.cov_)
            except np.linalg.LinAlgError:
                inv_cov = np.linalg.pinv(self.cov_)

            # Vectorized quadratic form computation
            quad_form = (diff @ inv_cov) * diff  # Shape: (n_samples, n_components)
            return np.sqrt(np.sum(quad_form, axis=1))  # Shape: (n_samples,)
        else:
            raise ValueError(f"Unknown method: {method}")

