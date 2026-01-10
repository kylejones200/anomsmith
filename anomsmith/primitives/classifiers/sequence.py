"""Sequence-based distress classifiers for predictive maintenance.

Classifies whether a sequence of sensor readings indicates distress.
"""

import logging
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from anomsmith.objects.health_state import HealthState, HealthStateView

if TYPE_CHECKING:
    try:
        from timesmith.typing import SeriesLike
    except ImportError:
        SeriesLike = None

logger = logging.getLogger(__name__)


class SequenceDistressClassifier:
    """Sequence-based classifier for detecting distress in time series.

    Uses a sliding window approach to classify whether a sequence of sensor
    readings indicates distress (health state Warning or Distress).

    Args:
        window_size: Size of sliding window for sequences (default 30)
        n_estimators: Number of trees in Random Forest (default 100)
        random_state: Random state for reproducibility
    """

    def __init__(
        self,
        window_size: int = 30,
        n_estimators: int = 100,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize SequenceDistressClassifier.

        Args:
            window_size: Size of sliding window for sequences
            n_estimators: Number of trees in Random Forest
            random_state: Random state for reproducibility
        """
        self.window_size = window_size
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.classifier_ = RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state
        )
        self.scaler_ = StandardScaler()
        self._fitted = False

    def fit(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
    ) -> "SequenceDistressClassifier":
        """Fit the sequence classifier.

        Args:
            sequences: Array of shape (n_samples, window_size) or (n_samples, window_size, n_features)
            labels: Array of shape (n_samples,) with binary labels (1 = distress, 0 = healthy)

        Returns:
            Self for method chaining
        """
        if sequences.ndim == 2:
            # Univariate sequences: (n_samples, window_size)
            X_flat = sequences
        elif sequences.ndim == 3:
            # Multivariate sequences: (n_samples, window_size, n_features)
            # Flatten to (n_samples, window_size * n_features)
            n_samples, window_size, n_features = sequences.shape
            X_flat = sequences.reshape(n_samples, window_size * n_features)
        else:
            raise ValueError(f"Sequences must be 2D or 3D, got shape {sequences.shape}")

        X_scaled = self.scaler_.fit_transform(X_flat)
        self.classifier_.fit(X_scaled, labels)
        self._fitted = True
        logger.debug(
            f"Fitted SequenceDistressClassifier: window_size={self.window_size}, "
            f"n_sequences={len(sequences)}"
        )
        return self

    def predict_health_states(
        self,
        sequences: np.ndarray,
        index: pd.Index | None = None,
    ) -> HealthStateView:
        """Predict health states from sequences.

        Args:
            sequences: Array of shape (n_samples, window_size) or (n_samples, window_size, n_features)
            index: Optional index for the health states

        Returns:
            HealthStateView with predicted health states
        """
        if not self._fitted:
            raise ValueError("Classifier must be fitted before prediction.")

        if sequences.ndim == 2:
            X_flat = sequences
        elif sequences.ndim == 3:
            n_samples, window_size, n_features = sequences.shape
            X_flat = sequences.reshape(n_samples, window_size * n_features)
        else:
            raise ValueError(f"Sequences must be 2D or 3D, got shape {sequences.shape}")

        X_scaled = self.scaler_.transform(X_flat)
        # Predict probabilities
        probas = self.classifier_.predict_proba(X_scaled)
        # Convert to health states: if probability of distress > 0.5, flag as Warning (1)
        # We'll use 0.5 threshold for Warning, 0.8 for Distress
        states = np.zeros(len(probas), dtype=int)
        if probas.shape[1] > 1:
            distress_proba = probas[:, 1]  # Probability of distress class
            states[distress_proba > 0.8] = HealthState.DISTRESS
            states[(distress_proba > 0.5) & (distress_proba <= 0.8)] = HealthState.WARNING
        else:
            # Binary case
            states[probas[:, 0] > 0.5] = HealthState.WARNING

        if index is None:
            index = pd.RangeIndex(start=0, stop=len(states))

        return HealthStateView(index=index, states=states)

    def predict_proba(self, sequences: np.ndarray) -> np.ndarray:
        """Predict distress probabilities.

        Args:
            sequences: Array of shape (n_samples, window_size) or (n_samples, window_size, n_features)

        Returns:
            Array of shape (n_samples, n_classes) with class probabilities
        """
        if not self._fitted:
            raise ValueError("Classifier must be fitted before prediction.")

        if sequences.ndim == 2:
            X_flat = sequences
        elif sequences.ndim == 3:
            n_samples, window_size, n_features = sequences.shape
            X_flat = sequences.reshape(n_samples, window_size * n_features)
        else:
            raise ValueError(f"Sequences must be 2D or 3D, got shape {sequences.shape}")

        X_scaled = self.scaler_.transform(X_flat)
        return self.classifier_.predict_proba(X_scaled)


def create_sequences(
    df: pd.DataFrame,
    sensor_cols: list[str],
    target_col: str,
    window_size: int = 30,
    entity_col: str = "unit",
    time_col: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create sequences from time series data.

    Args:
        df: DataFrame with sensor data
        sensor_cols: List of sensor column names
        target_col: Target column name (e.g., 'distress' or 'health_state')
        window_size: Size of sliding window
        entity_col: Column name for entity ID (e.g., 'unit')
        time_col: Optional column name for time ordering (if None, assumes sorted by entity)

    Returns:
        Tuple of (sequences, labels) where:
        - sequences: Array of shape (n_samples, window_size, n_features)
        - labels: Array of shape (n_samples,) with binary labels (1 = distress, 0 = healthy)
    """
    sequences = []
    labels = []

    for entity_id, group in df.groupby(entity_col):
        if time_col:
            group = group.sort_values(time_col)
        else:
            group = group.sort_index()

        values = group[sensor_cols].values
        targets = group[target_col].values

        # Create sequences
        for i in range(len(values) - window_size):
            sequences.append(values[i : i + window_size])
            labels.append(targets[i + window_size])

    sequences_array = np.array(sequences)
    labels_array = np.array(labels)

    # Convert to binary if needed (distress = 1, healthy = 0)
    if labels_array.dtype != int:
        labels_array = (labels_array > 0).astype(int)

    return sequences_array, labels_array

