"""LSTM-based distress classifiers for predictive maintenance.

Deep learning models for sequence-based distress detection.
Requires TensorFlow/Keras (optional dependency).
"""

import logging
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import pandas as pd

try:
    from tensorflow import keras
    from tensorflow.keras import layers, models
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    keras = None  # type: ignore
    layers = None  # type: ignore
    models = None  # type: ignore
    EarlyStopping = None  # type: ignore
    ReduceLROnPlateau = None  # type: ignore

from anomsmith.objects.health_state import HealthState, HealthStateView

if TYPE_CHECKING:
    try:
        from timesmith.typing import SeriesLike
    except ImportError:
        SeriesLike = None

logger = logging.getLogger(__name__)


class LSTMDistressClassifier:
    """LSTM-based classifier for detecting distress in time series sequences.

    Uses Long Short-Term Memory networks to learn patterns in sequences of
    sensor readings that indicate distress.

    Args:
        window_size: Size of sliding window for sequences (default 30)
        lstm_units: Number of LSTM units (default 64)
        n_features: Number of features per timestep (default 3 for PCA components)
        epochs: Number of training epochs (default 20)
        batch_size: Batch size for training (default 32)
        random_state: Random state for reproducibility
    """

    def __init__(
        self,
        window_size: int = 30,
        lstm_units: int = 64,
        n_features: int = 3,
        epochs: int = 20,
        batch_size: int = 32,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize LSTMDistressClassifier.

        Args:
            window_size: Size of sliding window for sequences
            lstm_units: Number of LSTM units
            n_features: Number of features per timestep
            epochs: Number of training epochs
            batch_size: Batch size for training
            random_state: Random state for reproducibility
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError(
                "TensorFlow is required for LSTMDistressClassifier. "
                "Install with: pip install tensorflow"
            )

        self.window_size = window_size
        self.lstm_units = lstm_units
        self.n_features = n_features
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.model_: keras.Model | None = None  # type: ignore
        self._fitted = False

        if random_state is not None:
            keras.utils.set_random_seed(random_state)  # type: ignore

    def fit(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
        verbose: int = 0,
    ) -> "LSTMDistressClassifier":
        """Fit the LSTM classifier.

        Args:
            sequences: Array of shape (n_samples, window_size, n_features)
            labels: Array of shape (n_samples,) with binary labels (1 = distress, 0 = healthy)
            validation_data: Optional tuple of (X_val, y_val) for validation
            verbose: Verbosity level (0 = silent, 1 = progress)

        Returns:
            Self for method chaining
        """
        if sequences.ndim != 3:
            raise ValueError(f"Sequences must be 3D (n_samples, window_size, n_features), got shape {sequences.shape}")

        n_samples, window_size, n_features = sequences.shape
        if n_features != self.n_features:
            logger.warning(
                f"Expected {self.n_features} features, got {n_features}. "
                f"Updating n_features to {n_features}."
            )
            self.n_features = n_features

        # Build LSTM model
        self.model_ = models.Sequential([  # type: ignore
            layers.LSTM(self.lstm_units, input_shape=(window_size, n_features)),  # type: ignore
            layers.Dense(1, activation="sigmoid"),  # type: ignore
        ])
        self.model_.compile(  # type: ignore
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

        # Callbacks
        callbacks = []
        if validation_data is not None:
            callbacks.append(
                EarlyStopping(  # type: ignore
                    monitor="val_loss", patience=3, restore_best_weights=True
                )
            )
            callbacks.append(
                ReduceLROnPlateau(  # type: ignore
                    monitor="val_loss", factor=0.5, patience=2, verbose=0
                )
            )

        # Train model
        self.model_.fit(  # type: ignore
            sequences,
            labels,
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=False,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose,
        )

        self._fitted = True
        logger.debug(
            f"Fitted LSTMDistressClassifier: window_size={window_size}, "
            f"n_features={n_features}, n_sequences={n_samples}"
        )
        return self

    def predict_health_states(
        self,
        sequences: np.ndarray,
        index: pd.Index | None = None,
    ) -> HealthStateView:
        """Predict health states from sequences.

        Args:
            sequences: Array of shape (n_samples, window_size, n_features)
            index: Optional index for the health states

        Returns:
            HealthStateView with predicted health states
        """
        if not self._fitted:
            raise ValueError("Classifier must be fitted before prediction.")

        if sequences.ndim != 3:
            raise ValueError(f"Sequences must be 3D (n_samples, window_size, n_features), got shape {sequences.shape}")

        # Predict probabilities
        probas = self.model_.predict(sequences, verbose=0).flatten()  # type: ignore

        # Convert to health states: 0.5 threshold for Warning, 0.8 for Distress
        states = np.zeros(len(probas), dtype=int)
        states[probas > 0.8] = HealthState.DISTRESS
        states[(probas > 0.5) & (probas <= 0.8)] = HealthState.WARNING

        if index is None:
            index = pd.RangeIndex(start=0, stop=len(states))

        return HealthStateView(index=index, states=states)

    def predict_proba(self, sequences: np.ndarray) -> np.ndarray:
        """Predict distress probabilities.

        Args:
            sequences: Array of shape (n_samples, window_size, n_features)

        Returns:
            Array of shape (n_samples,) with distress probabilities
        """
        if not self._fitted:
            raise ValueError("Classifier must be fitted before prediction.")

        if sequences.ndim != 3:
            raise ValueError(f"Sequences must be 3D (n_samples, window_size, n_features), got shape {sequences.shape}")

        probas = self.model_.predict(sequences, verbose=0).flatten()  # type: ignore
        return probas


class AttentionLSTMDistressClassifier:
    """Attention-based LSTM classifier for detecting distress.

    Uses LSTM with attention mechanism to focus on important parts of the sequence.

    Args:
        window_size: Size of sliding window for sequences (default 30)
        lstm_units: Number of LSTM units (default 64)
        n_features: Number of features per timestep (default 3 for PCA components)
        epochs: Number of training epochs (default 20)
        batch_size: Batch size for training (default 32)
        random_state: Random state for reproducibility
    """

    def __init__(
        self,
        window_size: int = 30,
        lstm_units: int = 64,
        n_features: int = 3,
        epochs: int = 20,
        batch_size: int = 32,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize AttentionLSTMDistressClassifier."""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError(
                "TensorFlow is required for AttentionLSTMDistressClassifier. "
                "Install with: pip install tensorflow"
            )

        self.window_size = window_size
        self.lstm_units = lstm_units
        self.n_features = n_features
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.model_: keras.Model | None = None  # type: ignore
        self._fitted = False

        if random_state is not None:
            keras.utils.set_random_seed(random_state)  # type: ignore

    def fit(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
        verbose: int = 0,
    ) -> "AttentionLSTMDistressClassifier":
        """Fit the attention-based LSTM classifier.

        Args:
            sequences: Array of shape (n_samples, window_size, n_features)
            labels: Array of shape (n_samples,) with binary labels (1 = distress, 0 = healthy)
            validation_data: Optional tuple of (X_val, y_val) for validation
            verbose: Verbosity level (0 = silent, 1 = progress)

        Returns:
            Self for method chaining
        """
        if sequences.ndim != 3:
            raise ValueError(f"Sequences must be 3D (n_samples, window_size, n_features), got shape {sequences.shape}")

        n_samples, window_size, n_features = sequences.shape
        if n_features != self.n_features:
            logger.warning(
                f"Expected {self.n_features} features, got {n_features}. "
                f"Updating n_features to {n_features}."
            )
            self.n_features = n_features

        # Build attention-based LSTM model
        # Custom attention layer matching article approach
        import tensorflow as tf

        class AttentionWithWeights(layers.Layer):  # type: ignore
            def build(self, input_shape):
                self.W = self.add_weight(
                    shape=(input_shape[-1], 1), initializer="random_normal", trainable=True
                )

            def call(self, inputs):
                # Compute attention scores: inputs shape (batch, seq_len, hidden_dim)
                # W shape (hidden_dim, 1)
                # scores shape (batch, seq_len, 1)
                scores = tf.matmul(inputs, self.W)
                # Squeeze to (batch, seq_len)
                scores = tf.squeeze(scores, axis=-1)
                # Apply softmax to get attention weights
                weights = tf.nn.softmax(scores, axis=1)
                # Expand for broadcasting: (batch, seq_len, 1)
                weights_expanded = tf.expand_dims(weights, axis=-1)
                # Weighted sum: (batch, hidden_dim)
                context = tf.reduce_sum(inputs * weights_expanded, axis=1)
                return context, weights

        input_seq = layers.Input(shape=(window_size, n_features))  # type: ignore
        x = layers.LSTM(self.lstm_units, return_sequences=True)(input_seq)  # type: ignore

        # Apply attention
        context, attention_weights = AttentionWithWeights()(x)  # type: ignore

        output = layers.Dense(1, activation="sigmoid", name="pred")(context)  # type: ignore

        self.model_ = models.Model(  # type: ignore
            inputs=input_seq, outputs={"pred": output, "attn": attention_weights}
        )
        self.model_.compile(  # type: ignore
            optimizer="adam",
            loss={"pred": "binary_crossentropy"},
            metrics={"pred": "accuracy"},
        )

        # Callbacks
        callbacks = []
        if validation_data is not None:
            callbacks.append(
                EarlyStopping(  # type: ignore
                    monitor="val_loss", patience=3, restore_best_weights=True
                )
            )
            callbacks.append(
                ReduceLROnPlateau(  # type: ignore
                    monitor="val_loss", factor=0.5, patience=2, verbose=0
                )
            )

        # Prepare validation data dict
        val_dict = None
        if validation_data is not None:
            val_dict = {"pred": validation_data[1]}

        # Train model
        self.model_.fit(  # type: ignore
            sequences,
            {"pred": labels},
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=False,
            validation_data=(validation_data[0], val_dict) if validation_data else None,
            callbacks=callbacks,
            verbose=verbose,
        )

        self._fitted = True
        logger.debug(
            f"Fitted AttentionLSTMDistressClassifier: window_size={window_size}, "
            f"n_features={n_features}, n_sequences={n_samples}"
        )
        return self

    def predict_health_states(
        self,
        sequences: np.ndarray,
        index: pd.Index | None = None,
    ) -> HealthStateView:
        """Predict health states from sequences.

        Args:
            sequences: Array of shape (n_samples, window_size, n_features)
            index: Optional index for the health states

        Returns:
            HealthStateView with predicted health states
        """
        if not self._fitted:
            raise ValueError("Classifier must be fitted before prediction.")

        if sequences.ndim != 3:
            raise ValueError(f"Sequences must be 3D (n_samples, window_size, n_features), got shape {sequences.shape}")

        # Predict probabilities
        pred_dict = self.model_.predict(sequences, verbose=0)  # type: ignore
        probas = pred_dict["pred"].flatten()

        # Convert to health states: 0.5 threshold for Warning, 0.8 for Distress
        states = np.zeros(len(probas), dtype=int)
        states[probas > 0.8] = HealthState.DISTRESS
        states[(probas > 0.5) & (probas <= 0.8)] = HealthState.WARNING

        if index is None:
            index = pd.RangeIndex(start=0, stop=len(states))

        return HealthStateView(index=index, states=states)

    def predict_proba(self, sequences: np.ndarray) -> np.ndarray:
        """Predict distress probabilities.

        Args:
            sequences: Array of shape (n_samples, window_size, n_features)

        Returns:
            Array of shape (n_samples,) with distress probabilities
        """
        if not self._fitted:
            raise ValueError("Classifier must be fitted before prediction.")

        if sequences.ndim != 3:
            raise ValueError(f"Sequences must be 3D (n_samples, window_size, n_features), got shape {sequences.shape}")

        pred_dict = self.model_.predict(sequences, verbose=0)  # type: ignore
        probas = pred_dict["pred"].flatten()
        return probas

    def get_attention_weights(self, sequences: np.ndarray) -> np.ndarray:
        """Get attention weights for sequences.

        Args:
            sequences: Array of shape (n_samples, window_size, n_features)

        Returns:
            Array of shape (n_samples, window_size) with attention weights
        """
        if not self._fitted:
            raise ValueError("Classifier must be fitted before getting attention weights.")

        if sequences.ndim != 3:
            raise ValueError(f"Sequences must be 3D (n_samples, window_size, n_features), got shape {sequences.shape}")

        pred_dict = self.model_.predict(sequences, verbose=0)  # type: ignore
        return pred_dict["attn"]

