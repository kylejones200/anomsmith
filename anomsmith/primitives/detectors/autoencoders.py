"""Autoencoder-based anomaly detectors (optional TensorFlow / PyTorch).

Migrated from the former *Anomaly Detection Toolkit*. Install ``anomsmith[deep]`` for
Keras (LSTM) and/or PyTorch backends.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from anomsmith.objects.views import LabelView, ScoreView
from anomsmith.primitives.base import BaseDetector
from anomsmith.primitives.detectors._utils import extract_index_and_values

if TYPE_CHECKING:
    try:
        from timesmith.typing import SeriesLike
    except ImportError:
        SeriesLike = None

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment,misc]
    nn = None  # type: ignore[assignment,misc]

try:
    from keras.callbacks import EarlyStopping
    from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
    from keras.models import Sequential

    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    EarlyStopping = None  # type: ignore[assignment,misc]
    LSTM = None  # type: ignore[assignment,misc]
    Dense = None  # type: ignore[assignment,misc]
    RepeatVector = None  # type: ignore[assignment,misc]
    TimeDistributed = None  # type: ignore[assignment,misc]
    Sequential = None  # type: ignore[assignment,misc]


def _univariate_series(
    y: np.ndarray | pd.Series | SeriesLike,
) -> tuple[pd.Index, np.ndarray]:
    """Return index and 1D float values for univariate input."""
    if isinstance(y, pd.Series):
        index, values = extract_index_and_values(y)
        flat = np.asarray(values, dtype=float).ravel()
        return index, flat
    arr = np.asarray(y, dtype=float)
    if arr.ndim > 1 and arr.shape[1] > 1:
        raise ValueError("Autoencoder detectors only support univariate series.")
    flat = arr.ravel()
    index = pd.RangeIndex(start=0, stop=flat.shape[0])
    return index, flat


class LSTMAutoencoderDetector(BaseDetector):
    """LSTM autoencoder: high reconstruction error ⇒ anomaly (univariate only)."""

    def __init__(
        self,
        window_size: int = 20,
        lstm_units: list[int] | None = None,
        contamination: float = 0.05,
        threshold_std: float = 3.0,
        epochs: int = 50,
        batch_size: int = 32,
        random_state: int | None = None,
    ) -> None:
        units = lstm_units or [32, 16]
        self.window_size = window_size
        self.lstm_units = units
        self.contamination = contamination
        self.threshold_std = threshold_std
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_: Sequential | None = None  # type: ignore[valid-type]
        self.scaler_ = MinMaxScaler()
        self.reconstruction_errors_: np.ndarray | None = None
        super().__init__(
            window_size=window_size,
            lstm_units=units,
            contamination=contamination,
            threshold_std=threshold_std,
            epochs=epochs,
            batch_size=batch_size,
            random_state=random_state,
        )

    def _create_windows(self, data: np.ndarray) -> np.ndarray:
        if len(data) < self.window_size:
            return np.empty((0, self.window_size, 1))
        if data.ndim > 1:
            data = data.flatten()
        n_windows = len(data) - self.window_size + 1
        indices = np.arange(self.window_size) + np.arange(n_windows)[:, np.newaxis]
        windows = data[indices]
        return windows[:, :, np.newaxis]

    def _create_model(self, input_shape: tuple[int, int]) -> Sequential:  # type: ignore[valid-type]
        if not KERAS_AVAILABLE or Sequential is None or LSTM is None:
            raise ImportError("Keras/TensorFlow is required for LSTMAutoencoderDetector.")
        if Dense is None or RepeatVector is None or TimeDistributed is None:
            raise ImportError("Keras/TensorFlow is required for LSTMAutoencoderDetector.")
        model = Sequential(
            [
                LSTM(
                    self.lstm_units[0],
                    activation="relu",
                    input_shape=input_shape,
                    return_sequences=True,
                ),
                LSTM(self.lstm_units[1], activation="relu", return_sequences=False),
                RepeatVector(self.window_size),
                LSTM(self.lstm_units[1], activation="relu", return_sequences=True),
                LSTM(self.lstm_units[0], activation="relu", return_sequences=True),
                TimeDistributed(Dense(1)),
            ]
        )
        model.compile(optimizer="adam", loss="mse")
        return model

    def _raw_window_scores(self, flat: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise ValueError("LSTMAutoencoderDetector must be fitted before scoring.")
        X_scaled = self.scaler_.transform(flat.reshape(-1, 1)).flatten()
        X_windows = self._create_windows(X_scaled)
        if len(X_windows) == 0:
            return np.array([])
        X_pred = self.model_.predict(X_windows, verbose=0)
        return np.mean(np.abs(X_windows - X_pred), axis=(1, 2))

    def fit(
        self,
        y: np.ndarray | pd.Series | SeriesLike,
        X: np.ndarray | pd.DataFrame | None = None,
    ) -> LSTMAutoencoderDetector:
        if not KERAS_AVAILABLE or Sequential is None or EarlyStopping is None:
            raise ImportError(
                "Keras/TensorFlow is required for LSTMAutoencoderDetector. "
                "Install with: pip install 'anomsmith[deep]'"
            )
        _, flat = _univariate_series(y)
        X_scaled = self.scaler_.fit_transform(flat.reshape(-1, 1)).flatten()
        X_windows = self._create_windows(X_scaled)
        if len(X_windows) == 0:
            raise ValueError(f"Series too short for window_size={self.window_size}")

        self.model_ = self._create_model((self.window_size, 1))
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=0
        )
        self.model_.fit(
            X_windows,
            X_windows,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0,
        )
        self._fitted = True
        logger.debug("Fitted LSTMAutoencoderDetector")
        return self

    def score(self, y: np.ndarray | pd.Series | SeriesLike) -> ScoreView:
        self._check_fitted()
        index, flat = _univariate_series(y)
        re = self._raw_window_scores(flat)
        self.reconstruction_errors_ = re
        full: np.ndarray = np.zeros(len(flat), dtype=float)
        if re.size > 0:
            full[self.window_size - 1 :] = re
        return ScoreView(index=index, scores=full)

    def predict(self, y: np.ndarray | pd.Series | SeriesLike) -> LabelView:
        sv = self.score(y)
        s = np.asarray(sv.scores, dtype=float)
        active = s[self.window_size - 1 :]
        if active.size == 0:
            return LabelView(index=sv.index, labels=np.zeros(len(s), dtype=int))
        mean = float(np.mean(active))
        std = float(np.std(active))
        std = std if std > 0 else 1.0
        thr = mean + self.threshold_std * std
        labels = (s > thr).astype(int)
        return LabelView(index=sv.index, labels=labels)


class PyTorchAutoencoderDetector(BaseDetector):
    """Feedforward autoencoder on sliding windows (PyTorch, univariate only)."""

    def __init__(
        self,
        window_size: int = 24,
        hidden_dims: list[int] | None = None,
        learning_rate: float = 1e-3,
        epochs: int = 200,
        batch_size: int = 32,
        threshold_std: float = 3.0,
        random_state: int | None = None,
    ) -> None:
        dims = hidden_dims or [64, 16, 4]
        self.window_size = window_size
        self.hidden_dims = dims
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.threshold_std = threshold_std
        if TORCH_AVAILABLE and torch is not None:
            self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device_ = None
        self.model_: nn.Module | None = None
        self.reconstruction_errors_: np.ndarray | None = None
        self.X_mean_: float | None = None
        self.X_std_: float | None = None
        if TORCH_AVAILABLE and torch is not None and random_state is not None:
            torch.manual_seed(random_state)
        super().__init__(
            window_size=window_size,
            hidden_dims=dims,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            threshold_std=threshold_std,
            random_state=random_state,
        )

    def _create_windows(self, data: np.ndarray) -> np.ndarray:
        if len(data) < self.window_size:
            return np.empty((0, self.window_size))
        n_windows = len(data) - self.window_size + 1
        shape = (n_windows, self.window_size)
        strides = (data.strides[0], data.strides[0])
        return np.lib.stride_tricks.as_strided(
            data, shape=shape, strides=strides, writeable=False
        ).copy()

    def _create_model(self, input_dim: int) -> nn.Module:
        if not TORCH_AVAILABLE or torch is None or nn is None:
            raise ImportError("PyTorch is required.")

        class Autoencoder(nn.Module):
            def __init__(self, in_dim: int, hidden_dims: list[int]) -> None:
                super().__init__()
                encoder_layers: list[nn.Module] = []
                prev_dim = in_dim
                for hidden_dim in hidden_dims:
                    encoder_layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU()])
                    prev_dim = hidden_dim
                decoder_layers: list[nn.Module] = []
                for i in range(len(hidden_dims) - 2, -1, -1):
                    decoder_layers.extend(
                        [nn.Linear(hidden_dims[i + 1], hidden_dims[i]), nn.ReLU()]
                    )
                decoder_layers.append(nn.Linear(hidden_dims[0], in_dim))
                self.encoder = nn.Sequential(*encoder_layers)
                self.decoder = nn.Sequential(*decoder_layers)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                z = self.encoder(x)
                return self.decoder(z)

        return Autoencoder(input_dim, self.hidden_dims).to(self.device_)

    def fit(
        self,
        y: np.ndarray | pd.Series | SeriesLike,
        X: np.ndarray | pd.DataFrame | None = None,
    ) -> PyTorchAutoencoderDetector:
        if not TORCH_AVAILABLE or torch is None or nn is None:
            raise ImportError(
                "PyTorch is required for PyTorchAutoencoderDetector. "
                "Install with: pip install 'anomsmith[deep]'"
            )
        if self.device_ is None:
            self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _, flat = _univariate_series(y)
        x_mean = float(np.mean(flat))
        x_std = float(np.std(flat))
        x_std = x_std if x_std > 0 else 1.0
        X_normalized = (flat - x_mean) / x_std
        X_windows = self._create_windows(X_normalized)
        if len(X_windows) == 0:
            raise ValueError(f"Series too short for window_size={self.window_size}")
        n = len(X_windows)
        lo, hi = int(0.1 * n), int(0.9 * n)
        X_train = X_windows[lo:hi]
        self.model_ = self._create_model(self.window_size)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()
        from torch.utils.data import DataLoader, TensorDataset

        dataset = TensorDataset(torch.from_numpy(X_train).float())
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.model_.train()
        for _ in range(self.epochs):
            for (batch,) in dataloader:
                batch = batch.to(self.device_)
                optimizer.zero_grad()
                recon = self.model_(batch)
                loss = loss_fn(recon, batch)
                loss.backward()
                optimizer.step()
        self.X_mean_ = x_mean
        self.X_std_ = x_std
        self._fitted = True
        logger.debug("Fitted PyTorchAutoencoderDetector")
        return self

    def score(self, y: np.ndarray | pd.Series | SeriesLike) -> ScoreView:
        self._check_fitted()
        if (
            self.model_ is None
            or self.X_mean_ is None
            or self.X_std_ is None
            or torch is None
        ):
            raise ValueError("PyTorchAutoencoderDetector is not fitted.")
        index, flat = _univariate_series(y)
        x_std = self.X_std_ if self.X_std_ > 0 else 1.0
        X_normalized = (flat - self.X_mean_) / x_std
        X_windows = self._create_windows(X_normalized)
        full: np.ndarray = np.zeros(len(flat), dtype=float)
        if len(X_windows) == 0:
            return ScoreView(index=index, scores=full)
        self.model_.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X_windows).float().to(self.device_)
            X_recon = self.model_(X_tensor).cpu().numpy()
        reconstruction_errors = np.mean((X_windows - X_recon) ** 2, axis=1)
        self.reconstruction_errors_ = reconstruction_errors
        full[self.window_size - 1 :] = reconstruction_errors
        return ScoreView(index=index, scores=full)

    def predict(self, y: np.ndarray | pd.Series | SeriesLike) -> LabelView:
        sv = self.score(y)
        s = np.asarray(sv.scores, dtype=float)
        active = s[self.window_size - 1 :]
        if active.size == 0:
            return LabelView(index=sv.index, labels=np.zeros(len(s), dtype=int))
        mean = float(np.mean(active))
        std = float(np.std(active))
        std = std if std > 0 else 1.0
        thr = mean + self.threshold_std * std
        labels = (s > thr).astype(int)
        return LabelView(index=sv.index, labels=labels)
