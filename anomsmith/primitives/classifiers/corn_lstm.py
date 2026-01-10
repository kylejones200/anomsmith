"""CORN (Continuous Ordinal Regression Networks) LSTM for ordinal classification.

Uses PyTorch with CORN loss function to train LSTM models that respect
the natural ordering of health states.
"""

import logging
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    nn = None  # type: ignore
    Dataset = None  # type: ignore
    DataLoader = None  # type: ignore

try:
    from coral_pytorch.losses import corn_loss
    from coral_pytorch.dataset import corn_label_from_logits

    CORAL_PYTORCH_AVAILABLE = True
except ImportError:
    CORAL_PYTORCH_AVAILABLE = False
    corn_loss = None  # type: ignore
    corn_label_from_logits = None  # type: ignore

from anomsmith.objects.health_state import HealthState, HealthStateView
from anomsmith.primitives.base import BaseEstimator

if TYPE_CHECKING:
    try:
        from timesmith.typing import SeriesLike
    except ImportError:
        SeriesLike = None

logger = logging.getLogger(__name__)

# Only define PyTorch-dependent classes if PyTorch is available
if TORCH_AVAILABLE:
    class SequenceDataset(Dataset):  # type: ignore
        """Dataset for time series sequences."""

        def __init__(self, sequences: np.ndarray, labels: np.ndarray):
            """Initialize sequence dataset.

            Args:
                sequences: Array of shape (n_samples, seq_len, n_features)
                labels: Array of shape (n_samples,) with ordinal labels (0, 1, 2)
            """
            self.X = torch.tensor(sequences, dtype=torch.float32)  # type: ignore
            self.y = torch.tensor(labels, dtype=torch.long)  # type: ignore

        def __len__(self) -> int:
            """Return dataset size."""
            return len(self.X)

        def __getitem__(self, idx: int) -> tuple:
            """Get item by index."""
            return self.X[idx], self.y[idx]


    class LSTMBackbone(nn.Module):  # type: ignore
        """LSTM backbone for sequence processing."""

        def __init__(self, input_size: int = 21, hidden_size: int = 64):
            """Initialize LSTM backbone.

            Args:
                input_size: Number of features per timestep
                hidden_size: Number of LSTM hidden units
            """
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)  # type: ignore
            self.hidden_size = hidden_size

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
            """Forward pass.

            Args:
                x: Input tensor of shape (batch, seq_len, input_size)

            Returns:
                Hidden state of shape (batch, hidden_size)
            """
            _, (h, _) = self.lstm(x)  # type: ignore
            return h[-1]  # Return last hidden state


    class CORNModel(nn.Module):  # type: ignore
        """CORN (Continuous Ordinal Regression Networks) model."""

        def __init__(self, input_size: int = 21, hidden_size: int = 64, num_classes: int = 3):
            """Initialize CORN model.

            Args:
                input_size: Number of features per timestep
                hidden_size: Number of LSTM hidden units
                num_classes: Number of ordinal classes (default 3 for health states)
            """
            super().__init__()
            self.backbone = LSTMBackbone(input_size, hidden_size)  # type: ignore
            # CORN output: num_classes - 1 thresholds
            self.out = nn.Linear(hidden_size, num_classes - 1)  # type: ignore

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
            """Forward pass.

            Args:
                x: Input tensor of shape (batch, seq_len, input_size)

            Returns:
                Logits of shape (batch, num_classes - 1) for ordinal thresholds
            """
            features = self.backbone(x)
            return self.out(features)
else:
    # Dummy classes when PyTorch is not available (won't be used due to __init__ check)
    SequenceDataset = None  # type: ignore
    LSTMBackbone = None  # type: ignore
    CORNModel = None  # type: ignore


class CORNLSTMClassifier(BaseEstimator):
    """CORN LSTM classifier for ordinal health state prediction.

    Uses Continuous Ordinal Regression Networks to predict ordered health states
    from time series sequences. Respects the natural ordering of classes.

    Args:
        seq_len: Sequence length (window size) (default 30)
        input_size: Number of features per timestep (default 21 for sensors)
        hidden_size: Number of LSTM hidden units (default 64)
        num_classes: Number of ordinal classes (default 3)
        epochs: Number of training epochs (default 10)
        batch_size: Batch size for training (default 64)
        learning_rate: Learning rate (default 0.001)
        random_state: Random state for reproducibility
    """

    def __init__(
        self,
        seq_len: int = 30,
        input_size: int = 21,
        hidden_size: int = 64,
        num_classes: int = 3,
        epochs: int = 10,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize CORN LSTM classifier."""
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for CORNLSTMClassifier. "
                "Install with: pip install torch"
            )
        if not CORAL_PYTORCH_AVAILABLE:
            raise ImportError(
                "coral-pytorch is required for CORNLSTMClassifier. "
                "Install with: pip install coral-pytorch"
            )

        super().__init__(
            seq_len=seq_len,
            input_size=input_size,
            hidden_size=hidden_size,
            num_classes=num_classes,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            random_state=random_state,
        )
        self.seq_len = seq_len
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model_: Optional[CORNModel] = None  # type: ignore
        self._fitted = False

        if random_state is not None:
            torch.manual_seed(random_state)  # type: ignore

    def fit(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        verbose: int = 0,
    ) -> "CORNLSTMClassifier":
        """Fit the CORN LSTM model.

        Args:
            sequences: Array of shape (n_samples, seq_len, input_size) or
                      (n_samples, seq_len) for univariate
            labels: Array of shape (n_samples,) with ordinal labels (0, 1, 2)

        Returns:
            Self for method chaining
        """
        if sequences.ndim == 2:
            # Univariate: add feature dimension
            sequences = sequences.reshape(sequences.shape[0], sequences.shape[1], 1)
        elif sequences.ndim != 3:
            raise ValueError(
                f"Sequences must be 2D or 3D, got shape {sequences.shape}"
            )

        n_samples, seq_len_actual, input_size_actual = sequences.shape

        if seq_len_actual != self.seq_len:
            logger.warning(
                f"Sequence length mismatch: expected {self.seq_len}, got {seq_len_actual}. "
                f"Updating seq_len to {seq_len_actual}."
            )
            self.seq_len = seq_len_actual

        if input_size_actual != self.input_size:
            logger.warning(
                f"Input size mismatch: expected {self.input_size}, got {input_size_actual}. "
                f"Updating input_size to {input_size_actual}."
            )
            self.input_size = input_size_actual

        labels_array = np.asarray(labels).astype(int)

        # Validate labels
        unique_vals = np.unique(labels_array)
        if not np.all(np.isin(unique_vals, range(self.num_classes))):
            raise ValueError(
                f"Labels must be in [0, {self.num_classes-1}], got unique values: {unique_vals}"
            )

        # Create dataset and dataloader
        dataset = SequenceDataset(sequences, labels_array)  # type: ignore
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)  # type: ignore

        # Create model
        self.model_ = CORNModel(  # type: ignore
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_classes=self.num_classes,
        )

        # Training setup
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)  # type: ignore
        self.model_.train()  # type: ignore

        # Training loop
        for epoch in range(self.epochs):
            total_loss = 0.0
            for X_batch, y_batch in loader:
                logits = self.model_(X_batch)  # type: ignore
                loss = corn_loss(logits, y_batch, self.num_classes)  # type: ignore
                optimizer.zero_grad()  # type: ignore
                loss.backward()  # type: ignore
                optimizer.step()  # type: ignore
                total_loss += loss.item()  # type: ignore

            if verbose > 0:
                logger.info(f"Epoch {epoch+1}/{self.epochs}: Loss = {total_loss:.4f}")

        self._fitted = True
        logger.debug(
            f"Fitted CORNLSTMClassifier: n_samples={n_samples}, "
            f"seq_len={self.seq_len}, input_size={self.input_size}"
        )
        return self

    def predict(self, sequences: np.ndarray) -> np.ndarray:
        """Predict health states from sequences.

        Args:
            sequences: Array of shape (n_samples, seq_len, input_size) or
                      (n_samples, seq_len) for univariate

        Returns:
            Array of predicted health states (0=Healthy, 1=Warning, 2=Distress)
        """
        self._check_fitted()

        if sequences.ndim == 2:
            sequences = sequences.reshape(sequences.shape[0], sequences.shape[1], 1)
        elif sequences.ndim != 3:
            raise ValueError(
                f"Sequences must be 2D or 3D, got shape {sequences.shape}"
            )

        self.model_.eval()  # type: ignore

        with torch.no_grad():  # type: ignore
            X_tensor = torch.tensor(sequences, dtype=torch.float32)  # type: ignore
            logits = self.model_(X_tensor)  # type: ignore
            # Convert logits to ordinal labels using CORN label function
            labels = corn_label_from_logits(logits).numpy()  # type: ignore

        return labels.astype(int)

    def predict_health_states(
        self, sequences: np.ndarray, index: Optional[pd.Index] = None
    ) -> HealthStateView:
        """Predict health states as HealthStateView.

        Args:
            sequences: Array of shape (n_samples, seq_len, input_size)
            index: Optional index for the health states

        Returns:
            HealthStateView with predicted health states
        """
        if index is None:
            index = pd.RangeIndex(start=0, stop=len(sequences))

        predictions = self.predict(sequences)
        return HealthStateView(index=index, states=predictions)

