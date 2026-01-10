"""Sequence-based classifiers for predictive maintenance."""

from anomsmith.primitives.classifiers.failure_risk import FailureRiskClassifier
from anomsmith.primitives.classifiers.sequence import (
    SequenceDistressClassifier,
    create_sequences,
)

try:
    from anomsmith.primitives.classifiers.lstm import (
        LSTMDistressClassifier,
        AttentionLSTMDistressClassifier,
    )
    _TENSORFLOW_AVAILABLE = True
except ImportError:
    _TENSORFLOW_AVAILABLE = False

try:
    from anomsmith.primitives.classifiers.ordinal import OrdinalLogisticClassifier
    _MORD_AVAILABLE = True
except ImportError:
    _MORD_AVAILABLE = False

try:
    from anomsmith.primitives.classifiers.lightgbm_ordinal import LightGBMOrdinalClassifier
    _LIGHTGBM_AVAILABLE = True
except ImportError:
    _LIGHTGBM_AVAILABLE = False

try:
    from anomsmith.primitives.classifiers.corn_lstm import CORNLSTMClassifier
    _CORAL_PYTORCH_AVAILABLE = True
except ImportError:
    _CORAL_PYTORCH_AVAILABLE = False

try:
    from anomsmith.primitives.classifiers.ensemble_ordinal import (
        AveragingOrdinalEnsemble,
        StackedOrdinalEnsemble,
    )
    _ENSEMBLE_AVAILABLE = True
except ImportError:
    _ENSEMBLE_AVAILABLE = False

__all__ = [
    "SequenceDistressClassifier",
    "create_sequences",
    "FailureRiskClassifier",
]

if _TENSORFLOW_AVAILABLE:
    __all__.extend(["LSTMDistressClassifier", "AttentionLSTMDistressClassifier"])

if _MORD_AVAILABLE:
    __all__.append("OrdinalLogisticClassifier")

if _LIGHTGBM_AVAILABLE:
    __all__.append("LightGBMOrdinalClassifier")

if _CORAL_PYTORCH_AVAILABLE:
    __all__.append("CORNLSTMClassifier")

if _ENSEMBLE_AVAILABLE:
    __all__.extend(["AveragingOrdinalEnsemble", "StackedOrdinalEnsemble"])

