"""Layer 2: Primitives.

This layer defines algorithm interfaces and thin utilities.
It must not know about tasks or evaluation.
Only numpy and pandas are allowed (no sklearn, matplotlib, etc.).
"""

from anomsmith.primitives.base import (
    BaseDetector,
    BaseEstimator,
    BaseObject,
    BaseScorer,
)
from anomsmith.primitives.model_persistence import (
    export_model_for_sagemaker,
    load_model,
    save_model,
)
from anomsmith.primitives.scaling import robust_zscore
from anomsmith.primitives.thresholding import ThresholdRule, apply_threshold

__all__ = [
    "BaseObject",
    "BaseEstimator",
    "BaseScorer",
    "BaseDetector",
    "ThresholdRule",
    "apply_threshold",
    "robust_zscore",
    "save_model",
    "load_model",
    "export_model_for_sagemaker",
]


