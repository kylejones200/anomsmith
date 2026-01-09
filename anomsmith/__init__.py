"""Anomsmith: A strict 4-layer architecture for anomaly detection."""

from anomsmith.primitives.base import BaseDetector, BaseScorer
from anomsmith.primitives.detectors.ml import (
    IsolationForestDetector,
    LOFDetector,
    RobustCovarianceDetector,
)
from anomsmith.primitives.detectors.pca import PCADetector
from anomsmith.primitives.scorers.statistical import IQRScorer, ZScoreScorer
from anomsmith.primitives.thresholding import ThresholdRule
from anomsmith.workflows import (
    backtest_detector,
    detect_anomalies,
    score_anomalies,
    sweep_thresholds,
)

# Re-export timesmith types for convenience
try:
    from timesmith.typing import PanelLike, SeriesLike
    
    __all__ = [
        # Workflows
        "score_anomalies",
        "detect_anomalies",
        "sweep_thresholds",
        "backtest_detector",
        # Base classes
        "BaseScorer",
        "BaseDetector",
        "ThresholdRule",
        # Statistical scorers
        "ZScoreScorer",
        "IQRScorer",
        # ML detectors
        "IsolationForestDetector",
        "LOFDetector",
        "RobustCovarianceDetector",
        # PCA detector
        "PCADetector",
        # Timesmith types (now core to anomsmith)
        "SeriesLike",
        "PanelLike",
    ]
except ImportError:
    __all__ = [
        # Workflows
        "score_anomalies",
        "detect_anomalies",
        "sweep_thresholds",
        "backtest_detector",
        # Base classes
        "BaseScorer",
        "BaseDetector",
        "ThresholdRule",
        # Statistical scorers
        "ZScoreScorer",
        "IQRScorer",
        # ML detectors
        "IsolationForestDetector",
        "LOFDetector",
        "RobustCovarianceDetector",
        # PCA detector
        "PCADetector",
    ]

