"""Anomsmith: Anomaly detection workflows that turn time series signals into actionable decisions."""

from anomsmith.primitives.base import BaseDetector, BaseScorer
from anomsmith.primitives.detectors.ml import (
    IsolationForestDetector,
    LOFDetector,
    RobustCovarianceDetector,
)
from anomsmith.primitives.detectors.ensemble import VotingEnsembleDetector
from anomsmith.primitives.detectors.drift import ARIMADriftDetector
from anomsmith.primitives.detectors.pca import PCADetector
from anomsmith.primitives.detectors.wavelet import WaveletDetector
from anomsmith.primitives.scorers.seasonal import SeasonalBaselineScorer
from anomsmith.primitives.scorers.statistical import IQRScorer, ZScoreScorer
from anomsmith.primitives.thresholding import ThresholdRule
from anomsmith.workflows import (
    ModelPerformanceTracker,
    aggregate_metrics_for_cloudwatch,
    apply_policy,
    assess_asset_health,
    assess_health_with_pca,
    backtest_detector,
    batch_predict,
    batch_score,
    classify_health_from_distance,
    compare_survival_models,
    compute_concordance_index,
    compute_pca_health_thresholds,
    compute_performance_metrics,
    detect_anomalies,
    detect_concept_drift,
    discretize_rul,
    evaluate_policy,
    evaluate_survival_model,
    fit_survival_model_for_maintenance,
    predict_health_states_from_survival,
    predict_rul_from_survival,
    rank_assets_by_risk,
    score_anomalies,
    sweep_thresholds,
    track_mahalanobis_distance,
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
        "discretize_rul",
        "apply_policy",
        "evaluate_policy",
        "assess_asset_health",
        "rank_assets_by_risk",
        "batch_score",
        "batch_predict",
        "compute_performance_metrics",
        "detect_concept_drift",
        "aggregate_metrics_for_cloudwatch",
        "ModelPerformanceTracker",
        "fit_survival_model_for_maintenance",
        "predict_rul_from_survival",
        "predict_health_states_from_survival",
        "compare_survival_models",
        "compute_concordance_index",
        "evaluate_survival_model",
        "track_mahalanobis_distance",
        "classify_health_from_distance",
        "assess_health_with_pca",
        "compute_pca_health_thresholds",
        # Base classes
        "BaseScorer",
        "BaseDetector",
        "ThresholdRule",
        # Statistical scorers
        "ZScoreScorer",
        "IQRScorer",
        "SeasonalBaselineScorer",
        # ML detectors
        "IsolationForestDetector",
        "LOFDetector",
        "RobustCovarianceDetector",
        # PCA detector
        "PCADetector",
        # Wavelet detector
        "WaveletDetector",
        # Drift detector
        "ARIMADriftDetector",
        # Ensemble detector
        "VotingEnsembleDetector",
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
        "discretize_rul",
        "apply_policy",
        "evaluate_policy",
        "assess_asset_health",
        "rank_assets_by_risk",
        "batch_score",
        "batch_predict",
        "compute_performance_metrics",
        "detect_concept_drift",
        "aggregate_metrics_for_cloudwatch",
        "ModelPerformanceTracker",
        "fit_survival_model_for_maintenance",
        "predict_rul_from_survival",
        "predict_health_states_from_survival",
        "compare_survival_models",
        "compute_concordance_index",
        "evaluate_survival_model",
        "track_mahalanobis_distance",
        "classify_health_from_distance",
        "assess_health_with_pca",
        "compute_pca_health_thresholds",
        # Base classes
        "BaseScorer",
        "BaseDetector",
        "ThresholdRule",
        # Statistical scorers
        "ZScoreScorer",
        "IQRScorer",
        "SeasonalBaselineScorer",
        # ML detectors
        "IsolationForestDetector",
        "LOFDetector",
        "RobustCovarianceDetector",
        # PCA detector
        "PCADetector",
        # Wavelet detector
        "WaveletDetector",
        # Drift detector
        "ARIMADriftDetector",
        # Ensemble detector
        "VotingEnsembleDetector",
    ]

