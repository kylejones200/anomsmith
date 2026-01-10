"""Layer 4: Workflows.

Workflows provide the public entry points users call.
Workflows can import matplotlib only if plots are added (not in first pass).
"""

from anomsmith.workflows.asset_health import assess_asset_health, rank_assets_by_risk
from anomsmith.workflows.batch_inference import batch_predict, batch_score, process_s3_batch
from anomsmith.workflows.detect import (
    detect_anomalies,
    report_detection,
    score_anomalies,
    sweep_thresholds,
)
from anomsmith.workflows.eval.backtest import backtest_detector
from anomsmith.workflows.eval.survival_metrics import (
    compute_concordance_index,
    evaluate_survival_model,
)
from anomsmith.workflows.model_monitoring import (
    ModelPerformanceTracker,
    aggregate_metrics_for_cloudwatch,
    compute_performance_metrics,
    detect_concept_drift,
)
from anomsmith.workflows.pca_pm import (
    assess_health_with_pca,
    classify_health_from_distance,
    compute_pca_health_thresholds,
    track_mahalanobis_distance,
)
from anomsmith.workflows.pm import apply_policy, discretize_rul, evaluate_policy
from anomsmith.workflows.survival import (
    compare_survival_models,
    fit_survival_model_for_maintenance,
    predict_health_states_from_survival,
    predict_rul_from_survival,
)

__all__ = [
    "score_anomalies",
    "detect_anomalies",
    "sweep_thresholds",
    "report_detection",
    "backtest_detector",
    "discretize_rul",
    "apply_policy",
    "evaluate_policy",
    "assess_asset_health",
    "rank_assets_by_risk",
    "batch_score",
    "batch_predict",
    "process_s3_batch",
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
]

