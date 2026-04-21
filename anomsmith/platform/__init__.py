"""Predictive maintenance *platform* layer (features, RUL helpers, alerts, ingestion).

This package consolidates the former standalone **Anomaly Detection Toolkit** workflow
code into the single ``anomsmith`` distribution. All **detector primitives** (PCA,
isolation forest, scorers, etc.) live under :mod:`anomsmith.primitives`; ``platform``
holds orchestration, dataset utilities, evaluation helpers, and optional matplotlib
visualizations that sit on top of those primitives.
"""

from anomsmith.platform.evaluation import (
    calculate_confusion_matrix_metrics,
    calculate_lead_time,
    compare_detectors,
    evaluate_detector,
)
from anomsmith.platform.predictive_maintenance import (
    Alert,
    AlertLevel,
    AlertSystem,
    DashboardVisualizer,
    FailureClassifier,
    FeatureExtractor,
    PredictiveMaintenanceSystem,
    RealTimeIngestion,
    RULEstimator,
    add_degradation_rates,
    add_rolling_statistics,
    calculate_rul,
    create_rul_labels,
    prepare_pm_features,
)
from anomsmith.platform.visualization import (
    plot_comparison_metrics,
    plot_pca_boundary,
    plot_reconstruction_error,
    plot_sensor_drift,
)

__all__ = [
    "calculate_confusion_matrix_metrics",
    "calculate_lead_time",
    "compare_detectors",
    "evaluate_detector",
    "Alert",
    "AlertLevel",
    "AlertSystem",
    "DashboardVisualizer",
    "FailureClassifier",
    "FeatureExtractor",
    "PredictiveMaintenanceSystem",
    "RealTimeIngestion",
    "RULEstimator",
    "add_degradation_rates",
    "add_rolling_statistics",
    "calculate_rul",
    "create_rul_labels",
    "prepare_pm_features",
    "plot_comparison_metrics",
    "plot_pca_boundary",
    "plot_reconstruction_error",
    "plot_sensor_drift",
]
