"""Named defaults for policy, health-state, and algorithm parameters.

Library code should import from here instead of scattering raw numeric literals.
Binary label values (0/1) and array axis indices are not duplicated here.
"""

from __future__ import annotations

# --- Health state probability thresholds (failure / distress classifiers) ---
DEFAULT_FAILURE_PROBA_WARNING_THRESHOLD: float = 0.5
DEFAULT_FAILURE_PROBA_DISTRESS_THRESHOLD: float = 0.8

# --- RUL discretization (same units as RUL / time-to-failure) ---
DEFAULT_RUL_HEALTHY_THRESHOLD: float = 30.0
DEFAULT_RUL_WARNING_THRESHOLD: float = 10.0

# --- Maintenance policy (illustrative defaults; tune per domain) ---
DEFAULT_POLICY_INTERVENE_COST: float = 100.0
DEFAULT_POLICY_REVIEW_COST: float = 30.0
DEFAULT_POLICY_WAIT_COST: float = 0.0
DEFAULT_POLICY_BASE_RISK_HEALTHY: float = 0.01
DEFAULT_POLICY_BASE_RISK_WARNING: float = 0.1
DEFAULT_POLICY_BASE_RISK_DISTRESS: float = 0.3
DEFAULT_POLICY_BASE_RISKS: tuple[float, float, float] = (
    DEFAULT_POLICY_BASE_RISK_HEALTHY,
    DEFAULT_POLICY_BASE_RISK_WARNING,
    DEFAULT_POLICY_BASE_RISK_DISTRESS,
)
DEFAULT_POLICY_INTERVENE_RISK_REDUCTION: float = 0.5
DEFAULT_POLICY_REVIEW_RISK_REDUCTION: float = 0.75

# --- Survival curves: survival probability at or below which median TTF is read ---
DEFAULT_SURVIVAL_PROBABILITY_AT_MEDIAN_TTF: float = 0.5

# --- Outlier / anomaly detectors ---
DEFAULT_OUTLIER_CONTAMINATION: float = 0.05
DEFAULT_PCA_VARIANCE_FRACTION: float = 0.95
DEFAULT_ISOLATION_FOREST_N_ESTIMATORS: int = 200
DEFAULT_LOF_N_NEIGHBORS: int = 20
DEFAULT_ELLIPTIC_ENVELOPE_SUPPORT_FRACTION: float = 0.8

# --- Asset health composite (classification vs anomaly score fusion) ---
DEFAULT_ASSET_HEALTH_CLASSIFICATION_WEIGHT: float = 0.6
DEFAULT_ASSET_HEALTH_ANOMALY_WEIGHT: float = 0.4
FUSION_WEIGHT_SUM_ABSOLUTE_TOLERANCE: float = 1e-9

# --- PCA predictive maintenance: percentile thresholds on training distances ---
DEFAULT_PCA_HEALTHY_DISTANCE_PERCENTILE: float = 75.0
DEFAULT_PCA_WARNING_DISTANCE_PERCENTILE: float = 95.0

# --- Random Forest (failure risk classifier; workflow n_estimators) ---
DEFAULT_RANDOM_FOREST_N_ESTIMATORS: int = 100

# --- Drift / monitoring ---
DEFAULT_DRIFT_DETECTION_STDDEV_THRESHOLD: float = 2.0
DEFAULT_MODEL_METRIC_DEGRADATION_THRESHOLD: float = 0.1

# --- Statistical methods ---
DEFAULT_CONFIDENCE_ALPHA: float = 0.05
TUKEY_FENCE_IQR_MULTIPLIER: float = 1.5
DEFAULT_UNIT_SCALE_FOR_ZERO_SPREAD: float = 1.0

# --- Scoring scale (binary IQR path) ---
IQR_BINARY_OUTLIER_SCORE: float = 1.0
IQR_BINARY_INLIER_SCORE: float = 0.0

# --- Division-by-zero guards ---
NUMERICAL_EPSILON: float = 1e-10

# --- Regularization / training defaults ---
DEFAULT_ORDINAL_REGULARIZATION_ALPHA: float = 0.0
DEFAULT_NEURAL_LEARNING_RATE: float = 0.001
DEFAULT_LIGHTGBM_LEARNING_RATE: float = 0.1
DEFAULT_LIGHTGBM_N_ESTIMATORS: int = 100

# --- Change-point detector ---
DEFAULT_CHANGE_POINT_WINDOW_SIZE: int = 10
DEFAULT_CHANGE_POINT_THRESHOLD_MULTIPLIER: float = 3.0

# --- CORN-LSTM architecture / training defaults ---
DEFAULT_CORN_LSTM_SEQ_LEN: int = 30
DEFAULT_CORN_LSTM_INPUT_SIZE: int = 21
DEFAULT_CORN_LSTM_HIDDEN_SIZE: int = 64
DEFAULT_CORN_LSTM_NUM_CLASSES: int = 3
DEFAULT_CORN_LSTM_EPOCHS: int = 10
DEFAULT_CORN_LSTM_BATCH_SIZE: int = 64

# --- Neural survival (pycox) ---
DEFAULT_NEURAL_SURVIVAL_N_BINS: int = 50
DEFAULT_NEURAL_SURVIVAL_HIDDEN_LAYERS: tuple[int, ...] = (32, 32)
DEFAULT_NEURAL_SURVIVAL_BATCH_SIZE: int = 128
DEFAULT_NEURAL_SURVIVAL_EPOCHS: int = 50
DEFAULT_SURVIVAL_CURVE_EXTRAPOLATION_FILL: float = 1.0

# --- lifelines CoxPH ---
DEFAULT_LIFELINES_PENALIZER: float = 0.0
DEFAULT_LIFELINES_L1_RATIO: float = 0.0

# --- Reporting ---
DEFAULT_DETECTION_REPORT_TOP_ANOMALIES: int = 10
