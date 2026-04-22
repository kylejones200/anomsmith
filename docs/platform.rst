Predictive maintenance platform
=================================

The ``anomsmith.platform`` subpackage is the **operational PM layer**: rolling and
spectral feature extraction, RandomForest-based RUL and failure models, threshold
alerts, streaming ingestion, and matplotlib dashboards. It was merged from the
former standalone **Anomaly Detection Toolkit** (historically packaged as
``anomaly_detection_toolkit``) so that one install delivers both low-level detectors
and higher-level maintenance workflows.

Detectors and scorers remain in ``anomsmith.primitives`` (single
:class:`~anomsmith.primitives.base.BaseDetector` hierarchy with ``LabelView`` /
``ScoreView`` outputs). The platform stack calls those detectors when you pass an
optional ``anomaly_detector`` into :class:`~anomsmith.platform.predictive_maintenance.PredictiveMaintenanceSystem`.

Key modules
-----------

* ``anomsmith.platform.predictive_maintenance`` — ``FeatureExtractor``, ``RULEstimator``,
  ``FailureClassifier``, ``AlertSystem``, ``PredictiveMaintenanceSystem``,
  ``RealTimeIngestion``, ``DashboardVisualizer``, and dataset helpers
  (``calculate_rul``, ``prepare_pm_features``, …).
* ``anomsmith.platform.evaluation`` — ``evaluate_detector``, ``compare_detectors``,
  lead-time and confusion helpers using the same label convention as primitives
  (``1`` = anomaly).
* ``anomsmith.platform.visualization`` — optional matplotlib plots wired to anomsmith
  detectors (e.g. :func:`~anomsmith.platform.visualization.plot_pca_boundary`).

Public re-exports also appear on the root ``anomsmith`` package for a single import
story (``from anomsmith import PredictiveMaintenanceSystem``, …).

Migration from ``anomaly_detection_toolkit``
--------------------------------------------

* Replace ``import anomaly_detection_toolkit`` with ``import anomsmith.platform`` or
  root ``anomsmith`` imports.
* Do **not** import duplicate detector names from the old toolkit; use
  ``anomsmith.primitives.detectors`` instead.
* ``prepare_pm_features(..., add_degradation_rates=...)`` was renamed to
  ``include_degradation_rates=...`` to avoid shadowing the ``add_degradation_rates``
  function inside that helper.

The standalone **asset_health** / **anomaly_detection_toolkit** repository is **no
longer required** for any documented capability; see :doc:`migration_from_anomaly_detection_toolkit`.
