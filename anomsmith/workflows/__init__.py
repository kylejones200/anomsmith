"""Layer 4: Workflows.

Workflows provide the public entry points users call.
Workflows can import matplotlib only if plots are added (not in first pass).
"""

from anomsmith.workflows.detect import (
    detect_anomalies,
    report_detection,
    score_anomalies,
    sweep_thresholds,
)
from anomsmith.workflows.eval.backtest import backtest_detector

__all__ = [
    "score_anomalies",
    "detect_anomalies",
    "sweep_thresholds",
    "report_detection",
    "backtest_detector",
]

