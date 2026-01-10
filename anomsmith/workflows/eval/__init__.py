"""Evaluation subpackage for metrics and backtesting."""

from anomsmith.workflows.eval.backtest import (
    ExpandingWindowSplit,
    SlidingWindowSplit,
    backtest_detector,
)
from anomsmith.workflows.eval.metrics import (
    average_run_length,
    compute_f1,
    compute_precision,
    compute_recall,
)

try:
    from anomsmith.workflows.eval.survival_metrics import (
        compute_concordance_index,
        evaluate_survival_model,
    )

    __all__ = [
        "compute_precision",
        "compute_recall",
        "compute_f1",
        "average_run_length",
        "ExpandingWindowSplit",
        "SlidingWindowSplit",
        "backtest_detector",
        "compute_concordance_index",
        "evaluate_survival_model",
    ]
except ImportError:
    __all__ = [
        "compute_precision",
        "compute_recall",
        "compute_f1",
        "average_run_length",
        "ExpandingWindowSplit",
        "SlidingWindowSplit",
        "backtest_detector",
    ]

