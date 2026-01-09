"""Layer 3: Tasks.

Tasks translate user intent into a sequence of primitive calls and outputs.
Tasks must not import matplotlib.
"""

from anomsmith.tasks.detect import DetectTask, run_detection, run_scoring
from anomsmith.tasks.helpers import make_panel_view, make_series_view

__all__ = [
    "DetectTask",
    "make_series_view",
    "make_panel_view",
    "run_scoring",
    "run_detection",
]

