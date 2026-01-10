"""Layer 1: Data and representations.

This layer uses timesmith's SeriesLike and PanelLike types for time series data.
ScoreView and LabelView are kept for anomaly-specific outputs.
No domain libraries (sklearn, matplotlib, etc.) are imported here.
Only numpy and pandas are allowed.
"""

from anomsmith.objects.health_state import (
    Action,
    ActionView,
    HealthState,
    HealthStateView,
    PolicyResult,
)
from anomsmith.objects.views import (
    LabelView,
    PanelView,
    ScoreView,
    SeriesView,
)
from anomsmith.objects.window import WindowSpec

# Re-export timesmith types if available
try:
    from timesmith.typing import PanelLike, SeriesLike
    
    __all__ = [
        "SeriesView",  # Alias for SeriesLike
        "PanelView",  # Alias for PanelLike
        "SeriesLike",  # Direct export from timesmith
        "PanelLike",  # Direct export from timesmith
        "ScoreView",
        "LabelView",
        "WindowSpec",
        # Health state objects
        "HealthState",
        "HealthStateView",
        "Action",
        "ActionView",
        "PolicyResult",
    ]
except ImportError:
    __all__ = [
        "SeriesView",
        "PanelView",
        "ScoreView",
        "LabelView",
        "WindowSpec",
        # Health state objects
        "HealthState",
        "HealthStateView",
        "Action",
        "ActionView",
        "PolicyResult",
    ]

