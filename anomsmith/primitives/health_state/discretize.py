"""RUL-based health state discretization."""

import logging
from typing import TYPE_CHECKING, Union

import numpy as np
import pandas as pd

from anomsmith.objects.health_state import HealthState, HealthStateView

if TYPE_CHECKING:
    try:
        from timesmith.typing import SeriesLike
    except ImportError:
        SeriesLike = None

logger = logging.getLogger(__name__)


def discretize_rul_to_health_states(
    rul: Union[np.ndarray, pd.Series, "SeriesLike"],
    healthy_threshold: float = 30.0,
    warning_threshold: float = 10.0,
    index: pd.Index | None = None,
) -> HealthStateView:
    """Discretize RUL values into health states.

    Maps RUL values to health states:
    - RUL > healthy_threshold: Healthy (0)
    - warning_threshold < RUL <= healthy_threshold: Warning (1)
    - RUL <= warning_threshold: Distress (2)

    Args:
        rul: Remaining Useful Life values
        healthy_threshold: RUL threshold for Healthy state (default 30)
        warning_threshold: RUL threshold for Warning state (default 10)
        index: Optional index for the health states (defaults to rul index if Series)

    Returns:
        HealthStateView with discretized states

    Examples:
        >>> import numpy as np
        >>> rul = np.array([50, 25, 5, 0])
        >>> states = discretize_rul_to_health_states(rul, healthy_threshold=30, warning_threshold=10)
        >>> states.states
        array([0, 1, 2, 2])
    """
    if isinstance(rul, pd.Series):
        index = rul.index
        rul_values = rul.values
    else:
        rul_values = np.asarray(rul)
        if index is None:
            index = pd.RangeIndex(start=0, stop=len(rul_values))

    # Discretize RUL - vectorized
    states = np.zeros(len(rul_values), dtype=int)
    states[(rul_values > warning_threshold) & (rul_values <= healthy_threshold)] = HealthState.WARNING
    states[rul_values <= warning_threshold] = HealthState.DISTRESS

    return HealthStateView(index=index, states=states)

