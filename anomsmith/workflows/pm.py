"""Predictive maintenance workflows.

Workflows for health state prediction, discretization, and policy evaluation.
"""

import logging
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import pandas as pd

from anomsmith.objects.health_state import HealthStateView, PolicyResult
from anomsmith.primitives.health_state.discretize import discretize_rul_to_health_states
from anomsmith.primitives.policy.simple import SimpleHealthPolicy

if TYPE_CHECKING:
    try:
        from timesmith.typing import SeriesLike
    except ImportError:
        SeriesLike = None

logger = logging.getLogger(__name__)


def discretize_rul(
    rul: Union[pd.Series, np.ndarray, "SeriesLike"],
    healthy_threshold: float = 30.0,
    warning_threshold: float = 10.0,
) -> pd.Series:
    """Discretize RUL values into health states.

    Maps RUL values to health states:
    - RUL > healthy_threshold: Healthy (0)
    - warning_threshold < RUL <= healthy_threshold: Warning (1)
    - RUL <= warning_threshold: Distress (2)

    Args:
        rul: Remaining Useful Life values
        healthy_threshold: RUL threshold for Healthy state (default 30)
        warning_threshold: RUL threshold for Warning state (default 10)

    Returns:
        pandas Series with health states aligned to input index

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> rul = pd.Series([50, 25, 5, 0])
        >>> states = discretize_rul(rul, healthy_threshold=30, warning_threshold=10)
        >>> states.values
        array([0, 1, 2, 2])
    """
    health_state_view = discretize_rul_to_health_states(
        rul, healthy_threshold=healthy_threshold, warning_threshold=warning_threshold
    )
    return health_state_view.to_series()


def apply_policy(
    health_states: Union[pd.Series, np.ndarray, HealthStateView],
    previous_states: Optional[Union[pd.Series, np.ndarray, HealthStateView]] = None,
    intervene_cost: float = 100.0,
    review_cost: float = 30.0,
    wait_cost: float = 0.0,
    base_risks: tuple[float, float, float] = (0.01, 0.1, 0.3),
    intervene_risk_reduction: float = 0.5,
    review_risk_reduction: float = 0.75,
) -> pd.DataFrame:
    """Apply decision policy to health states.

    Args:
        health_states: Current health states (0=Healthy, 1=Warning, 2=Distress)
        previous_states: Previous health states for transition detection (optional)
        intervene_cost: Cost of intervention action (default 100)
        review_cost: Cost of review action (default 30)
        wait_cost: Cost of wait action (default 0)
        base_risks: Base failure risks by state [healthy, warning, distress] (default [0.01, 0.1, 0.3])
        intervene_risk_reduction: Risk reduction factor for intervention (default 0.5)
        review_risk_reduction: Risk reduction factor for review (default 0.75)

    Returns:
        pandas DataFrame with health_states, actions, costs, and risks

    Examples:
        >>> import pandas as pd
        >>> states = pd.Series([0, 0, 1, 2, 2])
        >>> result = apply_policy(states)
        >>> result['action'].values
        array([0, 0, 1, 2, 2])
    """
    # Convert to HealthStateView if needed
    if isinstance(health_states, HealthStateView):
        health_state_view = health_states
    elif isinstance(health_states, pd.Series):
        health_state_view = HealthStateView(
            index=health_states.index, states=health_states.values
        )
    else:
        index = pd.RangeIndex(start=0, stop=len(health_states))
        health_state_view = HealthStateView(index=index, states=np.asarray(health_states))

    # Convert previous states if provided
    previous_state_view: Optional[HealthStateView] = None
    if previous_states is not None:
        if isinstance(previous_states, HealthStateView):
            previous_state_view = previous_states
        elif isinstance(previous_states, pd.Series):
            previous_state_view = HealthStateView(
                index=previous_states.index, states=previous_states.values
            )
        else:
            index = pd.RangeIndex(start=0, stop=len(previous_states))
            previous_state_view = HealthStateView(
                index=index, states=np.asarray(previous_states)
            )

    # Apply policy
    policy = SimpleHealthPolicy(
        intervene_cost=intervene_cost,
        review_cost=review_cost,
        wait_cost=wait_cost,
        base_risks=base_risks,
        intervene_risk_reduction=intervene_risk_reduction,
        review_risk_reduction=review_risk_reduction,
    )

    result = policy.apply(health_state_view, previous_state_view)
    return result.to_dataframe()


def evaluate_policy(
    health_states: Union[pd.Series, np.ndarray, HealthStateView],
    previous_states: Optional[Union[pd.Series, np.ndarray, HealthStateView]] = None,
    intervene_cost: float = 100.0,
    review_cost: float = 30.0,
    wait_cost: float = 0.0,
    base_risks: tuple[float, float, float] = (0.01, 0.1, 0.3),
    intervene_risk_reduction: float = 0.5,
    review_risk_reduction: float = 0.75,
) -> dict[str, float]:
    """Evaluate policy performance metrics.

    Args:
        health_states: Current health states (0=Healthy, 1=Warning, 2=Distress)
        previous_states: Previous health states for transition detection (optional)
        intervene_cost: Cost of intervention action (default 100)
        review_cost: Cost of review action (default 30)
        wait_cost: Cost of wait action (default 0)
        base_risks: Base failure risks by state [healthy, warning, distress] (default [0.01, 0.1, 0.3])
        intervene_risk_reduction: Risk reduction factor for intervention (default 0.5)
        review_risk_reduction: Risk reduction factor for review (default 0.75)

    Returns:
        Dictionary with total_cost, total_risk, interventions, reviews, waits

    Examples:
        >>> import pandas as pd
        >>> states = pd.Series([0, 0, 1, 2, 2])
        >>> metrics = evaluate_policy(states)
        >>> metrics['total_cost']
        230.0
    """
    # Convert to HealthStateView if needed
    if isinstance(health_states, HealthStateView):
        health_state_view = health_states
    elif isinstance(health_states, pd.Series):
        health_state_view = HealthStateView(
            index=health_states.index, states=health_states.values
        )
    else:
        index = pd.RangeIndex(start=0, stop=len(health_states))
        health_state_view = HealthStateView(index=index, states=np.asarray(health_states))

    # Convert previous states if provided
    previous_state_view: Optional[HealthStateView] = None
    if previous_states is not None:
        if isinstance(previous_states, HealthStateView):
            previous_state_view = previous_states
        elif isinstance(previous_states, pd.Series):
            previous_state_view = HealthStateView(
                index=previous_states.index, states=previous_states.values
            )
        else:
            index = pd.RangeIndex(start=0, stop=len(previous_states))
            previous_state_view = HealthStateView(
                index=index, states=np.asarray(previous_states)
            )

    # Evaluate policy
    policy = SimpleHealthPolicy(
        intervene_cost=intervene_cost,
        review_cost=review_cost,
        wait_cost=wait_cost,
        base_risks=base_risks,
        intervene_risk_reduction=intervene_risk_reduction,
        review_risk_reduction=review_risk_reduction,
    )

    return policy.evaluate(health_state_view, previous_state_view)

