"""Simple decision policy for health states.

Maps health states to actions based on state transitions and thresholds.
"""

import logging
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import pandas as pd

from anomsmith.objects.health_state import Action, ActionView, HealthStateView, PolicyResult

if TYPE_CHECKING:
    try:
        from timesmith.typing import SeriesLike
    except ImportError:
        SeriesLike = None

logger = logging.getLogger(__name__)


class SimpleHealthPolicy:
    """Simple decision policy for health state-based maintenance.

    Maps health states to actions:
    - Distress (2): Intervene immediately
    - Healthy (0) -> Warning (1): Flag for review
    - Otherwise: Wait

    Action costs and risk reductions are configurable.

    Args:
        intervene_cost: Cost of intervention action (default 100)
        review_cost: Cost of review action (default 30)
        wait_cost: Cost of wait action (default 0)
        base_risks: Base failure risks by state [healthy, warning, distress] (default [0.01, 0.1, 0.3])
        intervene_risk_reduction: Risk reduction factor for intervention (default 0.5)
        review_risk_reduction: Risk reduction factor for review (default 0.75)
    """

    def __init__(
        self,
        intervene_cost: float = 100.0,
        review_cost: float = 30.0,
        wait_cost: float = 0.0,
        base_risks: tuple[float, float, float] = (0.01, 0.1, 0.3),
        intervene_risk_reduction: float = 0.5,
        review_risk_reduction: float = 0.75,
    ) -> None:
        """Initialize SimpleHealthPolicy."""
        self.intervene_cost = intervene_cost
        self.review_cost = review_cost
        self.wait_cost = wait_cost
        self.base_risks = np.array(base_risks)
        self.intervene_risk_reduction = intervene_risk_reduction
        self.review_risk_reduction = review_risk_reduction

    def apply(
        self,
        health_states: HealthStateView,
        previous_states: HealthStateView | None = None,
    ) -> PolicyResult:
        """Apply policy to health states.

        Args:
            health_states: Current health states
            previous_states: Previous health states (for transition detection)

        Returns:
            PolicyResult with actions, costs, and risks
        """
        states = health_states.states
        index = health_states.index

        # Determine actions based on state transitions - vectorized
        actions = np.full(len(states), Action.WAIT, dtype=int)

        # Distress always triggers intervention
        actions[states == 2] = Action.INTERVENE

        # Check for transitions if previous states provided
        if previous_states is not None:
            if len(previous_states.states) != len(states):
                raise ValueError("Previous states must have same length as current states")

            # Healthy -> Warning transition triggers review
            transitions = (previous_states.states == 0) & (states == 1)
            actions[transitions] = Action.REVIEW

        # Compute costs - vectorized
        costs = np.zeros(len(actions))
        costs[actions == Action.INTERVENE] = self.intervene_cost
        costs[actions == Action.REVIEW] = self.review_cost
        costs[actions == Action.WAIT] = self.wait_cost

        # Compute risks after actions - vectorized
        base_risks_by_state = self.base_risks[states]
        risk_reductions = np.ones(len(actions))
        risk_reductions[actions == Action.INTERVENE] = self.intervene_risk_reduction
        risk_reductions[actions == Action.REVIEW] = self.review_risk_reduction
        risks = base_risks_by_state * risk_reductions

        action_view = ActionView(index=index, actions=actions)

        return PolicyResult(
            health_states=health_states,
            actions=action_view,
            costs=costs,
            risks=risks,
        )

    def evaluate(
        self,
        health_states: HealthStateView,
        previous_states: HealthStateView | None = None,
    ) -> dict[str, float]:
        """Evaluate policy performance.

        Args:
            health_states: Current health states
            previous_states: Previous health states (optional)

        Returns:
            Dictionary with total cost and total risk
        """
        result = self.apply(health_states, previous_states)
        return {
            "total_cost": float(result.costs.sum()),
            "total_risk": float(result.risks.sum()),
            "interventions": int((result.actions.actions == Action.INTERVENE).sum()),
            "reviews": int((result.actions.actions == Action.REVIEW).sum()),
            "waits": int((result.actions.actions == Action.WAIT).sum()),
        }

