"""Health state objects for predictive maintenance.

Defines health state categories and policy actions.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    try:
        from timesmith.typing import SeriesLike
    except ImportError:
        SeriesLike = None


class HealthState(IntEnum):
    """Health state categories for predictive maintenance.

    States are ordered from healthy (0) to distressed (highest value).
    """

    HEALTHY = 0
    WARNING = 1
    DISTRESS = 2

    def __str__(self) -> str:
        """Human-readable state name."""
        names = {0: "Healthy", 1: "Warning", 2: "Distress"}
        return names[self.value]


class Action(IntEnum):
    """Action categories for decision policies."""

    WAIT = 0
    REVIEW = 1
    INTERVENE = 2

    def __str__(self) -> str:
        """Human-readable action name."""
        names = {0: "wait", 1: "review", 2: "intervene"}
        return names[self.value]


@dataclass(frozen=True)
class HealthStateView:
    """Health state labels aligned to time series index.

    Attributes:
        index: Time series index
        states: Health state values (0=Healthy, 1=Warning, 2=Distress)
    """

    index: pd.Index
    states: np.ndarray

    def __post_init__(self) -> None:
        """Validate health state view."""
        if len(self.index) != len(self.states):
            raise ValueError(
                f"Index length ({len(self.index)}) must match states length ({len(self.states)})"
            )
        if not np.all((self.states >= 0) & (self.states <= 2)):
            raise ValueError("Health states must be in range [0, 2]")

    def to_series(self) -> pd.Series:
        """Convert to pandas Series."""
        return pd.Series(self.states, index=self.index, name="health_state")


@dataclass(frozen=True)
class ActionView:
    """Action labels aligned to time series index.

    Attributes:
        index: Time series index
        actions: Action values (0=wait, 1=review, 2=intervene)
    """

    index: pd.Index
    actions: np.ndarray

    def __post_init__(self) -> None:
        """Validate action view."""
        if len(self.index) != len(self.actions):
            raise ValueError(
                f"Index length ({len(self.index)}) must match actions length ({len(self.actions)})"
            )
        if not np.all((self.actions >= 0) & (self.actions <= 2)):
            raise ValueError("Actions must be in range [0, 2]")

    def to_series(self) -> pd.Series:
        """Convert to pandas Series."""
        return pd.Series(self.actions, index=self.index, name="action")


@dataclass(frozen=True)
class PolicyResult:
    """Result of applying a decision policy.

    Attributes:
        health_states: Predicted health states
        actions: Recommended actions
        costs: Action costs
        risks: Failure risks after actions
    """

    health_states: HealthStateView
    actions: ActionView
    costs: np.ndarray
    risks: np.ndarray

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        return pd.DataFrame(
            {
                "health_state": self.health_states.states,
                "action": self.actions.actions,
                "cost": self.costs,
                "risk": self.risks,
            },
            index=self.health_states.index,
        )

