"""Survival analysis primitives for predictive maintenance.

Provides survival modeling capabilities for estimating time-to-failure
and survival probabilities based on sensor readings and equipment attributes.
"""

from anomsmith.primitives.survival.cox import CoxSurvivalModel
from anomsmith.primitives.survival.neural import (
    DeepSurvModel,
    LogisticHazardModel,
)

try:
    from anomsmith.primitives.survival.kaplan_meier import KaplanMeierModel
    from anomsmith.primitives.survival.lifelines_cox import LifelinesCoxModel
    from anomsmith.primitives.survival.parametric import ParametricSurvivalModel

    __all__ = [
        "CoxSurvivalModel",
        "LogisticHazardModel",
        "DeepSurvModel",
        "LifelinesCoxModel",
        "KaplanMeierModel",
        "ParametricSurvivalModel",
    ]
except ImportError:
    __all__ = [
        "CoxSurvivalModel",
        "LogisticHazardModel",
        "DeepSurvModel",
    ]

