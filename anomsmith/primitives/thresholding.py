"""Thresholding primitives for converting scores to labels."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from anomsmith.objects.views import LabelView, ScoreView


@dataclass(frozen=True)
class ThresholdRule:
    """Rule for thresholding anomaly scores.

    Attributes:
        method: 'absolute' (use value directly) or 'quantile' (use quantile)
        value: Threshold value (absolute) or quantile (0-1)
        quantile: If method is 'quantile', this is the quantile to use
    """

    method: Literal["absolute", "quantile"]
    value: float
    quantile: float | None = None

    def __post_init__(self) -> None:
        """Validate inputs after initialization."""
        if self.method == "quantile":
            if self.quantile is None:
                raise ValueError("quantile must be provided when method is 'quantile'")
            if not 0 <= self.quantile <= 1:
                raise ValueError(f"quantile must be in [0, 1], got {self.quantile}")
        elif self.method == "absolute":
            if self.quantile is not None:
                raise ValueError("quantile should not be provided when method is 'absolute'")
        else:
            raise ValueError(f"method must be 'absolute' or 'quantile', got {self.method}")


def apply_threshold(score_view: ScoreView, rule: ThresholdRule) -> LabelView:
    """Apply threshold rule to scores to produce binary labels.

    Args:
        score_view: ScoreView with anomaly scores
        rule: ThresholdRule to apply

    Returns:
        LabelView with binary labels (1 = anomaly, 0 = normal)
    """
    scores = score_view.scores

    if rule.method == "absolute":
        threshold = rule.value
    elif rule.method == "quantile":
        if rule.quantile is None:
            raise ValueError("quantile must be provided when method is 'quantile'")
        threshold = np.quantile(scores, rule.quantile)
    else:
        raise ValueError(f"Unknown method: {rule.method}")

    labels = (scores >= threshold).astype(int)

    return LabelView(index=score_view.index, labels=labels)

