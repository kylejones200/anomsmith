"""Window specification for time series operations."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class WindowSpec:
    """Specification for sliding or expanding windows.

    Attributes:
        length: Window length in time steps
        step: Step size between windows (default: 1)
        alignment: 'left' (start at beginning), 'right' (end at current),
            or 'center' (centered on current point)
    """

    length: int
    step: int = 1
    alignment: Literal["left", "right", "center"] = "right"

    def __post_init__(self) -> None:
        """Validate inputs after initialization."""
        if self.length <= 0:
            raise ValueError(f"Window length must be positive, got {self.length}")
        if self.step <= 0:
            raise ValueError(f"Window step must be positive, got {self.step}")

