# Anomsmith

[![PyPI version](https://badge.fury.io/py/anomsmith.svg)](https://badge.fury.io/py/anomsmith)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/kylejones200/anomsmith/workflows/Tests/badge.svg)](https://github.com/kylejones200/anomsmith/actions)
[![Documentation](https://readthedocs.org/projects/anomsmith/badge/?version=latest)](https://anomsmith.readthedocs.io/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Anomaly detection workflows that turn time series signals into actionable decisions.

## Architecture

Anomsmith follows a strict 4-layer architecture that enforces clear separation of concerns:

### Layer 1: `anomsmith.objects` - Data and Representations

This layer defines immutable dataclasses for time series data structures. Only numpy and pandas are allowed - no domain libraries (sklearn, matplotlib, etc.).

**Components:**
- `SeriesView`: Single time series with index and values
- `PanelView`: Multi-entity series with entity key and time index
- `ScoreView`: Anomaly scores aligned to input index
- `LabelView`: Binary flags aligned to input index
- `WindowSpec`: Window specification for time series operations
- `validate`: Validators with clear error messages

### Layer 2: `anomsmith.primitives` - Algorithm Interfaces

This layer defines algorithm interfaces and thin utilities. It must not know about tasks or evaluation. Only numpy and pandas are allowed.

**Components:**
- `BaseObject`: Base class with parameter management
- `BaseEstimator`: Base class with fit and fitted state
- `BaseScorer`: Base class for anomaly scorers
- `BaseDetector`: Base class for anomaly detectors
- Tag system: Metadata about algorithm capabilities
- `ThresholdRule` and `apply_threshold`: Thresholding primitives
- `robust_zscore`: Robust score scaling using median and MAD

### Layer 3: `anomsmith.tasks` - Task Orchestration

Tasks translate user intent into a sequence of primitive calls and outputs. Tasks must not import matplotlib.

**Components:**
- `DetectTask`: Task specification dataclass
- `make_series_view` / `make_panel_view`: Helpers to convert pandas inputs
- `run_scoring`: Task runner for scoring
- `run_detection`: Task runner for detection

### Layer 4: `anomsmith.workflows` - Public API

Workflows provide the public entry points users call. Workflows can import matplotlib only if plots are added.

**Components:**
- `score_anomalies`: Score anomalies in a time series
- `detect_anomalies`: Detect anomalies with thresholding
- `sweep_thresholds`: Evaluate multiple threshold values
- `report_detection`: Generate detection report with summary stats
- `anomsmith.workflows.eval`: Evaluation subpackage
  - Metrics: precision, recall, f1, average_run_length
  - `ExpandingWindowSplit`: Time series splitter for backtesting
  - `backtest_detector`: Run backtests across expanding windows

## Installation

```bash
pip install anomsmith
```

## Quick Start

```python
import pandas as pd
import numpy as np
from anomsmith import detect_anomalies, RobustZScoreScorer, ThresholdRule

# Create time series
y = pd.Series(np.random.randn(100))

# Initialize scorer
scorer = RobustZScoreScorer()
scorer.fit(y.values)

# Define threshold
threshold_rule = ThresholdRule(method="quantile", value=0.95, quantile=0.95)

# Detect anomalies
result = detect_anomalies(y, scorer, threshold_rule)
print(result.head())
```

## Example

See `examples/basic_detect.py` for a complete example with synthetic data.

```bash
python examples/basic_detect.py
```

## Public API

The public API is exposed in `anomsmith.__init__`:

- `score_anomalies`: Score anomalies in a time series
- `detect_anomalies`: Detect anomalies with thresholding
- `sweep_thresholds`: Evaluate multiple threshold values
- `backtest_detector`: Run backtests across expanding windows
- `BaseScorer`: Base class for scorers
- `BaseDetector`: Base class for detectors
- `ThresholdRule`: Threshold rule dataclass

## Included Detectors

- `RobustZScoreScorer`: Robust z-score based anomaly scorer using median and MAD
- `ChangePointDetector`: Change point detector using rolling window statistics

## Testing

```bash
pytest tests/
```

## Migration Guide

### Migrating Existing Detectors

To migrate an existing detector to Anomsmith:

1. **Implement Base Interface**: Choose `BaseScorer` or `BaseDetector` based on whether your detector produces only scores or both scores and labels.

2. **Follow Layer Rules**:
   - Layer 1 (objects): Use `SeriesView`, `ScoreView`, `LabelView` for data structures
   - Layer 2 (primitives): Implement in `anomsmith.primitives.scorers` or `anomsmith.primitives.detectors`
   - Layer 3 (tasks): Use `run_scoring` or `run_detection` task runners
   - Layer 4 (workflows): Use public API functions like `detect_anomalies`

3. **Example Migration**:
   ```python
   from anomsmith.primitives.base import BaseScorer
   from anomsmith.objects.views import ScoreView
   import pandas as pd
   import numpy as np

   class MyScorer(BaseScorer):
       def fit(self, y, X=None):
           # Fit logic here
           self._fitted = True
           return self

       def score(self, y):
           # Score logic here
           index = pd.RangeIndex(len(y)) if not isinstance(y, pd.Series) else y.index
           scores = np.abs(y)  # Example scoring
           return ScoreView(index=index, scores=scores)
   ```

4. **Add Tests**: Create tests in `tests/` following the existing test patterns.

5. **Update Public API**: If the detector should be part of the public API, add it to `anomsmith.__init__`.

## License

MIT

## Contributing

See `CONTRIBUTING.md` for guidelines.
