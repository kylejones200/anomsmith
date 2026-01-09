# Getting Started

This guide walks through a full anomaly detection workflow using Anomsmith. The example uses a single time series with an injected anomaly. The focus stays on structure and meaning.

We begin with a pandas Series.

```python
import numpy as np
import pandas as pd

idx = pd.date_range("2022-01-01", periods=100, freq="D")
y = pd.Series(np.random.normal(0, 1, size=100), index=idx)
y.iloc[60] += 8
```

Anomsmith relies on shared typing from Timesmith. Validation happens at the boundary.

```python
from timesmith.typing.validators import assert_series_like

assert_series_like(y)
```

We choose a scorer. Scorers produce aligned anomaly scores. They do not decide what is anomalous.

```python
from anomsmith.primitives.scorers.robust_zscore import RobustZScoreScorer

scorer = RobustZScoreScorer()
scorer.fit(y)  # Fit on the data (no-op for RobustZScoreScorer, but required by interface)
```

We now run detection through a workflow. Thresholding stays explicit.

```python
from anomsmith.workflows import detect_anomalies
from anomsmith.primitives.thresholding import ThresholdRule

result = detect_anomalies(
    y=y,
    detector=scorer,
    threshold_rule=ThresholdRule(method="quantile", value=0.95, quantile=0.95)
)
```

The output is a table. Scores and flags align with the original index.

```python
print(result.tail())
```

This is the full Anomsmith loop. Scoring stays separate from decision rules. Evaluation and reporting build on the same objects. Nothing hides assumptions.

