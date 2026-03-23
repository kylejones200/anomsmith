# Production Hardening Checklist

Concrete, verifiable items before production deployment.

## Build & Install

- [ ] `pip install anomsmith` completes without error
- [x] `dca` (or equivalent) entry point runs: `python -m anomsmith` works
- [ ] `import anomsmith` succeeds with minimal deps
- [x] `from anomsmith import RobustZScoreScorer, detect_anomalies, ThresholdRule` succeeds

## Core API

- [ ] `detect_anomalies(y, scorer, rule)` works for `y` as pd.Series and np.ndarray
- [x] `sweep_thresholds(y, scorer, thresholds, labels)` works when `labels` is pd.Series and np.ndarray
- [x] `backtest_detector(y, detector, rule, labels=...)` works when `y` and `labels` are numpy arrays
- [ ] `score_anomalies` returns aligned index with input

## Batch & Cloud

- [x] `batch_score` raises or propagates on scorer failure (does not yield empty scores)
- [x] `batch_predict` raises or propagates on detector failure (does not yield empty labels)
- [x] `process_s3_batch(s3_keys, model, bucket="...")` requires bucket; no key-based extraction
- [ ] S3 processing validates keys are non-empty before iteration

## Model Persistence

- [ ] `save_model` raises if model not fitted
- [ ] `load_model` raises FileNotFoundError for missing path
- [ ] Round-trip: save → load → score produces same results (within tolerance)

## Validation

- [ ] `assert_series` raises ValueError for invalid input
- [ ] `assert_panel` raises ValueError for invalid input
- [ ] Timesmith validators resolve from single import path (no silent fallback loops)

## Testing

- [ ] `pytest tests/` passes
- [ ] Tests exist for: batch_score, batch_predict, sweep_thresholds with ndarray labels, backtest with ndarray
- [ ] No tests marked `@pytest.mark.skip` without ticket
- [ ] CI runs on every PR

## Observability

- [ ] Workflow entry points log at INFO
- [ ] Exceptions logged with traceback (no bare `except: pass`)
- [x] Batch errors do not silently produce empty/invalid output

## Security

- [ ] No secrets in code or config
- [ ] Pickle load documented as untrusted-input risk
- [ ] No `eval()` or `exec()` on user input

## Tooling

- [x] `mypy.ini` exists (not mypi.ini)
- [ ] CI uses ruff (or documented alternative)
- [ ] Pre-commit hooks run ruff
- [ ] Black/isort consistent with config

## Documentation

- [ ] README quick start runs without modification
- [ ] All public API members in __all__
- [ ] Breaking changes documented in CHANGELOG
