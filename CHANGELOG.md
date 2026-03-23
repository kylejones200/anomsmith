# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Breaking Changes

- **`process_s3_batch`**: `bucket` is now a required positional argument. The previous logic that attempted to infer bucket from key paths was incorrect (S3 keys do not include bucket). Call sites must be updated:
  ```python
  # Before
  process_s3_batch(keys, model, s3_client=client, bucket="my-bucket")
  # After
  process_s3_batch(keys, model, bucket="my-bucket", s3_client=client)
  ```

- **`batch_score`** and **`batch_predict`**: Removed unused parameters `batch_size` and `return_index`. If you were passing these, remove them from your calls.

### Added

- `anomsmith.__main__`: Entry point for `python -m anomsmith` and `dca` console script
- `RobustZScoreScorer` exported in public API (`from anomsmith import RobustZScoreScorer`)
- Production audit deliverables: `PRODUCTION_AUDIT.md`, `TECH_DEBT_REGISTER.md`, `PROD_HARDENING_CHECKLIST.md`, `DELETE_LIST.md`
- `CHANGELOG.md` for tracking breaking changes and releases

### Fixed

- `sweep_thresholds` now accepts numpy array labels (previously raised AttributeError)
- `backtest_detector` now accepts numpy array `y` and labels
- ARIMA drift detector re-raises exceptions instead of silently returning zeros
- Batch inference propagates errors instead of yielding empty scores/labels
- `process_s3_batch` validates non-empty `s3_keys`
- Renamed `mypi.ini` to `mypy.ini`

### Changed

- CI lint check switched from flake8 to ruff
- Pre-commit hooks: ruff and ruff-format now run on commit
- `ModelPerformanceTracker` uses list-based history instead of repeated `pd.concat`
- `load_model` docstring now documents pickle security considerations
- `datetime.utcnow()` replaced with `datetime.now(timezone.utc)` (Python 3.12 compatibility)
