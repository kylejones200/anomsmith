# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- **``anomsmith.platform``**: merged the predictive maintenance, evaluation, and
  visualization stack from the former **Anomaly Detection Toolkit** repository
  (``anomaly_detection_toolkit`` / ``asset_health``). One PyPI package now covers
  primitives plus PM orchestration; optional anomaly hooks use
  ``anomsmith.primitives.base.BaseDetector`` only.
- Documentation: :doc:`platform`.
- **Toolkit parity (retire ``asset_health``)**: ``LSTMAutoencoderDetector``,
  ``PyTorchAutoencoderDetector``, ``WaveletDenoiser``, score-combining
  ``EnsembleDetector`` / ``ScoreCombiningEnsembleDetector``, and ``VotingEnsemble``
  alias; see :doc:`migration_from_anomaly_detection_toolkit`.

### Breaking Changes

- **``prepare_pm_features``** (in ``anomsmith.platform``): the boolean flag
  ``add_degradation_rates`` was renamed to ``include_degradation_rates`` so the name
  no longer shadows the ``add_degradation_rates`` function.

### Changed

- CI now runs **Ruff** and **Mypy** as blocking steps, and **pytest** with a **coverage floor** (42% on `anomsmith`).
- `mypy.ini` fixes invalid `follow_imports` configuration and documents **typed ratchets** for survival, optional classifiers, and a few heavy workflow modules until their signatures are aligned with `BaseEstimator`.
- `plots` optional dependency: `plotsmith` is constrained to **Python 3.12+** via an environment marker so metadata resolves cleanly on older interpreters.
- `anomsmith.tasks.detect`: `DetectTask` is resolved without a static **no-redef** clash between Timesmith’s task type and the local fallback dataclass.
- Pre-commit continues to use Ruff and ruff-format on commit.
- `ModelPerformanceTracker` uses list-based history instead of repeated `pd.concat`.
- `load_model` docstring now documents pickle security considerations.
- `datetime.utcnow()` replaced with `datetime.now(timezone.utc)` (Python 3.12 compatibility).

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
