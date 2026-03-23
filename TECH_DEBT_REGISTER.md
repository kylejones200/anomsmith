# Technical Debt Register

| ID | Title | Severity | Area | Impact | Recommended Fix | Est. Effort |
|----|-------|----------|------|--------|-----------------|-------------|
| TD-001 | Broken dca entry point | Critical | Build | `pip install` creates non-functional CLI | ~~Add `__main__.py`~~ | ✅ RESOLVED |
| TD-002 | Batch inference silent failures | Critical | Reliability | Errors yield fake empty results | ~~Re-raise; no empty yields~~ | ✅ RESOLVED |
| TD-003 | process_s3_batch wrong bucket logic | Critical | API | Bucket extraction from key incorrect | ~~Require explicit bucket~~ | ✅ RESOLVED |
| TD-004 | sweep_thresholds fails on numpy labels | Critical | API | AttributeError when labels is ndarray | ~~Normalize labels~~ | ✅ RESOLVED |
| TD-005 | backtest_detector fails on numpy inputs | High | API | y.iloc/labels.reindex fail for ndarray | ~~Use make_series_view~~ | ✅ RESOLVED |
| TD-006 | ARIMA drift returns zeros on error | High | Reliability | Masks model failures | ~~Re-raise exception~~ | ✅ RESOLVED |
| TD-007 | RobustZScoreScorer not in public API | High | API | README import fails | ~~Add to __init__.py~~ | ✅ RESOLVED |
| TD-008 | Fragile timesmith validator imports | Medium | Dependencies | Multiple fallback paths | ~~Consolidate; helpers uses validate~~ | ✅ RESOLVED |
| TD-009 | mypi.ini typo | Medium | Tooling | Mypy config may not be used | ~~Rename to mypy.ini~~ | ✅ RESOLVED |
| TD-010 | CI uses flake8, project uses ruff | Medium | CI | Inconsistent linting | ~~Switch CI to ruff~~ | ✅ RESOLVED |
| TD-011 | datetime.utcnow() deprecated | Medium | Compatibility | Deprecation in Python 3.12+ | ~~Use datetime.now(timezone.utc)~~ | ✅ RESOLVED |
| TD-012 | ModelPerformanceTracker pd.concat in loop | Medium | Performance | O(n²) behavior for many updates | ~~Use list + single DataFrame build~~ | ✅ RESOLVED |
| TD-013 | Duplicate __all__ in __init__.py | Low | Code Quality | Maintenance burden | ~~Single __all__ with _BASE_EXPORTS~~ | ✅ RESOLVED |
| TD-014 | Unused batch_size in batch_score/batch_predict | Low | API | Dead parameter | ~~Removed unused params~~ | ✅ RESOLVED |

---

## Severity Definitions

- **Critical:** Data corruption, security risk, or system failure
- **High:** Incorrect behavior in common use
- **Medium:** Limits scalability or maintainability
- **Low:** Cleanup
