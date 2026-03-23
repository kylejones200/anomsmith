# Delete List

Items to remove or deprecate for production readiness.

## Dead Files

| File | Reason | Action |
|------|--------|--------|
| *(none confirmed)* | — | No orphan files identified |

## Dead Modules

| Module | Reason | Action |
|--------|--------|--------|
| *(none)* | — | All modules appear reachable from public API or examples |

## Unused Code

| Location | Item | Reason | Action |
|----------|------|--------|--------|
| `anomsmith/workflows/batch_inference.py` | `batch_size` param in `batch_score` | Never used; iterator yields per-item | Remove or implement batching |
| `anomsmith/workflows/batch_inference.py` | `batch_size` param in `batch_predict` | Same | Remove or implement |
| `anomsmith/workflows/batch_inference.py` | `return_index` param | Never used in implementation | Remove or implement |

## Duplicate Logic

| Location | Description | Action |
|----------|-------------|--------|
| `anomsmith/objects/validate.py` vs `anomsmith/tasks/helpers.py` | Both import timesmith validators with fallbacks | Consolidate to validate.py; helpers uses validate |
| `anomsmith/__init__.py` | Duplicate `__all__` in try/except | Single __all__ with conditional appends |

## Documentation / Audit Artifacts (Pre-Release)

| File | Reason | Action |
|------|--------|--------|
| `CODE_QUALITY_REVIEW.md` | Superseded by PRODUCTION_AUDIT | Archive or merge into audit |
| `PREDICTIVE_MAINTENANCE_IMPLEMENTATION.md` | Implementation notes | Keep for reference; not dead |
| `SURVIVAL_ANALYSIS_IMPLEMENTATION.md` | Implementation notes | Keep for reference |
| `AWS_INTEGRATION.md` | Integration notes | Keep for reference |

## Removal Plan

1. **Safe to delete now:** None (no confirmed dead files)
2. **Refactor (no delete):** Remove unused params from batch_score/batch_predict
3. **Consolidate:** Validator imports
4. **Document only:** Keep implementation docs; optionally move to docs/
