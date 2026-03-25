# Optional delete / consolidation list

Items here are **candidates** for removal or merge. Nothing in this file has been deleted unless noted in git history; this is a backlog for maintainers.

## Redundant or pass-through APIs

| Item | Location | Note |
|------|----------|------|
| `rank_assets_by_risk` | `anomsmith/workflows/asset_health.py` | Largely duplicates `sort_values("combined_risk")` already applied in `assess_asset_health`. **Options:** (1) delete and document `assess_asset_health` sort order, (2) keep for DataFrames produced outside `assess_asset_health`, (3) make it a one-liner alias with explicit docstring. |

## Duplication already removed (historical)

| Removed / consolidated | Was | Now |
|----------------------|-----|-----|
| Mahalanobis math in workflow | `track_mahalanobis_distance` manual numpy | `PCADetector.score` for `score_method="mahalanobis"` |
| Inline PR/F1 in threshold sweep | `sweep_thresholds` vectorized duplicate | `workflows.eval.metrics` functions |
| Dual `HealthStateView` coercion | `apply_policy` / `evaluate_policy` | `_coerce_health_state_view` in `pm.py` |
| Asset health hidden policy | Fixed thresholds/weights only in workflow body | Keyword-only args + `anomsmith.constants` defaults on `assess_asset_health` |

## Example / script overlap

| Pattern | Files | Recommendation |
|---------|-------|----------------|
| Turbofan-style feature selection | `examples/*turbofan*`, `examples/pca_predictive_maintenance_example.py`, etc. | Extract **optional** `examples/_turbofan_io.py` **only if** you need single maintenance; otherwise keep copy-paste in examples (lowest coupling). |

## Dynamic imports in survival workflow

| Item | Location | Note |
|------|----------|------|
| Model factory branches | `fit_survival_model_for_maintenance` | Could move to `anomsmith.primitives.survival.factory` to shrink workflow module; **not required** for correctness. |

## Documentation drift

| Item | Note |
|------|------|
| `SURVIVAL_ANALYSIS_IMPLEMENTATION.md`, `PREDICTIVE_MAINTENANCE_IMPLEMENTATION.md` | Implementation diaries; may contradict code over time. Prefer `docs/` + docstrings as source of truth or add “last reviewed” dates. |

## Dead code search (manual)

No automated dead-code pass was executed. Suggested command for maintainers: `vulture anomsmith` (optional dev dependency).
