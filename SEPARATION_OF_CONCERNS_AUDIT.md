# Separation of Concerns Audit — anomsmith

**Last updated:** 2026-03-24

## Executive summary

**Domain context:** This repository is a **Python library** for anomaly detection, predictive maintenance, and survival workflows. It is **not** a web application. There is **no SQL**, no HTTP route layer, and no separate frontend. Many checklist items from a classic “UI / API / DB” audit map to **N/A**; the relevant boundaries are the library’s documented **four layers** (objects → primitives → tasks → workflows) plus **configuration defaults** (`anomsmith.constants`) and **infrastructure** (S3, CloudWatch-shaped helpers, optional neural/survival backends).

**Layering verdict:** **Partial but improved.** Critical and most high-severity leaks called out in the first audit pass are **fixed**. Workflows still **import primitives directly** in several places (inconsistent with a strict “workflows → tasks only” rule); that remains **documented technical debt**, not a silent bug.

**Changes since original audit**

| Area | Action |
|------|--------|
| PCA / Mahalanobis | `track_mahalanobis_distance` delegates to `PCADetector.score` (no private sklearn state in workflow). |
| Metrics | `sweep_thresholds` uses `workflows.eval.metrics` only. |
| Batch / S3 | Tasks used for series-like batches; CSV body parsing split from scoring. |
| Survival | `predict_rul_from_survival` index contract fixed; optional `index` parameter. |
| Policy coercion | `_coerce_health_state_view` in `pm.py`. |
| Magic numbers | Centralized in `anomsmith.constants` with named defaults across primitives/workflows. |
| Asset health policy | `assess_asset_health` exposes keyword-only `risk_proba_warning_threshold`, `risk_proba_distress_threshold`, `classification_weight`, `anomaly_weight`, and `isolation_n_estimators` (defaults from constants); validates fusion weights sum to 1 and warning &lt; distress. |

---

## Phase 1 — Current layer map

| Layer (as implemented) | Role | Boundary quality |
|------------------------|------|------------------|
| **Constants** (`anomsmith.constants`) | Single source for default numerics (thresholds, weights, training defaults) | **Clear** — not a runtime “layer” but an explicit contract boundary |
| **Objects** (`anomsmith.objects`) | Views (`ScoreView`, `LabelView`), health state enums/views, validation hooks | **Clear** — structural validation; algorithms excluded by design |
| **Primitives** (`anomsmith.primitives`) | Scorers, detectors, threshold rules, policies, survival models | **Clear** — compute and rules; no I/O |
| **Tasks** (`anomsmith.tasks`) | Input coercion, validation, orchestration of primitives | **Partial** — batch multi-column panels still bypass tasks by necessity |
| **Workflows** (`anomsmith.workflows`) | Pandas in/out, reports, batch/S3, monitoring-flavored summaries | **Improved** — less duplicated math; still mixes optional infra (S3) |
| **Examples / scripts** | Demonstrations | **Acceptable** — duplication expected; not production API |
| **Infrastructure** | boto3 (S3), timesmith typing | **Mixed into workflows** for S3; no dedicated `infrastructure/` package |

**Stated rule (docs):** “Each layer imports only from layers below it.” **Violation:** workflows import **both** `tasks` and `primitives` (e.g. `detect.py` uses `apply_threshold` from primitives; `asset_health` constructs primitives directly). **Mitigation options:** (1) accept as pragmatic for a small library, (2) add task-level wrappers for every primitive touchpoint, (3) document allowed exceptions per module.

---

## SQL findings

**None.** No relational database access, ORM, or raw SQL in the tree. **Rule:** If SQL is introduced later, it must live behind a dedicated data-access module; workflows and examples must not embed queries.

---

## Violations by severity

### Critical

| ID | File | Status | Notes |
|----|------|--------|-------|
| SOC-001 | `anomsmith/workflows/survival.py` | **Fixed** | `predict_rul_from_survival` accepts optional `index`; DataFrame `X` yields aligned RUL index. Invalid `index=` call site removed. |

### High

| ID | File | Status | Notes |
|----|------|--------|-------|
| SOC-002 | `anomsmith/workflows/pca_pm.py` | **Fixed** | Delegates to `detector.score`; requires `score_method="mahalanobis"`. |
| SOC-003 | `anomsmith/workflows/detect.py` | **Fixed** | Sweep uses `compute_precision` / `compute_recall` / `compute_f1`. |
| SOC-004 | `anomsmith/workflows/batch_inference.py` | **Fixed (partial)** | `run_scoring` / `run_detection` for 1D series-like batches; multi-column DataFrame still calls model directly. |
| SOC-005 | `anomsmith/workflows/batch_inference.py` | **Fixed** | `_series_from_s3_csv_body` separates I/O from scoring. |

### Medium

| ID | File | Status | Notes |
|----|------|--------|-------|
| SOC-006 | `anomsmith/workflows/asset_health.py` | **Fixed** | Thresholds, fusion weights, and `isolation_n_estimators` are **explicit API** (keyword-only where noted) with defaults from `anomsmith.constants`; validation for ordered probabilities and fusion weights summing to 1. Min-max normalization remains in workflow (acceptable fusion adapter). |
| SOC-007 | `anomsmith/workflows/detect.py` | **Open** | `apply_threshold` still orchestrated in workflow; optional move to `tasks/detect.py`. |
| SOC-008 | `anomsmith/workflows/pm.py` | **Fixed** | `_coerce_health_state_view`. |
| SOC-009 | `anomsmith/workflows/model_monitoring.py` | **Open** | CloudWatch-oriented naming; optional `infrastructure/` or rename to `monitoring_adapters`. |
| SOC-010 | `anomsmith/workflows/survival.py` | **Open** | Dynamic imports / factory in workflow; optional `survival_factory` module. |

### Low

| ID | File | Status | Notes |
|----|------|--------|-------|
| SOC-011 | `anomsmith/workflows/asset_health.py` | **Open** | `rank_assets_by_risk` vs sort inside `assess_asset_health` — see `OPTIONAL_DELETE_LIST.md`. |
| SOC-012 | `examples/*.py` | **Accepted** | Tutorial duplication; do not import from package. |

### Cross-cutting (configuration)

| ID | Topic | Status | Notes |
|----|-------|--------|-------|
| SOC-CFG-001 | Default numerics | **Addressed** | `anomsmith/constants.py` holds named defaults; primitives and workflows reference constants instead of raw literals where refactoring landed. |

---

## UI / domain / API issues (adapted)

- **No UI:** N/A.
- **Client trust:** Callers must not treat library defaults as compliance-approved policy; `assess_asset_health` and `apply_policy` parameters are now the explicit levers.
- **Public API shape:** Mix of `pd.Series`, `pd.DataFrame`, `dict`, and views — acceptable for pandas-native design; column/index contracts belong in docstrings (ongoing).

---

## Data-access issues

- N/A (no DB). **S3:** keep boto3 and parsing at workflow edge or future `infrastructure/`.

---

## Trust-boundary issues

- **SOC-001** would have failed at runtime if exercised; now covered by tests.
- **Policy defaults** remain **illustrative**; domain owners must override thresholds/costs for production.

---

## Target architecture (Phase 3)

| Layer | Responsibilities | Disallowed |
|-------|------------------|------------|
| **Constants** | Default numerics, tolerances, shared training defaults | Business orchestration, I/O |
| **Presentation** | N/A in-repo | N/A |
| **Application (workflows)** | Compose tasks/primitives, pandas adapters, reporting | Reimplementing primitive math; private attribute access |
| **Domain** | Rules in primitives + explicit workflow parameters | boto3, raw SQL |
| **Tasks** | Validate/coerce, ordered primitive calls | Heavy reporting layout |
| **Data access** | Future only | Business rules |
| **Infrastructure** | S3, metrics SDKs | Scoring logic |

---

## Refactor plan (prioritized)

1. ~~SOC-001–005, SOC-006, SOC-008~~ — done.
2. **Next:** SOC-007 — `run_detect_with_threshold` (or equivalent) in `tasks/detect.py`; thin `detect_anomalies` to tasks + view mapping only.
3. **Next:** SOC-009 — relocate or document `model_monitoring` as adapter layer.
4. **Optional:** SOC-010 — survival model factory module.
5. **Optional:** SOC-011 — delete or narrow `rank_assets_by_risk`.

---

## Tests (boundary enforcement)

| Module | What is enforced |
|--------|------------------|
| `tests/test_layer_boundaries.py` | PCA → primitive score; sweep ↔ metrics; survival RUL index; batch → tasks (1D); detect scores ↔ `run_scoring`; asset health explicit params ↔ defaults; fusion weight / threshold validation. |
| Rest of `tests/` | Regression coverage for workflows, primitives, persistence. |

**Gaps:** No automated check that workflows never import primitives (would be a custom linter or import-graph test); optional follow-up.

---

## Final verdict

**Partial layering** remains by design tradeoff (small library, pandas-first API). **Correctness and duplication issues** from the original audit are **resolved or explicitly parameterized**. Remaining work is **structural polish** (task-only orchestration, factory/module splits, optional delete of redundant helpers), not emergency refactors.
