# Layer responsibilities — anomsmith

This project documents a **four-layer library architecture** (`docs/architecture.rst`). The table below maps those layers to the separation model you asked for (presentation / application / domain / data access / infrastructure) and states **allowed** vs **disallowed** behavior.

## Layer mapping

| anomsmith layer | Maps to | Owns |
|-----------------|---------|------|
| **Constants** (`anomsmith.constants`) | Configuration defaults | Named numeric defaults (thresholds, weights, training hyperparameter defaults); not orchestration |
| **Objects** | Domain data + invariants | Indices, view types, structural validation |
| **Primitives** | Domain logic (algorithms + rules) | Scoring, detection, thresholds, policies, survival math |
| **Tasks** | Application orchestration (thin) | Input coercion, alignment checks, primitive invocation order |
| **Workflows** | Application surface + adapters | pandas in/out, batching, reports, optional I/O glue |
| **Examples / scripts** | Presentation / usage demos | End-to-end scripts only; not imported by the package |

**Data access:** Not used today. If added, introduce an explicit module (e.g. `anomsmith.data` or `anomsmith.repositories`) and keep SQL/ORM out of workflows.

**Infrastructure:** External systems (S3, cloud metrics APIs). Prefer small dedicated functions or a subpackage; workflows may call them but should not embed protocol details in scoring logic.

**Constants:** Any default that encodes policy or calibration (e.g. failure probability bands, fusion weights, drift z-score threshold) belongs in `anomsmith.constants` or an explicit caller-supplied argument — not as unexplained literals in workflows.

---

## Rules by layer

### Constants (`anomsmith.constants`)

**Allowed:** Module-level named floats/ints/tuples used as defaults; tolerances for validation.

**Disallowed:** Importing primitives or workflows; I/O; stateful behavior.

### Objects (`anomsmith.objects`)

**Allowed:** Immutable views, enums, `__post_init__` structural validation, alignment helpers.

**Disallowed:** Fitting models, calling sklearn/torch, threshold tuning, file/network I/O, embedding default business policy (e.g. cost of intervention).

---

### Primitives (`anomsmith.primitives`)

**Allowed:** Algorithms, rules, training/scoring/predict, pure transformations, policy objects with explicit parameters.

**Disallowed:** Reading environment-specific data paths, boto3, constructing user-facing prose, pandas-specific report layouts (keep returns as arrays/views/Series where reasonable).

---

### Tasks (`anomsmith.tasks`)

**Allowed:** `make_series_view`, `assert_*`, `run_scoring`, `run_detection`, ordering of primitive calls, raising on invalid inputs.

**Disallowed:** Duplicating primitive formulas, S3/HTTP, defining new business constants without domain review.

---

### Workflows (`anomsmith.workflows`)

**Allowed:** Call tasks and primitives to build user-facing functions, return DataFrames/Series/dicts, logging, high-level batch loops.

**Disallowed:** Copy-pasting math that already exists on a primitive (e.g. recomputing Mahalanobis when `PCADetector.score` exists), depending on private attributes (`_*` fitted internals) instead of public APIs, embedding SQL.

**Exception (documented):** pandas-native convenience is intentional; workflows may format tabular outputs. Keep **decision rules** (thresholds, costs) parameterized or delegated to primitives/domain.

---

### Evaluation (`anomsmith.workflows.eval`)

**Allowed:** Metrics, backtests, survival evaluation — pure functions over arrays/Series.

**Disallowed:** Training production models, I/O side effects.

---

### Infrastructure (conceptual; not always a separate package)

**Allowed:** boto3 clients, CSV/stream parsing helpers, clock/timezone utilities for metrics export.

**Disallowed:** Anomaly scoring, health-state semantics.

---

## Cross-layer contracts

- Prefer **explicit types** (`ScoreView`, `LabelView`, `HealthStateView`) between tasks and primitives.
- Workflows may convert views to **pandas** at the boundary; avoid pushing pandas deep into primitives.
- **Do not** pass unfitted models across layers without documenting it; tasks/workflows should enforce `is_fitted` where applicable.

---

## Validation

- **Structural validation:** objects + tasks (`assert_series`, `assert_aligned`).
- **Semantic / business validation:** primitives and policy objects (e.g. threshold ranges).
- **Do not** reimplement the same numeric rule in both a primitive and a workflow; call one implementation.
