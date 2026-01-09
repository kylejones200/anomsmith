# Timesmith Integration Opportunities

This document outlines additional areas where anomsmith could integrate more closely with timesmith beyond the SeriesView/PanelView changes already made.

## ✅ Already Integrated

1. **SeriesView/PanelView** → Now use `SeriesLike`/`PanelLike` from timesmith
2. **Type validation** → Using timesmith's `assert_series_like` and `assert_panel_like`

## 🔄 Potential Integrations

### 1. Base Classes (`anomsmith/primitives/base.py`)

**Current State:**
- `BaseObject`: Parameter management (get_params, set_params, clone)
- `BaseEstimator`: Extends BaseObject, adds fit() and fitted state
- `BaseScorer`: Extends BaseEstimator, adds score() method
- `BaseDetector`: Extends BaseEstimator, adds predict() and score() methods

**Timesmith Provides:**
- `BaseObject`: Similar parameter management
- `BaseEstimator`: Similar fit/fitted pattern
- `BaseDetector`: May have different interface

**Recommendation:**
- ⚠️ **Keep anomsmith's base classes** - They have anomaly-specific methods (`score()`, `predict()`) that may not match timesmith's interface
- Consider making anomsmith's classes extend timesmith's if compatible, but maintain our own interface

### 2. DetectTask (`anomsmith/tasks/detect.py`)

**Current State:**
- Custom `DetectTask` dataclass with y, X, labels, window_spec, cutoff

**Timesmith Provides:**
- `DetectTask`: May have similar structure

**Recommendation:**
- ✅ **Use timesmith's DetectTask if available** - Already implemented with fallback
- ✅ **Updated to use `SeriesLike`** - DetectTask now uses `SeriesLike` type

### 3. ExpandingWindowSplit (`anomsmith/workflows/eval/backtest.py`)

**Current State:**
- Custom `ExpandingWindowSplit` class

**Timesmith Provides:**
- `ExpandingWindowSplit`: Likely similar implementation
- `SlidingWindowSplit`: Additional option

**Recommendation:**
- ✅ **Use timesmith's ExpandingWindowSplit** - Already implemented with fallback
- ✅ **Expose `SlidingWindowSplit` from timesmith** - Now available with fallback implementation

### 4. Type Hints Throughout Codebase

**Current State:**
- Many functions use `pd.Series | np.ndarray`
- `DetectTask.y` uses `pd.Series | np.ndarray`

**Recommendation:**
- ✅ **Update to use `SeriesLike`** - All function signatures now accept `SeriesLike`
- ✅ **Updated throughout codebase** - Base classes, workflows, tasks, and primitives all use `SeriesLike`
- This allows seamless passing of timesmith types

### 5. ScoreView and LabelView

**Current State:**
- Custom dataclasses for anomaly scores and labels
- Have `index` and `scores`/`labels` attributes

**Timesmith Provides:**
- May have similar output types, or may use pandas Series directly

**Recommendation:**
- ⚠️ **Keep ScoreView/LabelView** - They're anomaly-specific and provide clear structure
- Could potentially use pandas Series directly, but the dataclass provides validation

### 6. WindowSpec (`anomsmith/objects/window.py`)

**Current State:**
- Custom `WindowSpec` dataclass for window specifications

**Timesmith Provides:**
- May have windowing concepts in its API

**Recommendation:**
- ⚠️ **Keep WindowSpec** - If timesmith doesn't have an equivalent, keep ours
- If timesmith adds windowing, consider migration

## Implementation Status

- [x] SeriesView → SeriesLike
- [x] PanelView → PanelLike  
- [x] DetectTask → Try to use timesmith's (with fallback) + Updated to use SeriesLike
- [x] ExpandingWindowSplit → Try to use timesmith's (with fallback)
- [x] SlidingWindowSplit → Exposed from timesmith (with fallback)
- [x] Update all type hints to use SeriesLike → Completed throughout codebase
- [x] Base classes → Updated to accept SeriesLike in type hints
- [x] ScoreView/LabelView → Kept as-is (anomaly-specific)

## Next Steps

1. **Test compatibility** - Verify timesmith's DetectTask and ExpandingWindowSplit work with anomsmith
2. **Update type hints** - Systematically replace `pd.Series | np.ndarray` with `SeriesLike`
3. **Documentation** - Update docs to show timesmith integration examples
4. **Examples** - Create examples showing objects moving between libraries

