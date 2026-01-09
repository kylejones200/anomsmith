# Performance Analysis: Numba/JIT for Anomsmith

## Where Numba Could Help

### 1. **Pure NumPy Functions** ✅ Good Candidates
- `robust_zscore()` in `anomsmith/primitives/scaling.py`
  - Pure numpy operations (median, MAD calculation)
  - Called frequently in scoring
  - **Expected speedup: 2-5x for large arrays**

- `average_run_length()` in `anomsmith/workflows/eval/metrics.py`
  - NumPy array operations
  - Could benefit from JIT
  - **Expected speedup: 2-3x**

### 2. **Vectorized Operations** ⚠️ Limited Benefit
- `sweep_thresholds()` in `anomsmith/workflows/detect.py`
  - Already highly vectorized with numpy broadcasting
  - Numba might help slightly, but overhead could negate gains
  - **Expected speedup: 1.2-1.5x (marginal)**

### 3. **Rolling Window Calculations** ❌ Won't Help
- `ChangePointDetector.score()` uses `pandas.rolling()`
  - Numba doesn't work with pandas operations
  - Would need to rewrite with pure numpy loops
  - **Not recommended** - pandas rolling is already optimized

## Where Numba Won't Help

### 1. **ML Algorithms** ❌ No Benefit
- `IsolationForestDetector`, `LOFDetector`, `RobustCovarianceDetector`
  - Use sklearn which is already highly optimized (C/Cython)
  - Numba can't optimize sklearn internals
  - **No speedup expected**

### 2. **PCA Operations** ❌ No Benefit
- `PCADetector` uses sklearn.decomposition.PCA
  - Already optimized in sklearn
  - **No speedup expected**

### 3. **Already Vectorized Code** ⚠️ Minimal Benefit
- Most statistical scorers use vectorized numpy
- `sweep_thresholds` uses broadcasting
- Numba overhead might outweigh benefits for small-medium arrays

## Recommendation

### Option 1: Optional Numba (Recommended)
Make numba an **optional dependency** for workflows only:

```python
# In workflows/eval/metrics.py or workflows/detect.py
try:
    from numba import jit
    USE_NUMBA = True
except ImportError:
    USE_NUMBA = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

@jit(nopython=True, cache=True)
def _robust_zscore_numba(values, epsilon):
    # Pure numpy implementation
    ...
```

**Pros:**
- Follows ecosystem rule: "Only workflows may use optional heavy deps"
- Users can opt-in for performance
- Doesn't bloat core dependencies

**Cons:**
- Adds complexity
- Requires maintaining two code paths

### Option 2: Skip Numba
Most code is already optimized:
- ML algorithms use sklearn (C/Cython optimized)
- Statistical operations are vectorized
- Real bottlenecks are in sklearn, not our code

**Pros:**
- Simpler codebase
- No additional dependencies
- Current performance is likely sufficient

**Cons:**
- Miss potential 2-5x speedups in `robust_zscore` and `average_run_length`

## Benchmarking Recommendation

Before adding numba, benchmark:
1. Profile actual use cases to find real bottlenecks
2. Measure `robust_zscore` performance on large arrays (10k+ points)
3. Compare numba vs pure numpy for `average_run_length`
4. Test with realistic data sizes

## Conclusion

**Numba could help in 2-3 specific functions**, but:
- Most code is already optimized
- ML algorithms (main workload) won't benefit
- Should be **optional** if implemented
- Only add if profiling shows it's needed

**Recommendation:** Skip for now unless profiling shows `robust_zscore` or `average_run_length` are bottlenecks in real workloads.

