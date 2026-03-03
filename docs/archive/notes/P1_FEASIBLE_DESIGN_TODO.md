# P1 High-Fidelity Feasible Design: Comprehensive Bug Fixes & Action Items

**Created**: 2025-12-29
**Objective**: Achieve P1 (Geometrically Optimized Stellarator) high-fidelity feasible design
**Target**: Beat ALM-NGOpt baseline (`max_elongation = 2.10`)
**Current Status**: 0 feasible designs, only 2/200 candidates passing GeometerWorker

---

## Problem Definition Recap

**P1 Goal**: Minimize `max_elongation` subject to:
- `aspect_ratio ≤ 4.0`
- `average_triangularity ≤ -0.5`
- `edge_rotational_transform_over_n_field_periods ≥ 0.3`

---

## Phase 1: Critical Bugs (Must Fix First)

### Bug #1: None QI Causes TypeError Crash
- [ ] **CRITICAL** | `problems.py:144-147`
- **Impact**: System crashes when QI computation fails (common edge case)
- **Current Code**:
  ```python
  qi = metrics.get("qi", 1.0)  # Default to 1.0 if MISSING
  log_qi = np.log10(qi) if qi > 0 else 0.0  # CRASH if qi=None!
  ```
- **Fix**: Use null-coalescing pattern:
  ```python
  qi = metrics.get("qi")
  log_qi = np.log10(qi) if (qi is not None and qi > 0) else 0.0
  ```
- **Test**: Run P2/P3 evaluation with boundary that triggers "Not enough crossings found"

---

### Bug #2: Inconsistent Objective Sign Conventions
- [ ] **HIGH** | `problems.py` vs `forward_model.py`
- **Status (validated)**: The repo intentionally distinguishes multiple objective concepts (ALM objective vs physics objective vs ranking score). Mixing them is an easy source of semantic bugs, but `forward_model.compute_objective()` returning raw gradient for P2 is *by design*.
- **SSOT**: `ai_scientist/objective_types.py` documents the intended semantics and direction.
- **Action Items**:
  - [ ] Audit call sites that compare/aggregate objectives and ensure they use the intended objective kind
  - [ ] Add a small unit test that `objective_types` semantics match `forward_model.compute_objective()` for p1/p2/p3

---

### Bug #3: Silent Mock Backend Fallback in Production
- [ ] **HIGH** | `forward_model.py:297-298`
- **Status (validated)**:
  - Auto backend selection falls back to mock with a DEBUG log at `forward_model.py:297-298` (so it can be effectively silent at default INFO logging).
  - The repo already supports forcing real backend via `AI_SCIENTIST_PHYSICS_BACKEND=real` (raises if unavailable).
  - Tests also use `AI_SCIENTIST_REQUIRE_REAL_BACKEND=1` as an opt-in gate, but runtime backend auto-selection does not currently honor that env var.
- **Fix**: Consider elevating the auto-fallback log to WARNING and/or honoring `AI_SCIENTIST_REQUIRE_REAL_BACKEND` in backend auto-selection.
- **Action Items**:
  - [ ] Change log level to WARNING for auto mock fallback
  - [ ] (Optional) Honor `AI_SCIENTIST_REQUIRE_REAL_BACKEND` in `forward_model._auto_select_backend()`
  - [ ] (Optional) Print selected backend at startup in `cycle_executor.py`

---

### Bug #4: ALM Constraint Ordering Mismatch
- [ ] **HIGH** | `coordinator.py` vs `constraints.py`
- **Status (validated)**: This appears already addressed:
  - `ai_scientist/constraints.py` is SSOT for constraint names + ordering
  - `ai_scientist/coordinator._alm_constraint_names()` calls `get_constraint_names(..., for_alm=True)`
  - `ai_scientist/forward_model.compute_constraint_margins()` reorders margins into canonical order before returning
- **Action Items**:
  - [ ] Add a regression test asserting canonical constraint order is preserved end-to-end (including P3 `for_alm=True`)

---

## Phase 2: Math/Physics Bugs (Accuracy)

### Bug #5: 25% Elongation Error (Covariance Method)
- [ ] **MEDIUM** | `geometry.py:612-684`
- **Impact**: Covariance-based elongation systematically underestimates by ~25%
- **Root Cause**: Assumes elliptical cross-sections; stellarators have complex shapes
- **Current Method**: `elongation()` uses covariance eigenvalue ratio
- **Correct Method**: `elongation_isoperimetric()` uses Q = 4πA/P²
- **Status (validated)**: `elongation_isoperimetric()` already exists and is used in key call sites (e.g. `GeometerWorker`). `elongation()` remains covariance-based but appears unused elsewhere.
- **Fix**:
  - [ ] Decide whether to remove/rename `elongation()` (covariance) or keep it as a diagnostic
  - [ ] Add unit tests comparing isoperimetric elongation to a trusted reference (benchmark ellipse-fit or VMEC-derived)

---

### Bug #6: 5% Aspect Ratio Parametric Bias
- [ ] **MEDIUM** | `geometry.py:797-798`
- **Impact**: Aspect ratio biased by non-uniform θ sampling
- **Current Code**: Uniform θ samples, but poloidal arclength varies
- **Status (validated)**: `aspect_ratio_arc_length()` already exists (`geometry.py:816+`), but multiple call sites still use `aspect_ratio()` (unweighted mean).
- **Action Items**:
  - [ ] Wire `aspect_ratio_arc_length()` into benchmark-critical call sites (GeometerWorker / RL / differentiable / cycle_executor / prerelax)
  - [ ] Validate against constellaration reference implementation
  - [ ] Add tolerance check in tests (< 1% error)

---

### Bug #7: Hard-Coded Softmax Temperature
- [ ] **LOW** | `geometry.py:1070-1071`
- **Impact**: Feature importance weighting not tunable
- **Current Code (validated)**: `temperature = 100.0` and `Z_weights = torch.softmax(temperature * Z, dim=1)`
- **Fix**: Add `temperature` parameter with default 0.1
- **Action**: Low priority, purely cosmetic

---

### Bug #8: Perfect QI (0) Misinterpreted as Bad
- [ ] **MEDIUM** | `forward_model.py:526-529`
- **Impact**: Theoretically perfect QI=0 treated as constraint violation
- **Current Logic (validated)**: `_log10_or_large()` returns `10.0` for `qi is None or qi <= 0.0`, so `qi=0` is treated as a *large violation*.
- **Fix**: Add special case for qi ≈ 0:
  ```python
  if qi is not None and qi < 1e-10:
      log_qi = -10.0  # Perfect QI, well under threshold
  ```
- **Test**: Synthetic boundary with known perfect QI

---

### Bug #14: Eigenvalue Clamp Distorts Small Geometries
- [ ] **MEDIUM** | `geometry.py:656-665`
- **Impact**: Small boundary perturbations get clamped, losing gradient signal
- **Status (validated)**: The current `geometry.elongation()` implementation does not clamp a tensor named `eigenvalues`. It clamps intermediate eigenvalue-derived scalars (e.g. `l1_safe`, `l2_safe`) with small constants.
- **Fix**: If gradients are still distorted for small shapes, revisit those clamp constants and/or scale them by boundary size (e.g. via R00).

---

### Bug #15: abs(area) Hides Self-Intersection
- [ ] **LOW** | `geometry.py:804,856`
- **Impact**: Self-intersecting boundaries not detected, waste VMEC calls
- **Current Code (validated)**: `cross_section_area` is passed through `torch.abs(...)` in aspect-ratio computations
- **Fix**: Preserve sign, add warning/penalty for negative area
- **Action**: Add `self_intersection_penalty` to constraint system

---

## Phase 3: ML/Surrogate Bugs

### Bug #9: Fidelity Embedding Ignored in Ranking
- [ ] **MEDIUM** | `surrogate_v2.py:1130-1146`
- **Impact**: Multi-fidelity predictions don't account for fidelity level
- **Current Code**:
  ```python
  for model in self._models:
      outputs = model(X)  # NO fidelity argument!
  ```
- **Status (validated)**: `StellaratorNeuralOp.forward(..., fidelity=...)` exists and uses the embedding, but ranking currently doesn't pass `fidelity` (defaults to high-fidelity).
- **Fix**: Pass fidelity tensor to model:
  ```python
  fidelity_tensor = torch.full((X.shape[0],), fidelity_idx)
  outputs = model(X, fidelity=fidelity_tensor)
  ```
- **Action Items**:
  - [ ] Update `rank_candidates()` to accept fidelity parameter
  - [ ] Ensure `rank_candidates()` derives fidelity from candidate metadata (stage/status) where appropriate
  - [ ] Add test for fidelity-aware ranking

---

### Bug #10: Missing Normalization Stats → Garbage Predictions
- [ ] **HIGH** | `surrogate_v2.py:946-953`
- **Impact**: Predictions are meaningless if scaler not loaded
- **Current Behavior (validated)**: If `_y_*` denorm stats are missing, the code falls back to using normalized outputs (see the `else:` block at `surrogate_v2.py:946-953`) without raising.
- **Fix**: Raise explicit error:
  ```python
  if not hasattr(self, "_y_obj_mean"):
      raise RuntimeError("Normalization stats not loaded. Call load_checkpoint() first.")
  ```
- **Action Items**:
  - [ ] Add validation in `rank_candidates()` entry point
  - [ ] Log warning at checkpoint load if stats missing
  - [ ] Add test for missing normalization behavior

---

### Bug #11: Seed Time Collision (Low Entropy)
- [ ] **LOW** | `surrogate_v2.py:692-698`
- **Impact**: Same random seed if calls within 1 second
- **Current Code**: `seed = int(time.time()) % (2**32)`
- **Fix**: Use higher resolution time:
  ```python
  seed = int(time.time_ns()) % (2**32)
  ```

---

### Bug #12: No Inference Checkpointing for Large Batches
- [ ] **LOW** | `surrogate_v2.py:281-293`
- **Impact**: Memory spikes on large candidate batches
- **Fix**: Add optional batch chunking with intermediate results
- **Action**: Low priority, only relevant for >10k candidates

---

## Phase 4: Infrastructure Bugs

### Bug #13: Process Pool Serialization Overhead
- [ ] **LOW** | `forward_model.py:853-862`
- **Impact**: Slow parallel evaluation due to pickle overhead
- **Status (validated)**: `forward_model.py:853-862` is executor construction. Serialization overhead (when `pool_type=process`) comes from passing `boundary` payloads into `ProcessPoolExecutor` tasks in `forward_model_batch()`.
- **Current State**: Full objects serialized per evaluation
- **Fix Options**:
  - Use `multiprocessing.shared_memory` for boundary arrays
  - Pre-serialize with `cloudpickle` once
- **Action**: Document trade-offs, implement if bottleneck confirmed

---

## Phase 5: P1-Specific Pipeline Fixes

### Sampler Pipeline Failure
- [ ] **CRITICAL** | Only 2/200 candidates pass validation
- **Root Cause Analysis**:
  - [ ] Check if `checkpoints/diffusion_v2.pt` exists
  - [ ] Verify GeometerWorker validation thresholds
  - [ ] Audit seed generation in `ExplorationWorker` (fallback sampler seeds do not incorporate cycle index)
- **Action Items**:
  - [ ] Add logging for rejection reasons in GeometerWorker
  - [ ] Loosen initial constraints for exploration phase
  - [ ] Implement "warm-start" from known feasible P1 boundaries

---

### Missing P1-Specific Surrogate Training
- [ ] **HIGH** | Surrogate trained on P2/P3 data
- **Impact**: Poor prediction accuracy for P1-specific metrics
- **Action Items**:
  - [ ] Curate P1-focused training data (tight aspect ratio, low elongation)
  - [ ] Fine-tune surrogate on P1 distribution
  - [ ] Add P1-specific validation split

---

### Constraint Bound Mismatch
- [ ] **MEDIUM** | `constraints.py` vs problem definitions
- **Action Items**:
  - [ ] Verify `aspect_ratio_upper_bound = 4.0` matches P1Problem
  - [ ] Ensure `triangularity_upper_bound = -0.5` sign is correct
  - [ ] Add integration test: constraint values match across all modules

---

## Phase 6: Validation & Testing

### Unit Tests
- [ ] Test `problems.py` with None/NaN metrics
- [ ] Test `geometry.py` elongation against VMEC reference
- [ ] Test `constraints.py` normalized violations
- [ ] Test `surrogate_v2.py` with missing normalization stats

### Integration Tests
- [ ] End-to-end P1 cycle with mock backend (fast)
- [ ] End-to-end P1 cycle with real backend (slow, CI-skip)
- [ ] Constraint ordering consistency check

### Regression Tests
- [ ] Elongation calculation matches benchmark: `max_elongation ≈ 2.10`
- [ ] Aspect ratio calculation matches benchmark: `AR ≤ 4.0`
- [ ] No TypeError on QI=None scenarios

---

## Quick Reference: Priority Order

1. **Immediate (Today)**:
   - [ ] Bug #1: None QI crash
   - [ ] Bug #3: Silent mock backend
   - [ ] Sampler pipeline logging

2. **Short-term (This Week)**:
   - [ ] Bug #4: Constraint ordering
   - [ ] Bug #5: Elongation method
   - [ ] Bug #10: Normalization validation
   - [ ] P1 warm-start implementation

3. **Medium-term (Next Sprint)**:
   - [ ] Bug #2: Objective signs
   - [ ] Bug #6: Aspect ratio accuracy
   - [ ] Bug #9: Fidelity embedding
   - [ ] P1-specific surrogate training

4. **Low Priority (Backlog)**:
   - [ ] Bug #7, #11, #12, #13, #14, #15

---

## Success Criteria

- [ ] P1 pipeline produces >50 valid candidates per cycle
- [ ] At least 10% of candidates are feasible (meet all constraints)
- [ ] Best `max_elongation < 2.10` (beats ALM-NGOpt baseline)
- [ ] Zero crashes on edge cases (None QI, missing checkpoints)
- [ ] All 15 bugs addressed with tests

---

## References

- `ai_scientist/problems.py`: P1/P2/P3 problem definitions
- `ai_scientist/constraints.py`: Constraint bounds (SSOT)
- `ai_scientist/forward_model.py`: Physics evaluation pipeline
- `ai_scientist/optim/geometry.py`: Geometric calculations
- `ai_scientist/optim/surrogate_v2.py`: Neural operator surrogate
- `ai_scientist/backends/real.py`: VMEC++ integration
- `configs/experiment.p1.aso.yaml`: P1 experiment configuration
