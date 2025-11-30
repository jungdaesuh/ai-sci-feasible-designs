# ASO Surrogate → Real Physics Pipeline Implementation Plan

**Goal**: Implement a two-stage optimization workflow where the Neural Operator surrogate pre-screens candidate seeds before expensive real-physics ALM evaluation.

**Architecture**: Generate large seed pool → Surrogate ranking → Top-k selection → Real ALM optimization

---

## Phase 1: Configuration Setup

### 1.1 Add Surrogate Configuration to ASO Experiment Configs

- [ ] **Update `configs/experiment.p1.aso.yaml`**
  ```yaml
  surrogate:
    backend: neural_operator
    n_ensembles: 3
    learning_rate: 1e-3
    epochs: 50
    hidden_dim: 64
    use_offline_dataset: true
  ```

- [ ] **Update `configs/experiment.p2.aso.yaml`** with same surrogate section

- [ ] **Update `configs/experiment.p3.aso.yaml`** with same surrogate section

### 1.2 Ensure Surrogate Checkpoint Availability

- [ ] **Verify checkpoint exists**: `checkpoints/surrogate_physics_v2.pt`
  - If missing, either:
    - [ ] Run offline training script to generate checkpoint
    - [ ] Set `use_offline_dataset: false` for cold-start training

- [ ] **Add checkpoint validation** in `runner.py:_create_surrogate()`:
  - Log warning if `use_offline_dataset=true` but checkpoint missing
  - Provide clear error message with instructions

---

## Phase 2: Coordinator Enhancement

### 2.1 Modify Seed Pool Generation

**File**: `ai_scientist/coordinator.py`

- [ ] **Update `produce_candidates_aso()` to generate larger seed pool**

  Change:
  ```python
  seeds = self._prepare_seeds(initial_seeds, cycle, 1)
  ```

  To:
  ```python
  pool_multiplier = self.cfg.proposal_mix.surrogate_pool_multiplier  # default: 4.0
  pool_size = max(10, int(pool_multiplier * n_trajectories))
  raw_seeds = self._prepare_seeds(initial_seeds, cycle, pool_size)
  ```

- [ ] **Add configurable `n_trajectories` parameter** (default: 1 for single-trajectory ASO)

### 2.2 Implement Surrogate Pre-Screening

**File**: `ai_scientist/coordinator.py`

- [ ] **Add `_surrogate_rank_seeds()` method**:
  ```python
  def _surrogate_rank_seeds(
      self,
      seeds: List[Dict[str, Any]],
      top_k: int,
  ) -> List[Dict[str, Any]]:
      """Score seeds using surrogate predictions and return top-k."""
      if not self.surrogate or not hasattr(self.surrogate, 'predict_batch'):
          return seeds[:top_k]

      scores = []
      for seed in seeds:
          try:
              pred = self._get_surrogate_prediction(seed)
              # Lower objective + lower violation = better
              # Negate for descending sort
              score = -(pred.objective + 10.0 * max(0, pred.max_violation))
              scores.append((seed, score))
          except Exception as e:
              print(f"[Coordinator] Surrogate prediction failed: {e}")
              scores.append((seed, float('-inf')))

      scores.sort(key=lambda x: x[1], reverse=True)
      return [s[0] for s in scores[:top_k]]
  ```

- [ ] **Add `_get_surrogate_prediction()` helper**:
  ```python
  def _get_surrogate_prediction(self, seed: Dict[str, Any]) -> SurrogatePrediction:
      """Get surrogate prediction for a seed."""
      params = seed.get("params", seed)
      # Convert to format expected by surrogate
      return self.surrogate.predict(params)
  ```

### 2.3 Integrate Pre-Screening into ASO Flow

**File**: `ai_scientist/coordinator.py`

- [ ] **Modify `produce_candidates_aso()` main flow**:
  ```python
  def produce_candidates_aso(self, ...):
      # 1. Generate large seed pool
      pool_multiplier = self.cfg.proposal_mix.surrogate_pool_multiplier
      pool_size = max(10, int(pool_multiplier))
      raw_seeds = self._prepare_seeds(initial_seeds, cycle, pool_size)

      if not raw_seeds:
          print("[Coordinator] No valid seeds, returning empty")
          return []

      # 2. Surrogate pre-screening (NEW)
      if self.surrogate is not None:
          print(f"[Coordinator] Surrogate pre-screening {len(raw_seeds)} seeds...")
          ranked_seeds = self._surrogate_rank_seeds(raw_seeds, top_k=1)
          print(f"[Coordinator] Selected {len(ranked_seeds)} top seeds for ALM")
      else:
          print("[Coordinator] No surrogate available, using first seed")
          ranked_seeds = raw_seeds[:1]

      # 3. Run trajectory with top seed(s)
      candidates = []
      for i, seed in enumerate(ranked_seeds):
          traj = TrajectoryState(id=i, seed=seed)
          traj_candidates = self._run_trajectory_aso(
              traj=traj,
              eval_budget=eval_budget // len(ranked_seeds),
              cycle=cycle,
              experiment_id=experiment_id,
              config=config,
          )
          candidates.extend(traj_candidates)

      # 4. Persist telemetry
      self._persist_telemetry(experiment_id)

      return candidates
  ```

---

## Phase 3: Surrogate Interface Enhancement

### 3.1 Verify/Add Required Surrogate Methods

**File**: `ai_scientist/optim/surrogate_v2.py`

- [ ] **Verify `NeuralOperatorSurrogate.predict()` exists and returns proper structure**
  - Should return objective estimate
  - Should return constraint violation estimates (or max_violation)

- [ ] **Add `predict_batch()` method if missing** (for efficiency):
  ```python
  def predict_batch(self, params_list: List[Dict]) -> List[SurrogatePrediction]:
      """Batch prediction for multiple candidates."""
      # Implementation depends on model architecture
      pass
  ```

- [ ] **Ensure consistent return type** (`SurrogatePrediction` or similar dataclass)

### 3.2 Add Fallback for Non-Neural Surrogates

**File**: `ai_scientist/coordinator.py`

- [ ] **Handle case when surrogate is `SurrogateBundle` (random forest)**
  - Either skip pre-screening entirely
  - Or implement RF-compatible ranking interface

---

## Phase 4: Telemetry & Observability

### 4.1 Add Surrogate Screening Telemetry

**File**: `ai_scientist/coordinator.py`

- [ ] **Log pre-screening statistics**:
  ```python
  def _surrogate_rank_seeds(self, seeds, top_k):
      # ... ranking logic ...

      # Log statistics
      if scores:
          all_scores = [s[1] for s in scores if s[1] > float('-inf')]
          print(f"[Coordinator] Surrogate scores: "
                f"best={max(all_scores):.4f}, "
                f"worst={min(all_scores):.4f}, "
                f"selected_threshold={scores[top_k-1][1]:.4f}")

      return [s[0] for s in scores[:top_k]]
  ```

- [ ] **Add to telemetry events**:
  ```python
  self.telemetry.append({
      # ... existing fields ...
      "surrogate_pool_size": len(raw_seeds),
      "surrogate_selected": len(ranked_seeds),
      "surrogate_best_score": best_score,
      "surrogate_threshold_score": threshold_score,
  })
  ```

---

## Phase 5: Testing & Validation

### 5.1 Unit Tests

- [ ] **Test surrogate ranking logic**:
  - `test_surrogate_rank_seeds_returns_top_k`
  - `test_surrogate_rank_seeds_handles_empty_pool`
  - `test_surrogate_rank_seeds_handles_prediction_failure`

- [ ] **Test integration with produce_candidates_aso**:
  - `test_aso_uses_surrogate_when_available`
  - `test_aso_falls_back_when_no_surrogate`

### 5.2 Integration Tests

- [ ] **End-to-end ASO with surrogate pre-screening**:
  - Generate 100 seeds
  - Verify surrogate ranking is called
  - Verify only top-k proceed to ALM
  - Verify final candidates are returned

### 5.3 Performance Validation

- [ ] **Compare ASO performance with/without surrogate pre-screening**:
  - Track: evals_to_feasible, final_objective, wall_clock_time
  - Expected: fewer ALM evals with surrogate, similar or better final quality

---

## Phase 6: Documentation

- [ ] **Update `docs/run_protocol.md`** with surrogate configuration instructions

- [ ] **Update `docs/ASO_V4_IMPLEMENTATION_GUIDE.md`** with pre-screening architecture

- [ ] **Add inline code comments** explaining the surrogate pipeline flow

---

## Implementation Order

| Priority | Task | Effort | Dependencies |
|----------|------|--------|--------------|
| P0 | 1.1 Update YAML configs | Low | None |
| P0 | 1.2 Verify/create checkpoint | Medium | 1.1 |
| P1 | 2.1 Modify seed pool generation | Low | 1.1 |
| P1 | 2.2 Implement `_surrogate_rank_seeds()` | Medium | 3.1 |
| P1 | 2.3 Integrate into ASO flow | Medium | 2.1, 2.2 |
| P2 | 3.1 Verify surrogate interface | Low | None |
| P2 | 3.2 Add fallback logic | Low | 3.1 |
| P3 | 4.1 Add telemetry | Low | 2.3 |
| P3 | 5.x Testing | Medium | 2.3 |
| P3 | 6.x Documentation | Low | All |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Surrogate checkpoint missing | Add clear error message with training instructions |
| Surrogate predictions inaccurate | Keep fallback path (no pre-screening) |
| Pre-screening too slow | Add batch prediction, limit pool size |
| Top-k selection misses good candidates | Use larger pool multiplier, add diversity |

---

## Success Criteria

- [ ] ASO mode can run with `surrogate.backend: neural_operator`
- [ ] Surrogate pre-screens 10-100 seeds before ALM
- [ ] Only top-k (default 1) seeds proceed to expensive real-physics ALM
- [ ] Telemetry captures pre-screening statistics
- [ ] Fallback works when surrogate unavailable
- [ ] No regression in ASO convergence quality
