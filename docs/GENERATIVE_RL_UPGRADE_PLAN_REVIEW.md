# Review: Generative & RL Upgrade Plan (Updated)

**Date**: 2025-12-09 (Updated after codebase audit)
**Target Document**: `docs/GENERATIVE_RL_UPGRADE_PLAN.md`
**Reference Document**: `docs/STELLARFORGE_IMPLEMENTATION_PLAN.md` (v2 Paper-Backed)

## 1. Executive Summary

**CORRECTION**: An audit of the codebase reveals that **most StellarForge components are already implemented** (as of 2025-12-09). The original review incorrectly identified several features as "missing" when they actually exist in the code.

**Updated Verdict**: The architectural components are **85-90% complete**, but **integration gaps** remain. The system has all four modules (Quad-Hybrid architecture) but they are not fully wired together in the operational pipeline.

---

## 2. Implementation Status by Phase

### Phase 1: Generative Model (Diffusion) - ✅ 90% COMPLETE

**Files**: `ai_scientist/optim/generative.py`, `ai_scientist/config.py`, `ai_scientist/experiment_setup.py`

#### ✅ Implemented Features (Previously Incorrectly Marked as Missing)

1. **PCA Compression** (Lines 283-291 in generative.py)
   - ✅ `sklearn.decomposition.PCA` integrated
   - ✅ 661 → 50 dimensional reduction
   - ✅ `fit_transform()` during training
   - ✅ `inverse_transform()` during sampling

2. **Network Scaling** (Lines 54-116 in generative.py)
   - ✅ `hidden_dim=2048` (default parameter, line 139)
   - ✅ `n_layers=4` (default parameter, line 141)
   - ✅ `batch_size=4096` (default parameter, config.py:204)
   - ✅ Sinusoidal embeddings (lines 30-42)
   - ✅ GELU activation (line 90)

3. **Checkpoint Loading** (Lines 340-389 in generative.py)
   - ✅ `load_checkpoint()` method implemented
   - ✅ `state_dict()` includes PCA serialization
   - ✅ Integration in `experiment_setup.py:72-78`

4. **Conditional Sampling** (Lines 391-473 in generative.py)
   - ✅ Target metrics: (ι, A, nfp, N) - line 122-127
   - ✅ Dynamic context passing in workers.py:140-145

#### 🔴 Remaining Gaps (Phase 1)

1. **Config Loader Bug** (Critical)
   - File: `ai_scientist/config.py:612-623`
   - Issue: `_generative_config_from_dict()` doesn't parse new StellarForge fields
   - Missing fields: `checkpoint_path`, `device`, `hidden_dim`, `n_layers`, `pca_components`, `batch_size`, `diffusion_timesteps`
   - Impact: Config YAML values are ignored, defaults used instead

2. **Missing Offline Training Script**
   - No `scripts/train_generative_offline.py` to train on 160k ConStellaration dataset
   - Current `scripts/train_offline.py` trains surrogate, not generative model
   - Required for pre-training on offline data

3. **Epoch Count Discrepancy**
   - Paper spec: 250 epochs
   - Current default: 200 epochs
   - Minor, but should match paper for reproducibility

---

### Phase 2: Geometric Pre-relaxation - ✅ 100% MODULE COMPLETE, ❌ 0% INTEGRATION

**File**: `ai_scientist/optim/prerelax.py` (FULLY IMPLEMENTED)

#### ✅ Implemented Features (Incorrectly Marked as "MISSING" in Original Review)

1. **Geometric Energy Function** (Lines 18-57)
   - ✅ Mean curvature penalty (smoothness)
   - ✅ Aspect ratio target maintenance
   - ✅ Elongation constraint (>10 penalty)

2. **Optimization Function** (Lines 60-114)
   - ✅ `prerelax_boundary()` with gradient descent
   - ✅ Configurable steps, learning rate, target AR
   - ✅ Returns (optimized_params, final_energy)

3. **Device Support**
   - ✅ CPU, CUDA, MPS support

#### 🔴 Critical Integration Gap

**The pre-relaxation module exists but is NOT called in the operational pipeline.**

- ❌ Not integrated in `ai_scientist/forward_model.py` (as proposed in STELLARFORGE_IMPLEMENTATION_PLAN.md:298-323)
- ❌ Not instantiated in `ai_scientist/coordinator.py`
- ❌ No `PreRelaxWorker` class exists (should follow pattern of ExplorationWorker, OptimizationWorker)

**Required Actions**:
1. Create `PreRelaxWorker` in `workers.py`
2. Add to coordinator workflow between Dream and RL-Refine
3. Add config toggle: `prerelax.enabled`, `prerelax.steps`, `prerelax.threshold`

---

### Phase 3: RL Agent (PPO) - ✅ 95% COMPLETE

**Files**: `ai_scientist/rl_env.py`, `ai_scientist/optim/rl_ppo.py`, `ai_scientist/workers.py`

#### ✅ Implemented Features (Incorrectly Marked as "Vague" in Original Review)

1. **Gym Environment** (`rl_env.py:23-188`)
   - ✅ `StellaratorEnv` with surrogate integration
   - ✅ State space: Fourier coefficients (661-dim)
   - ✅ Action space: Continuous deltas
   - ✅ Reward shaping with cliff penalties (lines 84-141)
   - ✅ Target metrics support

2. **PPO Implementation** (`optim/rl_ppo.py:26-190`)
   - ✅ Actor-critic network (layer_init with orthogonal weights)
   - ✅ GAE advantages computation
   - ✅ PPO clipping (clip_coef=0.2)
   - ✅ Value function training

3. **Worker Integration** (`workers.py:273-387`)
   - ✅ `RLRefinementWorker` class
   - ✅ PPO training loop with buffer
   - ✅ Best-candidate tracking
   - ✅ Called in coordinator EXPLOIT and HYBRID modes

#### 🟠 Remaining Gaps (Phase 3)

1. **Workflow Inefficiency**
   - Current: RL refines ALL seeds (lines 169-170 in coordinator.py)
   - Optimal: RL should refine only top-K after surrogate ranking
   - Impact: Wasting compute on poor candidates

2. **Hyperparameter Tuning**
   - Need to validate against BoltzGen paper specs
   - Current: Generic PPO defaults
   - Should test: learning_rate, gamma, gae_lambda variations

---

## 3. Architectural Clarification: Quad-Hybrid, Not Tri-Hybrid

**Correction**: The system implements a **Quad-Hybrid** architecture with four distinct modules:

1. **Dreamer** (Generative Diffusion) - `DiffusionDesignModel`
2. **Pre-relaxer** (Geometric) - `prerelax_boundary()`
3. **Engineer** (RL PPO) - `RLRefinementWorker`
4. **Critic** (Surrogate) - `NeuralOperatorSurrogate`

The original review's suggestion to "make it Quad-Hybrid" was correct, but the implementation already exists—it's just not fully integrated.

---

## 4. Operational Workflow Correction

### Current Actual Workflow (coordinator.py:163-179)

```
1. Dream    → ExplorationWorker generates seeds (Diffusion/VAE)
2. RL       → RLRefinementWorker refines ALL seeds ❌ INEFFICIENT
3. Geometer → GeometerWorker filters geometrically valid
4. Optimize → OptimizationWorker (surrogate-guided GD)
```

**Missing**: Pre-relaxation step (module exists but not called)

### Proposed Optimal Workflow

```
1. Dream     → Generate N seeds (e.g., 1000)
2. Pre-relax → Fast geometric fix (filter ~30-40% invalid) [NEEDS INTEGRATION]
3. Geometer  → Validate remaining (~600-700 seeds)
4. Surrogate → Rank and select top-K (e.g., 100)
5. RL-Refine → Micro-surgery on top-K only (not all seeds) [NEEDS REORDERING]
6. Optimize  → Final GD refinement
7. Verify    → VMEC++ on best candidates
```

**Required Changes**:
- Insert Pre-relax between steps 1 and 2
- Move RL-Refine AFTER surrogate ranking (step 5)
- Reduce RL compute by only refining top-K

---

## 5. Updated Section Assessment

| Section | Original Assessment | Corrected Assessment |
| :--- | :--- | :--- |
| **Phase 1 (Generative)** | ❌ Incomplete | ✅ 90% Complete (Config bug + training script needed) |
| **Phase 2 (Pre-relax)** | 🟠 Missing | ✅ Module 100% Complete (Integration 0%) |
| **Phase 3 (RL Agent)** | ⚠️ Vague | ✅ 95% Complete (Workflow optimization needed) |
| **Architecture** | Tri-Hybrid | Quad-Hybrid (Corrected) |
| **Workflow** | Needs Pre-relax | Pre-relax exists, needs integration + RL reordering |

---

## 6. Prioritized Action Items

### 🔴 Critical (Blocking Production Use)

1. **Fix Config Loading** (`ai_scientist/config.py:612-623`)
   ```python
   def _generative_config_from_dict(data: Mapping[str, Any] | None) -> GenerativeConfig:
       config = data or {}
       return GenerativeConfig(
           enabled=bool(config.get("enabled", False)),
           backend=str(config.get("backend", "vae")),
           latent_dim=int(config.get("latent_dim", 16)),
           learning_rate=float(config.get("learning_rate", 1e-3)),
           epochs=int(config.get("epochs", 100)),
           kl_weight=float(config.get("kl_weight", 0.001)),
           # ADD THESE:
           checkpoint_path=Path(config["checkpoint_path"]) if config.get("checkpoint_path") else None,
           device=str(config.get("device", "cpu")),
           hidden_dim=int(config.get("hidden_dim", 2048)),
           n_layers=int(config.get("n_layers", 4)),
           pca_components=int(config.get("pca_components", 50)),
           batch_size=int(config.get("batch_size", 4096)),
           diffusion_timesteps=int(config.get("diffusion_timesteps", 200)),
       )
   ```

2. **Integrate Pre-relaxation** (3 sub-tasks)
   - Create `PreRelaxWorker(Worker)` in `workers.py`
   - Add config section: `prerelax: {enabled: bool, steps: int, threshold: float}`
   - Wire into coordinator.py workflow (after Dream, before Geometer)

3. **Create Offline Training Script**
   - File: `scripts/train_generative_offline.py`
   - Load 160k ConStellaration dataset
   - Train `DiffusionDesignModel` with paper specs
   - Save to `checkpoints/diffusion_paper_spec.pt`

### 🟠 High Priority (Performance Optimization)

4. **Fix RL Workflow Efficiency** (coordinator.py:169-170)
   - Move RL refinement AFTER surrogate ranking
   - Only refine top-K candidates (e.g., K=100)
   - Current: Wastes compute refining all seeds

5. **Add Periodic Retraining** (coordinator.py)
   - Implement "Learn" step from UPGRADE_PLAN operational workflow
   - Trigger: Every N cycles or on HV stagnation
   - Retrain both surrogate and generative model

### 🟡 Medium Priority (Reproducibility)

6. **Match Paper Epochs**
   - Change default epochs: 200 → 250
   - Align with Padidar et al. (2025) spec

7. **Hyperparameter Tuning Study**
   - Validate RL hyperparameters against BoltzGen
   - Document optimal settings in config

---

## 7. Conclusion

**Key Correction**: The original review was overly pessimistic. The codebase has made substantial progress:

- ✅ **All four architectural modules exist and are mostly complete**
- ✅ **Paper-spec network architectures are implemented**
- ✅ **PCA, checkpointing, conditional sampling all work**

**Remaining Work**: Integration, not implementation. The building blocks are there; they need to be properly wired together.

**Estimated Effort**:
- Critical fixes: 1-2 days
- High priority: 2-3 days
- **Total**: ~5 days to production-ready Quad-Hybrid system

**Next Actions**:
1. Update `GENERATIVE_RL_UPGRADE_PLAN.md` to reflect Quad-Hybrid architecture
2. Fix config loading (30 minutes)
3. Integrate pre-relaxation (4 hours)
4. Create offline training script (2 hours)
5. Optimize RL workflow (2 hours)
