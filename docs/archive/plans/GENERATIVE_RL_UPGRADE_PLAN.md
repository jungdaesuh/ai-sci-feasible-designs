# AI Scientist: Generative & RL Architecture Upgrade Plan

**Status**: Updated 2025-12-09 to reflect actual implementation state
**Architecture**: StellarForge Quad-Hybrid System

This document synthesizes the architectural decisions, analysis, and implementation roadmap for upgrading the `ai-sci-feasible-designs` repository. It focuses on integrating a pre-trained Generative Model ("The Dreamer"), a Geometric Pre-relaxer ("The Filter"), a Reinforcement Learning agent ("The Engineer"), and a Surrogate Critic ("The Frankenstein") to solve the ConStellaration Fusion Challenge.

---

## 1. The Core Problem: "Cold Start" & Optimization Valleys

The current system relies on "on-the-fly" training for its generative models and standard optimization (ALM/Gradient Descent) for refinement.

*   **Generative Failure:** The VAE/Diffusion models are initialized from scratch at the start of each experiment. Without access to the 160k offline dataset, they fail to learn valid stellarator topology, producing garbage seeds.
*   **Optimization Failure:** Standard optimizers struggle with the sharp discontinuities ("cliffs") of physics constraints (e.g., Vacuum Well stability), often getting stuck in local minima.

---

## 2. The Solution: "StellarForge" Quad-Hybrid Architecture

We are moving to a **four-stage pipeline** that leverages the strengths of different AI modalities.

### Component A: The "Dreamer" (Generative Model)
*   **Role:** Knowledge Base & Seed Generator.
*   **Model:** Conditional Diffusion Model (1D).
*   **Implementation**: `ai_scientist/optim/generative.py` - **✅ 90% COMPLETE**
*   **Workflow:**
    *   **Offline Pre-training:** Train on the massive ~160k Hugging Face dataset (`proxima-fusion/constellaration`) to learn the manifold of plausible physics shapes.
    *   **Inference:** During experiments, the agent "prompts" the model (e.g., "Generate designs with high stability") to get high-quality "warm start" seeds.
    *   **Fine-Tuning:** Periodically fine-tune on new, elite candidates discovered during the run ("Active Learning").

### Component B: The "Pre-relaxer" (Geometric Filter)
*   **Role:** Fast Geometric Validation & Smoothing.
*   **Model:** Gradient-based optimization on geometric energy (LEGO-xtal inspired).
*   **Implementation**: `ai_scientist/optim/prerelax.py` - **✅ 100% MODULE COMPLETE** (Integration pending)
*   **Workflow:**
    *   Receives "noisy" geometries from the Dreamer.
    *   Performs fast gradient descent (~50 steps, milliseconds per candidate) to minimize:
        - Mean curvature (smoothness)
        - Aspect ratio deviation from target
        - Elongation violations (>10 is invalid)
    *   Filters out geometrically impossible candidates BEFORE expensive physics simulation.
    *   Expected impact: 30-40% reduction in VMEC++ crashes.

### Component C: The "Critic" (Surrogate Environment)
*   **Role:** Fast Physics Simulator.
*   **Model:** Neural Operator Ensemble (or "Frankenstein" multi-output).
*   **Implementation**: `ai_scientist/optim/surrogate_v2.py` - **✅ COMPLETE**
*   **Workflow:**
    *   Combines predictions for separate metrics (Aspect Ratio, Stability, Coil Complexity) into a unified Reward Function.
    *   Provides millisecond-latency feedback to the RL agent, replacing the slow `VMEC++` code during the inner optimization loop.

### Component D: The "Engineer" (RL Agent)
*   **Role:** Precision Refinement.
*   **Model:** PPO (Proximal Policy Optimization).
*   **Implementation**: `ai_scientist/rl_env.py`, `ai_scientist/optim/rl_ppo.py` - **✅ 95% COMPLETE**
*   **Workflow:**
    *   **Input:** Receives a "Dreamed" and "Pre-relaxed" seed from Components A & B.
    *   **Action:** Performs "Micro-Surgery" (small delta adjustments) on Fourier coefficients.
    *   **Reward:** Optimizes the "Frankenstein" score, learning to navigate constraints via negative "Cliff Penalties."
    *   **Output:** Produces a highly refined candidate for final verification.

---

## 3. Implementation Roadmap

### Phase 1: Generative Model Upgrade (Immediate Priority) - ✅ 90% COMPLETE

**Status**: Core implementation exists, integration gaps remain.

**Completed Components:**
1.  ✅ **PCA Compression** (`ai_scientist/optim/generative.py:283-291`)
    *   Implemented: 661 → 50 latent dimensions
    *   Captures >85% variance
    *   Inverse transform for sampling
2.  ✅ **Network Architecture** (`ai_scientist/optim/generative.py:54-116`)
    *   `hidden_dim=2048` (4 layers)
    *   Sinusoidal embeddings
    *   GELU activation
3.  ✅ **Checkpoint Loading** (`ai_scientist/optim/generative.py:340-389`)
    *   `load_checkpoint()` / `state_dict()` methods
    *   Integration in `experiment_setup.py:72-78`
4.  ✅ **Conditional Sampling** (`ai_scientist/optim/generative.py:391-473`)
    *   Target metrics: (ι, A, nfp, N)
    *   Dynamic context passing

**Remaining Work:**
1.  🔴 **Config Loader Bug** (`ai_scientist/config.py:612-623`)
    *   Fix `_generative_config_from_dict()` to parse new fields
    *   Add: `checkpoint_path`, `device`, `hidden_dim`, `n_layers`, `pca_components`, `batch_size`, `diffusion_timesteps`
2.  🔴 **Offline Training Script**
    *   Create `scripts/train_generative_offline.py`
    *   Train on 160k ConStellaration dataset
    *   Save checkpoint to `checkpoints/diffusion_paper_spec.pt`
3.  🟡 **Minor Tweaks**
    *   Change default epochs: 200 → 250 (match paper)

### Phase 2: Pre-relaxation ("The Filter") - ✅ 100% MODULE, 🔴 0% INTEGRATION

**Status**: Fully implemented module, not integrated into pipeline.

**Completed Module** (`ai_scientist/optim/prerelax.py`):
1.  ✅ `geometric_energy()` function (lines 18-57)
    *   Mean curvature penalty
    *   Aspect ratio targeting
    *   Elongation constraint
2.  ✅ `prerelax_boundary()` function (lines 60-114)
    *   Gradient descent optimization
    *   Configurable steps, learning rate
    *   Device support (CPU/CUDA/MPS)

**Required Integration:**
1.  🔴 **Create PreRelaxWorker** (`ai_scientist/workers.py`)
    *   Follow pattern of `ExplorationWorker`, `OptimizationWorker`
    *   Input: List of candidate params
    *   Output: List of geometrically smoothed params
2.  🔴 **Add Config Section**
    ```yaml
    prerelax:
      enabled: true
      steps: 50
      lr: 0.01
      threshold: 0.1
      device: "cpu"  # or "cuda", "mps"
    ```
3.  🔴 **Wire into Coordinator** (`ai_scientist/coordinator.py`)
    *   Insert between Dream and RL-Refine
    *   Call: `self.prerelax_worker.run({"candidates": seeds})`

### Phase 3: RL Agent Integration - ✅ 95% COMPLETE

**Status**: Core implementation complete, workflow optimization needed.

**Completed Components:**
1.  ✅ **Gym Environment** (`ai_scientist/rl_env.py:23-188`)
    *   `StellaratorEnv` with surrogate integration
    *   State: Fourier coefficients (661-dim)
    *   Action: Continuous deltas
    *   Reward: Cliff-penalized composite score
2.  ✅ **PPO Agent** (`ai_scientist/optim/rl_ppo.py:26-190`)
    *   Actor-critic architecture
    *   GAE advantages
    *   PPO clipping (ε=0.2)
3.  ✅ **Worker Integration** (`ai_scientist/workers.py:273-387`)
    *   `RLRefinementWorker` class
    *   Training loop with buffer
    *   Best-candidate tracking

**Optimization Needed:**
1.  🟠 **Workflow Efficiency** (`coordinator.py:169-170`)
    *   Current: RL refines ALL seeds (wasteful)
    *   Optimal: Surrogate rank → Select top-K → RL refine top-K only
    *   Expected speedup: 5-10x
2.  🟡 **Hyperparameter Tuning**
    *   Validate against BoltzGen paper
    *   Test: learning_rate, gamma, gae_lambda

---

## 4. Operational Workflow: The "Fast-Slow" Loop

### Current Implementation (coordinator.py:163-179)

```
1. Dream    → ExplorationWorker (Diffusion/VAE) generates N seeds
2. RL       → RLRefinementWorker refines ALL seeds [INEFFICIENT]
3. Geometer → GeometerWorker filters geometrically valid
4. Optimize → OptimizationWorker (surrogate-guided GD)
```

### Proposed Optimal Workflow

| Stage | Action | Compute Cost | Status |
| :--- | :--- | :--- | :--- |
| **1. Dream** | Generative Model creates 1,000 "Warm Start" seeds. | Low (GPU inference) | ✅ Implemented |
| **2. Pre-relax** | Fast geometric smoothing (50 steps GD per seed). | Very Low (CPU/GPU) | ✅ Module ready, needs integration |
| **3. Geometer** | Validate geometric constraints (Jacobian, Elongation). | Negligible (vectorized) | ✅ Implemented |
| **4. Filter** | Surrogate ranks candidates, select top 100. | Low (GPU inference) | ⚠️ Partially implemented |
| **5. Refine** | RL Agent polishes top-100 in Surrogate Environment. | Medium (GPU training) | ✅ Implemented, needs reordering |
| **6. Verify** | Run `VMEC++` (Real Physics) on refined top-100. | High (CPU Cluster) | ✅ Implemented |
| **7. Learn** | Add real `VMEC++` results to dataset. Retrain Surrogate & Generative Model. | Medium (GPU training) | ❌ Missing periodic trigger |

**Key Changes from Current**:
- **Add Step 2**: Pre-relaxation (module exists, just wire it in)
- **Reorder Step 5**: RL refinement AFTER surrogate ranking (not before)
- **Add Step 7**: Periodic retraining (trigger on cycle % 5 == 0)

---

## 5. Decision Log

*   **Why Diffusion?** We need to model the complex, non-linear manifold of valid stellarators. VAEs often produce blurry/averaged outputs, while Diffusion preserves topological fidelity.
*   **Why Offline Training?** Online training on <100 samples is mathematically insufficient for deep learning. We must transfer learn from the 160k dataset.
*   **Why Pre-relaxation?** Generative models produce noisy geometries. Fast geometric filtering (milliseconds) prevents wasted compute on invalid candidates in VMEC++ (minutes).
*   **Why RL?** Gradient descent gets stuck on constraint cliffs (e.g., $W < 0$). RL policies trained with penalties learn to robustly avoid these regions.
*   **Why Quad-Hybrid?** Each component addresses a different failure mode:
    - Dreamer: Cold start problem
    - Pre-relaxer: Geometric invalidity
    - Critic: Fast feedback
    - Engineer: Constraint cliff navigation

---

## 6. Implementation Status Summary

| Component | Module Status | Integration Status | Priority |
|-----------|---------------|-------------------|----------|
| **Dreamer (Diffusion)** | ✅ 90% Complete | ⚠️ Config bug | 🔴 Critical |
| **Pre-relaxer (Geometric)** | ✅ 100% Complete | ❌ Not wired | 🔴 Critical |
| **Critic (Surrogate)** | ✅ 100% Complete | ✅ Integrated | ✅ Done |
| **Engineer (RL PPO)** | ✅ 95% Complete | ⚠️ Inefficient order | 🟠 High |
| **Offline Training** | ❌ Missing script | N/A | 🔴 Critical |
| **Periodic Retraining** | ❌ Not implemented | N/A | 🟠 High |

**Overall Progress**: **Architecture 85% complete, Integration 60% complete**

---

## 7. Next Steps (Prioritized)

### Week 1: Critical Fixes
1.  **Fix Config Loading** (30 minutes) - `config.py:612-623`
2.  **Integrate Pre-relaxation** (4 hours)
    - Create `PreRelaxWorker`
    - Add config section
    - Wire into coordinator
3.  **Create Offline Training Script** (2 hours) - `scripts/train_generative_offline.py`
4.  **Optimize RL Workflow** (2 hours) - Move RL after surrogate ranking

### Week 2: Production Readiness
5.  **Add Periodic Retraining** (3 hours) - Step 7 in operational workflow
6.  **Hyperparameter Tuning Study** (1 day) - Validate against papers
7.  **End-to-End Testing** (2 days) - Run full pipeline on P2/P3 benchmarks

### Week 3: Documentation & Deployment
8.  **Update User Documentation**
9.  **Create Training Tutorial**
10. **Production Deployment**

**Estimated Total Effort**: 5 days to production-ready Quad-Hybrid system

---

## 8. Expected Performance Improvements

Based on paper results (Padidar et al., Stark et al., Ridwan et al.) and current benchmarks:

| Metric | Current Baseline | After Integration | Target (Full Training) |
|--------|------------------|-------------------|----------------------|
| **Feasibility Rate** | ~5% | 30-40% (Pre-relax) | 70-85% (Offline Dreamer) |
| **VMEC++ Calls to Baseline** | ~500 | ~200 (Pre-relax filter) | ~40-60 (Full Quad-Hybrid) |
| **Wall-Clock Time** | ~8 hours | ~4 hours | ~1-2 hours |
| **L_∇B (P2 Objective)** | 8.61 | ~9.0 | 10-12 |

**Key Insight**: Pre-relaxation alone (already implemented!) can provide immediate 2x speedup by filtering invalid geometries. Full Quad-Hybrid unlocks order-of-magnitude improvements.

---

## Appendix A: File Locations

### Implemented Modules
- **Dreamer**: `ai_scientist/optim/generative.py` (DiffusionDesignModel)
- **Pre-relaxer**: `ai_scientist/optim/prerelax.py` (prerelax_boundary)
- **Critic**: `ai_scientist/optim/surrogate_v2.py` (NeuralOperatorSurrogate)
- **Engineer**: `ai_scientist/rl_env.py` + `ai_scientist/optim/rl_ppo.py`
- **Workers**: `ai_scientist/workers.py` (ExplorationWorker, RLRefinementWorker, etc.)
- **Coordinator**: `ai_scientist/coordinator.py` (Orchestration logic)

### Configuration
- **Experiment Config**: `ai_scientist/config.py` (GenerativeConfig, etc.)
- **Setup**: `ai_scientist/experiment_setup.py` (create_generative_model, create_surrogate)

### Scripts
- **Surrogate Training**: `scripts/train_offline.py` (✅ Exists)
- **Generative Training**: `scripts/train_generative_offline.py` (❌ Needs creation)

---

## Appendix B: References

1. **Padidar et al. (2025)** - "Diffusion Models for Fusion Reactor Design" (arxiv:2511.20445v1)
   - Conditional DDPM, 2048 hidden dim, 4 layers, PCA 661→50
2. **Stark et al. (2025)** - "BoltzGen: Structure Prediction with RL" (biorxiv:2025.11.20.689494v1)
   - PPO for design optimization, cliff penalties, 66% success rate
3. **Ridwan et al. (2025)** - "LEGO-xtal: Geometric Pre-relaxation" (arxiv:2506.08224v2)
   - Fast gradient descent on SO(3) descriptors, 25→1700x structure improvement

---

**Document Version**: 2.0 (Updated 2025-12-09)
**Status**: Ready for Implementation Sprint
**Contact**: See `docs/GENERATIVE_RL_UPGRADE_PLAN_REVIEW.md` for detailed audit
