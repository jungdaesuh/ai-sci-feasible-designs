# AI Scientist: Unified Execution Roadmap (V2.2)

**Status:** Active
**Context:** Synthesizes `V2_UPGRADE_PLAN`, `DATASET_INTEGRATION_PLAN`, and `HYBRID_ARCHITECTURE_PLAN`.
**Goal:** Enable "Gradient Descent on Inputs" using a Hybrid Neuro-Symbolic architecture pre-trained on 158k existing samples.

---

## 🛑 Critical Path (Sequential Dependencies)

You cannot skip steps. Phase 1 (Data) is required for Phase 2 (Training). Phase 2 is required for Phase 3 (Optimization).

### Phase 1: The Foundation (Data & Math)
*Focus: Establishing the "Engineer" (Analytic Math) and the "Map" (Clean Data).*

- [x] **1.1 Implement Differentiable Geometry (The Engineer)**
    - *Source:* `HYBRID_ARCHITECTURE_PLAN` (Phase 1)
    - *Action:* Create `ai_scientist/optim/geometry.py`.
    - *Specs:* Implement `elongation`, `curvature`, `aspect_ratio` using **pure PyTorch** tensor operations.
    - *Why:* We need perfect gradients for Problem 1 immediately.

- [x] **1.2 Build Offline Data Pipeline**
    - *Source:* `DATASET_INTEGRATION_PLAN` (Phase 0)
    - *Action:* Create `scripts/train_offline.py` and `ai_scientist/optim/data_loader.py`.
    - *Specs:*
        - Load 158k HF dataset.
        - Filter invalid geometry (using 1.1 checks).
        - Fit `LogRobustScaler` for Physics metrics.

### Phase 2: The Brain (Surrogate Training)
*Focus: Training the "Physicist" (Neural Network) on the cleaned data.*

- [ ] **2.1 Refactor Surrogate Architecture**
    - *Source:* `V2_UPGRADE_PLAN` (Phase 2) + `HYBRID_ARCHITECTURE_PLAN` (Phase 2)
    - *Action:* Update `NeuralOperatorSurrogate` in `ai_scientist/optim/surrogate.py`.
    - *Change:* **REMOVE** geometric output heads (Elongation, etc.). Keep only Physics heads ($W_{MHD}$, $QI$).
    - *Why:* The NN should not waste capacity learning math we implemented in 1.1.

- [ ] **2.2 Pre-train Physics Surrogate**
    - *Action:* Run `scripts/train_offline.py`.
    - *Output:* `checkpoints/surrogate_physics_v2.pt` and `checkpoints/scaler.pkl`.

- [ ] **2.3 Generate "Best-of-Failure" Seeds**
    - *Source:* `DATASET_INTEGRATION_PLAN` (Phase 0.3)
    - *Action:* Filter dataset for samples closest to Problem 1/2/3 targets.
    - *Output:* `seeds/p1_seeds.json`, `seeds/p2_seeds.json`.

### Phase 3: The Engine (Optimization Loop)
*Focus: assembling the Hybrid Loss function.*

- [ ] **3.1 Implement Hybrid Loss**
    - *Source:* `HYBRID_ARCHITECTURE_PLAN` (Phase 4)
    - *Action:* Update `ai_scientist/optim/differentiable.py`.
    - *Formula:* `Loss = w_p * NN(x) + w_e * Math(x)`.
    - *Detail:* Inject `geometry.elongation(x)` (from 1.1) directly into the loss graph.

- [ ] **3.2 Update Runner for Offline Artifacts**
    - *Source:* `DATASET_INTEGRATION_PLAN` (Phase 2)
    - *Action:* Update `ai_scientist/runner.py`.
    - *Logic:* If `use_offline_dataset=True`, load `checkpoints/surrogate_physics_v2.pt` and bypass online bootstrap training.

### Phase 4: The Autonomy (Hierarchical Agents)
*Focus: The "Manager" adjusting the weights.*

- [ ] **4.1 Implement Coordinator Logic**
    - *Source:* `V2_UPGRADE_PLAN` (Phase 5)
    - *Action:* Update `ai_scientist/coordinator.py`.
    - *Logic:*
        - Monitor stagnation.
        - If stuck, switch from "Optimization Worker" to "Exploration Worker" (VAE sampling).

---

## 📊 Progress Tracker

| Milestone | Goal | Dependency | Status |
| :--- | :--- | :--- | :--- |
| **M1** | Perfect Geometric Gradients | Phase 1.1 | ✅ Completed |
| **M2** | Data Pipeline & Seeds | Phase 1.2 | ⬜ Pending |
| **M3** | Trained Physics Model | Phase 2.2 | ⬜ Pending |
| **M4** | Hybrid Loop Running | Phase 3.1 | ⬜ Pending |
