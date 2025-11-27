# Codebase Verification Review: AI Scientist V2.2 Roadmap

**Date:** November 27, 2025
**Auditor:** Gemini CLI Agent

## 1. Executive Summary
The documentation (`AI_SCIENTIST_V2_ROADMAP.md`, `AI_SCIENTIST_HYBRID_ARCHITECTURE_PLAN.md`) describes a **future "V2.2" state** that is **not yet implemented**. The current codebase represents "V2.0".

While the fundamental infrastructure (PyTorch optimization, Neural Surrogates, Coordinator) is present, the key innovation—the **Hybrid Neuro-Symbolic Architecture** (splitting "Physicist" and "Engineer")—is missing. The codebase currently relies on a monolithic neural network for both physics and geometry, which the roadmap explicitly identifies as the bottleneck to be fixed.

---

## 2. Detailed Gap Analysis

### A. The "Engineer" (Differentiable Geometry)
*   **Doc Claim:** "Implement `elongation`, `curvature`, `aspect_ratio` using **pure PyTorch** tensor operations... in `ai_scientist/optim/geometry.py`."
*   **Codebase Reality:**
    *   `ai_scientist/optim/geometry.py` exists but only implements **Phase 1.1 (Coordinate Transformation)**: `fourier_to_real_space`, `to_cartesian`, `surface_to_point_cloud`.
    *   **Missing:** It does **not** implement the differentiable metric functions (`elongation`, `curvature`, etc.).
    *   **Impact:** The optimizer cannot currently obtain "perfect gradients" for geometry as claimed.

### B. The "Physicist" (Surrogate Models)
*   **Doc Claim:** "Refactor `NeuralOperatorSurrogate`... **REMOVE** geometric output heads... Keep only Physics heads ($W_{MHD}$, $QI$)."
*   **Codebase Reality:**
    *   `ai_scientist/optim/surrogate_v2.py` currently implements `StellaratorNeuralOp` with **all heads enabled**:
        ```python
        self.head_elongation = nn.Linear(hidden_dim, 1)
        ```
    *   **Impact:** The model is still wasting capacity learning geometry that should be analytical.

### C. The Engine (Optimization Loop)
*   **Doc Claim:** "Inject `geometry.elongation(x)` directly into the loss graph" in `ai_scientist/optim/differentiable.py`.
*   **Codebase Reality:**
    *   `gradient_descent_on_inputs` calculates loss using the **Surrogate's predictions**:
        ```python
        pred_elo, std_elo = surrogate.predict_torch(...)
        viol_elo = torch.relu(pred_elo.squeeze() + ...)
        ```
    *   **Impact:** The optimization loop is still subject to neural approximation error for geometric constraints.

### D. The Data (Offline Pipeline)
*   **Doc Claim:** "Build Offline Data Pipeline... Create `scripts/train_offline.py` and `ai_scientist/optim/data_loader.py`."
*   **Codebase Reality:**
    *   `scripts/train_offline.py` does **not exist**.
    *   `ai_scientist/optim/data_loader.py` does **not exist**.
    *   **Impact:** There is no automated mechanism to pre-train the V2 surrogate on the 158k dataset.

### E. Autonomy (Agents & Workers)
*   **Doc Claim:** "Create `GeometerWorker`... Update `Coordinator` logic."
*   **Codebase Reality:**
    *   `ai_scientist/workers.py` contains `OptimizationWorker` and `ExplorationWorker`. **`GeometerWorker` is missing.**
    *   `ai_scientist/coordinator.py` has the "Stagnation Detection" logic (`hv_delta < 0.005`), but it does not reference the `GeometerWorker` or the hybrid strategy specifics.

---

## 3. Verification Verdict

| Component | Status | Notes |
| :--- | :--- | :--- |
| **Differentiable Optim** | ✅ **Exists** | `differentiable.py` is functional but uses the wrong loss source. |
| **Neural Surrogate** | ⚠️ **Partial** | Exists (`surrogate_v2.py`) but needs refactoring (remove geo heads). |
| **Analytic Geometry** | ⚠️ **Partial** | `geometry.py` exists but lacks metric implementations. |
| **Data Pipeline** | ❌ **Missing** | No `train_offline.py` or `data_loader.py`. |
| **Hybrid Architecture** | ❌ **Missing** | The "Split" between Physicist and Engineer is not implemented. |

**Conclusion:** The documentation is a **valid roadmap for immediate implementation**, accurately identifying the current architectural limitations. The code changes required are substantial but well-defined by the documents.
