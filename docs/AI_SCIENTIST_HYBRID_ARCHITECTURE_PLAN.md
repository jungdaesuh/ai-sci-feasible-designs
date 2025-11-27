# AI Scientist V2.2: Hybrid Intelligence Architecture (Physicist & Engineer)

**Date:** November 27, 2025
**Status:** In Progress (Phase 1 Complete)
**Target System:** `ai_scientist` (V2 Integration)

---

## 1. Executive Summary

This plan defines the architecture for Phase 5 ("Hierarchical Autonomy") of the AI Scientist project. It implements the **"GOAT Technical Strategy"** by establishing a **Hybrid Intelligence System** that separates high-level reasoning (LLM-driven) from low-level execution (Numerical).

The core innovation is the architectural split of the optimization loop into two distinct experts:
1.  **The Physicist (Neural):** A Deep Learning Surrogate (FNO) that predicts expensive physics simulations ($W_{MHD}$, $QI$).
2.  **The Engineer (Analytical):** A Differentiable Geometry Module that calculates exact geometric constraints (Elongation, Curvature, Self-Intersection).

This split prevents "Gradient Starvation," ensures perfect gradients for geometric constraints, and significantly accelerates the discovery of feasible stellarator designs.

---

## 2. Architectural Components

### 2.1. The "Brain" (LLM-Driven Agents)
*   **Coordinator (`ai_scientist/coordinator.py`):**
    *   **Role:** Strategic Orchestrator.
    *   **Responsibilities:**
        *   Analyzes `WorldModel` state (stagnation detection).
        *   Decides high-level strategy ("Explore" vs. "Exploit").
        *   Adjusts weights in the loss function ($w_{physics}$ vs. $w_{engineer}$).
        *   Delegates tasks to Workers.
*   **Planner (`ai_scientist/planner.py`):**
    *   **Role:** Hypothesis Generator.
    *   **Responsibilities:**
        *   Queries the 182k dataset for "Goldilocks Zones."
        *   Formulates optimization goals (e.g., "Target $R_{2,1} > 0.1$").

### 2.2. The "Hands" (Numerical Workers)
*   **OptimizationWorker (`ai_scientist/workers.py`):**
    *   **Role:** The Execution Engine.
    *   **Mechanism:** Runs `gradient_descent_on_inputs` (PyTorch/Adam).
    *   **Input:** Initial Seed + Loss Function Weights.
    *   **Output:** Optimized Design Candidates.
*   **AnalyticalGeometerWorker (New):**
    *   **Role:** The Gatekeeper.
    *   **Mechanism:** Exact Python/NumPy Math.
    *   **Responsibilities:**
        *   **Pre-Filter:** Rejects invalid geometry (self-intersections) *before* the surrogate runs.
        *   **Fast Check:** Computes $Jacobian < 0$ to detect folding.

### 2.3. The Optimization Loop (The "Engine")
The critical refactoring of `surrogate_v2.py` and `differentiable.py` to implement the Hybrid Loss.

**Old Architecture (Monolithic):**
$$ \mathcal{L} = \text{NeuralNet}(x) \rightarrow [\text{MHD}, \text{QI}, \text{Elongation}] $$

**New Architecture (Hybrid):**
$$ \mathcal{L} = w_p \cdot \text{NeuralNet}(x)_{[\text{MHD}, \text{QI}]} + w_e \cdot \text{DiffGeometry}(x)_{[\text{Elongation}]} $$

*   **NeuralNet:** Specializes purely in `vmecpp` simulation approximation.
*   **DiffGeometry:** Provides **perfect, noiseless gradients** for surface smoothness and buildability.

---

## 3. Implementation Roadmap

### Phase 1: The "Engineer" (Differentiable Geometry)
*   **Task 1.1:** Create `ai_scientist/optim/geometry.py`.
    *   Implement fully differentiable PyTorch versions of geometric metrics:
        *   `elongation(r_cos, z_sin)`
        *   `mean_curvature(r_cos, z_sin)`
        *   `surface_area(r_cos, z_sin)`
    *   *Crucial:* Must use `torch.sin`, `torch.cos`, `torch.matmul` to maintain the autograd graph.

### Phase 2: The "Physicist" (Surrogate Refactoring)
*   [x] **Task 2.1:** Refactor `ai_scientist/optim/surrogate_v2.py`.
    *   Remove `head_elongation` and `head_curvature` from `StellaratorNeuralOp`.
    *   Retain only `head_mhd`, `head_qi`, and `head_objective` (if objective is physics-based).
    *   Retrain the model on the 182k dataset focusing *only* on physics targets.

### Phase 3: The "Gatekeeper" (Analytical Worker)
*   **Task 3.1:** Update `ai_scientist/workers.py`.
    *   Create `GeometerWorker` class.
    *   Implement `check_validity(boundary)` using `constellaration` utils (Jacobian check).
*   **Task 3.2:** Integrate into `Coordinator`.
    *   Add logic: `if not geometer.check(candidate): continue`.

### Phase 4: The "Brain" Integration
*   **Task 4.1:** Update `ai_scientist/optim/differentiable.py`.
    *   Modify `gradient_descent_on_inputs` to use the new Hybrid Loss formula.
    *   Inject `geometry.elongation(x)` directly into the loss calculation.

---

## 4. Technical Rationale (Why this wins)

1.  **Perfect Gradients:** By calculating geometry analytically, we remove approximation error from the "Engineering" half of the problem. The optimizer gets a clean signal: "Smooth this surface."
2.  **Reduced Model Complexity:** The Neural Network no longer needs to learn simple geometry. It can dedicate all its capacity to learning the chaotic, non-linear plasma physics.
3.  **Speed:** Geometric rejection happens in microseconds. We stop wasting GPU cycles predicting the plasma stability of "broken donuts."
4.  **Robustness:** The "Tug-of-War" is stabilized. The "Engineer" (Math) anchors the optimization in valid reality, while the "Physicist" (Neural) pulls towards high performance.

---

## 5. Compatibility Check
*   **Existing Code:** `ai_scientist/optim/differentiable.py` already uses PyTorch for optimization. This plan simply changes *what* function is being optimized (Surrogate $ightarrow$ Hybrid).
*   **Dataset:** The 182k dataset (`AI_SCIENTIST_DATASET_INTEGRATION_PLAN.md`) is perfectly suited to train the specialized "Physicist" model.

This plan represents the optimal synthesis of AI capability and Scientific Computing rigor.
