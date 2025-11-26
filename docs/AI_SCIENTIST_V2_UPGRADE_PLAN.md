# AI Scientist Version 2.0: Physics-Informed Autonomy Upgrade Plan

**Date:** November 26, 2025
**Status:** Proposed / Implementation Started
**Target System:** `ai_scientist` & `constellaration`

---

## 1. Executive Summary

This document outlines the strategic roadmap to evolve the `ai_scientist` from its current "Version 1.0" state (Random Forests + Derivative-Free Optimization) to a "Version 2.0" system powered by **Deep Learning (Geometric/Operator Learning)** and **Differentiable Optimization**.

The primary goal is to enable **"Gradient Descent on Inputs"** (Inverse Design) to solve the Constellaration Fusion Challenge, drastically reducing the number of function evaluations required to find feasible stellarator geometries.

We will implement an **A/B Testing Architecture** to allow Version 1.0 (Baseline) and Version 2.0 (Experimental) to run in parallel, logging to a shared World Model for real-time performance comparison.

### Comparison: V1.0 vs V2.0

| Feature | Version 1.0 (Baseline) | Version 2.0 (Target) |
| :--- | :--- | :--- |
| **Surrogate Model** | **Random Forests** (sklearn). Flattens geometry to 1D vectors. Non-differentiable. | **Equivariant Deep Learning** (FNO/GNN). Preserves 3D/Toroidal geometry. Fully differentiable. |
| **Optimization** | **Derivative-Free** (Nevergrad/CMA-ES). Treats physics as a "Black Box". | **Differentiable Inverse Design**. Backpropagates gradients from Objective $\to$ Inputs. |
| **Physics Constraints** | **Post-hoc Filtering**. Checks constraints after generation. | **Differentiable Physics**. Enforced via loss functions and Augmented Lagrangian gradients. |
| **Exploration** | **Heuristic Sampling** (Near-Axis + Noise). | **Generative Models** (Diffusion/VAEs) & Latent Space Optimization. |
| **Autonomy** | **Single Agent Loop**. Linear planning. | **Hierarchical Agents**. Coordinator + Specialized Workers (Optimizer, Explorer). |

---

## 2. Gap Analysis

The current codebase is a robust, production-ready orchestration system ("Version 1.0"), but it lacks the specific "Physics Brain" required for Version 2.0.

1.  **Surrogates (`ai_scientist/optim/surrogate.py`):** Currently hardcoded to `SurrogateBundle` (Random Forests). Needs to become a modular factory supporting Neural Operators.
2.  **Optimization (`ai_scientist/runner.py`):** Currently uses `nevergrad` for the SA-ALM inner loop. Needs to support a `torch.optim` / JAX-based loop for differentiable surrogates.
3.  **Geometry (`constellaration`):** The physics foundation exists but is not linked to the learning pipeline. We need Differentiable Physics utilities (calculating metrics from predicted fields in a differentiable way).

---

## 3. Implementation Roadmap (5 Phases)

### Phase 1: Infrastructure & Representation
*Goal: Establish the data representations and infrastructural backbone.*

*   **1.1 Hybrid Representation:** Utilities to convert between Fourier coefficients (Global) and 3D Mesh/Point Clouds (Local).
*   **1.2 Equivariance Integration:** Integrate `e3nn` or similar to enforce rotational symmetry ($N_{fp}$) in all networks.
*   **1.3 World Model Upgrade:** Expand `memory.db` schema to store advanced optimization states (ALM multipliers, surrogate checkpoints).

### Phase 2: Advanced Physics-Informed Surrogates (The Physics Core)
*Goal: Build fast, differentiable, uncertainty-aware replacements for VMEC++.*

*   **2.1 Modular Surrogate Interface:** Refactor `SurrogateBundle` into an abstract base class supporting both `sklearn` (V1) and `torch`/`jax` (V2) backends.
*   **2.2 Physics Core (FNOs):** [x] Implement Equivariant Fourier Neural Operators to predict magnetic fields ($\vec{B}$) from boundary coefficients.
*   **2.3 Geometric Surrogates (GNNs):** Implement GNNs for geometric constraints (curvature, elongation).
*   **2.4 Uncertainty Quantification:** Implement Deep Ensembles for robust active learning.

### Phase 3: Autonomous Optimization Engine (Inverse Design)
*Goal: Navigate the rugged landscape using gradients.*

*   **3.1 Differentiable Optimizer:** [x] Implement a "Gradient Descent on Inputs" loop.
    *   $\theta_{new} = \theta_{old} - \alpha \nabla_{\theta} \mathcal{L}(\text{Surrogate}(\theta))$
*   **3.2 SA-ALM Upgrade:** [x] Integrate the differentiable optimizer as the inner loop solver for the Augmented Lagrangian Method.

### Phase 4: Exploration & Pareto Mapping
*Goal: Discover novel designs and map trade-offs.*

*   **4.1 VAE / Latent Optimization:** [x] Train a VAE to learn a smooth latent space for optimization.
*   **4.2 Conditional Generation:** (Advanced) Diffusion models for $P(\text{Geometry} | \text{Metrics})$.

### Phase 5: Orchestration & Autonomy
*Goal: Self-driving research.*

*   **5.1 Hierarchical Agents:** [x] Refactor `planner.py` into a Coordinator that manages specialized Worker Agents (Optimization Worker, Exploration Worker).
*   **5.2 Adaptive Switching:** [x] Coordinator dynamically switches strategies based on World Model state (e.g., "Stuck in local minima" $\to$ "Switch to Exploration").

---

## 4. A/B Testing Architecture

To ensure safety and measurability, V1 and V2 will coexist.

### 4.1 Configuration Flags
We will introduce new flags in `ai_scientist/config.py`:

```python
@dataclass
class ExperimentConfig:
    # ... existing fields ...
    surrogate_backend: str = "random_forest"  # Options: "random_forest", "neural_operator"
    optimizer_backend: str = "nevergrad"      # Options: "nevergrad", "gradient_descent"
```

### 4.2 Decoupled Runner Logic
Refactor `runner.py` to instantiate components dynamically:

```python
# Pseudo-code for runner.py
if cfg.surrogate_backend == "neural_operator":
    surrogate = NeuralOperatorSurrogate(...)
else:
    surrogate = RandomForestSurrogate(...)

# In optimization loop:
if cfg.optimizer_backend == "gradient_descent":
    candidate = differentiable_optimize(surrogate, initial_guess)
else:
    candidate = derivative_free_optimize(surrogate, initial_guess)
```

### 4.3 Parallel Execution Strategy
1.  **Config A (`v1.yaml`):** `surrogate_backend="random_forest"`, `optimizer_backend="nevergrad"`
2.  **Config B (`v2.yaml`):** `surrogate_backend="neural_operator"`, `optimizer_backend="gradient_descent"`
3.  **Shared DB:** Both processes run concurrently:
    ```bash
    python -m ai_scientist.runner --config v1.yaml --memory-db shared.db &
    python -m ai_scientist.runner --config v2.yaml --memory-db shared.db &
    ```
4.  **Analysis:** Compare `hv` (Hypervolume) and `feasibility_rate` by `experiment_tag` in the database.

---

## 5. Recommended Tech Stack

*   **Deep Learning:** PyTorch (Mature ecosystem for optimization/equivariance).
*   **Equivariance:** `e3nn` (Euclidean Neural Networks).
*   **Optimization:** `torch.optim` (Adam/LBFGS) + `BoTorch` (Bayesian Optimization).
*   **Physics:** JAX (if tight integration with `constellaration` physics kernels is needed later).

## 6. Immediate Next Steps

- [x] **Refactor Config:** Update `config.py` to support backend flags.
- [x] **Decouple Runner:** Remove global `_SURROGATE_BUNDLE` in `runner.py` and implement the factory pattern.
- [x] **Implement V2 Skeleton:** Create the placeholder class for `DeepSurrogate` to prove the A/B architecture works.
- [x] **Implement Physics Core (FNOs):** Implemented `StellaratorNeuralOp` in `surrogate_v2.py`.
- [x] **Implement Differentiable Optimizer:** Implemented `optimize_alm_inner_loop` in `differentiable.py` and integrated into `runner.py`.
