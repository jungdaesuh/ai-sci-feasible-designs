# AI Scientist V2.1: Utilizing the 182k "Landscape of Failure" Dataset

**Date:** November 26, 2025
**Status:** Proposed / Implementation Started
**Target System:** `ai_scientist` (V2 Integration) & `constellaration`

---

## 1. Executive Summary

We possess a massive asset: the **~158k sample "Constellaration" dataset** hosted on Hugging Face. Currently, the `ai_scientist` V2 system ignores this data, starting every experiment from a "cold start" (random initialization).

This plan outlines the integration of this dataset to **Pre-train** the V2 Neural Surrogates and Generative Models. Crucially, it recognizes the operational challenges of such a large dataset (training is slow, data is messy) and the architectural necessity of separating "Physics" (learned) from "Geometry" (calculated).

The strategy prioritizes **Offline Pre-training** (Phase 0) to create a robust data pipeline and cached model checkpoints. It adopts a **Hybrid Neuro-Symbolic Architecture** where deep learning is reserved for expensive physics approximations ($W_{MHD}$, $QI$), while exact geometry (Elongation, Curvature) is handled by differentiable analytical functions.

---

## 2. The Strategic Value of Infeasible Data

We will utilize the "failed" data for three specific mathematical purposes:

### 2.1. Gradient Learning (The Surrogate)
The Differentiable Optimizer (`gradient_descent_on_inputs`) relies on $\nabla_{\text{inputs}} \mathcal{L}$. An untrained surrogate predicts noise. By training on 158k examples of inputs $\to$ metrics (even bad metrics), the surrogate learns the *direction* of steepest descent.

### 2.2. Manifold Learning (The Generator)
The valid design space for stellarators is a tiny manifold. Random sampling wastes 99.9% of evaluations on "broken" geometry. Pre-training the **VAE/Diffusion Model** on the geometrically valid subset of the dataset restricts the search to the valid manifold.

### 2.3. "Best-of-Failure" Initialization (The Seeds)
Starting from NAE (Near-Axis Expansion) is often far from the solution. We will filter the dataset to find the "Pareto Frontier of Failures"—designs that failed, but failed *least*—tailored to each specific benchmark problem (P1, P2, P3).

---

## 3. Architecture: Hybrid Neuro-Symbolic Experts

To maximize efficiency and gradient quality, we split the optimization feedback into two distinct experts:

### 3.1. The "Physicist" (Neural Network)
*   **Role:** Approximate expensive, non-linear plasma physics.
*   **Implementation:** `NeuralOperatorSurrogate` (PyTorch FNO).
*   **Targets:** $W_{MHD}$ (Vacuum Well), $QI$ (Quasisymmetry Error).
*   **Input:** Boundary Coefficients ($\theta$).
*   **Training:** Supervised learning on the 158k dataset.

### 3.2. The "Engineer" (Differentiable Math)
*   **Role:** Calculate exact geometric constraints.
*   **Implementation:** `DifferentiableGeometry` (PyTorch Module).
*   **Targets:** Elongation, Mean Curvature, Aspect Ratio.
*   **Input:** Boundary Coefficients ($\theta$).
*   **Training:** **None.** Calculated analytically using differentiable tensor operations.
*   **Why:** Prevents "gradient starvation" (where the NN lazily learns easy geometry and ignores hard physics) and provides noiseless, perfect gradients for engineering constraints.

---

## 4. Implementation Roadmap

### Phase 0: Offline Data Pipeline (The Foundation)
*Goal: Build a robust, standalone pipeline to clean, normalize, and pre-train models once.*

*   [x] **Task 0.1:** Create `ai_scientist/optim/data_loader.py`.
    *   [x] Wrap `constellaration` loader.
    *   [x] **Geometric QA:** Implement filters to reject geometrically invalid shapes (e.g., self-intersection heuristics, spectral decay checks) to prevent "garbage in, garbage out."
    *   [x] **Strict Schema:** Define required columns per benchmark (P1, P2, P3) and drop rows with missing critical metrics.
*   [x] **Task 0.2:** Implement **Robust Normalization**.
    *   [x] Use `LogRobustScaler` (log1p + RobustScaler) for heavy-tailed physics metrics (e.g., $W_{MHD}$).
    *   [x] Save the fitted scaler as `scaler.pkl`.
*   [x] **Task 0.3:** Create `scripts/train_offline.py`.
    *   [x] Load cleaned data.
    *   [x] Train `NeuralOperatorSurrogate` (Physics targets only) and save weights to `checkpoints/surrogate_v2.pt`.
    *   [ ] Train `DiffusionDesignModel` and save weights to `checkpoints/diffusion_v2.pt`.
    *   [x] Generate problem-specific seed files: `seeds_p1.json`, `seeds_p2.json`, `seeds_p3.json`.

### Phase 1: Differentiable Geometry (The Engineer)
*Goal: Implement the analytic expert.*

*   [x] **Task 1.1:** Create `ai_scientist/optim/geometry.py`.
    *   [x] Implement differentiable PyTorch versions of surface generation ($R(\theta, \phi), Z(\theta, \phi)$) from coefficients.
    *   [x] Implement differentiable metrics: Aspect Ratio, Elongation, Mean Curvature.
    *   [x] **Constraint:** Must use pure `torch` operations to maintain the autograd graph (no NumPy).

### Phase 2: Runtime Integration (The Consumer)
*Goal: Update `runner.py` to load artifacts and combine gradients.*

*   [x] **Task 2.1:** Modify `ai_scientist/runner.py` to accept `checkpoint_dir` in config.
*   [x] **Task 2.2:** Load the pre-trained `surrogate_v2.pt` and `scaler.pkl`.
*   [x] **Task 2.3:** Update `differentiable.py` (Optimizer) to use the Hybrid Loss:
    $$ \mathcal{L} = w_p \cdot \text{Neural}(x) + w_e \cdot \text{Analytic}(x) $$
*   [ ] **Task 2.4:** Update `SmartSeeder` to load problem-specific seeds.

---

## 5. Technical Integration Details

### 5.1 New Config Flags
In `ai_scientist/config.py`:
```python
@dataclass
class ExperimentConfig:
    # ...
    use_offline_dataset: bool = False  # Master switch
    offline_checkpoint_dir: Path = Path("checkpoints/v2_1")
    offline_seed_file: Path | None = None
```

### 5.2 The Offline Trainer (`scripts/train_offline.py`)
```python
def main():
    # 1. Load & Clean
    df = load_source_datasets_with_no_errors()
    df = filter_geometric_validity(df)
    
    # 2. Normalize (LogRobust for Physics, Standard for Geometry if needed)
    scaler = LogRobustScaler()
    physics_metrics = scaler.fit_transform(df[PHYSICS_COLS])
    joblib.dump(scaler, "checkpoints/scaler.pkl")
    
    # 3. Train Surrogate (PHYSICS ONLY)
    surrogate = NeuralOperatorSurrogate()
    surrogate.fit(df[PARAMS], physics_metrics)
    torch.save(surrogate.state_dict(), "checkpoints/surrogate_v2.pt")
    
    # 4. Smart Seeding
    seeds_p1 = select_best_seeds(df, problem="p1")
    # ...
```

### 5.3 Runner Logic
In `ai_scientist/runner.py`:
```python
def run(...):
    # ...
    if cfg.use_offline_dataset:
        # Load Pre-trained Physics Brain
        surrogate.load_state_dict(torch.load(...))
        surrogate.set_scaler(joblib.load(...))
        
        # Load Seeds
        initial_pool = load_seeds(...)
```

---

## 6. Expected Impact

| Metric | V2 (Cold Start) | V2.1 (Hybrid + Pre-trained) | Reason |
| :--- | :--- | :--- | :--- |
| **Startup Time** | Fast (0 min) | **Fast (0 min)** | Heavy training is moved offline. |
| **Cycle 1 Feasibility** | ~0% | **~5-10%** | Seeds are "best failures"; optimizer has valid gradients. |
| **Gradient Quality** | Noisy | **High** | Analytic geometry provides perfect gradients; NN focuses on physics. |
| **Stability** | Fragile | **Robust** | LogRobustScaler handles heavy tails; Hybrid architecture prevents gradient starvation. |

## 7. Agent Role (AGENTS.md)

*   **Data Engineer Agent:** Responsible for running `scripts/train_offline.py` and validating `scaler.pkl` statistics.
*   **Planning Agent:** Can request a "re-seed" from the offline cache if the runtime optimization stalls.
