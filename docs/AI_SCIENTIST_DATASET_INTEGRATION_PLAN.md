# AI Scientist V2.1: Utilizing the 182k "Landscape of Failure" Dataset

**Date:** November 26, 2025
**Status:** Proposed / Implementation Started
**Target System:** `ai_scientist` (V2 Integration) & `constellaration`

---

## 1. Executive Summary

We possess a massive asset: the **~158k sample "Constellaration" dataset** hosted on Hugging Face. Currently, the `ai_scientist` V2 system ignores this data, starting every experiment from a "cold start" (random initialization).

This plan outlines the integration of this dataset to **Pre-train** the V2 Neural Surrogates and Generative Models. Crucially, it recognizes the operational challenges of such a large dataset: training is slow, data is messy, and models are fragile.

Therefore, the strategy prioritizes **Offline Pre-training** (Phase 0) to create a robust data pipeline and cached model checkpoints, rather than attempting to train on 158k rows dynamically at runtime. This transforms the system from a "naive searcher" to an "expert navigator" without destroying developer velocity.

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

## 3. Implementation Roadmap

### Phase 0: Offline Data Pipeline (The Foundation)
*Goal: Build a robust, standalone pipeline to clean, normalize, and pre-train models once.*

*   **Task 0.1:** Create `ai_scientist/optim/data_loader.py`.
    *   Wrap `constellaration` loader.
    *   **Geometric QA:** Implement filters to reject geometrically invalid shapes (e.g., self-intersection heuristics, spectral decay checks) to prevent "garbage in, garbage out."
    *   **Strict Schema:** Define required columns per benchmark (P1, P2, P3) and drop rows with missing critical metrics.
*   **Task 0.2:** Implement **Robust Normalization**.
    *   Use `LogRobustScaler` (log1p + RobustScaler) for heavy-tailed physics metrics (e.g., $W_{MHD}$).
    *   Save the fitted scaler as `scaler.pkl`.
*   **Task 0.3:** Create `scripts/train_offline.py`.
    *   Load cleaned data.
    *   Train `NeuralOperatorSurrogate` and save weights to `checkpoints/surrogate_v2.pt`.
    *   Train `DiffusionDesignModel` and save weights to `checkpoints/diffusion_v2.pt`.
    *   Generate problem-specific seed files: `seeds_p1.json`, `seeds_p2.json`, `seeds_p3.json`.

### Phase 1: Runtime Integration (The Consumer)
*Goal: Update `runner.py` to load artifacts instead of training.*

*   **Task 1.1:** Modify `ai_scientist/runner.py` to accept `checkpoint_dir` in config.
*   **Task 1.2:** Load the pre-trained `surrogate_v2.pt` and `scaler.pkl` at startup.
*   **Task 1.3:** Load the pre-trained `diffusion_v2.pt` for the generative model.
*   **Task 1.4:** Update `SmartSeeder` to load the relevant `seeds_p{X}.json` based on the active `cfg.problem`.

---

## 4. Technical Integration Details

### 4.1 New Config Flags
In `ai_scientist/config.py`:
```python
@dataclass
class ExperimentConfig:
    # ...
    use_offline_dataset: bool = False  # Master switch
    offline_checkpoint_dir: Path = Path("checkpoints/v2_1") # Path to .pt and .pkl files
    offline_seed_file: Path | None = None # Optional override for seed file
```

### 4.2 The Offline Trainer (`scripts/train_offline.py`)
```python
def main():
    # 1. Load & Clean
    df = load_source_datasets_with_no_errors()
    df = filter_geometric_validity(df)
    
    # 2. Normalize
    scaler = LogRobustScaler()
    metrics_norm = scaler.fit_transform(df[METRIC_COLS])
    joblib.dump(scaler, "checkpoints/scaler.pkl")
    
    # 3. Train Surrogate
    surrogate = NeuralOperatorSurrogate()
    surrogate.fit(df[PARAMS], metrics_norm)
    torch.save(surrogate.state_dict(), "checkpoints/surrogate_v2.pt")
    
    # 4. Smart Seeding
    seeds_p1 = select_best_seeds(df, problem="p1")
    save_seeds(seeds_p1, "checkpoints/seeds_p1.json")
    # ... (repeat for p2, p3)
```

### 4.3 Runner Logic
In `ai_scientist/runner.py`:
```python
def run(...):
    # ... setup ...
    
    if cfg.use_offline_dataset:
        print(f"[runner] Loading V2.1 checkpoints from {cfg.offline_checkpoint_dir}...")
        
        # Load Scaler
        scaler = joblib.load(cfg.offline_checkpoint_dir / "scaler.pkl")
        
        # Load Surrogate
        if isinstance(surrogate, NeuralOperatorSurrogate):
            surrogate.load_state_dict(torch.load(cfg.offline_checkpoint_dir / "surrogate_v2.pt"))
            surrogate.set_scaler(scaler) # Inject scaler for inference
            
        # Load Generator
        if generative_model:
            generative_model.load_state_dict(torch.load(cfg.offline_checkpoint_dir / "diffusion_v2.pt"))
            
        # Load Seeds
        seed_file = cfg.offline_seed_file or (cfg.offline_checkpoint_dir / f"seeds_{cfg.problem}.json")
        initial_pool = load_seeds(seed_file)
        
    # ... proceed to Cycle 1 with populated pool ...
```

---

## 5. Expected Impact

| Metric | V2 (Cold Start) | V2.1 (Pre-trained) | Reason |
| :--- | :--- | :--- | :--- |
| **Startup Time** | Fast (0 min) | **Fast (0 min)** | Heavy training is moved offline. |
| **Cycle 1 Feasibility** | ~0% | **~5-10%** | Seeds are "best failures"; optimizer has valid gradients. |
| **Surrogate Accuracy** | Random | **High** | Model sees 158k examples before inference. |
| **Stability** | Fragile | **Robust** | LogRobustScaler handles heavy-tailed physics metrics. |

## 6. Agent Role (AGENTS.md)

*   **Data Engineer Agent:** Responsible for running `scripts/train_offline.py` periodically when new data is available and validating the `scaler.pkl` statistics.
*   **Planning Agent:** Can request a "re-seed" from the offline cache if the runtime optimization stalls.
