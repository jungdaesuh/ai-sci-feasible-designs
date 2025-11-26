# AI Scientist V2.1: Utilizing the 182k "Landscape of Failure" Dataset

**Date:** November 26, 2025
**Status:** Proposed / Ready for Implementation
**Target System:** `ai_scientist` (V2 Integration) & `constellaration`

---

## 1. Executive Summary

We possess a massive asset: the **182k sample "Constellaration" dataset** hosted on Hugging Face. Currently, the `ai_scientist` V2 system ignores this data, starting every experiment from a "cold start" (random initialization).

While the dataset contains **zero** fully feasible designs, it is critically valuable. It maps the **"Landscape of Failure."** By analyzing 182,000 ways to fail, Deep Learning models can learn the **gradients of improvement** required to succeed.

This plan outlines the integration of this dataset to **Pre-train** the V2 Neural Surrogates and Generative Models. Crucially, it includes a **Robust Preprocessing Pipeline** to handle data format mismatches, metric normalization, and geometric alignment, ensuring the "raw" data is consumable by our neural networks.

---

## 2. The Strategic Value of Infeasible Data

We will utilize the "failed" data for three specific mathematical purposes:

### 2.1. Gradient Learning (The Surrogate)
The Differentiable Optimizer (`gradient_descent_on_inputs`) relies on $\nabla_{\text{inputs}} \mathcal{L}$.
*   **Problem:** An untrained surrogate predicts random noise, meaning the gradients are random.
*   **Solution:** By training on 182k examples of inputs $\to$ metrics (even bad metrics), the surrogate learns the *direction* of steepest descent. It learns that "increasing $R_{mn}$ reduces MHD instability," even if it has never seen a stable design.

### 2.2. Manifold Learning (The Generator)
The valid design space for stellarators is a tiny manifold within a vast high-dimensional hypercube.
*   **Problem:** Random sampling wastes 99.9% of evaluations on "broken" geometry (self-intersecting surfaces, wild ripples).
*   **Solution:** The 182k designs, while physically imperfect, are mostly *geometrically valid* tori. Pre-training the **VAE/Diffusion Model** on them forces the generator to output only smooth, well-formed shapes, restricting the search to the valid geometric manifold.

### 2.3. "Best-of-Failure" Initialization (The Seeds)
*   **Problem:** Starting from NAE (Near-Axis Expansion) is mathematically elegant but often far from the solution in the $W_{MHD}$ landscape.
*   **Solution:** We will filter the 182k dataset to find the "Pareto Frontier of Failures"—designs that failed, but failed *least*. These will serve as the initial population for the Genetic Algorithms and ALM loops, placing us much closer to the target.

---

## 3. Implementation Roadmap

### Phase 1: Data Ingestion & Preprocessing (The Pipeline)
*Connect the loader and sanitize the data for Deep Learning.*

*   **Task 1.1:** Create `ai_scientist/optim/data_loader.py`.
    *   Wrap `constellaration.generative_model.bootstrap_dataset.load_source_datasets_with_no_errors`.
    *   Implement local caching to avoid downloading 182k rows on every run.
*   **Task 1.2:** Implement **Data Transformation & Cleaning**.
    *   **Schema Adaptation:** Convert flattened Pandas columns (e.g., `boundary.r_cos`) to the nested dictionary structure (`{'params': {'r_cos': ...}}`) expected by `runner.py`.
    *   **Type Casting:** Ensure all numerical fields are `float32` (crucial for PyTorch/JAX) and handle JSON string parsing for metrics.
    *   **Sanitization:** Filter out rows with `NaN` or `Inf` values in critical metric columns to prevent training instability.
*   **Task 1.3:** Implement **Metric Normalization**.
    *   Compute global Mean ($\mu$) and Standard Deviation ($\sigma$) for all 182k rows.
    *   Create a persistable `StandardScaler` (or use sklearn's) to transform raw metrics (e.g., $W_{MHD} \approx -0.01$) into normalized inputs (e.g., $z \approx -1.2$) for the neural networks. **This is critical** as raw metrics vary by orders of magnitude.

### Phase 2: The Brain (Surrogate Warm-Start)
*Teach the Neural Operator physics before the experiment starts.*

*   **Task 2.1:** Modify `ai_scientist/runner.py`.
    *   Add a startup routine `_warm_start_surrogate(cfg, surrogate)`.
    *   If `cfg.surrogate.pretrained` is True, load the preprocessed offline dataset.
    *   Call `surrogate.fit(history_182k)` **before** the first cycle.
*   **Task 2.2:** Schema Alignment.
    *   Ensure the feature vectorizer (`tools.structured_flatten`) in `ai_scientist` is compatible with the geometry representation in the Hugging Face dataset.

### Phase 3: The Imagination (Generative Warm-Start)
*Teach the Diffusion Model valid geometry.*

*   **Task 3.1:** Modify `ai_scientist/runner.py`.
    *   Add `_warm_start_generative(cfg, generative_model)`.
    *   Train `DiffusionDesignModel` or `VAE` on the **normalized** geometry columns of the 182k dataset.
    *   This effectively "compresses" the 182k designs into the model's weights.

### Phase 4: The Launchpad (Smart Seeding)
*Start closer to the finish line.*

*   **Task 4.1:** Implement `SmartSeeder` in `ai_scientist/optim/samplers.py`.
    *   Logic: `SELECT * FROM dataset ORDER BY (violation_mhd + violation_qi) ASC LIMIT 100`.
    *   Use these 100 "best failures" to populate the initial pool for Cycle 1, replacing the random `NearAxisSampler`.

---

## 4. Technical Integration Details

### 4.1 New Config Flags
In `ai_scientist/config.py`:
```python
@dataclass
class ExperimentConfig:
    # ...
    use_offline_dataset: bool = False  # Master switch
    offline_dataset_cache: Path = Path("~/.cache/constellaration_hf")
```

### 4.2 The `data_loader.py` Interface
```python
def load_offline_data(cache_dir: Path) -> tuple[list[dict], StandardScaler]:
    """
    1. Check cache.
    2. If missing, call constellaration.bootstrap_dataset.load_source...
    3. Preprocess: Unflatten, Cast Types, Remove NaNs.
    4. Normalize: Fit StandardScaler on metrics.
    5. Return (clean_rows, scaler).
    """
```

### 4.3 Runner Integration Logic
In `ai_scientist/runner.py`:
```python
def run(...):
    # ... setup ...
    
    # NEW: Warm Start
    if cfg.use_offline_dataset:
        print("[runner] Loading and Preprocessing 182k offline dataset...")
        offline_data, scaler = data_loader.load_offline_data(cfg.offline_dataset_cache)
        
        # 1. Train Surrogate (Learn Gradients)
        if isinstance(surrogate, NeuralOperatorSurrogate):
            print(f"[runner] Pre-training surrogate on {len(offline_data)} samples...")
            # Pass scaler to surrogate to handle un-normalization if needed
            surrogate.fit(offline_data, target_values=..., scaler=scaler)
            
        # 2. Train Generative Model (Learn Manifold)
        if generative_model:
            print(f"[runner] Pre-training generative model...")
            generative_model.fit(offline_data, scaler=scaler)
            
        # 3. Smart Seeding
        # (Optional: Inject best offline candidates into initial pool)
        
    # ... proceed to Cycle 1 ...
```

---

## 5. Expected Impact

| Metric | V2 (Cold Start) | V2.1 (Pre-trained) | Reason |
| :--- | :--- | :--- | :--- |
| **Cycle 1 Feasibility** | ~0% | **~5-10%** | Starting seeds are selected from the top 0.1% of the 182k dataset. |
| **Surrogate Accuracy** | Low (Random) | **High** | Model has seen the full physics landscape before inference. |
| **Convergence Speed** | Slow (Needs ~10 cycles to learn) | **Instant** | Differentiable optimizer has valid gradients immediately. |
| **Generative Quality** | Noise | **Smooth Tori** | Diffusion model generates valid shapes from T=0. |
| **Stability** | High Variance | **Robust** | Normalization prevents gradient explosions from raw metric scales. |

## 6. Agent Role (AGENTS.md)

*   **Data Engineer Agent (New Role/Capability):** Responsible for maintaining the `data_loader`, specifically the normalization logic (handling outliers) and schema synchronization.
*   **Planning Agent:** Can now decide *when* to pull from the offline dataset. For example, if it detects "Mode Collapse" (all designs looking the same), it can request "Inject diverse seeds from the offline dataset" to unstuck the optimization.