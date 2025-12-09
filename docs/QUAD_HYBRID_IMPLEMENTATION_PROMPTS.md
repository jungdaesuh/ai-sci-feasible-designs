# Quad-Hybrid Implementation Prompts

**Purpose**: Copy-paste prompts for AI coding agents to implement each task.
**Reference**: See `QUAD_HYBRID_IMPLEMENTATION_GUIDE.md` for full context.
**Updated**: 2025-12-10 (includes Opus 4.5 discoveries)

---

## Task 1: Fix Config Loading (15 min)

### Prompt

```
Fix the `_generative_config_from_dict` function in `ai_scientist/config.py` (lines 612-623).

**Problem**: The function is missing the StellarForge-specific fields, causing YAML config values to be ignored.

**Missing fields to add**:
- `checkpoint_path: Path | None` (handle empty string edge case)
- `device: str` (default "cpu")
- `hidden_dim: int` (default 2048)
- `n_layers: int` (default 4)
- `pca_components: int` (default 50)
- `batch_size: int` (default 4096)
- `diffusion_timesteps: int` (default 200)

**Edge case**: If `checkpoint_path` is empty string `""`, set it to `None` instead of `Path("")`.

**Reference**: The `GenerativeConfig` dataclass at lines 190-206 already has these fields defined - they just aren't being parsed from the YAML.

DO NOT change the dataclass definition. Only fix the `_generative_config_from_dict` function.
```

---

## Task 2: Fix experiment_setup.py (5 min)

### Prompt

```
Fix the `create_generative_model` function in `ai_scientist/experiment_setup.py`.

**Problem**: When creating a `DiffusionDesignModel`, the `diffusion_timesteps` parameter is not being passed from config, causing the model to use a default value instead of the configured value (e.g., 200).

**Fix**: Pass `diffusion_timesteps=cfg.generative.diffusion_timesteps` when instantiating `DiffusionDesignModel`.

Find where `DiffusionDesignModel(...)` is instantiated and add the missing parameter.

DO NOT change anything else. This is a one-line fix.
```

---

## Task 2.5: Fix NFP Propagation Bug (5 min) - NEW

### Prompt

```
Fix the NFP (n_field_periods) propagation bug in `ai_scientist/workers.py` (lines 359-362).

**Problem**: In `RLRefinementWorker.run()`, after calling `tools.structured_unflatten()`, the `n_field_periods` field is NOT restored. This causes downstream validation (Geometer) to fail or compute wrong metrics.

**Location**: Inside the training loop where `best_params` is set:
```python
best_params = tools.structured_unflatten(next_obs, env.schema)
# ❌ n_field_periods is NOT restored!
```

**Fix**: After `structured_unflatten`, restore the metadata fields:
```python
best_params = tools.structured_unflatten(next_obs, env.schema)
best_params["n_field_periods"] = params.get("n_field_periods", params.get("nfp", 3))
best_params["is_stellarator_symmetric"] = params.get("is_stellarator_symmetric", True)
```

Note: The existing code at lines 361-362 DOES try to restore `n_field_periods`, but it may be using `nfp` inconsistently. Ensure BOTH field names are checked.

DO NOT change anything else. This is a 2-line fix.
```

---

## Task 3A: Add Basic PreRelaxWorker (20 min)

### Prompt

```
Add a new `PreRelaxWorker` class to `ai_scientist/workers.py`.

**Location**: Insert after `GeometerWorker` class (around line 271).

**Purpose**: Geometric pre-relaxation worker that applies fast gradient-based smoothing to candidates before they reach the Surrogate/RL stages.

**Requirements**:
1. Inherit from `Worker` base class (like other workers in the file)
2. Import `logging`, `prerelax_boundary` from `ai_scientist.optim.prerelax`
3. Import `tools` from `ai_scientist` for `FlattenSchema`
4. Use `ThreadPoolExecutor` for parallel processing (16 workers) in `_relax_single()`
5. Include a `_normalize_params` method that pads/truncates `r_cos` and `z_sin` to match the surrogate's `FlattenSchema` dimensions

**CRITICAL - NFP Bug Fix (Opus 4.5)**:
The `prerelax_boundary` function has `nfp` hardcoded to 3. You MUST:
1. Extract NFP from `params.get("n_field_periods", params.get("nfp", 3))`
2. Pass it explicitly to `prerelax_boundary(..., nfp=nfp, ...)`
3. Restore it in the output: `optimized["n_field_periods"] = nfp`

**Exception handling** (use these specific types):
- `RuntimeError`: GPU OOM or tensor operation failure
- `ValueError`: Shape mismatch or invalid input
- `Exception`: Catch-all (log at error level)

**Constructor parameters**:
- `cfg: ai_config.ExperimentConfig`
- `surrogate_schema: Optional[tools.FlattenSchema] = None`

**Run method**:
- Input context: `{"candidates": List[Dict], "schema": Optional[FlattenSchema]}`
- Output: `{"candidates": List[Dict], "status": str}`
- Filter out candidates with geometric_energy > 1.0 (threshold)
- Print progress similar to other workers: `[PreRelaxWorker] ...`

**Key settings** (hardcoded for now):
- steps = 50
- lr = 0.01
- target_ar = 8.0
- energy_threshold = 1.0
- max_workers = 16
- device = "cpu"

Follow the existing code patterns in `workers.py` exactly. Look at `GeometerWorker` for reference.
```

**Rollback**: If this breaks the build, revert with:
```bash
git checkout HEAD -- ai_scientist/workers.py
```

---

## Task 3B: Add Batched Processing (10 min)

### Prompt

```
Extend the `PreRelaxWorker` class in `ai_scientist/workers.py` with a `_prerelax_batch()` method.

**Purpose**: Batched tensor processing for 10× speedup on large candidate sets.

**Requirements**:
1. Add `_prerelax_batch(self, candidates, schema, target_ar) -> List[Dict]`
2. PARTITION candidates by NFP before batching (fixes lossy approximation issue)
3. Process each NFP group separately with vectorized gradient descent
4. Pass `schema` as a parameter, not instance attribute (thread safety)

**Key implementation detail** (from reviewer):
```python
# Partition by NFP to avoid lossy approximation
nfp_groups: Dict[int, List[Tuple[int, Dict]]] = defaultdict(list)
for idx, cand in enumerate(candidates):
    params = cand.get("params") or cand
    nfp = int(params.get("n_field_periods", params.get("nfp", 3)))
    nfp_groups[nfp].append((idx, cand))

# Process each NFP group separately
for nfp, group in nfp_groups.items():
    # ... batch processing with correct NFP value
```

**Update run() method**:
- If `len(candidates) >= 100` and `context.get("use_batched", True)`, use `_prerelax_batch()`
- Otherwise use `_relax_single()` with ThreadPoolExecutor

DO NOT modify the `_relax_single()` or `_normalize_params()` methods from Task 3A.
```

**Rollback**: If this breaks the build, revert with:
```bash
git checkout HEAD -- ai_scientist/workers.py
```

---

## Task 4: Wire PreRelaxWorker into Coordinator (30 min)

### Prompt

```
Integrate `PreRelaxWorker` into `ai_scientist/coordinator.py`.

**Part A: Add import** (top of file, around line 35-40)
Add `PreRelaxWorker` to the import from `ai_scientist.workers`.

**Part B: Initialize worker** (in `__init__`, around line 89)
After `self.rl_worker = RLRefinementWorker(cfg, self.surrogate)`, add:
```python
self.prerelax_worker = PreRelaxWorker(
    cfg,
    surrogate_schema=self.surrogate._schema if self.surrogate else None,
)
```

**Part C: Update `produce_candidates` HYBRID branch** (lines 162-179)
Replace the current HYBRID workflow with the Quad-Hybrid pipeline:

Current order (WRONG):
1. Dream → 2. RL (all seeds) → 3. Geometer → 4. Optimize

New order (CORRECT):
1. Dream → 2. Pre-relax (NEW) → 3. Geometer → 4. Surrogate Rank → 5. RL (top-K only) → 6. Optimize

**Key changes**:
- Insert `self.prerelax_worker.run()` after explore_worker, before geo_worker
- Pass surrogate schema: `{"candidates": seeds, "schema": self.surrogate._schema if self.surrogate else None}`
- MOVE RL refinement AFTER surrogate ranking
- Only refine top-K candidates (k = min(100, len(ranked)))
- Call `self._surrogate_rank_seeds()` to rank before RL

**Print statements** (for debugging):
- `[Coordinator] Dreamer generated N seeds`
- `[Coordinator] Pre-relaxer smoothed N candidates`
- `[Coordinator] Geometer passed N/M candidates`
- `[Coordinator] Surrogate selected top-K for RL refinement`
- `[Coordinator] RL Agent refined N candidates`
- `[Coordinator] Quad-Hybrid pipeline complete: N final candidates`

Also update the EXPLOIT branch (lines 144-160) with the same pattern.
```

**Rollback**: If this breaks the build, revert with:
```bash
git checkout HEAD -- ai_scientist/coordinator.py
```

---

## Task 5: Create Offline Training Script (2 hours)

### Prompt

```
Create a new script `scripts/train_generative_offline.py` for pre-training the Diffusion model on the ConStellaration dataset.

**Purpose**: Train `DiffusionDesignModel` on the 160k offline dataset from Hugging Face (`proxima-fusion/constellaration`).

**Requirements**:
1. Load dataset from Hugging Face using `datasets` library
2. Extract Fourier coefficients (r_cos, z_sin) and target metrics
3. Initialize `DiffusionDesignModel` with paper specs:
   - hidden_dim=2048
   - n_layers=4
   - pca_components=50
   - diffusion_timesteps=200
4. Train for 250 epochs (paper spec)
5. Save checkpoint to `checkpoints/diffusion_paper_spec.pt`

**Reference**: Look at `scripts/train_offline.py` for the surrogate training pattern.

**CLI arguments**:
- `--epochs` (default 250)
- `--batch-size` (default 4096)
- `--device` (default "cuda" if available)
- `--output` (default "checkpoints/diffusion_paper_spec.pt")
- `--dataset` (default "proxima-fusion/constellaration")

**Logging**: Print training progress every 10 epochs.
```

---

## Task 6: Add Periodic Retraining (3 hours)

### Prompt

```
Add periodic retraining of the Generative model in `ai_scientist/coordinator.py`.

**Location**: At the end of each cycle in `produce_candidates` or in a new method.

**Trigger conditions** (implement as config options):
1. Every N cycles (e.g., cycle % 5 == 0)
2. On HV stagnation (HV delta < 0.005 for 3 cycles)

**Retraining logic**:
1. Collect elite candidates from the cycle (top 32 by objective)
2. Call `self.generative_model.fine_tune_on_elites(elites)` (method may need to be added to DiffusionDesignModel)
3. Also retrain surrogate: `self.surrogate.fit(...)` with updated data

**Config additions** (in `ai_scientist/config.py`):
Add a new `RetrainingConfig` dataclass:
```python
@dataclass(frozen=True)
class RetrainingConfig:
    enabled: bool = True
    cycle_cadence: int = 5
    min_elites: int = 32
    hv_stagnation_threshold: float = 0.005
```

Add `retraining: RetrainingConfig` to `ExperimentConfig`.

**Print statement**: `[Coordinator] Periodic retraining triggered (cycle N)`
```

---

## Verification Task

### Prompt

```
Run a smoke test to verify the Quad-Hybrid pipeline works end-to-end.

**Command**:
```bash
python -m ai_scientist.experiment_runner \
  --config configs/experiment.yaml \
  --cycles 1 \
  --n-candidates 10
```

**Expected output** (verify these lines appear in order):
1. `[Coordinator] Dreamer generated 10 seeds`
2. `[PreRelaxWorker] Pre-relaxing 10 candidates...`
3. `[PreRelaxWorker] X/10 survived`
4. `[Coordinator] Pre-relaxer smoothed X candidates`
5. `[GeometerWorker] Retained Y/X candidates`
6. `[Coordinator] Surrogate selected top-Z for RL refinement`
7. `[RLRefinementWorker] Refining Z candidates...`
8. `[Coordinator] Quad-Hybrid pipeline complete`

**Expected timing** (10 candidates):
- Dreamer: <1s
- Pre-relaxer: <2s
- Geometer: <0.5s
- Surrogate: <0.5s
- RL Agent: <5s

**Error diagnosis**:
| Error Message | Task to Fix |
|--------------|-------------|
| `KeyError: n_field_periods` | Task 2.5 |
| `AttributeError: 'NoneType' object has no attribute '_schema'` | Task 4 (surrogate init) |
| `ValueError: shape mismatch` | Task 3A (_normalize_params) |
| PreRelaxWorker rejects >80% | Increase energy_threshold in Task 3A |

If any step fails, report the error message and which task needs fixing.
```

---

## Execution Order

Run these tasks in order. Each task depends on the previous:

1. ✅ Task 1: Config loading (unblocks checkpoint loading)
2. ✅ Task 2: experiment_setup (unblocks correct model init)
3. ✅ Task 2.5: NFP propagation fix (Opus 4.5 - prevents downstream failures)
4. ✅ Task 3A: Basic PreRelaxWorker (creates the new worker)  *← DONE*
5. ✅ Task 3B: Batched processing (10× speedup for large batches)  *← DONE*
6. ✅ Task 4: Coordinator wiring (integrates everything)  *← DONE*
7. [x] Verification (confirms it works)
8. ✅ Task 5: Offline training (optional, for production)  *← DONE*
9. ✅ Task 6: Periodic retraining (optional, for production)  *← DONE*

---

## Source Attribution

- **Task 1**: Claude Sonnet (empty string edge case)
- **Task 2**: DeepThink (diffusion_timesteps)
- **Task 2.5**: Opus 4.5 (NFP propagation bug)
- **Task 3A**: Grok (drop-in patterns) + GPT-5.1 (schema normalization)
- **Task 3B**: Opus 4.5 (batched processing + NFP partitioning)
- **Task 4**: Claude Sonnet + Grok (pipeline reorder)
