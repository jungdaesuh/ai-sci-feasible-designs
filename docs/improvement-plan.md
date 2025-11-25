Short answer: the code you pasted is a _really_ solid skeleton, but it’s still mostly a “random-search + logging + hypervolume” system with only stubbed adaptation. The big wins now are:

- make the search less dumb (surrogates / curriculum instead of pure RNG),
- actually _use_ the adaptation + PEFT hooks,
- tighten the agent architecture (who calls which tools, with what context),
- and squeeze more out of compute (parallelism, caching, stage-gating).

# Actionable reminder
Use `ai_scientist/roadmap.md` as the single source for the phase-by-phase homework, especially the Phase 6 “Documentation & DX” checklist at the bottom of that file. Treat this markdown plan as the narrative companion—each section below explains *why* the roadmap lists those tasks, and the roadmap shows *how* to execute them.

I’ll go through concrete, code-level ways to do that.

---

## 1. Make the search much smarter than pure random

Right now, new designs per cycle are basically:

```python
candidate_seeds = [
    cfg.random_seed + cycle_index * cfg.budgets.screen_evals_per_cycle + i
    for i in range(cfg.budgets.screen_evals_per_cycle)
]
candidates = [
    _generate_candidate_params(cfg.boundary_template, seed)
    for seed in candidate_seeds
]
```

This is symmetric Gaussian noise around a simple base boundary ⇒ no memory of what worked.

### 1.1. Use the _existing_ surrogate hooks

You already have:

```python
history = world_model.surrogate_training_data(target="hv", problem=cfg.problem)
```

You’re just not calling it anywhere.

**Upgrade**:

- At the start of each cycle (once you have some data), train a tiny surrogate:

  - X = metrics or parameter summary (e.g. gradient, aspect, a few shape stats)
  - y = hv / objective / -feasibility

- Use that surrogate to bias proposals: pick top-N “promising” perturbations, not purely random.

Pseudo-flow inside `_run_cycle` before `_generate_candidate_params`:

```python
history = world_model.surrogate_training_data(target="hv", problem=cfg.problem)
if len(history) > MIN_SURROGATE_POINTS:
    surrogate = fit_small_regressor(history)  # sklearn RandomForest / XGBoost etc
    candidates = propose_with_surrogate(
        surrogate, cfg.boundary_template, rng, cfg.budgets.screen_evals_per_cycle
    )
else:
    candidates = random_candidates(...)
```

Even a crude model (RF, shallow NN, RBF) will massively outperform pure RNG on such expensive evals.

### 1.2. Use `normalized_constraint_distance_sampler` for curriculum

You already wrote a nice constraint-aware sampler:

```python
normalized_constraint_distance_sampler(
    base_designs,
    normalized_distances,
    proposal_count,
    jitter_scale=0.01,
)
```

But it’s never called.

Use it to:

- Take near-feasible designs from previous cycles,
- Prefer those with small feasibility,
- Jitter around them to exploit the feasible frontier.

Example integration:

1. Build `base_designs` = recent feasible designs’ `params`.
2. `normalized_distances` = normalized feasibility values.
3. Ask sampler to give `proposal_count ≈ screen_evals_per_cycle` designs.

You can then mix:

- 70% from constraint sampler,
- 30% pure random for exploration.

---

## 2. Actually use PEFT / LoRA for agents

`adapter.py` is currently a stub:

- `AdapterState.load_lora_weights` and `push_updates` just log.
- `is_peft_enabled()` toggles via env var.

### 2.1. Real LoRA loading

Implement:

```python
def load_lora_weights(self, label: str, stage: str) -> None:
    path = os.path.join("lora_weights", f"{label}_{stage}.pt")
    if os.path.exists(path):
        model.load_adapter(path)  # pseudo: integrate with whatever LLM you’re using
    ...
```

And similarly, `push_updates` should:

- write updated adapter weights,
- or schedule an offline fine-tune run based on logged preference pairs (see below).

This lets:

- `evaluate_p3` calls at stage "p3" gradually benefit from RLHF/RLAIF on your preference logs.

### 2.2. Close the loop with `adaptation_helpers`

`runner._run_cycle` is already writing:

- preference pairs (`append_preference_pair`)
- per-cycle reward diffs (`append_preference_record`)
- statements + hv deltas in `world_model.statements`

The missing bit is a job that:

1. Reads these preference logs.
2. Runs a DPO / RLAIF / simple regression fine-tune on an offline dataset of (input → better/worse).
3. Saves updated LoRA bundles to disk per tool/stage.
4. PEFT hook loads them automatically next run/next cycle.

Architecturally:

- Keep tuning separate from evaluation loop (e.g., nightly job).
- But make sure tuner writes to a known folder the adapter hook can read.

---

## 3. Improve the multi-agent architecture itself

### 3.1. Use roles + gates more aggressively

In `agent.py`:

- roles → model tiers:

```python
_ROLE_ALIAS_MAP = {
    "screen": lambda cfg: cfg.instruct_model,
    "short_loop": lambda cfg: cfg.instruct_model,
    "prompt": lambda cfg: cfg.instruct_model,
    "planning": lambda cfg: cfg.thinking_model,
    "report": lambda cfg: cfg.thinking_model,
    "verification": lambda cfg: cfg.thinking_model,
}
```

And `AgentGate` controls:

- which tools allowed,
- system prompt per model_alias (from `configs/model.yaml`).

Right now the runner doesn’t call the agent layer at all: it calls `tools.evaluate_p*` directly.

**Upgrade pattern**:

- For _search/planning_:

  - Use a K2-Thinking agent (`role="planning"`) that:

    - Reads the current P3 summary + stage history from RAG / world_model.
    - Decides which proposals to generate (parameters) via `make_boundary` tool.
    - Chooses when to call `evaluate_p1/p2/p3`.

- For _screening_:

  - Use a K2-Instruct agent (`role="screen"`) limited to:

    - local perturbations of promising designs,
    - simple tool calls (`evaluate_p3`, no RAG).

- For _reporting_:

  - Use a `role="report"` agent that only has `log_citation` and `write_report` enabled.

Concretely, you’d:

- Add an “agent driver” layer (e.g. `ai_scientist/planner.py`) that:

  1. Calls `provision_model_tier(role="planning")`.
  2. Builds a tool call spec from `tools_api`.
  3. Sends it through `model_provider.build_chat_request` to K2 backend.
  4. Interprets the tool outputs and feeds them into the runner.

This transforms the system from “Python chooses seeds, LLM only for reports” into “LLM plans experiments in the loop.”

### 3.2. Give the agent structured context instead of raw text

You already have:

- RAG (`rag.retrieve`),
- compact `P3Summary`,
- stage history,
- pareto archive snapshots.

Include these as **structured inputs** to the planning agent:

- Provide top-K pareto entries (gradient, aspect, feasibility, seed, design_hash).
- Provide last N cycle summaries (objective, hv).
- Provide a few RAG-retrieved chunks from the Proxima paper / docs.

Prompt pattern:

> You are the planning agent for P3.
> See structured context:
>
> - current P3 summary: `<json>`
> - last 3 cycles: `<json>`
> - retrieved baseline snippets: `<text>`
>   Your job: propose `N` new designs by calling tool `make_boundary` with sensible `r_cos/z_sin` perturbations around the best and most diverse Pareto entries.

Use tools_api schemas for `make_boundary` to force well-formed parameters.

---

## 4. Sharpen stage-gating & budgeting logic

Your stage gates are already configurable via `StageGateConfig`, but logic is simple:

- S1→S2:

  - either feasibility margin small enough, or
  - relative objective improvement ≥ threshold.

- S2→S3:

  - either HV delta ≤ threshold, or
  - cycles exhausted.

### 4.1. Add safety checks on “fake” progress

Right now hypervolume improvements might be noise. You can:

- Require **minimum number of feasible points** before allowing S2→S3:

```python
if p3_summary.feasible_count < MIN_FEASIBLE_FOR_S3:
    return False
```

- Track a **moving average** of hv improvements; only gate when the _average_ improvement < epsilon across lookback.

### 4.2. Make budgets adaptive

In `_run_cycle`, `screen_evals_per_cycle` and `max_high_fidelity_evals_per_cycle` are fixed config.

You can adapt them:

- If hv improves a lot, temporarily increase promote_top_k or high-fidelity budget.
- If hv stalls and feasibility is poor, allocate more budget to S1 (screening / exploration).

Implementation:

- Maintain running stats in `stage_history` or a separate table.
- Derive per-cycle “bonus budget multipliers” fed into `_run_cycle`.

---

## 5. System-level performance improvements

### 5.1. Parallelism / Process vs Thread pool

`_evaluate_stage` currently uses:

```python
executor_cls = ThreadPoolExecutor if budgets.pool_type == "thread" else ProcessPoolExecutor
```

But `forward_model.forward_model` is CPU-heavy; threads in CPython won’t parallelize CPU much due to GIL unless forward_model is fully in C / releases GIL. Try:

- Default `pool_type="process"` for heavy physics.
- Limit `n_workers` ≈ number of physical cores (or set from env).
- If `forward_model` already uses internal multithreading, avoid oversubscription by setting OMP_NUM_THREADS=1 for the pool.

### 5.2. Smarter cache usage

You have a global cache in `tools`:

- `_EVALUATION_CACHE`
- `get_cache_stats(stage)`

But you only:

- clear it once at `run()` start,
- never inspect hit/miss counts.

Ideas:

- Log `get_cache_stats` each cycle; if hits are low, consider:

  - canonicalizing params better (avoid tiny jitter that changes hash),
  - reusing proposals from previous cycles more aggressively.

- For repeated seeds or re-evaluations (e.g., verification runs), always pass `use_cache=True` (you already do for main calls; verification uses `use_cache=False` correctly).

---

## 6. RAG & reporting as levers for agent quality

`reporting._collect_positioning_quotes` already uses RAG to pull context from the challenge docs for the **reports**.

You can extend this idea to the **planning agent**:

- Add a new OpenAI-style tool `retrieve_rag`:

  ```python
  {
    "name": "retrieve_rag",
    "description": "Retrieve up to k chunks from local challenge docs relevant to query.",
    "parameters": { ... }
  }
  ```

- Allow this tool in `AgentGate` for `role="planning"`.

Then the agent can:

- Ask: “good baseline hypervolume strategies”,
- Get relevant doc snippets,
- Ground its planning in the challenge’s recommended methods.

This should improve “reasoned” experiment design vs hallucinated physics.

---

## 7. Quick correctness / hygiene fixes

Two concrete things worth fixing:

1. **Duplicate field in `_P3Entry`** (bug):

   ```python
   @dataclass(frozen=True)
   class _P3Entry:
       gradient: float
       aspect: float
       seed: int
       evaluation: Mapping[str, Any]
       feasibility: float
       design_hash: str
       design_hash: str   # <- duplicate
   ```

   This will crash or override the field; remove the duplicate.

2. **PEFT environment switch**:

   Make sure `AI_SCIENTIST_PEFT` usage is documented in `configs/model.yaml` / README so you don’t accidentally run different behaviors without noticing.

---

## 8. If you want a concrete “next steps” plan

If you want something actionable _right now_, I’d do them in this order:

1. **Hook up `normalized_constraint_distance_sampler`** into `_run_cycle` to bias proposals around near-feasible P3 designs.
2. Implement a minimal surrogate in Python using `world_model.surrogate_training_data` and use it to rank candidate jitters before evaluation.
3. Add a simple offline script that reads preference logs and updates LoRA weights, and modify `adapter` to actually load those weights.
4. Introduce a “planning agent” tier that:

   - sees P3 summary + Pareto archive + a few RAG chunks,
   - calls `make_boundary` + `evaluate_p3` tools to propose candidate params.

---

Nice, let’s wire this thing up. I’ll give you **drop-in code** you can paste into `runner.py`, with comments where you need to adapt it to your actual `world_model` APIs.

I’ll do two things:

1. Add a **constraint-aware proposal generator** that calls `normalized_constraint_distance_sampler` and mixes it with random proposals.
2. Add a **tiny NumPy linear surrogate** that ranks candidates and keeps only the best `screen_evals_per_cycle` before evaluating.

---

## 1. New helpers in `runner.py`

Put these near the top of `runner.py` (after imports), adjusting import paths to match your repo:

```python
import numpy as np

# NEW: import your constraint sampler
from ai_scientist.sampling import normalized_constraint_distance_sampler  # <-- adjust if needed


# --- Simple param <-> vector utilities for surrogate ---

def _params_to_vector(params: Mapping[str, Any]) -> np.ndarray:
    """
    Flatten a hierarchical params dict into a 1D numeric vector.
    Assumes values are array-like / numeric.
    You may want to customize the key order to match your boundary structure.
    """
    pieces: list[np.ndarray] = []
    for key in sorted(params.keys()):
        value = params[key]
        arr = np.asarray(value, dtype=float).ravel()
        pieces.append(arr)
    return np.concatenate(pieces) if pieces else np.zeros(0, dtype=float)


def _fit_linear_surrogate(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Very simple linear surrogate: solve least-squares w for y ≈ X @ w.
    Returns weight vector w. No regularization; you can upgrade to ridge/LGBM/etc.
    """
    # Add bias term
    X_design = np.concatenate([X, np.ones((X.shape[0], 1), dtype=X.dtype)], axis=1)
    w, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    return w  # shape (d+1,)


def _surrogate_score_candidates(
    candidate_params: Sequence[Mapping[str, Any]],
    w: np.ndarray,
) -> np.ndarray:
    """
    Score candidate params using the linear surrogate weights.
    Higher score = better (e.g. predicted higher hypervolume).
    """
    if not candidate_params:
        return np.zeros((0,), dtype=float)

    X_cand = []
    for p in candidate_params:
        X_cand.append(_params_to_vector(p))
    X_cand = np.asarray(X_cand, dtype=float)

    # Add bias term
    X_design = np.concatenate([X_cand, np.ones((X_cand.shape[0], 1), dtype=X_cand.dtype)], axis=1)
    scores = X_design @ w
    return scores


def _build_surrogate_dataset_for_p3(
    world_model: "WorldModel",  # type: ignore[name-defined]
    max_points: int = 2000,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Build (X, y) for a surrogate that predicts P3 hypervolume from params.
    This uses world_model.surrogate_training_data(...) as a backend.

    You WILL need to adapt the key names here to your actual world_model API.
    """
    # ---- ADAPT THIS BLOCK TO YOUR API ----
    # Example assumption: each record has keys
    #   "params": dict used to run forward model
    #   "metrics": {"hv": float, "feasibility": float, ...}
    records = world_model.surrogate_training_data(target="hv", problem="p3")
    if not records:
        return None

    X_list: list[np.ndarray] = []
    y_list: list[float] = []

    # Take the *latest* max_points samples
    for rec in records[-max_points:]:
        params = rec["params"]          # <-- adjust if different
        metrics = rec["metrics"]        # <-- adjust if different
        hv = metrics.get("hv", None)    # <-- adjust metric name if needed
        if hv is None:
            continue
        X_list.append(_params_to_vector(params))
        y_list.append(float(hv))

    if not X_list:
        return None

    X = np.vstack(X_list)
    y = np.asarray(y_list, dtype=float)
    if X.shape[0] < 20:  # need at least some points
        return None

    return X, y


def _get_constraint_sampler_bases_for_p3(
    world_model: "WorldModel",  # type: ignore[name-defined]
    max_points: int = 256,
) -> tuple[list[Mapping[str, Any]], np.ndarray] | None:
    """
    Extract near-feasible P3 designs + their normalized constraint distances
    for use with normalized_constraint_distance_sampler.
    """
    # ---- ADAPT THIS BLOCK TO YOUR API ----
    # You want: a list of recent P3 evaluations with feasibility info.
    # Example pseudo-API: world_model.p3_evaluations() -> list[{"params":..., "metrics": {...}}]
    history = world_model.p3_evaluations()  # <-- rename / replace with your real accessor
    if not history:
        return None

    base_designs: list[Mapping[str, Any]] = []
    distances: list[float] = []

    # Filter for near-feasible designs: feasibility in (0, some_threshold)
    FEAS_TOL = 0.1
    for rec in history[-max_points:]:
        params = rec["params"]                # <-- adjust
        metrics = rec["metrics"]              # <-- adjust
        feas = metrics.get("feasibility", None)  # or use "max_constraint_violation"
        if feas is None:
            continue
        # We want *small but positive* distances: just outside feasibility
        if 0.0 < feas < FEAS_TOL:
            base_designs.append(params)
            distances.append(float(feas))

    if not base_designs:
        return None

    distances_arr = np.asarray(distances, dtype=float)
    # Normalize to [0, 1] so sampler can treat them as relative distances
    if distances_arr.ptp() > 0:
        distances_arr = (distances_arr - distances_arr.min()) / distances_arr.ptp()
    else:
        distances_arr = np.ones_like(distances_arr)

    return base_designs, distances_arr
```

Again: **the key pieces you must adapt** are:

- `world_model.surrogate_training_data(target="hv", problem="p3")` record format.
- `world_model.p3_evaluations()` or whatever method you actually have for history.

The rest is self-contained.

---

## 2. New proposal generator that mixes constraint + random + surrogate

Replace your current “make random candidates” logic with a new helper. Add this function in `runner.py` (below the helpers above):

```python
def _propose_p3_candidates_for_cycle(
    cfg: "AIConfig",                     # type: ignore[name-defined]
    world_model: "WorldModel",          # type: ignore[name-defined]
    cycle_index: int,
) -> list[Mapping[str, Any]]:
    """
    Propose candidate params for this cycle by mixing:
      - constraint-aware perturbations around near-feasible designs
      - pure random designs
      - optional surrogate ranking on top
    """
    budgets = cfg.budgets
    total_needed = budgets.screen_evals_per_cycle

    rng = np.random.default_rng(cfg.random_seed + cycle_index)

    # -----------------------
    # 1) Create a pool of candidates
    # -----------------------
    candidates: list[Mapping[str, Any]] = []

    # 1a) Constraint-based proposals (exploitation around near-feasible)
    constraint_bases = _get_constraint_sampler_bases_for_p3(world_model)
    if constraint_bases is not None:
        base_designs, norm_dists = constraint_bases
        n_constraint = int(total_needed * 0.7)  # 70% exploitation
        n_constraint = min(n_constraint, total_needed)

        if n_constraint > 0:
            constraint_props = normalized_constraint_distance_sampler(
                base_designs=base_designs,
                normalized_distances=norm_dists,
                proposal_count=n_constraint,
                jitter_scale=0.01,  # tune this hyperparameter
            )
            candidates.extend(constraint_props)

    # 1b) Random proposals (exploration)
    n_random = total_needed - len(candidates)
    if n_random > 0:
        for i in range(n_random):
            seed = (
                cfg.random_seed
                + cycle_index * budgets.screen_evals_per_cycle
                + i
            )
            params = _generate_candidate_params(cfg.boundary_template, seed)
            candidates.append(params)

    # If we somehow still have fewer than needed, pad with more random
    while len(candidates) < total_needed:
        seed = rng.integers(0, 2**31 - 1)
        candidates.append(_generate_candidate_params(cfg.boundary_template, int(seed)))

    # -----------------------
    # 2) Optional surrogate ranking on top
    # -----------------------
    surrogate_data = _build_surrogate_dataset_for_p3(world_model)
    if surrogate_data is None:
        # Not enough history – return candidates as-is
        return candidates

    X, y = surrogate_data
    try:
        w = _fit_linear_surrogate(X, y)
    except np.linalg.LinAlgError:
        # Fallback if something goes wrong in least squares
        return candidates

    scores = _surrogate_score_candidates(candidates, w)
    # Higher score = better. Keep the top `total_needed`.
    order = np.argsort(scores)[::-1]
    top_indices = order[:total_needed]
    ranked_candidates = [candidates[i] for i in top_indices]

    return ranked_candidates
```

Now `_propose_p3_candidates_for_cycle`:

- Uses **constraint sampler** if history exists.
- Fills the rest with random.
- If there is enough P3 history, **trains a tiny linear surrogate** to predict hv and ranks candidates.

---

## 3. Wire it into `_run_cycle`

Find your current `_run_cycle` in `runner.py`. You probably have something like (simplified):

```python
def _run_cycle(cfg, cycle_index, world_model, adapters):
    rng = np.random.default_rng(cfg.random_seed + cycle_index)
    budgets = cfg.budgets

    # ...
    candidate_seeds = [
        cfg.random_seed + cycle_index * cfg.budgets.screen_evals_per_cycle + i
        for i in range(cfg.budgets.screen_evals_per_cycle)
    ]
    candidates = [
        _generate_candidate_params(cfg.boundary_template, seed)
        for seed in candidate_seeds
    ]

    p3_results = _evaluate_stage(
        stage=3,
        params=candidates,
        cfg=cfg,
        world_model=world_model,
        adapters=adapters,
    )
    # ...
```

Replace the **candidate generation block** with a call to the new helper:

```python
def _run_cycle(cfg, cycle_index, world_model, adapters):
    rng = np.random.default_rng(cfg.random_seed + cycle_index)
    budgets = cfg.budgets

    # ... all your existing bookkeeping / logging ...

    # --- OLD ---
    # candidate_seeds = [
    #     cfg.random_seed + cycle_index * cfg.budgets.screen_evals_per_cycle + i
    #     for i in range(cfg.budgets.screen_evals_per_cycle)
    # ]
    # candidates = [
    #     _generate_candidate_params(cfg.boundary_template, seed)
    #     for seed in candidate_seeds
    # ]

    # --- NEW: constraint-aware + surrogate-ranked proposals ---
    candidates = _propose_p3_candidates_for_cycle(cfg, world_model, cycle_index)

    p3_results = _evaluate_stage(
        stage=3,
        params=candidates,
        cfg=cfg,
        world_model=world_model,
        adapters=adapters,
    )

    # ... rest of your cycle logic unchanged ...
```

That’s it: structurally, everything else (evaluation, Pareto archive updates, stage gating, reporting) stays the same. You’ve just made the **proposal step** much smarter.

---

## 4. What you’ll likely need to tweak

You’ll almost certainly have to adjust:

1. **`world_model` accessors**:

   - Replace `world_model.surrogate_training_data(target="hv", problem="p3")`’s expected record structure with your actual one.
   - Replace `world_model.p3_evaluations()` with whatever you actually have (could be `world_model.stage_history(stage=3)` or a dedicated getter).

2. **Metric names**:

   - I assumed `metrics["hv"]` and `metrics["feasibility"]`. If your repo uses e.g. `"hypervolume"` or `"max_constraint_violation"`, adjust.

3. **Ratios / hyperparams**:

   - `n_constraint = 0.7 * total_needed`
  - `jitter_scale=0.01`
  - `FEAS_TOL = 0.1`
    These are starting points; you can tune them based on how quickly you see feasibility improve vs exploration.

## Phase 6 – Documentation & DX

### 6.1 Refresh `improvement-plan.md`

- Rephrase this file’s introduction so the actionable steps at the top point directly to the checklist in `ai_scientist/roadmap.md` (the Phase 6 subsection there in particular). That way the roadmap becomes the canonical “what to do” list and this document explains “why we’re doing it.”
- In each numbered section above, insert inline references back to the rationale paragraphs that motivated that work (e.g., link the surrogate/curriculum ideas to “## 1. Make the search much smarter than pure random,” the PEFT loop to “## 2. Actually use PEFT / LoRA for agents,” and the planner descriptions to “## 3. Improve the multi-agent architecture itself”). These cross-links ensure someone reading this file can trace every recommendation back to the motivating context.
- Define success by stating that: (1) the top summary references `ai_scientist/roadmap.md`, (2) each section carries a rationale cross-link, and (3) Phase 6’s documentation pointers are embedded in the narrative rather than left as checklist items elsewhere.

### 6.2 Author onboarding snippet

- Create `docs/ai_scientist_onboarding.md` with explicit quickstart commands (activating the `.venv`, running `python -m ai_scientist.runner --config ... --problem ...`, invoking `hatch` targets, etc.), the most important environment variables (like `AI_SCIENTIST_PEFT`, any budget override flags, planner selection overrides), and a concise explanation of how the new agent planner differs from the legacy deterministic runner mentioned in “## 3. Improve the multi-agent architecture itself.”
- Reference `ai_scientist/roadmap.md` in that doc and spell out that contributors should address the phases in order (Phase 1 → Phase 6) so they understand how each stack of work builds on the previous phase.
- Close the onboarding snippet with a “what success looks like” paragraph: a reader can run both the agent planner and deterministic runner, knows where each phase’s deliverables live, and can locate the Phase 6 docs that describe the reasoning behind the current architecture.
