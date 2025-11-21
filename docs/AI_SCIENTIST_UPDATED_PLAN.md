# AI Scientist – Unified Implementation Plan (v0.1)

Purpose: merge prior plans and feedback into a concise, actionable blueprint to make the AI scientist autonomously find feasible P2/P3 stellarator designs, using the existing `ai_scientist` scaffold.

## Guiding Principles
- Single source of truth for constraints and margins shared by evaluators, surrogates, and logging.
- Feasibility-first, then objective (HV) optimization.
- Multi-fidelity VMEC scheduling with strict budgets and caching.
- Lightweight, swappable surrogates (start simple; keep interface stable).
- Persistent world-model (SQLite) for reproducibility, dedupe, and analytics.
- Deterministic, bounded, and observable runs (timeouts, seeds, logs).

## Fast Wins (do in order)
1) **Constraint margins helper**: implement `compute_constraint_margins(metrics, problem)` using the exact thresholds from `evaluate_p2/p3`; unit-test it. Use everywhere (surrogate labels + feasibility flag).
2) **Minimal SQLite world-model**: tables `boundaries`, `evaluations`, `cycles`; helpers for add/log/query; tests for read/write roundtrip.
3) **Surrogate interface upgrade**: replace `SimpleSurrogateRanker` with a pluggable `SurrogateBundle` (vectorizer + metric predictor + feasibility predictor). Start with sklearn RF/ET + logistic calibrator; keep PyTorch ensemble optional via flag.
4) **Feasibility loop in runner**: wire a feasibility phase that samples seeds + Gaussian Fourier perturbations, ranks via surrogate feasibility prob, evaluates with `stage="screen"`, logs per-cycle feasible count and HV (when feasible points exist).

## Architecture (incremental detail)
### Constraint handling
- Define margins as signed violations (positive = violates) per constraint; share helper between evaluation and surrogate training.
- Store both metrics and margins in DB; expose `max_violation` and `l2_distance`.

### Candidate sampling
- Baseline: (a) template/seed JSONs, (b) Gaussian low-mode perturbations around seeds or near-feasible archive entries.
- Optional: hook in near-axis / rotating-ellipse generators once verified in repo.
- Oversample then rank by predicted feasibility; keep some uncertainty/exploration weight.

### Surrogates
- `BoundaryScaler`: normalize concatenated `[r_cos, z_sin]` (and optional extras); store mean/std.
- `SurrogateBundle`: `predict_metrics`, `predict_constraint_margins_and_prob`, `update_online`.
- Default impl: sklearn RF/ET regressors + classifier (with probability); PyTorch ensemble behind a feature flag.
- Retrain policy: trigger when ≥K new labeled points or every N cycles; hard timeout; run in separate process/thread if needed.

### World-model (SQLite)
- Tables:
  - `boundaries(hash, p, nfp, r_cos_blob, z_sin_blob, source, parent_id, created_at)`
  - `evaluations(boundary_id, stage, vmec_status, runtime_sec, metrics..., margins..., is_feasible, created_at)`
  - `cycles(cycle_idx, phase, p, new_evals, new_feasible, cumulative_feasible, hv, notes, created_at)`
- Hashing: canonicalize Fourier coefficients (e.g., round to 1e-6) before hashing to reduce dup noise.
- Provide queries: feasible count, best feasible, near-feasible (by l2 margin), recent candidates.
- Add schema version field or pragma for future compatibility.

### Multi-fidelity + budgets
- Stages: `screen` (very low), `refine` (low), `final` (default/high).
- `BudgetController` uses HV delta, feasibility rate, cache hit rate to adapt:
  - Low feasibility → allocate more `screen` exploration.
  - High feasibility but flat HV → promote more to `refine/final`.
- Promotion rule (initial): take top `promote_top_k` by predicted feasibility, then by objective (P2: L∇B; P3: surrogate HV contribution).

### Feasibility phase (P2/P3)
- Loop per cycle:
  1) Sample 2× budget candidates.
  2) Rank by feasibility prob − distance-to-constraints; keep some uncertainty bonus if available.
  3) Evaluate top-K at `stage="screen"`; log VMEC status; treat failures as infeasible with max violation.
  4) Update surrogates with new data (respect retrain cadence/timeout).
  5) Record cycle summary (feasible count, new feasible, HV if any).
- Exit when feasible count ≥ target (e.g., P2: 5–10, P3: 3+).

### HV optimization phase
- Choose a concrete selector: start with NSGA-II (from nevergrad/pygmo) or NGOpt; surrogate-evaluate large batch, VMEC top-K per stage.
- Objective:
  - P2: maximize L∇B within feasibility prob ≥ threshold.
  - P3: maximize HV over (A, L∇B) with feasibility prob penalty.
- Maintain Pareto archive; compute HV each cycle; stop on HV plateau or budget exhaustion.

### Operational guardrails
- Determinism: set and log seeds (python/numpy/jax/torch); log code and schema version.
- VMEC robustness: wrap calls; on failure mark infeasible, log error, and continue.
- Timeouts: per-VMEC call and per-retrain; fail fast, don’t block queue.
- Resource observability: log runtime per stage, retrain time, queue lengths, cache hit rate.

## Deliverables (thin slices)
1) Margin helper + tests (unit).
2) SQLite world-model + IO tests.
3) SurrogateBundle (sklearn impl) + adapter replacing SimpleSurrogateRanker.
4) Feasibility phase loop in runner + logging.
5) Hash canonicalization + cache reuse.
6) HV phase wiring with chosen optimizer + promotion policy.
7) Optional: PyTorch surrogate flag; near-axis sampler once verified.

## Acceptance checks
- Unit tests: margins, world-model IO, HV on toy data, surrogate interface smoke.
- Dry-run: one-cycle feasibility run with tiny budgets completes without crash, logs feasible count/HV.
- Metrics tracked per cycle: feasible count, HV, VMEC failure rate, retrain time, cache hit rate.

## Notes on dependencies
- Default to sklearn-only footprint; gate PyTorch behind a config flag to avoid bloat when not needed.
- Document HF dataset locality/caching strategy if used for offline pretrain.

